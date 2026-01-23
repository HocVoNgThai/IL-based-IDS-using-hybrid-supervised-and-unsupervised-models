import sys, os
import zmq
from io import StringIO
# TIME
import time # Cần thiết cho việc kiểm tra RCVTIMEO
from datetime import datetime

# THR
import queue
import threading
import signal
import gc

# IO
import json
import pandas as pd
from pathlib import Path
from collections import deque

# ========= TOÀN CỤC =========
CURR_DIR = Path.cwd()
# Kích thước timeout (miligiây)
TIMEOUT_MS = 500
BATCH_SIZE = 500000

# ========= IMPORT FROM MINE =====
from src.config.settings import settings
from src.Utils.FlowFlushTransform import FlowFlushTransformer, STANDARD_COLS, COLS_TO_DROP, STANDARD_SCALER_PATH, MINMAX_SCALER_PATH, MINMAX_COLS, SAMPLE_COLS_TO_REMOVE, DO_NOT_TOUCH_COLS, DECIMAL_BIN

from src.Worker.DetectorWorker import DetectorWorker
from src.Worker.FlushWorker import FlushWorker
from src.Worker.LogWorker import LogWorker

from src.Components.Detector import Detectors_Pipeline
from src.Components.Models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.Components.Manager import Manager


# ===========================
# CLASS 
# ===========================
class FlowZmqServer:
    def __init__( self, 
                 bind_addr="tcp://*:5555", 
                 detect_batch_size=10, 
                 detect_queue_size= 100000,
                 log_queue_size = 100000,
                 flush_batch_size=1000, 
                 flush_queue_size = 100000,
                 n_workers= 2, 
                 output_dir="flows_parquet"):
        
        # self.bind_addr = bind_addr
        
        # BATCH SIZE
        self.detect_batch_size = detect_batch_size
        self.flush_batch_size = flush_batch_size
        
        # OUT
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # SOCKET
        self.ctx = zmq.Context() #zmq.Content.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.bind(bind_addr)
        
        # Điều này khiến sock.recv() thoát ra sau 500ms nếu không có tin nhắn
        self.sock.setsockopt(zmq.RCVTIMEO, TIMEOUT_MS)

        # self.flowFlushTransformer = FlowFlushTransformer(
        #     MINMAX_SCALER_PATH, STANDARD_SCALER_PATH, MINMAX_COLS, STANDARD_COLS, decimal_bin=6
        # )
        
        self.curr_index_parquet = 0
        
        # HEADER - RUNNING STATE
        self.header = None
        self.running = True

        # QUEUE
        self.detect_queue = queue.Queue(maxsize = detect_queue_size)
        self.flush_queue = queue.Queue(maxsize=flush_queue_size)
        # self.detect_queue = deque(maxlen= detect_queue_size)
        # self.flush_queue = deque(maxlen= flush_queue_size) # Discard Old Policy
        self.log_queue = queue.Queue(maxsize= log_queue_size)
        
        # PIPE
        ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.75)
        self.mgr = Manager(settings.MODEL_DIR)
        self.mgr.load_models([xgb, ocsvm, ae])
        print(f"[PY] Model State: {xgb.loaded}, {ocsvm.loaded}, {ae.loaded}")
        self.pipelineDetector = Detectors_Pipeline(xgb= xgb, ocsvm=ocsvm , ae= ae)
        
        
        # LẤY HEADER TRƯỚC KHI THỰC SỰ CHẠY
        # msg = self.sock.recv(flags=1)
        # self.header = msg.decode("utf-8")
        # self.header = self.header.split(',')
        self.header = self.fetch_header_from_java()
        
        # WORKERS INITIALIZATION
        self.detect_workers = [
            DetectorWorker(i, self.detect_queue, self.log_queue, self.pipelineDetector, self.header) #, self.alert_callback
            for i in range(n_workers)
        ]
        
        self.flush_workers = [
            FlushWorker(i, self.flush_queue, self.header) for i in range(1)
        ]
        
        self.log_workers = [
            LogWorker(i, self.log_queue) for i in range (1)
        ]
        
        #
        print(f"[PY] ZMQ listening on {bind_addr}")

    # =========================
    # GET header
    # =========================
    def fetch_header_from_java(self, addr="tcp://127.0.0.1:5556"):
        
        req_sock = self.ctx.socket(zmq.REQ)
        req_sock.connect(addr)
        
        # Cấu hình timeout để không bị treo nếu Java chưa bật
        req_sock.setsockopt(zmq.RCVTIMEO, 5000) 
        
        print("[PY] Đang yêu cầu Header từ Java...")
        try:
            req_sock.send_string("GET_HEADER")
            header_raw = req_sock.recv_string()
            print("[PY] Đã lấy được Header thành công.")
            return header_raw.split(',')
        
        except zmq.Again:
            print("[PY] Lỗi: Java không phản hồi Header sau 5 giây.")
            return None
        
        finally:
            req_sock.close()
        
    # =========================
    # Core loop
    # =========================
    def run(self):
        print("[PY] Server running, press Ctrl+C to stop...")
        
        # START WORKER THREAD
        self.start_workers()
    
        print("[PY] ZMQ SERVER - HEADER: ", self.header)
        # LOOP
        while self.running:
            try:
                msg = self.sock.recv(flags=zmq.NOBLOCK)
                self.handle_message(msg)
                
            except zmq.error.Again:
                # Xảy ra khi timeout (RCVTIMEO) hết hạn và không có tin nhắn. 
                # Đây là cách chúng ta cho phép vòng lặp kiểm tra self.running.
                continue 
                
            except zmq.ZMQError as e:
                # Xảy ra khi socket bị đóng từ bên ngoài (ví dụ: ZContext.term())
                # Hoặc khi có lỗi ZMQ khác
                if self.running: # Nếu lỗi không phải do lệnh tắt, log nó
                     print(f"[ERROR] ZMQ Error: {e}")
                break
                
            except Exception as e:
                print(f"[ERROR] General Error: {e}")
                self.running = False
        
        self.close()

    # =========================
    # Message handler
    # =========================
    def handle_message(self, msg: bytes):
        
        try:
            self.detect_queue.put(msg, timeout=0.2)
            self.flush_queue.put(msg, timeout=0.2)
        except queue.Full:
            print("[WARN] Queue full - Dropping batch to maintain real-time speed")

        except Exception as e:
            print(f"[ERROR] handle_message: {e}")

    # =========================
    # WORKERS
    # =========================
    def start_workers(self):
        for w in self.detect_workers:
            w.start()
        for w in self.log_workers:
            w.start()
        for w in self.flush_workers:
            w.start()
        
    # =========================
    # Graceful shutdown
    # =========================
    def stop(self, *_):
        if self.running:
            print("\n[PY] Shutdown signal received. Starting graceful termination...")
            self.running = False
            # Nếu ZMQ.term() được gọi, nó sẽ hủy bỏ sock.recv()
            # Tùy thuộc vào phiên bản ZMQ/Python, có thể cần gọi ctx.term()
        
    def close(self):
        print("[PY] Closing worker, zmq, ...")
        
        # ================================
        # STOP WORKERS
        # ================================ 
        for worker in self.detect_workers:
            worker.stop()
        for worker in self.log_workers:
            worker.stop()
        for worker in self.flush_workers:
            worker.stop()
        
        for w in self.detect_workers:
            w.join(timeout=5)

        for w in self.log_workers:
            w.join(timeout=5)
        
        for w in self.flush_workers:
            w.join(timeout=5)
        
        gc.collect()
        print("[PY] All detector workers stopped")
        
        # ====================
        # CLOSE SOCKET
        # ====================
        
        try:
            self.sock.close(linger=1000) #(mili sec)
        except ValueError as e:
            print("[PY] ZMQ Socket close error: ", e)
            
        try:
            # ctx.term() sẽ hủy bất kỳ tác vụ chặn nào (như sock.recv())
            self.ctx.term()
        except ValueError as e:
            print("[PY] ZMQ Content terminate error: ", e)
            
        gc.collect()
        print("[PY] ZMQ closed")
        return

# =========================
# Entry point
# =========================
if __name__ == "__main__":
    server = FlowZmqServer(
        bind_addr="tcp://*:5555",
        detect_batch_size=256,
        detect_queue_size= 100000, # 
        log_queue_size= 100000,
        flush_batch_size= 1280,
        flush_queue_size= 1000000,
        n_workers=5
    )

    # Đăng ký hàm stop cho các tín hiệu
    signal.signal(signal.SIGINT, server.stop)
    signal.signal(signal.SIGTERM, server.stop)

    server.run()