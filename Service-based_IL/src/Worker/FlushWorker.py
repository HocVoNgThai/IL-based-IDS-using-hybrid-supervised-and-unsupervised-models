# STANDARD LIBS
import os, sys
import threading
import queue
import time
from io import StringIO

# 3rd LIBS
import joblib
import zmq
import gc
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Local Import
from src.Utils.FlowFlushTransform import FlowFlushTransformer, STANDARD_COLS, STANDARD_SCALER_PATH, MINMAX_SCALER_PATH, MINMAX_COLS

class FlushWorker(threading.Thread):
    def __init__(self, worker_id, task_queue, header, batch_size = 100):
        super().__init__(daemon=True)

        # WORKER - QUEUE
        self.worker_id = worker_id
        self.q = task_queue
        
        # BATCH BUFFER
        self.batch_size= batch_size
        self.batch_buffer = []
        # self.buffer_lock = threading.Lock()
        self.last_flush_time = time.time()
        
        # HEADER pd
        self.header = header

        # FlowTransform
        self.flowFlushTransformer = FlowFlushTransformer(
            MINMAX_SCALER_PATH, STANDARD_SCALER_PATH, MINMAX_COLS, STANDARD_COLS, decimal_bin=6, header= header
        )
                       
        self.running = True

    def run(self):
        print(f"[Flush-{self.worker_id}] started")
        
        try:
            while self.running: # self.running
                try:
                    data = self.q.get(timeout=0.2)
                    self.batch_proc(data)
                    self.q.task_done()
                except queue.Empty:
                    if time.time() - self.last_flush_time > 30:
                        self.flush()
                    continue
                except Exception as e:
                    print(f"[Flush-{self.worker_id}] error:", e)
                
            
            print(f"[Flush-{self.worker_id}] Exiting Signal received! Flushing data left..." )  
            while not self.q.empty():
                try:
                    data = self.q.get(timeout=0.2)
                    self.batch_proc(data)
                    self.q.task_done()
                except Exception as e:
                    print(f"[Flush-{self.worker_id}] error:", e)
                    
        finally:    
            print(f"[Flush-{self.worker_id}] Exit cleanly!")
            self.flush()
        
    def batch_proc(self, raw_bytes):
        try:
            lines = raw_bytes.decode('utf-8').strip().split('\n')
            data = [l.split(',') for l in lines]
            df = pd.DataFrame(data, columns=self.header)
            
            if df.empty:
                return
            
            self.batch_buffer.append(df)    
            # Kiểm tra nếu đã đủ số lượng batch
            if len(self.batch_buffer) >= self.batch_size:
                self.flush()
            
        except Exception as e:
            print(f"[Flush-{self.worker_id}] Parse error: {e}")
            
        
    def flush(self):
        # with self.buffer_lock:
        if len(self.batch_buffer)<1 or not self.batch_buffer:
            return
        
        combined_df = pd.concat(self.batch_buffer, ignore_index=True)
        self.batch_buffer.clear()
        self.last_flush_time = time.time()

        # Gọi hàm lưu file (Parquet/JSONL)
        fname = self.flowFlushTransformer.flush(combined_df)
        if fname:
            print(f"[Flush-{self.worker_id}] ✔ Saved flows → {fname}")
        else:
            print(f"[Flush-{self.worker_id}] ✘ Flush failed (Transformer returned None)")
        
        return
            
            
    def flush_old(self):
        # with self.buffer_lock:
        if len(self.batch_buffer) < 1:
            print(f"[Flush-{self.worker_id}] X Flush Buffer Empty !")
            return
    
        df = pd.concat(self.batch_buffer, ignore_index=True)
        self.batch_buffer.clear()
            
            
        fname = self.flowFlushTransformer.flush(df)
        if fname != None:
            print(f"[Flush-{self.worker_id}] ✔ Flushed {len(df)} batch  → {fname}")
            
        else:
            print("f[Flush-{self.worker_id}] X Cannot Save !")
    
        del df
        
        return

    def stop(self):
        self.running = False