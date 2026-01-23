# Standard libs
try:
    import os, sys
    import time
    import fcntl
    from pathlib import Path
    from datetime import datetime

    # 3rd libs
    import json
    import gc

    # Local Import
    from src.config.incremental_config import incremental_settings
    from src.Components.Incremental import IncrementalLearning

    # CONST
    LOG_PREFIX = "[IL-JOB]"
    MIN_INTERVAL_SEC = incremental_settings.MIN_INTERVAL_SEC
    LOCK_FILE = incremental_settings.IL_LOCK_FILE
    STATE_FILE = incremental_settings.IL_STATE_FILE

except Exception as e:
    with open("/tmp/il_error.log", "a") as f:
        f.write(f"Crash at import: {str(e)}\n")
    sys.exit(1)

# ===== LOCK =====

# Toàn cục
lock_file_handle = None

def acquire_lock_old():
    if LOCK_FILE.exists():
        print(f"{LOG_PREFIX} Another IL job is running. Exit.")
        return False
    LOCK_FILE.touch()
    return True

def acquire_lock():
    global lock_file_handle
    try:
        # Mở file lock (tạo mới nếu chưa có)
        lock_file_handle = open(LOCK_FILE, 'w')
        
        # Cố gắng lấy lock độc quyền (LOCK_EX) 
        # Chế độ Non-blocking (LOCK_NB): Nếu đang có người giữ lock, trả về lỗi ngay thay vì đợi
        fcntl.flock(lock_file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # Ghi PID vào file để tiện theo dõi (Optional)
        lock_file_handle.write(str(os.getpid()))
        lock_file_handle.flush()
        return True
    
    except (IOError, OSError):
        print(f"{LOG_PREFIX} Another IL job is running. Exit.")
        if lock_file_handle:
            lock_file_handle.close()
        return False

def release_lock():
    global lock_file_handle
    try:
        if lock_file_handle:
            # Giải phóng lock và đóng file
            fcntl.flock(lock_file_handle, fcntl.LOCK_UN)
            lock_file_handle.close()
    except Exception as e:
        print(f"{LOG_PREFIX} Error releasing lock: {e}")

def release_lock_old():
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


# ===== STATE =====
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    
    return {
        "last_retrain": {},
        "model_version": 0
    }

def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))

# ===== IL LOGIC =====
def should_retrain(state, current_update_time: datetime):
    if incremental_settings.enable_training:
        
        if len(state["last_retrain"]) < 1:
            return True
        last = datetime.strptime(state["last_retrain"][str(state["model_version"])], "%Y-%m-%d %H:%M:%S")
        return (current_update_time - last).total_seconds() >= MIN_INTERVAL_SEC
    return False

def run_incremental_learning(state, last_update_time, current_update_time):
    print(f"{LOG_PREFIX} Running incremental learning...")
    
    il_job = IncrementalLearning(last_update_time=last_update_time,
                                 current_update_time=current_update_time,
                                 il_data_dir=incremental_settings.IL_DATA_DIR,
                                 il_logs_dir=incremental_settings.IL_LOGS,
                                 current_index= state["model_version"]
                                 )
    
    if il_job.should_train:
        il_job.workflow()
        time.sleep(2)  # mock
        print(f"{LOG_PREFIX} Incremental learning done.")
        return True
    
    else:
        print(f"{LOG_PREFIX}  Skip retrain (No data fouded).")
    
    return False
   

# ===== MAIN =====
def main():
    if not acquire_lock():
        return 0
    
    try:
        state = load_state()
        current_update_time = datetime.now()
        
        if not should_retrain(state, current_update_time=current_update_time):
            print(f"{LOG_PREFIX} Skip retrain (interval not reached)")
            return 0
        
        last_update_time = datetime.strptime(
            state["last_retrain"][str(state["model_version"])], 
            "%Y-%m-%d %H:%M:%S"
        ) if len(state["last_retrain"]) > 0 else datetime.min
        
        # print("Come")
        res = run_incremental_learning(state, last_update_time, current_update_time)
        
        # NẾU CÓ RETRAIN - lưu state
        if res:
            state["model_version"] += 1
            state["last_retrain"][state["model_version"]] = datetime.strftime(current_update_time, "%Y-%m-%d %H:%M:%S")
            save_state(state)
            print(f"{LOG_PREFIX} Model updated to v{state['model_version']}")

        return 0
    
    except Exception as e:
        print(f"{LOG_PREFIX} ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        release_lock()

if __name__ == "__main__":
    print(f"{LOG_PREFIX} Triggered at {datetime.now()}")
    sys.exit(main())
