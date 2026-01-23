# src/utils.py
import os
import time
import psutil
import threading
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import warnings

try: import GPUtil
except ImportError: GPUtil = None

warnings.filterwarnings('ignore')

# ==================== 1. RESOURCE TRACKER ====================
class ResourceTracker:
    def __init__(self):
        self.tracking = False
        self.history = {'time': [], 'cpu': [], 'ram': [], 'gpu': [], 'gpu_mem': []}
        self.thread = None
        self.start_time = 0

    def _monitor(self):
        while self.tracking:
            current_time = time.time() - self.start_time
            self.history['time'].append(current_time)
            self.history['cpu'].append(psutil.cpu_percent())
            self.history['ram'].append(psutil.virtual_memory().percent)
            
            gpu_load = 0; gpu_mem = 0
            try:
                if GPUtil and GPUtil.getGPUs(): 
                    gpu = GPUtil.getGPUs()[0]
                    gpu_load = gpu.load * 100
                    gpu_mem = gpu.memoryUsed
            except: pass
            
            self.history['gpu'].append(gpu_load)
            self.history['gpu_mem'].append(gpu_mem)
            time.sleep(1)

    def start(self):
        self.tracking = True
        self.history = {'time': [], 'cpu': [], 'ram': [], 'gpu': [], 'gpu_mem': []}
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.tracking = False
        if self.thread: self.thread.join()
        duration = time.time() - self.start_time
        
        # [UPDATE] Tính thêm avg_gpu để vẽ biểu đồ
        avg_gpu = np.mean(self.history['gpu']) if self.history['gpu'] else 0
        
        return {
            'duration': duration,
            'avg_cpu': np.mean(self.history['cpu']) if self.history['cpu'] else 0,
            'max_ram': np.max(self.history['ram']) if self.history['ram'] else 0,
            'avg_gpu': avg_gpu, # New metric
            'history': self.history
        }

# ==================== 2. IL METRICS (Nâng cao) ====================
class ILMetrics:
    def __init__(self): 
        self.R = {} 
        # Lưu lịch sử để vẽ biểu đồ biến thiên
        self.history = {'Avg_Acc': [], 'BWT': [], 'Forgetting': []}

    def record(self, tr_sess, te_sess, acc):
        if tr_sess not in self.R: self.R[tr_sess] = {}
        self.R[tr_sess][te_sess] = acc

    def calculate_metrics(self, current_step):
        # 1. Avg Accuracy
        acc_list = [self.R[current_step].get(i, 0) for i in range(current_step + 1)]
        avg_acc = np.mean(acc_list)
        
        # 2. BWT & Forgetting
        if current_step > 0:
            bwt = np.mean([self.R[current_step][i] - self.R[i][i] for i in range(current_step)])
            f_vals = []
            for j in range(current_step):
                max_prev = max([self.R[l].get(j, 0) for l in range(current_step)])
                f_vals.append(max_prev - self.R[current_step][j])
            forgetting = np.mean(f_vals)
        else:
            bwt = 0.0; forgetting = 0.0
            
        self.history['Avg_Acc'].append(avg_acc)
        self.history['BWT'].append(bwt)
        self.history['Forgetting'].append(forgetting)
        return avg_acc, bwt, forgetting

# ==================== 3. PLOTTING FUNCTIONS (All Types) ====================

def plot_resource_usage(log, save_path):
    """Biểu đồ cột tổng hợp (Time, CPU, RAM, GPU) với logic hiển thị text thông minh không chồng chéo"""
    Scenarios = list(log.keys())
    times = [log[s]['duration'] for s in Scenarios]
    cpus = [log[s]['avg_cpu'] for s in Scenarios]
    rams = [log[s]['max_ram'] for s in Scenarios]
    gpus = [log[s].get('avg_gpu', 0) for s in Scenarios]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # --- 1. Bar Chart (Time) ---
    bars = ax1.bar(Scenarios, times, color='#87CEEB', alpha=0.3, label='Time (s)', width=0.5)
    ax1.set_ylabel('Time (s)', color='blue', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # In giá trị Time (nằm giữa cột)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, 
                 f'{height:.1f}s', ha='center', va='center', color='blue', fontweight='bold', fontsize=11)

    # --- 2. Line Charts (CPU, RAM, GPU) ---
    ax2 = ax1.twinx()
    
    # Vẽ các đường
    line_cpu = ax2.plot(Scenarios, cpus, 'r-o', label='CPU %', linewidth=2.5, markersize=9, markerfacecolor='white', markeredgewidth=2)
    line_ram = ax2.plot(Scenarios, rams, 'g-s', label='RAM %', linewidth=2.5, markersize=9, markerfacecolor='white', markeredgewidth=2)
    line_gpu = ax2.plot(Scenarios, gpus, 'm-^', label='GPU %', linewidth=2.5, markersize=9, markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_ylabel('Usage %', color='black', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 115) 
    ax2.tick_params(axis='y', labelcolor='black')

    # --- LOGIC HIỂN THỊ TEXT THÔNG MINH (Smart Text Placement) ---
    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)

    for i, (c, r, g) in enumerate(zip(cpus, rams, gpus)):
        # Khoảng cách tối thiểu để tránh dính nhau
        vertical_spacing = 6 
        
        # 1. Xử lý CPU và RAM (Tránh đè nhau theo chiều dọc)
        # Nguyên tắc: Cái nào cao hơn thì text nằm trên, cái nào thấp hơn thì text nằm dưới
        if c >= r:
            # CPU cao hơn -> Text CPU ở trên, RAM ở dưới
            ax2.text(i, c + 4, f'{c:.1f}%', color='#D32F2F', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)
            ax2.text(i, r - 4, f'{r:.1f}%', color='#388E3C', ha='center', va='top', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)
        else:
            # RAM cao hơn -> Text RAM ở trên, CPU ở dưới
            ax2.text(i, r + 4, f'{r:.1f}%', color='#388E3C', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)
            ax2.text(i, c - 4, f'{c:.1f}%', color='#D32F2F', ha='center', va='top', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)

        # 2. Xử lý GPU (Tránh đè vào CPU/RAM)
        if g > 0.5:
            # GPU luôn đẩy sang bên phải một chút (offset x = 0.15)
            # Điều này giúp nó thoát khỏi trục dọc của CPU/RAM
            ax2.text(i + 0.15, g, f'{g:.1f}%', color='#9C27B0', ha='left', va='center', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)

    # --- Legend & Title ---
    lines = [bars] + line_cpu + line_ram + line_gpu
    labels = [l.get_label() for l in lines]
    
    ax2.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True, fontsize=11)
    
    plt.title('Resource Usage Summary (Time & Hardware)', fontsize=16, pad=40, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_detailed_resource_usage(history_dict, Scenario_name, save_path):
    """Biểu đồ đường chi tiết theo thời gian"""
    time_axis = history_dict['time']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(time_axis, history_dict['cpu'], label='CPU %', color='red'); ax1.plot(time_axis, history_dict['ram'], label='RAM %', color='green')
    ax1.set_ylabel('Usage (%)'); ax1.set_title(f'Resource Detail - {Scenario_name}'); ax1.legend()
    ax2.plot(time_axis, history_dict['gpu_mem'], label='GPU Mem (MB)', color='purple'); ax2.set_ylabel('MB'); ax2.set_xlabel('Time (s)'); ax2.legend()
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300); plt.close()

def plot_il_metrics_trends(il, save_path):
    """Biểu đồ biến động F, BWT, AvgAcc với Text Annotations"""
    steps = range(len(il.history['Avg_Acc']))
    labels = [f"Scenario {i}" for i in steps]
    
    plt.figure(figsize=(10, 6))
    
    # Vẽ các đường
    plt.plot(steps, il.history['Avg_Acc'], 'o-', label='Avg Accuracy', linewidth=2, color='#1f77b4')
    plt.plot(steps, il.history['BWT'], 's--', label='Backward Transfer (BWT)', linewidth=2, color='#ff7f0e')
    plt.plot(steps, il.history['Forgetting'], 'x-.', label='Forgetting Measure (F)', linewidth=2, color='#2ca02c')
    
    # In giá trị lên từng điểm
    for i in steps:
        # Avg Acc
        val_acc = il.history['Avg_Acc'][i]
        plt.text(i, val_acc + 0.005, f"{val_acc:.4f}", ha='center', va='bottom', color='#1f77b4', fontweight='bold', fontsize=9)
        
        # BWT
        val_bwt = il.history['BWT'][i]
        plt.text(i, val_bwt - 0.015, f"{val_bwt:.4f}", ha='center', va='top', color='#ff7f0e', fontweight='bold', fontsize=9)
        
        # Forgetting
        val_f = il.history['Forgetting'][i]
        plt.text(i, val_f + 0.005, f"{val_f:.4f}", ha='center', va='bottom', color='#2ca02c', fontweight='bold', fontsize=9)

    plt.title('Incremental Learning Metrics Evolution')
    plt.xlabel('Session')
    plt.ylabel('Score')
    plt.xticks(steps, labels)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi= 300)
    plt.close()

def plot_il_matrix(il, save_path):
    """Heatmap Accuracy"""
    Scenarios = sorted(il.R.keys()); n = len(Scenarios); matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n): matrix[i, j] = il.R[i].get(j, 0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlGnBu', xticklabels=[f'Test C{s}' for s in Scenarios], yticklabels=[f'Train C{s}' for s in Scenarios])
    plt.title('IL Accuracy Matrix'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300); plt.close()

# ==================== PER-CLASS ANALYSIS FUNCTIONS ====================

def analyze_and_plot_class_details(y_true, y_pred, title, save_dir, merge_labels_list=None, hide_unknown=False):
    """
    Tính toán và vẽ Bảng TP/FP/TN/FN (có kèm %) + Biểu đồ Metrics.
    """
    # 1. Gộp nhãn
    y_true_mapped = list(y_true)
    y_pred_mapped = list(y_pred)
    
    if merge_labels_list:
        y_true_mapped = ["Unknown" if lbl in merge_labels_list else lbl for lbl in y_true_mapped]
        y_pred_mapped = ["Unknown" if lbl in merge_labels_list else lbl for lbl in y_pred_mapped]

    # 2. Danh sách nhãn
    all_labels = sorted(list(set(y_true_mapped) | set(y_pred_mapped)))
    
    # 3. Lọc hiển thị
    labels_to_show = set(all_labels)
    if hide_unknown and "Unknown" in labels_to_show:
        labels_to_show.remove("Unknown")
        
    ordered = [l for l in all_labels if l in labels_to_show and l != "Benign" and l != "Unknown"]
    if "Benign" in labels_to_show: ordered.insert(0, "Benign")
    if "Unknown" in labels_to_show: ordered.append("Unknown")
    
    if not ordered: return

    # 4. Tính toán Metrics
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=all_labels)
    metrics_data = []
    
    label_to_idx = {l: i for i, l in enumerate(all_labels)}

    for label in ordered:
        idx = label_to_idx[label]
        TP = cm[idx, idx]
        FP = cm[:, idx].sum() - TP
        FN = cm[idx, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        
        # Metrics
        denom_acc = TP + TN + FP + FN
        acc = (TP + TN) / denom_acc if denom_acc > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        denom_f1 = prec + rec
        f1 = 2 * (prec * rec) / denom_f1 if denom_f1 > 0 else 0
        
        metrics_data.append([label, TP, FP, TN, FN, acc, prec, rec, f1])

    df = pd.DataFrame(metrics_data, columns=["Label", "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1"])
    
    # --- 5. DRAW TABLE (WITH PERCENTAGES) ---
    # Tăng chiều rộng hình để chứa text dài hơn
    fig_tbl, ax_tbl = plt.subplots(figsize=(20, len(ordered) * 0.6 + 2)) 
    ax_tbl.axis("off")
    
    table_vals = []
    for row in df[["Label", "TP", "FP", "TN", "FN"]].values:
        label = row[0]
        # Lấy giá trị số
        values = row[1:] 
        # Tổng số mẫu trong dataset (TP+FP+TN+FN cho 1 lớp luôn bằng tổng mẫu)
        total_samples = sum(values) 
        
        fmt_row = [label]
        for val in values:
            pct = (val / total_samples) * 100 if total_samples > 0 else 0
            # Format: "123,456 (12.5%)"
            fmt_row.append(f"{int(val):,} ({pct:.1f}%)")
            
        table_vals.append(fmt_row)
    
    table = ax_tbl.table(cellText=table_vals, colLabels=["Label", "TP", "FP", "TN", "FN"], 
                         loc="center", cellLoc="center", colColours=["#f0f0f0"] * 5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6) # Scale cao hơn chút cho thoáng
    ax_tbl.set_title(f"Confusion Details: {title}", fontweight='bold', pad=10)
    
    os.makedirs(os.path.dirname(f"{save_dir}/table_"), exist_ok=True)
    plt.savefig(f"{save_dir}/table_confusion_{title.lower().replace(' ', '_')}.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    # --- 6. DRAW BAR CHART ---
    metrics_plot = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(ordered)); width = 0.2
    
    fig_bar, ax_bar = plt.subplots(figsize=(max(16, len(ordered)*2), 8))
    colors = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c']
    
    for i, metric in enumerate(metrics_plot):
        vals = df[metric].values
        rects = ax_bar.bar(x + (i - 1.5) * width, vals, width, label=metric, color=colors[i])
        if len(rects) <= 5:
            for rect in rects:
                if rect.get_height() > 0.01:
                    ax_bar.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.01, 
                            f"{rect.get_height():.4f}", ha='center', va='bottom', 
                            fontsize=12, fontweight='bold')
        else:
            for rect in rects:
                if rect.get_height() > 0.01:
                    ax_bar.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.01, 
                            f"{rect.get_height():.4f}", ha='center', va='bottom', 
                            fontsize=10, fontweight='bold')
            

    ax_bar.set_title(f"Per-Class Performance Metrics: {title}", fontweight='bold', fontsize=16)
    
    ax_bar.set_xticks(x);
    ax_bar.set_xticklabels(ordered, rotation=0, fontsize=15, fontweight='bold')
    
    ax_bar.set_ylabel("Score", fontsize=15, fontweight='bold')
    ax_bar.set_ylim(0, 1.12)
    
    ax_bar.legend(loc='upper right', ncol=4, fontsize=14)
    ax_bar.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(f"{save_dir}/chart_per_class_{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    print(f"   -> Saved details for {title}")
    return


def plot_all_models_performance(all_metrics_history, save_dir):
    """
    Vẽ 4 biểu đồ cột (Bar Charts) riêng biệt cho từng Model (Pipeline, XGBoost, AE, OCSVM).
    Mỗi biểu đồ so sánh 4 chỉ số (Accuracy, Precision, Recall, F1) qua các Scenario.
    Hiển thị giá trị cụ thể trên đầu cột.
    """
    Scenarios = list(all_metrics_history.keys()) # ['Scenario 0', 'Scenario 1', 'Scenario 2']
    models = ['Pipeline', 'XGBoost', 'AE', 'OCSVM']
    
    # Mapping tên hiển thị sang key trong dict
    metrics_map = {
        'Accuracy': 'accuracy', 
        'Precision': 'precision', 
        'Recall': 'recall', 
        'F1-Score': 'f1-score'
    }
    metric_names = list(metrics_map.keys())
    
    # 1. Chuẩn bị dữ liệu: data[model][metric_name] = [val_c0, val_c1, val_c2]
    data = {mod: {m: [] for m in metric_names} for mod in models}
    
    for c in Scenarios:
        for mod in models:
            rep = all_metrics_history[c].get(mod, {})
            
            # Lấy giá trị (Mặc định 0 nếu không có)
            acc = rep.get('accuracy', 0)
            prec = rep.get('weighted avg', {}).get('precision', 0)
            rec = rep.get('weighted avg', {}).get('recall', 0)
            f1 = rep.get('weighted avg', {}).get('f1-score', 0)
            
            data[mod]['Accuracy'].append(acc)
            data[mod]['Precision'].append(prec)
            data[mod]['Recall'].append(rec)
            data[mod]['F1-Score'].append(f1)

    # 2. Vẽ 4 Subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 14)) # Kích thước lớn để chứa text
    axes = axes.flatten()
    
    x = np.arange(len(Scenarios))
    width = 0.2 # Độ rộng của mỗi cột con
    
    # Màu sắc cho 4 metrics
    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c'] # Xanh, Cam, Xanh lá, Đỏ

    for i, mod in enumerate(models):
        ax = axes[i]
        
        # Vẽ nhóm cột cho từng metric
        for j, metric in enumerate(metric_names):
            values = data[mod][metric]
            
            # Tính toán vị trí cột để chúng nằm cạnh nhau
            # j=0 -> -1.5w, j=1 -> -0.5w, j=2 -> 0.5w, j=3 -> 1.5w
            offset = x + (j - 1.5) * width 
            
            rects = ax.bar(offset, values, width, label=metric, color=colors[j])
            
            # In giá trị lên đầu mỗi cột
            for rect in rects:
                height = rect.get_height()
                if height > 0: # Chỉ in nếu giá trị > 0
                    ax.text(rect.get_x() + rect.get_width()/2., height + 0.02,
                            f'{height:.4f}',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(f'{mod} Performance Evolution', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(Scenarios, fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 1.1) # Tăng trần trục Y để chứa text xoay dọc
        ax.grid(axis='y', alpha=0.3)
        
        # Legend
        ax.legend(loc='upper left', ncol=2, fontsize=10)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = f"{save_dir}/all_models_comparison_bar.png"
    plt.savefig(save_path, dpi = 300)
    plt.close()
    print(f"Comparison bar charts saved to {save_path}")

def plot_unknown_detection_performance(unknown_stats, save_dir):
    """Biểu đồ hiệu suất phát hiện Unknown (Pre-IL)"""
    Scenarios = list(unknown_stats.keys())
    pre_recalls = [unknown_stats[c]['Pre']['recall'] for c in Scenarios]
    x = np.arange(len(Scenarios))
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, pre_recalls, width=0.5, label='Unknown Detection (Recall)', color='#e74c3c')
    plt.xticks(x, Scenarios); plt.ylim(0, 1.1); plt.title('Unknown Class Detection (Pre-IL)'); plt.legend()
    for bar in bars: plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2%}', ha='center', va='bottom')
    os.makedirs(save_dir, exist_ok=True); plt.savefig(f"{save_dir}/unknown_performance.png", dpi =300); plt.close()

# ... [SessionDataLoader, SessionManager GIỮ NGUYÊN] ...
class SessionDataLoader:
    def __init__(self): self.scaler = StandardScaler(); self.is_fitted = False
    def load_data_raw(self, path, label_col='Label', binary_label_col='Binary Label'):
        if not os.path.exists(path): raise ValueError(f"{path} not found")
        df = pd.read_parquet(path)
        y = df[label_col].astype(int) if label_col in df else None
        cols = [c for c in [label_col, binary_label_col] if c in df]
        X = df.drop(columns=cols).fillna(0).values
        return X, y
    def apply_scaling(self, X, fit=False):
        if fit: print("Fitting Global Scaler..."); X_scaled = self.scaler.fit_transform(X); self.is_fitted = True; return X_scaled
        else:
            if not self.is_fitted:
                try: self.load_scaler('sessions/global_scaler.joblib')
                except: pass
            if not self.is_fitted: return X
            return self.scaler.transform(X)
    def save_scaler(self, p): os.makedirs(os.path.dirname(p), exist_ok=True); joblib.dump(self.scaler, p)
    def load_scaler(self, p): self.scaler = joblib.load(p); self.is_fitted = True

class SessionManager:
    def __init__(self, base='sessions'):
        self.base = base; self.curr = 0; self.info = {}; os.makedirs(base, exist_ok=True)
        for i in range(3): os.makedirs(f"{base}/Scenario{i}", exist_ok=True)
    def initialize_Scenario_0(self, tr, te): self.curr=0; self.info[0]={'tr':tr,'te':te}; self.save()
    def advance_to_Scenario_1(self, tr, te): self.curr=1; self.info[1]={'tr':tr,'te':te}; self.save()
    def advance_to_Scenario_2(self, tr, te): self.curr=2; self.info[2]={'tr':tr,'te':te}; self.save()
    def save_models(self, models, sid):
        p = f"{self.base}/Scenario{sid}/models"; os.makedirs(p, exist_ok=True)
        for k, v in models.items():
            if hasattr(v, 'save_model'): v.save_model(f"{p}/{k}")
            else: joblib.dump(v, f"{p}/{k}.joblib")
    def load_models(self, sid, models):
        p = f"{self.base}/Scenario{sid}/models"
        for k, v in models.items():
            if hasattr(v, 'load_model'): v.load_model(f"{p}/{k}")
            elif os.path.exists(f"{p}/{k}.joblib"): models[k] = joblib.load(f"{p}/{k}.joblib")
        return models
    def save(self): joblib.dump({'c': self.curr, 'i': self.info}, f"{self.base}/info.joblib")

# ==================== 4. EVALUATION HELPERS ====================
def get_label_name(y):
    try: val = int(y)
    except: return str(y)
    return {0: "Benign", 1: "DDoS", 2: "DoS", 3: "Reconnaisance", 4: "MITM_ArpSpoofing", 5: "DNS_Spoofing"}.get(val, "Unknown")

def plot_cm(y_true, y_pred, title, save_path):
    labels = sorted(list(set(y_true) | set(y_pred)))
    ordered = [l for l in labels if l != "Benign" and l != "Unknown"]
    if "Benign" in labels: ordered.insert(0, "Benign")
    if "Unknown" in labels: ordered.append("Unknown")
    
    cm = confusion_matrix(y_true, y_pred, labels=ordered)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', xticklabels=ordered, yticklabels=ordered, vmin=0, vmax=1)
    
    plt.title(f"{title} (%)", fontsize=17, fontweight='bold', pad=20) 
    plt.ylabel('True Label', fontsize=17, fontweight = 'bold')
    plt.xlabel('Predicted Label', fontsize =15, fontweight='bold')
    
    plt.xticks(fontsize =14 , fontweight='bold', rotation=45, ha='right')
    plt.yticks(fontsize=14, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    plt.savefig(save_path, dpi=300);
    plt.close()

def plot_binary_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Reds', xticklabels=['Abnormal', 'Normal'], yticklabels=['Abnormal', 'Normal'], vmin=0, vmax=1)
    plt.title(f"{title} (%)"); plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi =300); plt.close()

def plot_unknown_binary_cm(y_true, preds, unknown_label, save_path, session_name):
    y_true = np.array(y_true); preds = np.array(preds)
    if isinstance(unknown_label, list): is_actual = np.isin(y_true, unknown_label).astype(int); label_text = f"List {unknown_label}"
    else: is_actual = (y_true == unknown_label).astype(int); label_text = get_label_name(unknown_label)
    y_bin_pred = (preds == "Unknown").astype(int)
    cm = confusion_matrix(is_actual, y_bin_pred)
    print(f"\n{'-'*40}\n  Unknown DETECTION TABLE: {label_text}\n{'-'*40}")
    try:
        print(f"True OTHER | {cm[0,0]:<6} | {cm[0,1]:<6} (Pred UNK)")
        print(f"True UNK   | {cm[1,0]:<6} | {cm[1,1]:<6} (Pred UNK)")
        tn, fp, fn, tp = cm.ravel(); print(f"Recall: {tp/(tp+fn):.2%}")
    except: pass
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Oranges', xticklabels=['Others', 'Pred Unknown'], yticklabels=['True Others', 'True Unknown'], vmin=0, vmax=1)
    plt.title(f"Unknown Detection Confusion Matrix - {session_name}"); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300); plt.close()

def plot_metrics_bar(report_dict, title, save_path):
    metrics = ['precision', 'recall', 'f1-score']
    try: values = [report_dict['weighted avg'][m] for m in metrics]
    except: values = [0,0,0]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    for bar in bars: plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom')
    plt.legend(bars, metrics, loc='upper right', frameon=True, shadow=True)
    plt.title(title); plt.ylim(0, 1.1); plt.ylabel('Score'); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300); plt.close()

def calculate_unknown_metrics(y_true, preds, unknown_label, save_dir, session_name):
    y_true = np.array(y_true); preds = np.array(preds)
    if isinstance(unknown_label, list): is_actual = np.isin(y_true, unknown_label).astype(int); label_text = f"List {unknown_label}"
    else: is_actual = (y_true == unknown_label).astype(int); label_text = get_label_name(unknown_label)
    y_bin_pred = (preds == "Unknown").astype(int)
    plot_unknown_binary_cm(y_true, preds, unknown_label, f"{save_dir}/unknown_cm_{session_name}.png", session_name)
    return {'precision': precision_score(is_actual, y_bin_pred, zero_division=0), 'recall': recall_score(is_actual, y_bin_pred, zero_division=0), 'f1': f1_score(is_actual, y_bin_pred, zero_division=0)}

def evaluate_supervised_with_unknown(y_true, y_pred, y_conf, atk_thres=0.7, ben_thres=0.7, session_name="", save_dir="", model_name="XGBoost", target_unknown=None):
    print(f"\n--- [METRICS] {model_name} Dual-Thres (Atk>={atk_thres}, Ben>={ben_thres}) - {session_name} ---")
    def map_label(y):
        if isinstance(target_unknown, list): return "Unknown" if y in target_unknown else get_label_name(y)
        elif target_unknown is not None: return "Unknown" if y == target_unknown else get_label_name(y)
        return get_label_name(y)
    y_str_true = [map_label(y) for y in y_true]; y_str_pred = []; stats = {"unk_atk": 0, "unk_ben": 0}
    for p, c in zip(y_pred, y_conf):
        p_val = int(p)
        if p_val != 0:
            if c < atk_thres: y_str_pred.append("Unknown"); stats["unk_atk"] += 1
            else: y_str_pred.append(get_label_name(p_val))
        else:
            if c < ben_thres: y_str_pred.append("Unknown"); stats["unk_ben"] += 1
            else: y_str_pred.append("Benign")
    print(f"   -> XGBoost flagged Unknown: {stats['unk_atk']} (Low Conf Atk) + {stats['unk_ben']} (Low Conf Ben)")
    print(classification_report(y_str_true, y_str_pred, digits=4, zero_division=0))
    plot_cm(y_str_true, y_str_pred, f"CM {model_name} (Unknown) - {session_name}", f"{save_dir}/cm_{model_name}_unknown_{session_name}.png")
    rep = classification_report(y_str_true, y_str_pred, output_dict=True, zero_division=0)
    plot_metrics_bar(rep, f"Metrics {model_name} - {session_name}", f"{save_dir}/metrics_{model_name}_unknown_{session_name}.png")
    return rep

def evaluate_supervised_model(y_true, y_pred, session_name, save_dir, model_name="Supervised", return_f1=False):
    print(f"\n--- [METRICS] {model_name} - {session_name} ---")
    y_str_true = [get_label_name(y) for y in y_true]; y_str_pred = [get_label_name(y) for y in y_pred]
    print(classification_report(y_str_true, y_str_pred, digits=4, zero_division=0))
    plot_cm(y_str_true, y_str_pred, f"CM {model_name} - {session_name}", f"{save_dir}/cm_{model_name.lower().replace(' ', '_')}_{session_name}.png")
    rep = classification_report(y_str_true, y_str_pred, digits=4, zero_division=0, output_dict=True)
    plot_metrics_bar(rep, f"Metrics {model_name} - {session_name}", f"{save_dir}/metrics_{model_name.lower().replace(' ', '_')}_{session_name}.png")
    return rep

def evaluate_final_pipeline(y_true, y_pred, sess, save_dir, return_f1=False, map_new_to_unknown=None):
    print(f"\n--- [METRICS] Final Pipeline --- {sess} ---")
    
    # Chuyển đổi y_true, y_pred sang String Label cơ bản (Chưa gộp)
    # Gộp sẽ được thực hiện bên trong hàm analyze_and_plot_class_details
    y_str_true = []
    if map_new_to_unknown is not None:
        y_str_true = [ get_label_name(val) if val not in map_new_to_unknown else "Unknown" for val in y_true]
    else:
        y_str_true = [get_label_name(val) for val in y_true]
    
    y_str_pred = [ val if isinstance(val, str) else get_label_name(val) for val in y_pred ]
    # In report tổng quan
    print(classification_report(y_str_true, y_str_pred, digits=4, zero_division=0))
    
    # Vẽ Confusion Matrix (Cũ) - Nếu muốn gộp cả ở đây thì cần thêm logic, nhưng ta tập trung vào hàm chi tiết bên dưới
    plot_cm(y_str_true, y_str_pred, f"CM Pipeline - {sess}", f"{save_dir}/cm_pipe_{sess}.png")
    
    rep = classification_report(y_str_true, y_str_pred, digits=4, zero_division=0, output_dict=True)
    plot_metrics_bar(rep, f"Metrics Pipeline - {sess}", f"{save_dir}/metrics_pipe_{sess}.png")
    
    # --- LOGIC CHUẨN BỊ THAM SỐ CHO HÀM VẼ CHI TIẾT ---
    
    # 1. Xác định danh sách tên nhãn cần gộp (Merge List)
    # Nếu map_new_to_unknown = [3] -> merge_list = ['Reconn']
    merge_labels_list = []
    if map_new_to_unknown:
        merge_labels_list = [get_label_name(v) for v in map_new_to_unknown]
    
    # 2. Xác định cờ ẩn Unknown (Hide Unknown)
    # - Pre-IL (có merge_list): KHÔNG ẩn Unknown (False) -> Để hiện cột Unknown (đã gộp các nhãn mới)
    # - Post-IL (không merge_list): ẨN Unknown (True) -> Để tập trung vào các nhãn cụ thể
    hide_unk = (map_new_to_unknown is None)
    
    # Gọi hàm vẽ chi tiết (Đã cập nhật logic bên trên)
    analyze_and_plot_class_details(
        y_str_true, y_str_pred, 
        title=f"Details_{sess}", 
        save_dir=save_dir, 
        merge_labels_list=merge_labels_list, # Truyền danh sách nhãn cần gộp
        hide_unknown=hide_unk              # Truyền cờ ẩn/hiện Unknown
    )
    
    if return_f1: return rep['weighted avg']['f1-score']
    return rep

def evaluate_unsupervised_detailed(y_true, ae_pred, ocsvm_pred, sess, save_dir, return_f1=False):
    y_bin = (y_true == 0).astype(int) 
    print("\n>>> AE (Target: 0=Abn, 1=Nor):"); print(classification_report(y_bin, ae_pred, target_names=['Abn', 'Nor'], digits=4))
    plot_binary_cm(y_bin, ae_pred, f"AE - {sess}", f"{save_dir}/cm_ae_{sess}.png")
    ae_rep = classification_report(y_bin, ae_pred, output_dict=True, zero_division=0)
    print("\n>>> OCSVM (Target: 0=Abn, 1=Nor):"); print(classification_report(y_bin, ocsvm_pred, target_names=['Abn', 'Nor'], digits=4))
    plot_binary_cm(y_bin, ocsvm_pred, f"OCSVM - {sess}", f"{save_dir}/cm_ocsvm_{sess}.png")
    oc_rep = classification_report(y_bin, ocsvm_pred, output_dict=True, zero_division=0)
    return ae_rep, oc_rep

def evaluate_gray_zone(y_true, xgb_pred, xgb_conf, ae_pred, ocsvm_pred, c_min, c_max, sess, save_dir):
    print(f"\n--- [ANALYSIS] Gray Zone ({c_min} <= Conf < {c_max}) ---")
    mask = (xgb_pred == 0) & (xgb_conf >= c_min) & (xgb_conf < c_max)
    if np.sum(mask) == 0: print("No samples in Gray Zone."); return
    y_g = y_true[mask]; ae_g = ae_pred[mask]; oc_g = ocsvm_pred[mask]; y_bin = (y_g == 0).astype(int)
    print(">>> AE Gray:"); print(classification_report(y_bin, ae_g, target_names=['Abn', 'Nor'], digits=4, zero_division=0))
    plot_binary_cm(y_bin, ae_g, f"AE Gray - {sess}", f"{save_dir}/cm_ae_gray_{sess}.png")
    print(">>> OCSVM Gray:"); print(classification_report(y_bin, oc_g, target_names=['Abn', 'Nor'], digits=4, zero_division=0))
    plot_binary_cm(y_bin, oc_g, f"OCSVM Gray - {sess}", f"{save_dir}/cm_ocsvm_gray_{sess}.png")

def plot_scenarios_comparison(results_dict, save_path, Scenario_name):
    """
    results_dict: {'XGB Only': {'f1': 0.8}, 'XGB+AE': {'f1': 0.85}, ...}
    """
    scenarios = list(results_dict.keys())
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    # Chuẩn bị data
    data = {m: [] for m in metrics}
    for sc in scenarios:
        report = results_dict[sc]
        data['Precision'].append(report['weighted avg']['precision'])
        data['Recall'].append(report['weighted avg']['recall'])
        data['F1-Score'].append(report['weighted avg']['f1-score'])
        
    x = np.arange(len(scenarios))
    width = 0.25
    
    plt.figure(figsize=(12, 7))
    
    # Vẽ 3 cột
    plt.bar(x - width, data['Precision'], width, label='Precision', color='#3498db')
    plt.bar(x, data['Recall'], width, label='Recall', color='#2ecc71')
    plt.bar(x + width, data['F1-Score'], width, label='F1-Score', color='#e74c3c')
    
    plt.xlabel('Ablation Scenarios')
    plt.ylabel('Score')
    plt.title(f'Performance Comparison of Different Pipeline Configurations - {Scenario_name}')
    plt.xticks(x, scenarios)
    plt.ylim(0, 1.15)
    plt.legend(loc='upper right', ncol=3)
    plt.grid(axis='y', alpha=0.3)
    
    # Thêm text giá trị F1 lên cột
    for i in range(len(scenarios)):

        plt.text(i - width, data['Precision'][i] + 0.01, 
                 f"{data['Precision'][i]:.4f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=9, color='black')
        
        plt.text(i, data['Recall'][i] + 0.01, 
                 f"{data['Recall'][i]:.4f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=9, color='black')

        plt.text(i + width, data['F1-Score'][i] + 0.01, 
                 f"{data['F1-Score'][i]:.4f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=9, color='black')
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comparison chart saved to {save_path}")

def plot_ablation_evolution(ablation_history, save_dir):
    """
    ablation_history: {
        'Scenario 0': {'XGB Only': rep, 'Full Pipeline': rep, ...},
        'Scenario 1': {...},
        'Scenario 2': {...}
    }
    """
    Scenarios = sorted(list(ablation_history.keys())) # ['Scenario 0', 'Scenario 1', 'Scenario 2']
    scenarios = list(ablation_history[Scenarios[0]].keys()) # ['XGB Only', 'Full Pipeline'...]
    metrics = ['Precision', 'Recall', 'F1-Score'] # Bỏ Accuracy nếu muốn, hoặc thêm vào
    
    # Chuẩn bị dữ liệu: data[metric][scenario] = [val_c0, val_c1, val_c2]
    data = {m: {sc: [] for sc in scenarios} for m in metrics}
    
    for c in Scenarios:
        for sc in scenarios:
            rep = ablation_history[c].get(sc, {})
            data['Precision'][sc].append(rep.get('weighted avg', {}).get('precision', 0))
            data['Recall'][sc].append(rep.get('weighted avg', {}).get('recall', 0))
            data['F1-Score'][sc].append(rep.get('weighted avg', {}).get('f1-score', 0))

    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = range(len(Scenarios))
    markers = ['o', 's', '^', 'D']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'] # Red, Blue, Green, Purple
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, sc in enumerate(scenarios):
            ax.plot(x, data[metric][sc], marker=markers[j], label=sc, color=colors[j], linewidth=2.5)
            
            # Annotate điểm cuối
            if data[metric][sc]:
                last_val = data[metric][sc][-1]
                ax.text(x[-1], last_val + 0.01, f'{last_val:.2f}', color=colors[j], fontweight='bold', ha='left')

        ax.set_title(f'{metric} Evolution')
        ax.set_xticks(x); ax.set_xticklabels(Scenarios)
        ax.set_ylim(0.6, 1.1); ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper left') # Chỉ hiện legend ở hình đầu cho gọn

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/ablation_evolution.png", dpi=300)
    plt.close()
    print(f"Ablation evolution chart saved to {save_dir}/ablation_evolution.png")

def plot_unknown_detection_comparison(results_data, save_path):
    """
    Vẽ biểu đồ cột nhóm so sánh Unknown Detection Rate giữa các kịch bản ở Scenario 1 và Scenario 2.
    
    results_data format:
    {
        "XGB Only":      {"Scenario 1": 0.55, "Scenario 2": 0.60},
        "Full Pipeline": {"Scenario 1": 0.85, "Scenario 2": 0.90},
        ...
    }
    """
    scenarios = list(results_data.keys())
    Scenarios = ['Scenario 1', 'Scenario 2']
    
    # Chuẩn bị dữ liệu vẽ
    Scenario1_vals = [results_data[sc].get('Scenario 1', 0) for sc in scenarios]
    Scenario2_vals = [results_data[sc].get('Scenario 2', 0) for sc in scenarios]
    
    x = np.arange(len(scenarios))  # Vị trí các nhóm trên trục X
    width = 0.35  # Độ rộng của mỗi cột

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Vẽ 2 nhóm cột
    rects1 = ax.bar(x - width/2, Scenario1_vals, width, label='Scenario 1 (Target: Reconn)', color='#3498db')
    rects2 = ax.bar(x + width/2, Scenario2_vals, width, label='Scenario 2 (Target: MITM/DNS_Spoofing)', color='#e74c3c')

    # Thêm nhãn, tiêu đề
    ax.set_ylabel('Detection Rate (Recall of New Attacks as Unknown)', fontweight='bold')
    ax.set_xlabel('Ablation Scenarios', fontweight='bold')
    ax.set_title('Unknown Threat Detection Comparison (Scenario 1 vs Scenario 2)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 1) # Tăng trần để chứa text
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Hàm phụ để in giá trị lên đầu cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Unknown detection comparison chart saved to {save_path}")