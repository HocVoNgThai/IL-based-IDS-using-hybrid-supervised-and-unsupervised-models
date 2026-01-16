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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, precision_recall_fscore_support
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
        
        avg_gpu = np.mean(self.history['gpu']) if self.history['gpu'] else 0
        
        return {
            'duration': duration,
            'avg_cpu': np.mean(self.history['cpu']) if self.history['cpu'] else 0,
            'max_ram': np.max(self.history['ram']) if self.history['ram'] else 0,
            'avg_gpu': avg_gpu,
            'history': self.history
        }

# ==================== 2. IL METRICS ====================
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
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5, 
                 f'{height:.1f}s', ha='center', va='bottom', color='blue', fontweight='bold', fontsize=11)

    # --- 2. Line Charts (CPU, RAM, GPU) ---
    ax2 = ax1.twinx()
    
    # Vẽ các đường
    line_cpu = ax2.plot(Scenarios, cpus, 'r-o', label='CPU %', linewidth=2.5, markersize=9, markerfacecolor='white', markeredgewidth=2)
    line_ram = ax2.plot(Scenarios, rams, 'g-s', label='RAM %', linewidth=2.5, markersize=9, markerfacecolor='white', markeredgewidth=2)
    line_gpu = ax2.plot(Scenarios, gpus, 'm-^', label='GPU %', linewidth=2.5, markersize=9, markerfacecolor='white', markeredgewidth=2)
    
    ax2.set_ylabel('Usage %', color='black', fontweight='bold', fontsize=12)
    ax2.set_ylim(0, 115) 
    ax2.tick_params(axis='y', labelcolor='black')

    bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)

    for i, (c, r, g) in enumerate(zip(cpus, rams, gpus)):
        vertical_spacing = 6 
        
        if c >= r:
            ax2.text(i, c + 4, f'{c:.1f}%', color='#D32F2F', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)
            ax2.text(i, r - 4, f'{r:.1f}%', color='#388E3C', ha='center', va='top', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)
        else:
            ax2.text(i, r + 4, f'{r:.1f}%', color='#388E3C', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)
            ax2.text(i, c - 4, f'{c:.1f}%', color='#D32F2F', ha='center', va='top', 
                     fontweight='bold', fontsize=10, bbox=bbox_props)

        if g > 0.5:
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
    plt.legend(ncol=1, fontsize=15)
    # In giá trị lên từng điểm
    for i in steps:
        # Avg Acc
        val_acc = il.history['Avg_Acc'][i]
        plt.text(i, val_acc + 0.005, f"{val_acc:.4f}", ha='center', va='bottom', color='#1f77b4', fontweight='bold', fontsize=12)
        
        # BWT
        val_bwt = il.history['BWT'][i]
        plt.text(i, val_bwt - 0.015, f"{val_bwt:.4f}", ha='center', va='top', color='#ff7f0e', fontweight='bold', fontsize=12)
        
        # Forgetting
        val_f = il.history['Forgetting'][i]
        plt.text(i, val_f + 0.005, f"{val_f:.4f}", ha='center', va='bottom', color='#2ca02c', fontweight='bold', fontsize=12)

    plt.title('Incremental Learning Metrics Evolution', fontweight='bold', fontsize=16)
    plt.xlabel('Scenario', fontweight='bold', fontsize=14)
    plt.ylabel('Score', fontweight='bold', fontsize=14)
    plt.xticks(steps, labels, fontweight='bold', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_il_matrix(il, save_path):
    """Heatmap Accuracy"""
    Scenarios = sorted(il.R.keys()); n = len(Scenarios); matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n): matrix[i, j] = il.R[i].get(j, 0)
    
    # Tăng kích thước hình lên (10, 8) cho thoáng
    plt.figure(figsize=(10, 8))
    
    # annot_kws={"size": 14, "weight": "bold"} -> Làm số trong ô to và đậm
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlGnBu', 
                xticklabels=[f'Test C{s}' for s in Scenarios], 
                yticklabels=[f'Train C{s}' for s in Scenarios],
                annot_kws={"size": 16, "weight": "bold"}) 
    
    # Cập nhật font chữ cho trục X, Y
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold', rotation=0)
    
    # Cập nhật tiêu đề
    plt.title('IL Accuracy Matrix', fontweight='bold', fontsize=18, pad=20)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==================== PER-CLASS ANALYSIS FUNCTIONS ====================

def analyze_and_plot_class_details(y_true, y_pred, title, save_dir, merge_labels_list=None, hide_unknown=False):
    """
    Tính toán và vẽ Bảng TP/FP/TN/FN (có kèm %) + Biểu đồ Metrics.
    (Phiên bản cập nhật: Font to, Đậm, Rõ ràng cho báo cáo)
    """
    # 1. Gộp nhãn
    y_true_mapped = list(y_true)
    y_pred_mapped = list(y_pred)
    
    if merge_labels_list:
        y_true_mapped = ["UNKNOWN" if lbl in merge_labels_list else lbl for lbl in y_true_mapped]
        y_pred_mapped = ["UNKNOWN" if lbl in merge_labels_list else lbl for lbl in y_pred_mapped]

    # 2. Danh sách nhãn
    all_labels = sorted(list(set(y_true_mapped) | set(y_pred_mapped)))
    
    # 3. Lọc hiển thị
    labels_to_show = set(all_labels)
    if hide_unknown and "UNKNOWN" in labels_to_show:
        labels_to_show.remove("UNKNOWN")
        
    ordered = [l for l in all_labels if l in labels_to_show and l != "BENIGN" and l != "UNKNOWN"]
    if "BENIGN" in labels_to_show: ordered.insert(0, "BENIGN")
    if "UNKNOWN" in labels_to_show: ordered.append("UNKNOWN")
    
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
    
    # --- 5. DRAW TABLE (Cập nhật Font size) ---
    # Tăng kích thước hình để chứa bảng chữ to
    fig_tbl, ax_tbl = plt.subplots(figsize=(20, len(ordered) * 0.8 + 4)) 
    ax_tbl.axis("off")
    
    table_vals = []
    for row in df[["Label", "TP", "FP", "TN", "FN"]].values:
        label = row[0]
        values = row[1:] 
        total_samples = sum(values) 
        
        fmt_row = [label]
        for val in values:
            pct = (val / total_samples) * 100 if total_samples > 0 else 0
            fmt_row.append(f"{int(val):,} ({pct:.1f}%)")
            
        table_vals.append(fmt_row)
    
    table = ax_tbl.table(cellText=table_vals, colLabels=["Label", "TP", "FP", "TN", "FN"], 
                         loc="center", cellLoc="center", colColours=["#f0f0f0"] * 5)
    
    table.auto_set_font_size(False)
    table.set_fontsize(14) # Tăng fontsize bảng từ 11 lên 14
    table.scale(1.2, 2.2)  # Tăng chiều cao dòng (scale y) để chữ to không bị chèn
    
    # Làm đậm tiêu đề cột (Hàng đầu tiên)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight='bold')

    ax_tbl.set_title(f"Confusion Details: {title}", fontweight='bold', fontsize=20, pad=20)
    
    os.makedirs(os.path.dirname(f"{save_dir}/table_"), exist_ok=True)
    plt.savefig(f"{save_dir}/table_confusion_{title.lower().replace(' ', '_')}.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    # --- 6. DRAW BAR CHART (Cập nhật Font size) ---
    metrics_plot = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(ordered)); width = 0.2
    
    # Tăng width của biểu đồ để chứa text
    fig_bar, ax_bar = plt.subplots(figsize=(max(16, len(ordered)*3), 10))
    colors = ['#9b59b6', '#3498db', '#2ecc71', '#e74c3c']
    
    for i, metric in enumerate(metrics_plot):
        vals = df[metric].values
        rects = ax_bar.bar(x + (i - 1.5) * width, vals, width, label=metric, color=colors[i])
        
        # In giá trị lên đầu cột (Font to 10-12, Đậm)
        for rect in rects:
            if rect.get_height() > 0.01:
                ax_bar.text(rect.get_x() + rect.get_width()/2., rect.get_height() + 0.01, 
                            f"{rect.get_height():.4f}", ha='center', va='bottom', 
                            fontsize=10, fontweight='bold', rotation=0)

    ax_bar.set_title(f"Per-Class Performance Metrics: {title}", fontweight='bold', fontsize=20, pad=20)
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ordered, rotation=0, fontsize=14, fontweight='bold')
    
    ax_bar.set_ylabel("Score", fontsize=16, fontweight='bold')
    plt.yticks(fontsize=14)
    ax_bar.set_ylim(0, 1.15)
    
    ax_bar.legend(loc='upper right', ncol=4, fontsize=14)
    ax_bar.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/chart_per_class_{title.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()
    print(f"   -> Saved details for {title}")

def plot_all_models_performance(all_metrics_history, save_dir):
    """
    Vẽ 4 biểu đồ cột (Bar Charts) riêng biệt cho từng Model (Pipeline, XGBoost, AE, OCSVM).
    Mỗi biểu đồ so sánh 4 chỉ số (Accuracy, Precision, Recall, F1) qua các Scenario.
    Hiển thị giá trị cụ thể trên đầu cột.
    """
    Scenarios = list(all_metrics_history.keys())
    models = ['Pipeline', 'XGBoost', 'AE', 'OCSVM']
    
    metrics_map = {
        'Accuracy': 'accuracy', 
        'Precision': 'precision', 
        'Recall': 'recall', 
        'F1-Score': 'f1-score'
    }
    metric_names = list(metrics_map.keys())
    
    data = {mod: {m: [] for m in metric_names} for mod in models}
    
    for c in Scenarios:
        for mod in models:
            rep = all_metrics_history[c].get(mod, {})
            
            acc = rep.get('accuracy', 0)
            prec = rep.get('weighted avg', {}).get('precision', 0)
            rec = rep.get('weighted avg', {}).get('recall', 0)
            f1 = rep.get('weighted avg', {}).get('f1-score', 0)
            
            data[mod]['Accuracy'].append(acc)
            data[mod]['Precision'].append(prec)
            data[mod]['Recall'].append(rec)
            data[mod]['F1-Score'].append(f1)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    
    x = np.arange(len(Scenarios))
    width = 0.2 
    
    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c'] 

    for i, mod in enumerate(models):
        ax = axes[i]
        
        for j, metric in enumerate(metric_names):
            values = data[mod][metric]
            
            offset = x + (j - 1.5) * width 
            
            rects = ax.bar(offset, values, width, label=metric, color=colors[j])
            
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.text(rect.get_x() + rect.get_width()/2., height + 0.02,
                            f'{height:.4f}',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(f'{mod} Performance Evolution', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(Scenarios, fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Legend
        ax.legend(loc='upper left', ncol=2, fontsize=10)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = f"{save_dir}/all_models_comparison_bar.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Comparison bar charts saved to {save_path}")

def plot_unknown_detection_performance(unknown_stats, save_dir):
    """Biểu đồ hiệu suất phát hiện Unknown (Pre-IL)"""
    Scenarios = list(unknown_stats.keys())
    pre_recalls = [unknown_stats[c]['Pre']['recall'] for c in Scenarios]
    x = np.arange(len(Scenarios))
    plt.figure(figsize=(8, 6))
    bars = plt.bar(x, pre_recalls, width=0.5, label='Unknown Detection (Recall)', color='#e74c3c')
    plt.legend(fontsize=14)
    plt.xticks(x, Scenarios, fontsize=13, fontweight='bold'); plt.yticks(fontsize=13)
    plt.ylim(0, 1.1); plt.title('Unknown Class Detection (Pre-IL)', fontsize=16, fontweight='bold'); plt.legend()
    for bar in bars: plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    os.makedirs(save_dir, exist_ok=True); plt.savefig(f"{save_dir}/unknown_performance.png", dpi=300); plt.close()

class ScenarioDataLoader:
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
                try: self.load_scaler('Scenarios/global_scaler.joblib')
                except: pass
            if not self.is_fitted: return X
            return self.scaler.transform(X)
    def save_scaler(self, p): os.makedirs(os.path.dirname(p), exist_ok=True); joblib.dump(self.scaler, p)
    def load_scaler(self, p): self.scaler = joblib.load(p); self.is_fitted = True

class ScenarioManager:
    def __init__(self, base='Scenarios'):
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
    return {0: "BENIGN", 1: "DDoS", 2: "DoS", 3: "Reconn", 4: "MITM", 5: "DNS Spoofing"}.get(val, "UNKNOWN")

def plot_cm(y_true, y_pred, title, save_path):
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Sắp xếp lại thứ tự nhãn: BENIGN đầu, UNKNOWN cuối
    ordered = [l for l in labels if l != "BENIGN" and l != "UNKNOWN"]
    if "BENIGN" in labels: ordered.insert(0, "BENIGN")
    if "UNKNOWN" in labels: ordered.append("UNKNOWN")
    
    cm = confusion_matrix(y_true, y_pred, labels=ordered)
    
    # Chuẩn hóa matrix sang %
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    plt.figure(figsize=(10, 8))
    
    # --- Cập nhật heatmap ---
    # annot_kws: Chỉnh font chữ số bên trong các ô
    ax = sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', 
                     xticklabels=ordered, yticklabels=ordered, vmin=0, vmax=1,
                     annot_kws={"size": 15, "weight": "bold"}) 
    
    # --- Cập nhật Tiêu đề và Nhãn trục ---
    plt.title(f"{title} (%)", fontsize=17, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=15, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=15, fontweight='bold')
    
    # --- Cập nhật Font chữ cho các nhãn (BENIGN, DDoS...) ---
    plt.xticks(fontsize=14, fontweight='bold', rotation=45, ha='right')
    plt.yticks(fontsize=14, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_binary_cm(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    # Chuẩn hóa ma trận sang %
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
    plt.figure(figsize=(8, 6)) # Tăng kích thước hình
    
    # Vẽ Heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Reds', 
                xticklabels=['Abnormal', 'Normal'], 
                yticklabels=['Abnormal', 'Normal'], 
                vmin=0, vmax=1,
                annot_kws={"size": 15, "weight": "bold"}) # Số liệu trong ô to, đậm
    
    # Cập nhật Tiêu đề và Nhãn trục
    plt.title(f"{title} (%)", fontsize=17, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=15, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=15, fontweight='bold')
    
    # Cập nhật Font chữ cho nhãn (Abnormal, Normal)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold', rotation=0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_unknown_binary_cm(y_true, preds, unknown_label, save_path, Scenario_name):
    y_true = np.array(y_true); preds = np.array(preds)
    if isinstance(unknown_label, list): is_actual = np.isin(y_true, unknown_label).astype(int); label_text = f"List {unknown_label}"
    else: is_actual = (y_true == unknown_label).astype(int); label_text = get_label_name(unknown_label)
    y_bin_pred = (preds == "UNKNOWN").astype(int)
    cm = confusion_matrix(is_actual, y_bin_pred)
    print(f"\n{'-'*40}\n  UNKNOWN DETECTION TABLE: {label_text}\n{'-'*40}")
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
    plt.title(f"Unknown Detection Confusion Matrix - {Scenario_name}"); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300); plt.close()

def plot_metrics_bar(report_dict, title, save_path):
    metrics = ['precision', 'recall', 'f1-score']
    try: values = [report_dict['weighted avg'][m] for m in metrics]
    except: values = [0,0,0]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'])
    for bar in bars: plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom',fontsize=12, fontweight='bold')
    plt.title(title,fontsize=14, fontweight='bold'); plt.ylim(0, 1.15); plt.ylabel('Score',fontsize=12, fontweight='bold'); plt.tight_layout()
    plt.xticks(fontsize=11, fontweight='bold') # Nhãn trục X (tên metrics) lớn hơn
    plt.yticks(fontsize=11)
    os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300); plt.close()

def calculate_unknown_metrics(y_true, preds, unknown_label, save_dir, Scenario_name):
    y_true = np.array(y_true); preds = np.array(preds)
    if isinstance(unknown_label, list): is_actual = np.isin(y_true, unknown_label).astype(int); label_text = f"List {unknown_label}"
    else: is_actual = (y_true == unknown_label).astype(int); label_text = get_label_name(unknown_label)
    y_bin_pred = (preds == "UNKNOWN").astype(int)
    plot_unknown_binary_cm(y_true, preds, unknown_label, f"{save_dir}/unknown_cm_{Scenario_name}.png", Scenario_name)
    return {'precision': precision_score(is_actual, y_bin_pred, zero_division=0), 'recall': recall_score(is_actual, y_bin_pred, zero_division=0), 'f1': f1_score(is_actual, y_bin_pred, zero_division=0)}

def evaluate_supervised_with_unknown(y_true, y_pred, y_conf, atk_thres=0.7, ben_thres=0.7, Scenario_name="", save_dir="", model_name="XGBoost", target_unknown=None):
    print(f"\n--- [METRICS] {model_name} Dual-Thres (Atk>={atk_thres}, Ben>={ben_thres}) - {Scenario_name} ---")
    def map_label(y):
        if isinstance(target_unknown, list): return "UNKNOWN" if y in target_unknown else get_label_name(y)
        elif target_unknown is not None: return "UNKNOWN" if y == target_unknown else get_label_name(y)
        return get_label_name(y)
    y_str_true = [map_label(y) for y in y_true]; y_str_pred = []; stats = {"unk_atk": 0, "unk_ben": 0}
    for p, c in zip(y_pred, y_conf):
        p_val = int(p)
        if p_val != 0:
            if c < atk_thres: y_str_pred.append("UNKNOWN"); stats["unk_atk"] += 1
            else: y_str_pred.append(get_label_name(p_val))
        else:
            if c < ben_thres: y_str_pred.append("UNKNOWN"); stats["unk_ben"] += 1
            else: y_str_pred.append("BENIGN")
    print(f"   -> XGBoost flagged UNKNOWN: {stats['unk_atk']} (Low Conf Atk) + {stats['unk_ben']} (Low Conf Ben)")
    print(classification_report(y_str_true, y_str_pred, digits=4, zero_division=0))
    plot_cm(y_str_true, y_str_pred, f"CM {model_name} (Unknown) - {Scenario_name}", f"{save_dir}/cm_{model_name}_unknown_{Scenario_name}.png")
    rep = classification_report(y_str_true, y_str_pred, output_dict=True, zero_division=0)
    plot_metrics_bar(rep, f"Metrics {model_name} - {Scenario_name}", f"{save_dir}/metrics_{model_name}_unknown_{Scenario_name}.png")
    return rep

def evaluate_supervised_model(y_true, y_pred, Scenario_name, save_dir, model_name="Supervised", return_f1=False):
    print(f"\n--- [METRICS] {model_name} - {Scenario_name} ---")
    y_str_true = [get_label_name(y) for y in y_true]; y_str_pred = [get_label_name(y) for y in y_pred]
    print(classification_report(y_str_true, y_str_pred, digits=4, zero_division=0))
    plot_cm(y_str_true, y_str_pred, f"CM {model_name} - {Scenario_name}", f"{save_dir}/cm_{model_name.lower().replace(' ', '_')}_{Scenario_name}.png")
    rep = classification_report(y_str_true, y_str_pred, digits=4, zero_division=0, output_dict=True)
    plot_metrics_bar(rep, f"Metrics {model_name} - {Scenario_name}", f"{save_dir}/metrics_{model_name.lower().replace(' ', '_')}_{Scenario_name}.png")
    return rep

def evaluate_final_pipeline(y_true, y_pred, sess, save_dir, return_f1=False, map_new_to_unknown=None):
    print(f"\n--- [METRICS] Final Pipeline - {sess} ---")
    
    # Chuyển đổi y_true, y_pred sang String Label cơ bản (Chưa gộp)
    # Gộp sẽ được thực hiện bên trong hàm analyze_and_plot_class_details
    y_str_true = [get_label_name(val) for val in y_true]
    y_str_pred = [val if isinstance(val, str) else get_label_name(val) for val in y_pred]

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
    Vẽ biểu đồ so sánh hiệu năng giữa các cấu hình khác nhau (Ablation Study).
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
    
    # Tăng kích thước hình (16x9)
    plt.figure(figsize=(16, 9))
    
    # Vẽ 3 cột
    plt.bar(x - width, data['Precision'], width, label='Precision', color='#3498db')
    plt.bar(x, data['Recall'], width, label='Recall', color='#2ecc71')
    plt.bar(x + width, data['F1-Score'], width, label='F1-Score', color='#e74c3c')
    
    # Cập nhật font chữ to và đậm hơn
    plt.xlabel('Ablation Scenarios', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=16, fontweight='bold')
    plt.title(f'Performance Comparison of Different Pipeline Configurations - {Scenario_name}', 
              fontsize=20, fontweight='bold', pad=20)
    
    plt.xticks(x, scenarios, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    plt.ylim(0, 1.15)
    
    # Legend to hơn
    plt.legend(loc='upper right', ncol=3, fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Thêm text giá trị lên cột (Font to 12, Đậm)
    for i in range(len(scenarios)):
        # Precision
        plt.text(i - width, data['Precision'][i] + 0.01, 
                 f"{data['Precision'][i]:.4f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=12, color='black')
        
        # Recall
        plt.text(i, data['Recall'][i] + 0.01, 
                 f"{data['Recall'][i]:.4f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=12, color='black')

        # F1-Score
        plt.text(i + width, data['F1-Score'][i] + 0.01, 
                 f"{data['F1-Score'][i]:.4f}", 
                 ha='center', va='bottom', fontweight='bold', fontsize=12, color='black')
        
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
        if i == 0: ax.legend(loc='lower left') # Chỉ hiện legend ở hình đầu cho gọn

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
    ax.set_ylabel('Detection Rate (Recall of New Attacks as UNKNOWN)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Ablation Scenarios', fontsize=14, fontweight='bold')
    ax.set_title('Unknown Threat Detection Comparison (Scenario 1 vs Scenario 2)', fontsize=17, fontweight='bold')
    ax.set_xticks(x, scenarios, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12)
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
                        ha='center', va='bottom', fontsize=14, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Unknown detection comparison chart saved to {save_path}")

def calculate_weighted_metrics(y_true, y_pred, map_new_to_unknown=None):
    """
    Tính toán Accuracy, Precision, Recall, F1 (Weighted) với logic gộp nhãn.
    Giúp tính điểm Pre-IL chính xác: Coi (True: Reconn -> UNKNOWN) == (Pred: UNKNOWN) là ĐÚNG.
    """
    # 1. Map Ground Truth
    y_str_true = []
    for val in y_true:
        # Nếu nhãn nằm trong danh sách cần gộp (vd: [3] cho Reconn) -> Đổi thành "UNKNOWN"
        if map_new_to_unknown and int(val) in map_new_to_unknown:
            y_str_true.append("UNKNOWN")
        else:
            y_str_true.append(get_label_name(val))
            
    # 2. Map Prediction (Pipeline đã trả về string, nhưng cần đảm bảo)
    y_str_pred = []
    # Lấy tên các nhãn target để xử lý
    target_names = [get_label_name(v) for v in (map_new_to_unknown or [])]
    
    for val in y_pred:
        label_name = val if isinstance(val, str) else get_label_name(val)
        # Trong trường hợp hãn hữu pipeline đoán ra nhãn mới ở phase cũ
        if map_new_to_unknown and label_name in target_names:
             y_str_pred.append("UNKNOWN")
        else:
             y_str_pred.append(label_name)

    # 3. Tính toán Metrics
    acc = accuracy_score(y_str_true, y_str_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_str_true, y_str_pred, average='weighted', zero_division=0)
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    }

def plot_pipeline_evolution_comparison(evolution_data, save_path):
    """
    Vẽ biểu đồ cột so sánh tiến trình (Scenario 0 -> 1 Pre -> 1 Post...).
    Input: dictionary dạng:
    {
        'Scenario 0': {'Accuracy': 0.99, ...},
        'Scenario 1 (Pre-IL)': {'Accuracy': 0.98, ...},
        ...
    }
    """
    phases = list(evolution_data.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Chuẩn bị dữ liệu vẽ
    data = {m: [] for m in metrics}
    for phase in phases:
        for m in metrics:
            data[m].append(evolution_data[phase].get(m, 0))
            
    x = np.arange(len(phases))
    width = 0.2
    
    # Tăng kích thước hình (18x10) để các cột và chữ thoáng hơn
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Màu sắc: Xanh dương, Cam, Xanh lá, Đỏ
    colors = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']
    
    for i, metric in enumerate(metrics):
        values = data[metric]
        offset = x + (i - 1.5) * width
        rects = ax.bar(offset, values, width, label=metric, color=colors[i])
        
        # In giá trị lên đầu cột (Font to, Đậm)
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                        f'{height:.4f}',
                        ha='center', va='bottom', 
                        fontsize=12, fontweight='bold') # Tăng fontsize từ 9 lên 12

    # Cập nhật Tiêu đề và Nhãn trục (Font to, Đậm)
    ax.set_ylabel('Score', fontweight='bold', fontsize=16) # Tăng từ 12 lên 16
    ax.set_title('Pipeline Performance Evolution (Pre-IL vs Post-IL)', fontweight='bold', fontsize=20, pad=20) # Tăng lên 20
    
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=14, fontweight='bold') # Tăng lên 14
    
    # Chỉnh font số trục Y
    plt.yticks(fontsize=14)
    
    ax.set_ylim(0, 1.15) # Nới rộng trần thêm chút nữa để số không bị cắt
    
    # Legend to và rõ hơn
    ax.legend(loc='upper right', ncol=4, fontsize=14)
    
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300) # 300 dpi là chuẩn nét, 1000 sẽ rất nặng file
    plt.close()
    print(f"Evolution chart saved to {save_path}")