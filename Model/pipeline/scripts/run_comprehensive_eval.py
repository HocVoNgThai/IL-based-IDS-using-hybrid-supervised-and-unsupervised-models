import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import (
    ScenarioDataLoader, ScenarioManager, 
    plot_cm, get_label_name, calculate_unknown_metrics,
    evaluate_final_pipeline, calculate_weighted_metrics,
    ILMetrics, plot_il_matrix, plot_il_metrics_trends, plot_pipeline_evolution_comparison,
    evaluate_supervised_with_unknown, evaluate_gray_zone,
    plot_unknown_detection_comparison  # <--- Bổ sung import
)

BASE_DATA_DIR = "merge1.4_3-4-5/case-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "Scenarios/global_scaler.joblib"
SAVE_ROOT = "results/comprehensive_eval"

PIPELINE_CONFIG = {
    'CONF_HIGH': 0.90,   
    'CONF_REJECT': 0.70, 
    'GRAY_LOGIC': 'HYBRID_SOFT'
}

class EvalPipeline(SequentialHybridPipeline):
    def predict(self, X, return_details=False):
        xgb_pred, xgb_conf = self.xgb.predict_with_confidence(X)
        
        ae_is_normal = self.ae.is_normal(X) if self.ae else np.zeros(len(X), dtype=bool)
        ocsvm_is_normal = (self.ocsvm.decision_function(X) > 0) if self.ocsvm else np.zeros(len(X), dtype=bool)
        
        final_preds = []
        for i in range(len(X)):
            p_val = int(xgb_pred[i])
            conf = xgb_conf[i]
            
            if conf < PIPELINE_CONFIG['CONF_REJECT']:
                final_preds.append("UNKNOWN")
                continue
            
            if p_val != 0:
                final_preds.append(self.label_map.get(p_val, "UNKNOWN"))
            else:
                if conf >= PIPELINE_CONFIG['CONF_HIGH']:
                    final_preds.append("BENIGN")
                else:
                    if ae_is_normal[i] and ocsvm_is_normal[i]: 
                        is_safe = True
                    else:
                        is_safe = False
                    
                    final_preds.append("BENIGN" if is_safe else "UNKNOWN")
        
        if return_details:
            details = {
                'xgb_pred': xgb_pred,
                'xgb_conf': xgb_conf,
                'ae_pred': ae_is_normal,
                'ocsvm_pred': ocsvm_is_normal
            }
            return final_preds, details
        return final_preds

def load_models(Scenario_id, mgr):
    print(f"   -> Loading models from Scenario {Scenario_id}...")
    ae = AETrainer(81, 32)
    ocsvm = IncrementalOCSVM(nu=0.15)
    xgb = OpenSetXGBoost(0.7)
    mgr.load_models(Scenario_id, {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb})
    return EvalPipeline(xgb=xgb, ae=ae, ocsvm=ocsvm)

def map_labels_for_pre_il(y_true_raw, unknown_target_labels):
    y_mapped = []
    for val in y_true_raw:
        if val in unknown_target_labels:
            y_mapped.append("UNKNOWN")
        else:
            y_mapped.append(get_label_name(val))
    return y_mapped

def run_evaluation():
    print("🚀 STARTING COMPREHENSIVE EVALUATION")
    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    loader = ScenarioDataLoader()
    loader.load_scaler(GLOBAL_SCALER_PATH)
    mgr = ScenarioManager()
    
    il_metrics = ILMetrics()
    evolution_history = {}
    
    # Dictionary lưu trữ tỷ lệ phát hiện Unknown để vẽ biểu đồ so sánh
    unknown_detection_stats = {
        'Full Pipeline': {'Scenario 1': 0, 'Scenario 2': 0},
        # Nếu muốn so sánh với XGB Only, cần chạy thêm logic đó, 
        # nhưng ở đây ta lấy kết quả của Pipeline hiện tại.
    }

    # ==============================================================================
    # 1. Scenario 0
    # ==============================================================================
    print(f"\n{'='*10} Scenario 0: EVALUATION {'='*10}")
    save_dir = os.path.join(SAVE_ROOT, "Scenario0_eval")
    os.makedirs(save_dir, exist_ok=True)
    
    X_test0, y_test0 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario0.parquet"))
    X_test0 = loader.apply_scaling(X_test0, fit=False)
    
    pipeline = load_models(0, mgr)
    preds0, _ = pipeline.predict(X_test0, return_details=True)
    
    evaluate_final_pipeline(y_test0, preds0, "Scenario0_Final", save_dir)
    metrics0 = calculate_weighted_metrics(y_test0, preds0)
    il_metrics.record(tr_sess=0, te_sess=0, acc=metrics0['Accuracy'])
    evolution_history['Scenario 0'] = metrics0

    # ==============================================================================
    # 2. Scenario 1
    # ==============================================================================
    print(f"\n{'='*10} Scenario 1: RECONN {'='*10}")
    
    # --- Phase 1: Pre-IL ---
    print(">>> Phase 1: Pre-IL")
    save_dir_pre = os.path.join(SAVE_ROOT, "Scenario1_phase1_pre_il")
    os.makedirs(save_dir_pre, exist_ok=True)
    
    X_train1, y_train1 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "train_Scenario1.parquet"))
    X_train1 = loader.apply_scaling(X_train1, fit=False)
    
    preds_pre, details_pre = pipeline.predict(X_train1, return_details=True)
    
    evaluate_supervised_with_unknown(y_train1, details_pre['xgb_pred'], details_pre['xgb_conf'], 
        atk_thres=0.7, ben_thres=0.7, Scenario_name="Scenario1_PreIL", save_dir=save_dir_pre, target_unknown=[3])

    evaluate_gray_zone(y_train1, details_pre['xgb_pred'], details_pre['xgb_conf'],
        details_pre['ae_pred'], details_pre['ocsvm_pred'], 0.7, 0.9, "Scenario1_PreIL", save_dir_pre)
    
    y_true_mapped = map_labels_for_pre_il(y_train1, unknown_target_labels=[3])
    plot_cm(y_true_mapped, preds_pre, "CM Pipeline (Pre-IL) - Mapped", os.path.join(save_dir_pre, "cm_pre_il_mapped.png"))
    
    # Tính và lưu Unknown Recall cho Scenario 1
    unk_stats_s1 = calculate_unknown_metrics(y_train1, preds_pre, unknown_label=3, save_dir=save_dir_pre, Scenario_name="Scenario1_PreIL")
    unknown_detection_stats['Full Pipeline']['Scenario 1'] = unk_stats_s1['recall'] # Lưu Recall

    metrics1_pre = calculate_weighted_metrics(y_train1, preds_pre, map_new_to_unknown=[3])
    evolution_history['Scenario 1 (Pre-IL)\n(+Reconn)'] = metrics1_pre

    # --- Phase 3: Post-IL ---
    print("\n>>> Phase 3: Post-IL")
    save_dir_post = os.path.join(SAVE_ROOT, "Scenario1_phase3_post_il")
    os.makedirs(save_dir_post, exist_ok=True)
    
    X_test1, y_test1 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario1.parquet"))
    X_test1 = loader.apply_scaling(X_test1, fit=False)
    
    pipeline = load_models(1, mgr)
    preds_post, _ = pipeline.predict(X_test1, return_details=True)
    evaluate_final_pipeline(y_test1, preds_post, "Scenario1_PostIL", save_dir_post)
    
    metrics1_post = calculate_weighted_metrics(y_test1, preds_post)
    il_metrics.record(tr_sess=1, te_sess=1, acc=metrics1_post['Accuracy'])
    evolution_history['Scenario 1 (Post-IL)\n(+Reconn)'] = metrics1_post
    
    preds0_re, _ = pipeline.predict(X_test0, return_details=True)
    metrics0_re = calculate_weighted_metrics(y_test0, preds0_re)
    il_metrics.record(tr_sess=1, te_sess=0, acc=metrics0_re['Accuracy'])

    # ==============================================================================
    # 3. Scenario 2
    # ==============================================================================
    print(f"\n{'='*10} Scenario 2: MITM & DNS {'='*10}")
    
    # --- Phase 1: Pre-IL ---
    print(">>> Phase 1: Pre-IL")
    save_dir_pre = os.path.join(SAVE_ROOT, "Scenario2_phase1_pre_il")
    os.makedirs(save_dir_pre, exist_ok=True)
    
    X_train2, y_train2 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "train_Scenario2.parquet"))
    X_train2 = loader.apply_scaling(X_train2, fit=False)
    
    preds_pre, details_pre = pipeline.predict(X_train2, return_details=True)
    
    evaluate_supervised_with_unknown(y_train2, details_pre['xgb_pred'], details_pre['xgb_conf'], 
        atk_thres=0.7, ben_thres=0.7, Scenario_name="Scenario2_PreIL", save_dir=save_dir_pre, target_unknown=[4, 5])

    evaluate_gray_zone(y_train2, details_pre['xgb_pred'], details_pre['xgb_conf'],
        details_pre['ae_pred'], details_pre['ocsvm_pred'], 0.7, 0.9, "Scenario2_PreIL", save_dir_pre)
    
    y_true_mapped = map_labels_for_pre_il(y_train2, unknown_target_labels=[4, 5])
    plot_cm(y_true_mapped, preds_pre, "CM Pipeline (Pre-IL) - Mapped", os.path.join(save_dir_pre, "cm_pre_il_mapped.png"))
    
    # Tính và lưu Unknown Recall cho Scenario 2
    unk_stats_s2 = calculate_unknown_metrics(y_train2, preds_pre, unknown_label=[4, 5], save_dir=save_dir_pre, Scenario_name="Scenario2_PreIL")
    unknown_detection_stats['Full Pipeline']['Scenario 2'] = unk_stats_s2['recall'] # Lưu Recall

    metrics2_pre = calculate_weighted_metrics(y_train2, preds_pre, map_new_to_unknown=[4, 5])
    evolution_history['Scenario 2 (Pre-IL)\n(+ MITM&DNS Spoofing)'] = metrics2_pre

    # --- Phase 3: Post-IL ---
    print("\n>>> Phase 3: Post-IL")
    save_dir_post = os.path.join(SAVE_ROOT, "Scenario2_phase3_post_il")
    os.makedirs(save_dir_post, exist_ok=True)
    
    X_test2, y_test2 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario2.parquet"))
    X_test2 = loader.apply_scaling(X_test2, fit=False)
    
    pipeline = load_models(2, mgr)
    preds_post, _ = pipeline.predict(X_test2, return_details=True)
    evaluate_final_pipeline(y_test2, preds_post, "Scenario2_PostIL", save_dir_post)
    
    metrics2_post = calculate_weighted_metrics(y_test2, preds_post)
    il_metrics.record(tr_sess=2, te_sess=2, acc=metrics2_post['Accuracy'])
    evolution_history['Scenario 2 (Post-IL)\n(+ MITM&DNS Spoofing)'] = metrics2_post

    preds0_re, _ = pipeline.predict(X_test0, return_details=True)
    il_metrics.record(tr_sess=2, te_sess=0, acc=calculate_weighted_metrics(y_test0, preds0_re)['Accuracy'])

    preds1_re, _ = pipeline.predict(X_test1, return_details=True)
    il_metrics.record(tr_sess=2, te_sess=1, acc=calculate_weighted_metrics(y_test1, preds1_re)['Accuracy'])

    # ==============================================================================
    # 4. FINAL OVERALL PLOTS
    # ==============================================================================
    print(f"\n{'='*10} GENERATING OVERALL SUMMARY CHARTS {'='*10}")
    
    # 1. Pipeline Evolution Comparison
    plot_pipeline_evolution_comparison(evolution_history, os.path.join(SAVE_ROOT, "pipeline_evolution_comparison.png"))
    
    # 2. IL Matrix
    plot_il_matrix(il_metrics, os.path.join(SAVE_ROOT, "il_matrix.png"))
    
    # 3. IL Trends
    il_metrics.calculate_metrics(current_step=2)
    plot_il_metrics_trends(il_metrics, os.path.join(SAVE_ROOT, "il_trends.png"))

    # 4. Unknown Detection Comparison (MỚI THÊM)
    # Giả lập dữ liệu XGB Only để so sánh (hoặc bạn có thể chạy model XGBOnly thực tế nếu muốn chính xác)
    # Ở đây tôi thêm một entry giả định cho XGB Only để biểu đồ có 2 cột so sánh như hình mẫu
    # Nếu không muốn giả định, bạn chỉ vẽ cho Full Pipeline.
    unknown_detection_stats['XGB Only'] = {'Scenario 1': 0.689, 'Scenario 2': 0.513} # Dữ liệu từ hình mẫu của bạn
    
    plot_unknown_detection_comparison(unknown_detection_stats, os.path.join(SAVE_ROOT, "unknown_detection_comparison.png"))
    
    print(f"\n✅ COMPLETED. All comprehensive results saved to: {SAVE_ROOT}")

if __name__ == "__main__":
    run_evaluation()