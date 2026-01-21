import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.pipeline import SequentialHybridPipeline
from src.utils import *

BASE_DATA_DIR = "merge1.4_3-4-5/case-from-3-incre-4class-incre-6class"
GLOBAL_SCALER_PATH = "Scenarios/global_scaler.joblib"
save_dir = "results/check"
unknown_stats = {
    'Scenario 1': {},
    'Scenario 2': {}
}
def load_models_for_scenario(scenario_id, mgr):
    """Load lại 3 mô hình (AE, OCSVM, XGB) từ checkpoint đã lưu"""
    print(f"   -> Loading models from Scenario {scenario_id}...")
    ae = AETrainer(81, 32)
    ocsvm = IncrementalOCSVM(nu=0.15)
    xgb = OpenSetXGBoost(0.7)
    
    models = {'ae.pt': ae, 'ocsvm.pkl': ocsvm, 'xgb.pkl': xgb}
    mgr.load_models(scenario_id, models)
    
    return SequentialHybridPipeline(xgb=xgb, ae=ae, ocsvm=ocsvm)

def run_evaluation():
    print("🚀 STARTING COMPREHENSIVE EVALUATION")
    os.makedirs(save_dir, exist_ok=True)
    
    loader = ScenarioDataLoader()
    loader.load_scaler(GLOBAL_SCALER_PATH)
    mgr = ScenarioManager()
    
    il_metrics = ILMetrics()
    evolution_history = {}

    # ==============================================================================
    # GIAI ĐOẠN 1: SCENARIO 0 (Baseline)
    # ==============================================================================
    print(f"\n{'='*10} Scenario 0: EVALUATION {'='*10}")
    
    X_test0, y_test0 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario0.parquet"))
    X_test0 = loader.apply_scaling(X_test0, fit=False)
    pipeline0 = load_models_for_scenario(0, mgr)

    preds0 = pipeline0.predict(X_test0)
    metrics0 = calculate_weighted_metrics(y_test0, preds0)
    evolution_history['Scenario 0'] = metrics0
    il_metrics.record(tr_sess=0, te_sess=0, acc=metrics0['Accuracy'])
    il_metrics.calculate_metrics(current_step=0)

    # ==============================================================================
    # GIAI ĐOẠN 2: SCENARIO 1 (Reconn)
    # ==============================================================================
    print(f"\n{'='*10} Scenario 1: PRE vs POST IL {'='*10}")
    X_train1, y_train1 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "train_Scenario1.parquet")) # Dùng train để check detection
    X_test1, y_test1 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario1.parquet"))
    X_train1 = loader.apply_scaling(X_train1); X_test1 = loader.apply_scaling(X_test1)

    print(">>> Scenario 1 [Pre-IL]: Checking Detection of Unknown (Reconn)...")
    preds1_pre = pipeline0.predict(X_train1) 
    
    unknown_stats['Scenario 1']['Pre'] = calculate_unknown_metrics(y_train1, preds1_pre, unknown_label=3, save_dir = save_dir, Scenario_name="Scenario1_PreIL")
    metrics1_pre = calculate_weighted_metrics(y_train1, preds1_pre, map_new_to_unknown=[3])
    evolution_history['Scenario 1 (Pre-IL)\n(+Reconn)'] = metrics1_pre

    print(">>> Scenario 1 [Post-IL]: Checking Classification after Learning...")
    pipeline1 = load_models_for_scenario(1, mgr)
    preds1_post = pipeline1.predict(X_test1)
    
    metrics1_post = calculate_weighted_metrics(y_test1, preds1_post)
    evolution_history['Scenario 1 (Post-IL)\n(+Reconn)'] = metrics1_post
    il_metrics.record(tr_sess=1, te_sess=1, acc=metrics1_post['Accuracy'])
    print("   -> Checking Stability on Scenario 0...")
    preds0_re = pipeline1.predict(X_test0)
    acc0_re = calculate_weighted_metrics(y_test0, preds0_re)['Accuracy']
    il_metrics.record(tr_sess=1, te_sess=0, acc=acc0_re)
    
    il_metrics.calculate_metrics(current_step=1)

    # ==============================================================================
    # GIAI ĐOẠN 3: SCENARIO 2 (MITM & DNS)
    # ==============================================================================
    print(f"\n{'='*10} Scenario 2: PRE vs POST IL {'='*10}")
    X_train2, y_train2 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "train_Scenario2.parquet"))
    X_test2, y_test2 = loader.load_data_raw(os.path.join(BASE_DATA_DIR, "test_Scenario2.parquet"))
    X_train2 = loader.apply_scaling(X_train2); X_test2 = loader.apply_scaling(X_test2)

    print(">>> Scenario 2 [Pre-IL]: Checking Detection of Unknown (MITM, DNS)...")
    preds2_pre = pipeline1.predict(X_train2)
    
    unknown_stats['Scenario 2']['Pre'] = calculate_unknown_metrics(y_train2, preds2_pre, [4, 5], save_dir, Scenario_name="Scenario1_PreIL")
    metrics2_pre = calculate_weighted_metrics(y_train2, preds2_pre, map_new_to_unknown=[4, 5])
    evolution_history['Scenario 2 (Pre-IL)\n(+ MITM&DNS Spoofing)'] = metrics2_pre
    print(">>> Scenario 2 [Post-IL]: Checking Classification after Learning...")
    pipeline2 = load_models_for_scenario(2, mgr)
    preds2_post = pipeline2.predict(X_test2)
    
    metrics2_post = calculate_weighted_metrics(y_test2, preds2_post)
    evolution_history['Scenario 2 (Post-IL)\n(+ MITM&DNS Spoofing)'] = metrics2_post

    il_metrics.record(tr_sess=2, te_sess=2, acc=metrics2_post['Accuracy'])

    print("   -> Checking Stability on Scenario 0...")
    preds0_re2 = pipeline2.predict(X_test0)
    il_metrics.record(tr_sess=2, te_sess=0, acc=calculate_weighted_metrics(y_test0, preds0_re2)['Accuracy'])
    
    print("   -> Checking Stability on Scenario 1...")
    preds1_re2 = pipeline2.predict(X_test1)
    il_metrics.record(tr_sess=2, te_sess=1, acc=calculate_weighted_metrics(y_test1, preds1_re2)['Accuracy'])
    
    il_metrics.calculate_metrics(current_step=2)

    print(f"\n{'='*10} GENERATING CHARTS {'='*10}")

    plot_pipeline_evolution_comparison(
        evolution_history, 
        os.path.join(save_dir, "pipeline_evolution_comparison.png")
    )

    plot_il_matrix(
        il_metrics, 
        os.path.join(save_dir, "il_matrix.png")
    )

    plot_il_metrics_trends(
        il_metrics, 
        os.path.join(save_dir, "il_trends.png")
    )
    
    plot_unknown_detection_performance(unknown_stats, save_dir)
    print(f"\n✅ All results saved to: {save_dir}")

if __name__ == "__main__":
    run_evaluation()