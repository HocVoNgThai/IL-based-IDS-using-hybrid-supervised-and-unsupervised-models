# STANDARD LIBS
import os, sys
import shutil
from datetime import datetime

# 3rd libs
import gc
import pandas as pd
import numpy as np
from pathlib import Path

# from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

# LOCAL IMPORT 
from src.config.settings import settings
from src.config.incremental_config import incremental_settings

from src.Components.Manager import Manager
from src.Components.Models import AETrainer, IncrementalOCSVM, OpenSetXGBoost
from src.Components.Detector import SequentialHybridPipeline
from src.Utils.utils import *
from src.Utils.func_convert import astype_il


class IncrementalLearning():
    def __init__(self, last_update_time: datetime, current_update_time: datetime, il_data_dir :Path, il_logs_dir:Path, current_index) -> None:
        # self.last_update_time = datetime.strptime(last_update_time, "%Y-%m-%d %H-%M-%S")
        # self.current_update_time = datetime.strptime(current_update_time, "%Y-%m-%d %H-%M-%S")
        self.last_update_time = last_update_time
        self.current_update_time = current_update_time
        self.current_index = current_index
        
        # DIR
        self.il_data_dir = il_data_dir
        self.il_logs_dir = il_logs_dir / f"{datetime.strftime(self.current_update_time, '%Y-%m-%d')}_{self.current_index}"
        self.il_logs_dir.mkdir(parents=True, exist_ok=True)
              
        self.corrupt_dir = Path(il_data_dir / "corrupt_files")  
        # DATA
        self.df = self.load_current_train_data()
        self.should_train= True
        if self.df is None:
            self.should_train = False
            return
        
        # BỎ NHỮNG MẪU CÒN XÓT CHƯA XỬ LÝ ĐI
        labels_to_remove = ["NeedManualLabel", "Unknown"]
        self.df = self.df[~self.df['Label'].isin(labels_to_remove)]
        # print(type(self.df))
        self.preDf = self.load_last_train_data()
        print("[IL] \n", self.preDf["Label"].value_counts())
        gc.collect()
        
        self.dict_current_df , self.sample_per_label = self.calculate_samples()
        
        # Lấy mẫu
        self.df = self.replay_buffer()
        print(self.df["Label"].value_counts())

    def get_timestr(self, name):
        
        name = list(name.split(" "))
        if len(name)>=2:
            name[1]= name[1].replace('-', ':')
            name = ' '.join(name)
            name = name.replace(".parquet", "")
            
            return name
        return ' '.join(name)
    
    def load_current_train_data(self):
        dfs = []
        print("[IL] Các file được tham gia: ")
        for file in self.il_data_dir.glob("*.parquet"):
            
            if file.name.replace(".parquet","") in ["session0", "fixed_budget", "fixed_budget_duphong"]:
                continue
            try:
                name =self.get_timestr(file.name)
                timeobj=datetime.strptime(name, "%Y-%m-%d %H:%M:%S")
            
                # print(self.last_update_time, "" , timeobj, "", self.current_update_time)
                if self.last_update_time < timeobj and timeobj < self.current_update_time:
                    print(" - ", file.name)
                    # print(timeobj)
                    # print(True)
                    dfs.append(pd.read_parquet(file))
            
            except Exception as e:
                print("[INCREMENTAL] Error while loaddata: ", e)
                try:
                    self.corrupt_dir.mkdir(exist_ok=True)
                    shutil.move(str(file), str(self.corrupt_dir / os.path.basename(file)))
                    print(f"[INCREMENTAL] Moved corrupt file to {self.corrupt_dir}")
                except Exception as move_err:
                    print(f"[INCREMENTAL] Could not move corrupt file: {move_err}")
    
                continue
            
            gc.collect()
            
            
        if len(dfs) < 1:
            return None
        
        return pd.concat(dfs, ignore_index= True)
        
    def load_last_train_data(self):
        return pd.read_parquet(incremental_settings.IL_DATA_DIR/"fixed_budget.parquet")
    
    def save_budget(self):
        return
    
    def calculate_samples(self):
        # num_labels = len(incremental_settings.IL_LABEL())
        # target_per_label = incremental_settings.IL_FIXED_MEM_BUDGET // num_labels
        
        dict_df = self.df["Label"].value_counts()
        dict_predf = incremental_settings.IL_LABEL()
        
        dict_newdf = dict_predf.copy()
        for key, val in dict_df.items():
            if key not in dict_predf.keys():
                dict_newdf[key]=val
            else:
                dict_newdf[key] += val
                
        num_labels = len(dict_newdf)
        target_per_label = incremental_settings.IL_FIXED_MEM_BUDGET //num_labels
        
        
        for key, val in dict_predf.items():
            if dict_newdf[key] >= target_per_label:
                dict_newdf[key] = target_per_label
                
        for key, val in dict_df.items():
            if key not in dict_predf.keys():
                if dict_newdf[key] > round(1.2*val):
                    dict_newdf[key] = round(1.2*val)
        
        print(f"[IL] Current (old & new) Labels: {dict_newdf}")
        
        del dict_df, dict_predf
        gc.collect()
        
        return dict_newdf, target_per_label
        
    def replay_buffer(self):
        # NÊN LƯU LẠI VÀO BUDGET CÁI NÀY ĐỂ DỄ DÙNG
        self.df = pd.concat([self.df,self.preDf])
        
        feature_cols = [col for col in self.df.columns if col not in ["Flow ID", "Timestamp", "Label", "Binary Label"]]
        
        size_herding =  int(self.sample_per_label* incremental_settings.herding_replay_ratio)
        # size_random = self.sample_per_label - size_herding
        
        chunks = []
        
        for key, val in self.dict_current_df.items():
            samples_key = self.df[self.df["Label"]==key]
            
            # Nếu mẫu ít hơn sample/label thì append luôn, k cần lựa
            if len(samples_key) < self.sample_per_label:
                chunks.append(samples_key)
                continue
            
            selected_indices = set()

            # HERDING LẤY NHỮNG MẪU GẦN MEAN
            class_features = samples_key[feature_cols].values # get features
            # class_mean = class_features.mean(axis=0) # mean đặc trưng
            class_mean = class_features.mean(axis=0).reshape(1, -1)  # dùng với cdist
            
            # distances = np.linalg.norm(class_features - class_mean, axis=1) # khoảng cách each mẫu -> mean
            distances = cdist(class_features, class_mean, metric='euclidean').flatten() # nhiều features + mẫu
            
            samples_key = samples_key.assign(dist_to_mean=distances) 
        
            herding_idx = samples_key.nsmallest(size_herding, 'dist_to_mean').index
            selected_indices.update(herding_idx)
            
            samples_key = samples_key.drop(columns=['dist_to_mean'])

            remaining_needed = self.sample_per_label - len(selected_indices)
            if remaining_needed > 0:
                potential_random = samples_key.drop(index=list(selected_indices))
                random_idx = potential_random.sample(min(remaining_needed, len(potential_random))).index
                selected_indices.update(random_idx)
                
            chunks.append(samples_key.loc[list(selected_indices)])
            gc.collect()
        
        
        combined_df = pd.concat(chunks, ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return combined_df
        # return pd.concat(chunks).sample(frac=1).reset_index(drop=True)
    
    def split_train_test(self):
        label_mapping = {}
        for index, key in enumerate(self.dict_current_df):
            label_mapping[key]= index
        
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        print("[src.Components.Incremental] Current Label dict: ",label_mapping)
        self.df["Label"]= self.df["Label"].map(label_mapping).astype(np.int8)
        
        
        drop_cols = [col for col in self.df.columns if col in ["Flow ID", "Timestamp", "Label", "Binary Label"]]
        X= self.df.drop(columns=drop_cols, axis = 1)
        
        # X = astype_il(X)
        scaler  = StandardScaler()
        scaler = joblib.load("src/pkl/scaler.joblib")
        
        # X = .astype(np.float32)
        
        X = pd.DataFrame(
            scaler.transform(X.astype(np.float64)),
            columns=X.columns,
            index=X.index
        )
        
        y = self.df["Label"]
        
        del drop_cols
        gc.collect()
        # trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        return train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    def workflow(self):
        print(f"====== Incremental Learning {self.current_index}")
        trainX, testX, trainy, testy = self.split_train_test()
        gc.collect()    
        
        ae = AETrainer(81, 32); ocsvm = IncrementalOCSVM(nu=0.15); xgb = OpenSetXGBoost(0.75)
        mgr = Manager(settings.MODEL_DIR, incremental_settings.IL_MODEL_DIR, self.current_update_time)
        mgr.load_models([xgb, ocsvm, ae])
        print(f"[IL] Model State Loaded: AE: {ae.loaded}, OCSVM: {ocsvm.loaded}, XGB: {xgb.loaded}")
        pipeline = SequentialHybridPipeline(xgb= xgb, ocsvm=ocsvm , ae= ae)
        
        # ==========================
        # PREIL
        # ==========================
        results = {'metrics': {}, 'unknown_stats': {}}
        print("\n--- Phase 1: Detection (Pre-IL) ---")
        preds, details = pipeline.predict(trainX, return_details=True)
        map_unknown = [3]
        # print(f"[IL] UNKNOWN LABELS: {map_unknown}" )
        evaluate_final_pipeline(trainy, preds, f"Scenario_PreIL", self.il_logs_dir, map_new_to_unknown=map_unknown)
        results['unknown_stats']['Pre'] = calculate_unknown_metrics(trainy, preds, unknown_label=3, save_dir=self.il_logs_dir, session_name= f"Scenario_PreIL")

        xgb_pre, xgb_conf = pipeline.xgb.predict_with_confidence(trainX)
        evaluate_supervised_with_unknown(trainy, xgb_pre, xgb_conf, atk_thres=0.7, ben_thres=0.7, session_name="", save_dir=self.il_logs_dir, target_unknown=3)
        evaluate_gray_zone(trainy, xgb_pre, xgb_conf, details['ae_pred'], details['ocsvm_pred'], 0.7, 0.9, f"Scenario_PreIL ", self.il_logs_dir)
        evaluate_unsupervised_detailed(trainy, details['ae_pred'], details['ocsvm_pred'],  f"Scenario_PreIL ", self.il_logs_dir, return_f1=True)
        
        # ===================================
        # INCREMENTAL LEARNING
        # ===================================
        print("\n--- Phase 2: Incremental Learning" )
        pipeline.incremental_learning(trainX, trainy)
        
        # ==================================
        # POST IL
        # ==================================
        print("\n--- Phase 3: Post Incremental Learning")
        final_preds, details_test = pipeline.predict(testX, return_details=True)
        results['metrics']['Pipeline'] = evaluate_final_pipeline(testy, final_preds, "Scenario_PostIL", self.il_logs_dir)
        
        xgb_post, _ = pipeline.xgb.predict_with_confidence(testX)
        results['metrics']['XGBoost'] = evaluate_supervised_model(testy, xgb_post, "Scenario_PostIL", self.il_logs_dir, "XGBoost")
        
        ae_rep, oc_rep = evaluate_unsupervised_detailed(testy, details_test['ae_pred'], details_test['ocsvm_pred'], "Scenario_PostIL", self.il_logs_dir)
        results['metrics']['AE'] = ae_rep
        results['metrics']['OCSVM'] = oc_rep
        
        mgr.save_models([xgb, ocsvm, ae], self.current_index +1)
        
        # loader0 = SessionDataLoader(); loader0.load_scaler(GLOBAL_SCALER_PATH)
        # X0 = loader0.apply_scaling(loader0.load_data_raw(s0_test)[0], fit=False); y0 = loader0.load_data_raw(s0_test)[1]
        # acc0 = accuracy_score([get_label_name(y) for y in y0], pipeline.predict(X0))
        # acc1 = accuracy_score([get_label_name(y) for y in testy], final_preds)
        # results['acc_s0'] = acc0; results['acc_s1'] = acc1
        return # pipeline, results
    
