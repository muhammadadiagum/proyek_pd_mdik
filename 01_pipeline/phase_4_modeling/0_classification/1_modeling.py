#!/usr/bin/env python3
"""
1_modeling.py — Phase 4: Modeling (4 Models × 5 Imbalance Methods)
=====================================================================
Models: Logistic Regression, Random Forest, XGBoost, MLP
Imbalance: Original, Random Oversampling, SMOTE, ADASYN, Random Undersampling
"""

import os, sys, json, time, gc, warnings
warnings.filterwarnings('ignore')
# ============================================================
# LOGGING: Tee stdout -> console + file
# ============================================================
class Tee:
    def __init__(self, log_path):
        self.stdout = sys.stdout
        self.log_file = open(log_path, 'w')
    def write(self, data):
        self.stdout.write(data)
        self.log_file.write(data)
        self.stdout.flush()
        self.log_file.flush()
    def flush(self):
        self.stdout.flush()
        self.log_file.flush()
    def close(self):
        self.log_file.close()
        sys.stdout = self.stdout

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.script_logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, '1_modeling.py.log')
tee = Tee(LOG_PATH)
sys.stdout = tee


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../../../00_data/processed/modeling_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
CAT_COLS = ['tectonic_zone', 'era', 'season']
FEATURE_COLS = ['mag', 'depth', 'nst', 'gap', 'dmin', 'rms',
                'latitude', 'longitude', 'abs_latitude',
                'depth_mag_ratio', 'monitoring_quality']
FN_COST = 100
FP_COST = 1


def evaluate(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    r = {
        'f1': round(float(f1_score(y_true, y_pred)), 4),
        'recall': round(float(recall_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'cost': int(fn * FN_COST + fp * FP_COST),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        r['auc'] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return r


def get_models():
    return {
        'LogisticRegression': {
            'model_class': LogisticRegression,
            'params': dict(max_iter=2000, random_state=42,
                           class_weight='balanced', C=1.0, penalty='l2'),
            'needs_scaling': True
        },
        'RandomForest': {
            'model_class': RandomForestClassifier,
            'params': dict(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            'needs_scaling': False
        },
        'XGBoost': {
            'model_class': XGBClassifier,
            'params': dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                           random_state=42,
                           eval_metric='logloss', use_label_encoder=False),
            'needs_scaling': False
        },
        # SVM excluded: O(n²-n³) kernel matrix — empirically 44+ min
        # on 248k rows (Original), F1=0.106. Not practical for this
        # dataset size.
        'MLP': {
            'model_class': MLPClassifier,
            'params': dict(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                           random_state=42, early_stopping=True),
            'needs_scaling': True
        },
    }


def get_imbalance_methods():
    return {
        'Original': None,
        'Random Oversampling': RandomOverSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'Random Undersampling': RandomUnderSampler(random_state=42),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_start = time.time()

    print("=" * 70)
    print("PHASE 4: MODELING (4 Models × 5 Imbalance Methods)")
    print("=" * 70)

    # Load
    df = pd.read_csv(DATA_PATH)
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    feat_cols = [c for c in FEATURE_COLS + [c for c in df_enc if c.startswith(('tectonic_zone_', 'era_', 'season_'))] if c in df_enc.columns]
    X = np.nan_to_num(df_enc[feat_cols].values, nan=0.0)
    y = df_enc['label'].values
    print(f"Data: {len(y):,} rows, {X.shape[1]} features")
    print(f"Tsunami: {sum(y==1)} ({sum(y==1)/len(y)*100:.3f}%)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(y_train):,}, Test: {len(y_test):,}")
    del X, y; gc.collect()

    # scale_pos_weight dihitung dari data latih (bukan konstanta warisan label lama)
    spw = float((len(y_train) - y_train.sum()) / max(int(y_train.sum()), 1))

    models_cfg = get_models()
    models_cfg['XGBoost']['params']['scale_pos_weight'] = spw
    print(f"  XGB scale_pos_weight: {spw:.2f}")
    imbalance = get_imbalance_methods()
    results = {}
    run_num = 0
    total_runs = len(imbalance) * len(models_cfg)

    for method_name, sampler in imbalance.items():
        print(f"\n{'='*70}")
        print(f"  METHOD: {method_name}")
        print(f"{'='*70}")

        t0 = time.time()
        if sampler is not None:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            print(f"  Resampled: {len(y_train):,} → {len(y_res):,}")
        else:
            X_res, y_res = X_train, y_train
            print(f"  Original: {len(y_res):,}")
        print(f"  Resample time: {time.time()-t0:.1f}s")

        method_results = {}
        for model_name, cfg in models_cfg.items():
            run_num += 1
            print(f"\n  [{run_num}/{total_runs}] {model_name} ({method_name})")
            t0 = time.time()

            if cfg['needs_scaling']:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X_res)
                Xs_test = scaler.transform(X_test)
                del scaler
            else:
                Xs = X_res
                Xs_test = X_test

            model = cfg['model_class'](**cfg['params'])
            model.fit(Xs, y_res)
            y_pred = model.predict(Xs_test)
            y_prob = model.predict_proba(Xs_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)

            elapsed = time.time() - t0
            m = evaluate(y_test, y_pred, y_prob)
            method_results[model_name] = m
            print(f"    F1={m['f1']:.4f}, Recall={m['recall']:.4f}, AUC={m.get('auc','N/A')}, Cost={m['cost']:,}  ({elapsed:.1f}s)")

            del model, y_pred, y_prob
            if cfg['needs_scaling']:
                del Xs, Xs_test
            gc.collect()

        results[method_name] = method_results

        # Save incrementally — crash-safe
        with open(os.path.join(OUTPUT_DIR, "model_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  [SAVED] results/model_results.json ({method_name} done)")

        del X_res, y_res
        gc.collect()

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print("FULL RESULTS TABLE")
    print(f"{'='*70}")
    model_names = list(models_cfg.keys())
    header = f"{'Method':25s}"
    for n in model_names:
        header += f" {'F1('+n[:4]+')':>12s}"
    print(header)
    print("-" * (25 + 13 * len(model_names)))
    for method_name in results:
        row = f"{method_name:25s}"
        for n in model_names:
            row += f" {results[method_name][n]['f1']:>12.4f}"
        print(row)

    # Best per model
    print(f"\n{'='*70}")
    print("BEST PER MODEL")
    print(f"{'='*70}")
    for n in model_names:
        best_m = max(results.keys(), key=lambda m: results[m][n]['f1'])
        b = results[best_m][n]
        print(f"  {n:20s}: Best={best_m}, F1={b['f1']:.4f}, Cost={b['cost']:,}")

    # Save
    with open(os.path.join(OUTPUT_DIR, "model_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: results/model_results.json")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 70)


if __name__ == '__main__':
    main()
