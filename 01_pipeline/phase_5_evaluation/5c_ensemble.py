#!/usr/bin/env python3
"""
5c — Ensemble Models
Soft Voting, Hard Voting, Stacking, Threshold Tuning
"""

import os, json, warnings
from datetime import datetime

# === SCRIPT LOGGING (auto-injected) ===
import sys

class _TeeLogger:
    """Write stdout to both terminal and .script_logs/<script>.log"""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')
        self.log.write(f"\n{'='*70}\n")
        self.log.write(f"RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*70}\n")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.write(f"\nRUN FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.close()

_LOG_PATH = os.path.join(".script_logs", os.path.basename(__file__) + ".log")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
sys.stdout = _TeeLogger(_LOG_PATH)
sys.stderr = sys.stdout
# === END SCRIPT LOGGING ===

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import xgboost as xgb


def _spw(y):
    """scale_pos_weight = rasio negatif/positif pada data latih (bukan konstanta)."""
    y = np.asarray(y)
    return float((len(y) - y.sum()) / max(y.sum(), 1))


DATA_PATH = "../../00_data/processed/modeling_dataset.csv"
OUTPUT_DIR = "results"
PLOTS_DIR = "plots"
CAT_COLS = ['tectonic_zone', 'era', 'season']
FEATURE_COLS = ['mag', 'depth', 'nst', 'gap', 'dmin', 'rms',
                'latitude', 'longitude', 'abs_latitude',
                'depth_mag_ratio', 'monitoring_quality']


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    feat_cols = [c for c in FEATURE_COLS + [c for c in df_enc if c.startswith(('tectonic_zone_', 'era_', 'season_'))] if c in df_enc.columns]
    X = np.nan_to_num(df_enc[feat_cols].values, nan=0.0)
    y = df_enc['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)
    return Xs_train, Xs_test, y_train, y_test


def eval_model(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    result = {
        'f1': round(float(f1_score(y_true, y_pred)), 4),
        'recall': round(float(recall_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred)), 4),
        'cost': int(fn * 100 + fp * 1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }
    if y_prob is not None:
        result['auc'] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("5c: ENSEMBLE MODELS")
    print("=" * 60)

    Xs_train, Xs_test, y_train, y_test = prepare_data()

    # Base models
    lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=42, scale_pos_weight=_spw(y_train),
                               eval_metric='logloss', use_label_encoder=False)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42,
                                n_jobs=-1)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                        random_state=42, early_stopping=True)

    lr.fit(Xs_train, y_train)
    xgb_m.fit(Xs_train, y_train)
    rf.fit(Xs_train, y_train)
    mlp.fit(Xs_train, y_train)

    results = {}

    # --- Single models ---
    print("\n--- Single Models ---")
    for name, model in [('XGBoost', xgb_m), ('LogisticRegression', lr), ('RandomForest', rf), ('MLP', mlp)]:
        y_pred = model.predict(Xs_test)
        y_prob = model.predict_proba(Xs_test)[:, 1]
        results[name] = eval_model(y_test, y_pred, y_prob)
        print(f"  {name:20s}: F1={results[name]['f1']:.4f}, Recall={results[name]['recall']:.4f}")

    # --- XGBoost threshold 0.36 ---
    print("\n--- XGBoost (threshold=0.36) ---")
    y_prob_xgb = xgb_m.predict_proba(Xs_test)[:, 1]
    y_pred_t = (y_prob_xgb >= 0.36).astype(int)
    results['xgb_threshold_036'] = eval_model(y_test, y_pred_t, y_prob_xgb)
    print(f"  F1={results['xgb_threshold_036']['f1']:.4f}, "
          f"Recall={results['xgb_threshold_036']['recall']:.4f}, "
          f"Cost={results['xgb_threshold_036']['cost']:,}")

    # --- Soft Voting ---
    print("\n--- Soft Voting (LR + XGB + RF) ---")
    vote_soft = VotingClassifier(
        estimators=[('lr', lr), ('xgb', xgb_m), ('rf', rf)],
        voting='soft', weights=[1, 2, 1])
    vote_soft.fit(Xs_train, y_train)
    y_pred = vote_soft.predict(Xs_test)
    y_prob = vote_soft.predict_proba(Xs_test)[:, 1]
    results['soft_voting'] = eval_model(y_test, y_pred, y_prob)
    print(f"  F1={results['soft_voting']['f1']:.4f}, Recall={results['soft_voting']['recall']:.4f}")

    # --- Hard Voting ---
    print("\n--- Hard Voting (LR + XGB + RF) ---")
    vote_hard = VotingClassifier(
        estimators=[('lr', lr), ('xgb', xgb_m), ('rf', rf)],
        voting='hard')
    vote_hard.fit(Xs_train, y_train)
    y_pred = vote_hard.predict(Xs_test)
    results['hard_voting'] = eval_model(y_test, y_pred)
    print(f"  F1={results['hard_voting']['f1']:.4f}, Recall={results['hard_voting']['recall']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "ensemble_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/ensemble_results.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
