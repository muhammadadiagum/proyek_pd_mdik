#!/usr/bin/env python3
"""
5e — Deeper Imbalance Analysis
5 metode sampling + perbandingan XGBoost
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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

plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 150,
})


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 60)
    print("5e: DEEPER IMBALANCE ANALYSIS")
    print("=" * 60)

    Xs_train, Xs_test, y_train, y_test = prepare_data()

    methods = {
        'Original': (Xs_train, y_train),
    }

    ros = RandomOverSampler(random_state=42)
    X_ros, y_ros = ros.fit_resample(Xs_train, y_train)
    methods['Random Oversampling'] = (X_ros, y_ros)

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(Xs_train, y_train)
    methods['SMOTE'] = (X_smote, y_smote)

    adasyn = ADASYN(random_state=42)
    X_adasyn, y_adasyn = adasyn.fit_resample(Xs_train, y_train)
    methods['ADASYN'] = (X_adasyn, y_adasyn)

    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(Xs_train, y_train)
    methods['Random Undersampling'] = (X_rus, y_rus)

    results = {}
    print(f"\n  {'Method':30s} | {'Train':>10s} | {'Pos%':>7s} | {'F1':>7s} | {'Recall':>7s} | {'Cost':>8s}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

    for name, (X_res, y_res) in methods.items():
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, scale_pos_weight=_spw(y_res),
            eval_metric='logloss', use_label_encoder=False)
        model.fit(X_res, y_res)
        y_pred = model.predict(Xs_test)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        f1 = float(f1_score(y_test, y_pred))
        recall = float(recall_score(y_test, y_pred))
        cost = fn * 100 + fp * 1
        pos_pct = sum(y_res == 1) / len(y_res) * 100

        results[name] = {
            'train_size': int(len(y_res)),
            'positive_pct': round(pos_pct, 2),
            'f1': round(f1, 4),
            'recall': round(recall, 4),
            'cost': int(cost),
        }
        print(f"  {name:30s} | {len(y_res):>10,} | {pos_pct:>6.1f}% | {f1:>7.4f} | {recall:>7.4f} | {cost:>8,}")

    # Plot
    names = list(results.keys())
    f1_vals = [results[n]['f1'] for n in names]
    cost_vals = [results[n]['cost'] for n in names]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(names, f1_vals, color=colors, edgecolor='white')
    axes[0].set_xlabel('F1-Score')
    axes[0].set_title('F1-Score by Imbalance Method (XGBoost)', fontweight='bold')
    axes[1].barh(names, cost_vals, color=colors, edgecolor='white')
    axes[1].set_xlabel('Cost (FN×100 + FP×1)')
    axes[1].set_title('Cost by Imbalance Method (XGBoost)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'imbalance_analysis.png'), bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] plots/imbalance_analysis.png")

    with open(os.path.join(OUTPUT_DIR, "imbalance_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: results/imbalance_results.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
