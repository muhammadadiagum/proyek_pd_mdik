#!/usr/bin/env python3
"""
5g — Ablation Study: monitoring_quality
Compare model WITH vs WITHOUT monitoring/quality features
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
BASE_FEATURES = ['mag', 'depth', 'latitude', 'longitude', 'abs_latitude', 'depth_mag_ratio']
QUALITY_FEATURES = ['nst', 'gap', 'dmin', 'rms', 'monitoring_quality']

plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 150,
})


def prepare_consistent_split(df_train, df_test, feature_cols):
    """Encode train+test together for consistent columns."""
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_enc = pd.get_dummies(df_all, columns=CAT_COLS, drop_first=False)
    all_cols = [c for c in feature_cols + [c for c in df_enc if c.startswith(('tectonic_zone_', 'era_', 'season_'))] if c in df_enc.columns]
    n_train = len(df_train)
    X_all = np.nan_to_num(df_enc[all_cols].values, nan=0.0)
    y_all = df_enc['label'].values
    return X_all[:n_train], y_all[:n_train], X_all[n_train:], y_all[n_train:], all_cols


def eval_model(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    r = {
        'f1': round(float(f1_score(y_true, y_pred)), 4),
        'recall': round(float(recall_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred)), 4),
        'cost': int(fn * 100 + fp * 1),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        r['auc'] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return r


def train_and_eval(X_train, y_train, X_test, y_test, model_name):
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    if model_name == 'XGBoost':
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=42, scale_pos_weight=_spw(y_train),
                               eval_metric='logloss', use_label_encoder=False)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        y_prob = m.predict_proba(X_test)[:, 1]
    elif model_name == 'XGB (th=0.36)':
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=42, scale_pos_weight=_spw(y_train),
                               eval_metric='logloss', use_label_encoder=False)
        m.fit(X_train, y_train)
        y_prob = m.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.36).astype(int)
    elif model_name == 'Soft Voting':
        lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
        xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                   random_state=42, scale_pos_weight=_spw(y_train),
                                   eval_metric='logloss', use_label_encoder=False)
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42,
                                    n_jobs=-1)
        lr.fit(X_train, y_train)
        xgb_m.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        p1 = lr.predict_proba(X_test)[:, 1]
        p2 = xgb_m.predict_proba(X_test)[:, 1]
        p3 = rf.predict_proba(X_test)[:, 1]
        y_prob = (p1 + p2 + p3) / 3
        y_pred = (y_prob >= 0.5).astype(int)

    return eval_model(y_test, y_pred, y_prob)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 70)
    print("5g: ABLATION STUDY — monitoring_quality")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    year_col = [c for c in df.columns if 'year' in c.lower()][0]
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    # Time-based split
    train_mask = df[year_col] <= 2015
    test_mask = df[year_col] >= 2016
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    print(f"Train: {len(df_train)}, Test: {len(df_test)}")

    # 3 Feature Configurations
    configs = {
        'A: All Features (24)': BASE_FEATURES + QUALITY_FEATURES,
        'B: No monitoring_quality (23)': BASE_FEATURES + ['nst', 'gap', 'dmin', 'rms'],
        'C: No Quality Features (20)': BASE_FEATURES,
    }

    models = ['XGBoost', 'XGB (th=0.36)', 'Soft Voting']
    results = {}

    for config_name, features in configs.items():
        print(f"\n--- {config_name} ---")
        X_train, y_train, X_test, y_test, feat_cols = prepare_consistent_split(df_train, df_test, features)

        print(f"  Features: {len(feat_cols)}")

        for model_name in models:
            key = f"{config_name}|{model_name}"
            r = train_and_eval(X_train, y_train, X_test, y_test, model_name)
            results[key] = r
            print(f"  {model_name:18s}: F1={r['f1']:.4f}, Recall={r['recall']:.4f}, Cost={r['cost']:,}")

    # --- Comparison Table ---
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON TABLE")
    print("=" * 70)

    comparison = {}
    for model_name in models:
        baseline_key = f"A: All Features (24)|{model_name}"
        b = results[baseline_key]

        no_mq_key = f"B: No monitoring_quality (23)|{model_name}"
        nb = results[no_mq_key]

        no_quality_key = f"C: No Quality Features (20)|{model_name}"
        nc = results[no_quality_key]

        comparison[model_name] = {
            'A_all': b,
            'B_no_mq': nb,
            'C_no_quality': nc,
            'delta_B_A_f1': round(nb['f1'] - b['f1'], 4),
            'delta_C_A_f1': round(nc['f1'] - b['f1'], 4),
            'delta_B_A_recall': round(nb['recall'] - b['recall'], 4),
            'delta_C_A_recall': round(nc['recall'] - b['recall'], 4),
        }

        print(f"\n  {model_name}:")
        print(f"    A (all 24):       F1={b['f1']:.4f}, Recall={b['recall']:.4f}")
        print(f"    B (no mq 23):     F1={nb['f1']:.4f}, Recall={nb['recall']:.4f} (ΔF1={nb['f1']-b['f1']:+.4f})")
        print(f"    C (no quality 20):F1={nc['f1']:.4f}, Recall={nc['recall']:.4f} (ΔF1={nc['f1']-b['f1']:+.4f})")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(models))
    w = 0.25

    a_f1 = [comparison[m]['A_all']['f1'] for m in models]
    b_f1 = [comparison[m]['B_no_mq']['f1'] for m in models]
    c_f1 = [comparison[m]['C_no_quality']['f1'] for m in models]

    a_rec = [comparison[m]['A_all']['recall'] for m in models]
    b_rec = [comparison[m]['B_no_mq']['recall'] for m in models]
    c_rec = [comparison[m]['C_no_quality']['recall'] for m in models]

    axes[0].bar(x - w, a_f1, w, label='A: All (24)', color='#2ecc71', edgecolor='white')
    axes[0].bar(x, b_f1, w, label='B: No MQ (23)', color='#f39c12', edgecolor='white')
    axes[0].bar(x + w, c_f1, w, label='C: No Quality (20)', color='#e74c3c', edgecolor='white')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_title('Ablation: F1-Score (Time-Based Split)', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha='right')
    axes[0].legend()

    axes[1].bar(x - w, a_rec, w, label='A: All (24)', color='#2ecc71', edgecolor='white')
    axes[1].bar(x, b_rec, w, label='B: No MQ (23)', color='#f39c12', edgecolor='white')
    axes[1].bar(x + w, c_rec, w, label='C: No Quality (20)', color='#e74c3c', edgecolor='white')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Ablation: Recall (Time-Based Split)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha='right')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'ablation_study.png'), bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] plots/ablation_study.png")

    # Save
    all_results = {
        'config': {
            'split': 'time-based (train≤2015, test≥2016)',
            'A': f'All features ({len(BASE_FEATURES + QUALITY_FEATURES)} + encoded)',
            'B': f'No monitoring_quality ({len(BASE_FEATURES + ["nst","gap","dmin","rms"])} + encoded)',
            'C': f'No quality features ({len(BASE_FEATURES)} + encoded)',
        },
        'results': results,
        'comparison': comparison,
    }
    with open(os.path.join(OUTPUT_DIR, "ablation_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: results/ablation_results.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
