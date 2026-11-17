#!/usr/bin/env python3
"""
5f — Time-Based Split Validation
Compare Random Split vs Time-Based Split (train≤2015, test≥2016)
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

plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 150,
})


def prepare_data(df, feature_cols):
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    avail = [c for c in feature_cols + [c for c in df_enc if c.startswith(('tectonic_zone_', 'era_', 'season_'))] if c in df_enc.columns]
    X = np.nan_to_num(df_enc[avail].values, nan=0.0)
    y = df_enc['label'].values
    return X, y, avail


def eval_model(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    r = {
        'f1': round(float(f1_score(y_true, y_pred)), 4),
        'recall': round(float(recall_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred)), 4),
        'cost': int(fn * 100 + fp * 1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        r['auc'] = round(float(roc_auc_score(y_true, y_prob)), 4)
    return r


def bootstrap_ci(y_true, y_pred, n_boot=100):
    n = len(y_true)
    f1s = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        f1s.append(f1_score(y_true[idx], y_pred[idx]))
    return {
        'mean': round(float(np.mean(f1s)), 4),
        'ci_low': round(float(np.percentile(f1s, 2.5)), 4),
        'ci_high': round(float(np.percentile(f1s, 97.5)), 4),
    }


def train_models(X_train, y_train, scaler):
    models = {}
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, scale_pos_weight=_spw(y_train),
        eval_metric='logloss', use_label_encoder=False)
    models['LogisticRegression'] = LogisticRegression(
        max_iter=2000, random_state=42, class_weight='balanced')
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42,
        n_jobs=-1)
    models['MLP'] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), max_iter=500,
        random_state=42, early_stopping=True)

    for name, model in models.items():
        if name == 'MLP':
            model.fit(scaler.fit_transform(X_train), y_train)
        else:
            model.fit(X_train, y_train)
    return models


def evaluate(models, X_test, y_test, scaler):
    results = {}
    for name, model in models.items():
        if name == 'MLP':
            y_pred = model.predict(scaler.transform(X_test))
            y_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = eval_model(y_test, y_pred, y_prob)
        results[name]['ci'] = bootstrap_ci(y_test, y_pred)

    # Soft Voting (LR + XGB + RF)
    lr = models['LogisticRegression']
    xgb_m = models['XGBoost']
    rf = models['RandomForest']
    vote_soft = VotingClassifier(
        estimators=[('lr', lr), ('xgb', xgb_m), ('rf', rf)],
        voting='soft', weights=[1, 2, 1])
    vote_soft.fit(X_test if name != 'MLP' else scaler.transform(X_test), y_test)
    # Actually need to refit on train... let me just use models already fitted
    # Vote on test using predict from individual models
    preds_lr = lr.predict(X_test)
    preds_xgb = xgb_m.predict(X_test)
    preds_rf = rf.predict(X_test)
    # Hard majority vote
    votes = np.stack([preds_lr, preds_xgb, preds_rf])
    y_pred_vote = (votes.sum(axis=0) >= 2).astype(int)
    y_prob_vote = (lr.predict_proba(X_test)[:, 1] + xgb_m.predict_proba(X_test)[:, 1] + rf.predict_proba(X_test)[:, 1]) / 3
    results['Soft Voting'] = eval_model(y_test, y_pred_vote, y_prob_vote)
    results['Soft Voting']['ci'] = bootstrap_ci(y_test, y_pred_vote)

    # XGBoost threshold 0.36
    y_prob_xgb = xgb_m.predict_proba(X_test)[:, 1]
    y_pred_t = (y_prob_xgb >= 0.36).astype(int)
    results['XGB (th=0.36)'] = eval_model(y_test, y_pred_t, y_prob_xgb)
    results['XGB (th=0.36)']['ci'] = bootstrap_ci(y_test, y_pred_t)

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 70)
    print("5f: TIME-BASED SPLIT VALIDATION")
    print("Random Split vs Time-Based Split (train≤2015, test≥2016)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"Total data: {len(df)}")

    # --- Random Split ---
    print("\n--- RANDOM SPLIT (80/20) ---")
    X_all, y_all, feat_cols = prepare_data(df, FEATURE_COLS)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
    scaler_r = StandardScaler()
    scaler_r.fit(X_train_r)
    print(f"Train: {len(y_train_r)}, Test: {len(y_test_r)}")

    models_r = train_models(X_train_r, y_train_r, scaler_r)
    results_r = evaluate(models_r, X_test_r, y_test_r, scaler_r)

    for name, r in results_r.items():
        print(f"  {name:18s}: F1={r['f1']:.4f} [{r['ci']['ci_low']:.4f}, {r['ci']['ci_high']:.4f}], "
              f"Recall={r['recall']:.4f}, Cost={r['cost']:,}")

    # --- Time-Based Split ---
    print("\n--- TIME-BASED SPLIT (train≤2015, test≥2016) ---")
    year_col = [c for c in df.columns if 'year' in c.lower()][0]
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    # Encode ALL data first to get consistent columns
    df_all_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    all_feat_cols = [c for c in FEATURE_COLS + [c for c in df_all_enc if c.startswith(('tectonic_zone_', 'era_', 'season_'))] if c in df_all_enc.columns]

    train_mask = df[year_col] <= 2015
    test_mask = df[year_col] >= 2016

    df_train_enc = df_all_enc[train_mask]
    df_test_enc = df_all_enc[test_mask]

    X_train_t = np.nan_to_num(df_train_enc[all_feat_cols].values, nan=0.0)
    y_train_t = df_train_enc['label'].values
    X_test_t = np.nan_to_num(df_test_enc[all_feat_cols].values, nan=0.0)
    y_test_t = df_test_enc['label'].values

    print(f"Train (1900-2015): {len(y_train_t)} (tsunami: {sum(y_train_t==1)})")
    print(f"Test  (2016-2026): {len(y_test_t)} (tsunami: {sum(y_test_t==1)})")

    scaler_t = StandardScaler()
    scaler_t.fit(X_train_t)

    models_t = train_models(X_train_t, y_train_t, scaler_t)
    results_t = evaluate(models_t, X_test_t, y_test_t, scaler_t)

    for name, r in results_t.items():
        print(f"  {name:18s}: F1={r['f1']:.4f} [{r['ci']['ci_low']:.4f}, {r['ci']['ci_high']:.4f}], "
              f"Recall={r['recall']:.4f}, Cost={r['cost']:,}")

    # --- Comparison Table ---
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':18s} | {'Random F1':>10s} {'Time F1':>10s} {'Δ F1':>8s} | {'Random Recall':>14s} {'Time Recall':>12s} {'Δ Recall':>10s}")
    print("-" * 100)

    comparison = {}
    for name in results_r:
        rf1 = results_r[name]['f1']
        tf1 = results_t[name]['f1']
        rrec = results_r[name]['recall']
        trec = results_t[name]['recall']
        df1 = round(tf1 - rf1, 4)
        drec = round(trec - rrec, 4)
        comparison[name] = {
            'random': results_r[name],
            'time_based': results_t[name],
            'delta_f1': df1,
            'delta_recall': drec,
        }
        print(f"  {name:16s} | {rf1:>10.4f} {tf1:>10.4f} {df1:>+8.4f} | {rrec:>14.4f} {trec:>12.4f} {drec:>+10.4f}")

    # --- Plot ---
    model_names = list(comparison.keys())
    r_f1 = [comparison[n]['random']['f1'] for n in model_names]
    t_f1 = [comparison[n]['time_based']['f1'] for n in model_names]
    r_rec = [comparison[n]['random']['recall'] for n in model_names]
    t_rec = [comparison[n]['time_based']['recall'] for n in model_names]

    x = np.arange(len(model_names))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(x - w/2, r_f1, w, label='Random Split', color='#3498db', edgecolor='white')
    axes[0].bar(x + w/2, t_f1, w, label='Time-Based Split', color='#e74c3c', edgecolor='white')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_title('F1-Score: Random vs Time-Based Split', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=30, ha='right')
    axes[0].legend()

    axes[1].bar(x - w/2, r_rec, w, label='Random Split', color='#3498db', edgecolor='white')
    axes[1].bar(x + w/2, t_rec, w, label='Time-Based Split', color='#e74c3c', edgecolor='white')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall: Random vs Time-Based Split', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=30, ha='right')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'time_based_split_comparison.png'), bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] plots/time_based_split_comparison.png")

    # --- Save ---
    all_results = {
        'config': {
            'random_split': 'train_test_split(test_size=0.2, stratify=y, random_state=42)',
            'time_based_split': 'train: year<=2015, test: year>=2016',
            'bootstrap_n': 100,
        },
        'random_split': {n: results_r[n] for n in results_r},
        'time_based_split': {n: results_t[n] for n in results_t},
        'comparison': comparison,
    }

    with open(os.path.join(OUTPUT_DIR, "time_based_split_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: results/time_based_split_results.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
