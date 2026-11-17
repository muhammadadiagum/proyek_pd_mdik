#!/usr/bin/env python3
"""
5h — Temporal Era Analysis
Evaluate model performance per decade
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
from sklearn.metrics import f1_score, recall_score, confusion_matrix
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

ERAS = [
    ('1900-1930', 1900, 1930),
    ('1931-1960', 1931, 1960),
    ('1961-1990', 1961, 1990),
    ('1991-2015', 1991, 2015),
    ('2016-2026', 2016, 2026),
]


def prepare_all(df):
    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    all_cols = [c for c in FEATURE_COLS + [c for c in df_enc if c.startswith(('tectonic_zone_', 'era_', 'season_'))] if c in df_enc.columns]
    X = np.nan_to_num(df_enc[all_cols].values, nan=0.0)
    y = df_enc['label'].values
    return X, y, all_cols


def train_models(X_train, y_train):
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

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    for name, model in models.items():
        if name == 'MLP':
            model.fit(Xs, y_train)
        else:
            model.fit(X_train, y_train)

    # Soft Voting
    lr = models['LogisticRegression']
    xgb_m = models['XGBoost']
    rf = models['RandomForest']
    p1 = lr.predict_proba(X_train)[:, 1]
    p2 = xgb_m.predict_proba(X_train)[:, 1]
    p3 = rf.predict_proba(X_train)[:, 1]
    y_vote = ((p1 + p2 + p3) / 3 >= 0.5).astype(int)

    return models, scaler


def evaluate(models, scaler, X_test, y_test):
    results = {}
    for name, model in models.items():
        if name == 'MLP':
            y_pred = model.predict(scaler.transform(X_test))
        else:
            y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        results[name] = {
            'f1': round(float(f1_score(y_test, y_pred)), 4) if sum(y_test) > 0 else None,
            'recall': round(float(recall_score(y_test, y_pred)), 4) if sum(y_test) > 0 else None,
            'cost': int(fn * 100 + fp * 1),
            'n_samples': int(len(y_test)),
            'n_tsunami': int(sum(y_test)),
        }

    # XGB threshold 0.36
    xgb_m = models['XGBoost']
    y_prob = xgb_m.predict_proba(X_test)[:, 1]
    y_pred_t = (y_prob >= 0.36).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)
    tn, fp, fn, tp = cm.ravel()
    results['XGB (th=0.36)'] = {
        'f1': round(float(f1_score(y_test, y_pred_t)), 4) if sum(y_test) > 0 else None,
        'recall': round(float(recall_score(y_test, y_pred_t)), 4) if sum(y_test) > 0 else None,
        'cost': int(fn * 100 + fp * 1),
        'n_samples': int(len(y_test)),
        'n_tsunami': int(sum(y_test)),
    }

    # Soft Voting
    lr = models['LogisticRegression']
    rf = models['RandomForest']
    p1 = lr.predict_proba(X_test)[:, 1]
    p2 = xgb_m.predict_proba(X_test)[:, 1]
    p3 = rf.predict_proba(X_test)[:, 1]
    y_vote_prob = (p1 + p2 + p3) / 3
    y_vote_pred = (y_vote_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_vote_pred)
    tn, fp, fn, tp = cm.ravel()
    results['Soft Voting'] = {
        'f1': round(float(f1_score(y_test, y_vote_pred)), 4) if sum(y_test) > 0 else None,
        'recall': round(float(recall_score(y_test, y_vote_pred)), 4) if sum(y_test) > 0 else None,
        'cost': int(fn * 100 + fp * 1),
        'n_samples': int(len(y_test)),
        'n_tsunami': int(sum(y_test)),
    }

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 70)
    print("5h: TEMPORAL ERA ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    year_col = [c for c in df.columns if 'year' in c.lower()][0]
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    feat_cols = [c for c in FEATURE_COLS + [c for c in df_enc if c.startswith(('tectonic_zone_', 'era_', 'season_'))] if c in df_enc.columns]
    df_enc['_year'] = df[year_col]
    X_all = np.nan_to_num(df_enc[feat_cols].values, nan=0.0)
    y_all = df_enc['label'].values
    years_all = df_enc['_year'].values

    # Train on all data 1900-2015
    train_mask = years_all <= 2015
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]

    print(f"Training data: {len(y_train)} (tsunami: {sum(y_train==1)})")

    models, scaler = train_models(X_train, y_train)

    # Evaluate per era
    era_results = {}

    for era_name, y_start, y_end in ERAS:
        era_mask = (years_all >= y_start) & (years_all <= y_end)
        X_era = X_all[era_mask]
        y_era = y_all[era_mask]

        n_total = len(y_era)
        n_tsunami = int(sum(y_era == 1))
        pct_tsunami = round(n_tsunami / n_total * 100, 3) if n_total > 0 else 0

        print(f"\n--- Era {era_name}: {n_total:,} rows, {n_tsunami} tsunami ({pct_tsunami}%) ---")

        if n_tsunami == 0:
            print("  No tsunami events — skipping")
            era_results[era_name] = {'n_samples': n_total, 'n_tsunami': 0, 'models': {}}
            continue

        era_results[era_name] = {
            'n_samples': n_total,
            'n_tsunami': n_tsunami,
            'pct_tsunami': pct_tsunami,
            'models': evaluate(models, scaler, X_era, y_era),
        }

        for mname, mr in era_results[era_name]['models'].items():
            f1_str = f"{mr['f1']:.4f}" if mr['f1'] is not None else "N/A"
            rec_str = f"{mr['recall']:.4f}" if mr['recall'] is not None else "N/A"
            print(f"  {mname:18s}: F1={f1_str}, Recall={rec_str}")

    # --- Plot ---
    model_names = ['XGBoost', 'XGB (th=0.36)', 'Soft Voting']
    era_names = [e[0] for e in ERAS]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(era_names)))

    for mname in model_names:
        f1s = []
        recs = []
        for ename in era_names:
            if ename in era_results and mname in era_results[ename].get('models', {}):
                f1s.append(era_results[ename]['models'][mname].get('f1'))
                recs.append(era_results[ename]['models'][mname].get('recall'))
            else:
                f1s.append(None)
                recs.append(None)

        x = np.arange(len(era_names))
        valid_f1 = [(i, v) for i, v in enumerate(f1s) if v is not None]
        valid_rec = [(i, v) for i, v in enumerate(recs) if v is not None]

        if valid_f1:
            axes[0].plot([v[0] for v in valid_f1], [v[1] for v in valid_f1], 'o-', label=mname, linewidth=2)
        if valid_rec:
            axes[1].plot([v[0] for v in valid_rec], [v[1] for v in valid_rec], 'o-', label=mname, linewidth=2)

    axes[0].set_xticks(range(len(era_names)))
    axes[0].set_xticklabels(era_names, rotation=20, ha='right')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_title('F1-Score by Era', fontweight='bold')
    axes[0].legend()

    axes[1].set_xticks(range(len(era_names)))
    axes[1].set_xticklabels(era_names, rotation=20, ha='right')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall by Era', fontweight='bold')
    axes[1].legend()

    plt.suptitle('Temporal Era Analysis (Train: 1900-2015)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'era_analysis.png'), bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] plots/era_analysis.png")

    with open(os.path.join(OUTPUT_DIR, "era_analysis_results.json"), 'w') as f:
        json.dump(era_results, f, indent=2, default=str)
    print(f"  Saved: results/era_analysis_results.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
