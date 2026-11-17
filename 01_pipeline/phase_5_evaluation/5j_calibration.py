#!/usr/bin/env python3
"""
5j — Calibration / Reliability Analysis
Is the model's predicted probability trustworthy?
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
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("=" * 70)
    print("5j: CALIBRATION / RELIABILITY ANALYSIS")
    print("=" * 70)

    # Load + encode
    df = pd.read_csv(DATA_PATH)
    year_col = [c for c in df.columns if 'year' in c.lower()][0]
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')

    df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
    feat_cols = [c for c in FEATURE_COLS + [c for c in df_enc if c.startswith(('tectonic_zone_', 'era_', 'season_'))] if c in df_enc.columns]
    df_enc['_year'] = df[year_col].values

    X_all = np.nan_to_num(df_enc[feat_cols].values, nan=0.0)
    y_all = df_enc['label'].values
    years = df_enc['_year'].values

    # Time-based split
    train_mask = years <= 2015
    test_mask = years >= 2016

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"Test tsunami: {sum(y_test==1)}")

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test = scaler.transform(X_test)

    # Train models
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
            model.fit(Xs_train, y_train)
        else:
            model.fit(X_train, y_train)

    # Get probabilities and calibration curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    results = {}

    colors = {'XGBoost': '#e74c3c', 'LogisticRegression': '#3498db',
              'RandomForest': '#2ecc71', 'MLP': '#9b59b6'}

    for name, model in models.items():
        if name == 'MLP':
            y_prob = model.predict_proba(Xs_test)[:, 1]
        else:
            y_prob = model.predict_proba(X_test)[:, 1]

        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=10, strategy='uniform')

        # Brier score (lower = better calibrated)
        brier = float(np.mean((y_prob - y_test) ** 2))

        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, 11)
        ece = 0.0
        for i in range(len(bin_edges) - 1):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
            if mask.sum() > 0:
                bin_acc = y_test[mask].mean()
                bin_conf = y_prob[mask].mean()
                ece += mask.sum() / len(y_test) * abs(bin_acc - bin_conf)
        ece = float(ece)

        results[name] = {
            'brier_score': round(brier, 4),
            'ece': round(ece, 4),
            'mean_predicted': round(float(y_prob.mean()), 4),
            'actual_positive_rate': round(float(y_test.mean()), 4),
        }

        print(f"\n  {name}:")
        print(f"    Brier Score: {brier:.4f} (lower = better)")
        print(f"    ECE: {ece:.4f} (lower = better)")
        print(f"    Mean predicted prob: {y_prob.mean():.4f}")
        print(f"    Actual positive rate: {y_test.mean():.4f}")

        # Plot calibration curve
        axes[0].plot(mean_predicted_value, fraction_of_positives, 's-',
                     label=name, color=colors[name], linewidth=2)

    # Perfect calibration line
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=1.5)
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Calibration Curve (Time-Based Split)', fontweight='bold')
    axes[0].legend(loc='upper left')

    # Probability distribution for TP vs FN
    xgb_m = models['XGBoost']
    y_prob_xgb = xgb_m.predict_proba(X_test)[:, 1]
    y_pred_xgb = (y_prob_xgb >= 0.36).astype(int)

    tp_mask = (y_test == 1) & (y_pred_xgb == 1)
    fn_mask = (y_test == 1) & (y_pred_xgb == 0)
    tn_mask = (y_test == 0) & (y_pred_xgb == 0)
    fp_mask = (y_test == 0) & (y_pred_xgb == 1)

    axes[1].hist(y_prob_xgb[tn_mask], bins=30, alpha=0.7, label=f'TN (n={sum(tn_mask)})',
                 color='#2ecc71', edgecolor='white', density=True)
    axes[1].hist(y_prob_xgb[tp_mask], bins=30, alpha=0.7, label=f'TP (n={sum(tp_mask)})',
                 color='#3498db', edgecolor='white', density=True)
    axes[1].hist(y_prob_xgb[fn_mask], bins=10, alpha=0.7, label=f'FN (n={sum(fn_mask)})',
                 color='#e74c3c', edgecolor='white', density=True)
    if sum(fp_mask) > 0:
        axes[1].hist(y_prob_xgb[fp_mask], bins=10, alpha=0.7, label=f'FP (n={sum(fp_mask)})',
                     color='#f39c12', edgecolor='white', density=True)
    axes[1].axvline(0.36, color='black', linestyle='--', label='Threshold=0.36')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Density')
    axes[1].set_title('XGBoost Probability Distribution', fontweight='bold')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'calibration_analysis.png'), bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] plots/calibration_analysis.png")

    with open(os.path.join(OUTPUT_DIR, "calibration_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: results/calibration_results.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
