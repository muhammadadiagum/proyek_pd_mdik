#!/usr/bin/env python3
"""
5i — Error Analysis: False Negatives
Why does the model MISS tsunamis?
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
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def _spw(y):
    """scale_pos_weight = rasio negatif/positif pada data latih (bukan konstanta)."""
    y = np.asarray(y)
    return float((len(y) - y.sum()) / max(y.sum(), 1))


DATA_PATH = "../../00_data/processed/modeling_dataset.csv"
MERGED_PATH = "../../00_data/interim/merged_dataset.csv"
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
    print("5i: ERROR ANALYSIS — FALSE NEGATIVES")
    print("=" * 70)

    # Load + encode once
    df = pd.read_csv(DATA_PATH)
    merged = pd.read_csv(MERGED_PATH)
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

    print(f"Test set (2016-2026): {len(y_test)} rows, {sum(y_test==1)} tsunami")

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, scale_pos_weight=_spw(y_train),
        eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    # Predict with threshold 0.36
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.36).astype(int)

    # Identify FN: actual=1, predicted=0
    fn_mask = (y_test == 1) & (y_pred == 0)
    tp_mask = (y_test == 1) & (y_pred == 1)

    n_tp = int(tp_mask.sum())
    n_fn = int(fn_mask.sum())
    print(f"True Positives: {n_tp}")
    print(f"False Negatives: {n_fn}")

    # Get corresponding rows from merged dataset for context
    # The merged dataset has label column — filter test rows
    test_indices = np.where(test_mask)[0]

    fn_indices = test_indices[fn_mask]
    tp_indices = test_indices[tp_mask]

    # Build analysis dataframe from df (original features before encoding)
    df_test = df.iloc[test_indices].copy()
    df_test['y_true'] = y_test
    df_test['y_pred'] = y_pred
    df_test['y_prob'] = y_prob

    fn_df = df_test[(df_test['y_pred'] == 0) & (df_test['y_true'] == 1)]
    tp_df = df_test[(df_test['y_pred'] == 1) & (df_test['y_true'] == 1)]

    print(f"\n{'='*70}")
    print("FALSE NEGATIVES ANALYSIS")
    print(f"{'='*70}")

    # --- 1. Cause Code Analysis ---
    print("\n1. CAUSE CODE DISTRIBUTION:")
    cause_col = [c for c in merged.columns if 'cause' in c.lower()][0]

    # Match FN events to NOAA cause codes
    fn_years = fn_df[year_col].values
    fn_lats = fn_df['latitude'].values
    fn_lons = fn_df['longitude'].values

    tp_years = tp_df[year_col].values
    tp_lats = tp_df['latitude'].values
    tp_lons = tp_df['longitude'].values

    print(f"\n  FN events (missed tsunamis): {n_fn}")
    print(f"  TP events (detected tsunamis): {n_tp}")

    # --- 2. Magnitude Distribution ---
    print("\n2. MAGNITUDE DISTRIBUTION:")
    fn_mag = fn_df['mag'].values
    tp_mag = tp_df['mag'].values
    all_mag = df_test[df_test['label']==1]['mag'].values

    print(f"  FN mag: mean={fn_mag.mean():.2f}, median={np.median(fn_mag):.2f}, "
          f"min={fn_mag.min():.1f}, max={fn_mag.max():.1f}")
    print(f"  TP mag: mean={tp_mag.mean():.2f}, median={np.median(tp_mag):.2f}, "
          f"min={tp_mag.min():.1f}, max={tp_mag.max():.1f}")

    # --- 3. Depth Distribution ---
    print("\n3. DEPTH DISTRIBUTION:")
    fn_depth = fn_df['depth'].values
    tp_depth = tp_df['depth'].values

    print(f"  FN depth: mean={fn_depth.mean():.1f}, median={np.median(fn_depth):.1f}, "
          f"min={fn_depth.min():.1f}, max={fn_depth.max():.1f}")
    print(f"  TP depth: mean={tp_depth.mean():.1f}, median={np.median(tp_depth):.1f}, "
          f"min={tp_depth.min():.1f}, max={tp_depth.max():.1f}")

    # --- 4. Monitoring Quality ---
    print("\n4. MONITORING QUALITY:")
    fn_mq = fn_df['monitoring_quality'].values
    tp_mq = tp_df['monitoring_quality'].values

    print(f"  FN quality: mean={fn_mq.mean():.3f}, median={np.median(fn_mq):.3f}")
    print(f"  TP quality: mean={tp_mq.mean():.3f}, median={np.median(tp_mq):.3f}")

    # --- 5. Region Distribution ---
    print("\n5. TECTONIC ZONE DISTRIBUTION:")
    zone_col = [c for c in df.columns if 'zone' in c.lower() and 'tectonic' in c.lower()]
    if not zone_col:
        # reconstruct from encoded
        for col in df.columns:
            if col == 'tectonic_zone':
                zone_col = col
                break

    for col in df.columns:
        if col == 'tectonic_zone':
            fn_zones = fn_df[col].value_counts()
            tp_zones = tp_df[col].value_counts()
            print(f"\n  FN zones:")
            for z, c in fn_zones.items():
                print(f"    {z}: {c}")
            print(f"\n  TP zones:")
            for z, c in tp_zones.items():
                print(f"    {z}: {c}")
            break

    # --- 6. Year Distribution ---
    print("\n6. YEAR DISTRIBUTION OF FN:")
    fn_years_series = fn_df[year_col].value_counts().sort_index()
    for y, c in fn_years_series.items():
        print(f"  {int(y)}: {c} FN")

    # --- 7. Prediction Probability Distribution ---
    print("\n7. PREDICTION PROBABILITY:")
    fn_prob = fn_df['y_prob'].values
    tp_prob = tp_df['y_prob'].values

    print(f"  FN probability: mean={fn_prob.mean():.3f}, median={np.median(fn_prob):.3f}, "
          f"max={fn_prob.max():.3f}")
    print(f"  TP probability: mean={tp_prob.mean():.3f}, median={np.median(tp_prob):.3f}, "
          f"min={tp_prob.min():.3f}")

    # How many FN have prob > 0.36 but < 0.5?
    fn_near_threshold = ((fn_prob >= 0.30) & (fn_prob < 0.36)).sum()
    fn_very_low = (fn_prob < 0.10).sum()
    print(f"\n  FN with prob 0.30-0.36 (near threshold): {fn_near_threshold}/{n_fn}")
    print(f"  FN with prob < 0.10 (very low confidence): {fn_very_low}/{n_fn}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Magnitude distribution
    axes[0, 0].hist(tp_mag, bins=15, alpha=0.7, label=f'TP (n={n_tp})', color='#2ecc71', edgecolor='white')
    axes[0, 0].hist(fn_mag, bins=15, alpha=0.7, label=f'FN (n={n_fn})', color='#e74c3c', edgecolor='white')
    axes[0, 0].set_xlabel('Magnitude')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Magnitude Distribution: TP vs FN', fontweight='bold')
    axes[0, 0].legend()

    # Depth distribution
    axes[0, 1].hist(tp_depth, bins=15, alpha=0.7, label=f'TP (n={n_tp})', color='#2ecc71', edgecolor='white')
    axes[0, 1].hist(fn_depth, bins=15, alpha=0.7, label=f'FN (n={n_fn})', color='#e74c3c', edgecolor='white')
    axes[0, 1].set_xlabel('Depth (km)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Depth Distribution: TP vs FN', fontweight='bold')
    axes[0, 1].legend()

    # Prediction probability
    axes[1, 0].hist(tp_prob, bins=20, alpha=0.7, label=f'TP (n={n_tp})', color='#2ecc71', edgecolor='white')
    axes[1, 0].hist(fn_prob, bins=20, alpha=0.7, label=f'FN (n={n_fn})', color='#e74c3c', edgecolor='white')
    axes[1, 0].axvline(0.36, color='black', linestyle='--', label='Threshold=0.36')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Prediction Probability: TP vs FN', fontweight='bold')
    axes[1, 0].legend()

    # Monitoring quality
    axes[1, 1].hist(tp_mq, bins=15, alpha=0.7, label=f'TP (n={n_tp})', color='#2ecc71', edgecolor='white')
    axes[1, 1].hist(fn_mq, bins=15, alpha=0.7, label=f'FN (n={n_fn})', color='#e74c3c', edgecolor='white')
    axes[1, 1].set_xlabel('Monitoring Quality')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Monitoring Quality: TP vs FN', fontweight='bold')
    axes[1, 1].legend()

    plt.suptitle('Error Analysis: False Negatives (Test: 2016-2026)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'error_analysis.png'), bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] plots/error_analysis.png")

    # Save
    results = {
        'test_set': {'n_total': int(len(y_test)), 'n_tsunami': int(sum(y_test==1))},
        'tp': {'count': n_tp},
        'fn': {'count': n_fn},
        'magnitude': {
            'fn_mean': round(float(fn_mag.mean()), 2),
            'fn_median': round(float(np.median(fn_mag)), 2),
            'tp_mean': round(float(tp_mag.mean()), 2),
            'tp_median': round(float(np.median(tp_mag)), 2),
        },
        'depth': {
            'fn_mean': round(float(fn_depth.mean()), 1),
            'tp_mean': round(float(tp_depth.mean()), 1),
        },
        'monitoring_quality': {
            'fn_mean': round(float(fn_mq.mean()), 3),
            'tp_mean': round(float(tp_mq.mean()), 3),
        },
        'probability': {
            'fn_mean': round(float(fn_prob.mean()), 3),
            'fn_median': round(float(np.median(fn_prob)), 3),
            'fn_near_threshold': int(fn_near_threshold),
            'fn_very_low': int(fn_very_low),
        },
    }

    with open(os.path.join(OUTPUT_DIR, "error_analysis_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: results/error_analysis_results.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
