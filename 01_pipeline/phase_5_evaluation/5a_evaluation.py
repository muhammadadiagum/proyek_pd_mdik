#!/usr/bin/env python3
"""
5a — Evaluation: SHAP, Threshold Tuning, Regional Analysis
================================================================
Evaluasi mendalam: explainability, threshold optimization,
dan analisis regional.

Output: results/evaluation_results.json, plots/*.png
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

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

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, make_scorer
)

import xgboost as xgb
import shap


def _spw(y):
    """scale_pos_weight = rasio negatif/positif pada data latih (bukan konstanta)."""
    y = np.asarray(y)
    return float((len(y) - y.sum()) / max(y.sum(), 1))


warnings.filterwarnings('ignore')

# Paths
DATA_PATH = "../../00_data/processed/modeling_dataset.csv"
OUTPUT_DIR = "results"
PLOTS_DIR = "plots"

# Feature config (same as modeling)
FEATURE_COLS = [
    'mag', 'depth', 'nst', 'gap', 'dmin', 'rms',
    'latitude', 'longitude', 'abs_latitude',
    'depth_mag_ratio', 'monitoring_quality',
]
CATEGORICAL_COLS = ['tectonic_zone', 'era', 'season']
TARGET_COL = 'label'

# Style
plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 150,
})


def prepare_features(df):
    """Persiapan fitur (sama seperti modeling)."""
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    all_feature_cols = FEATURE_COLS.copy()
    for col in df_encoded.columns:
        if col.startswith(('tectonic_zone_', 'era_', 'season_')):
            all_feature_cols.append(col)

    available_cols = [c for c in all_feature_cols if c in df_encoded.columns]
    X = df_encoded[available_cols].values
    y = df_encoded[TARGET_COL].values
    return X, y, available_cols, df_encoded


# ============================================================
# 1. SHAP ANALYSIS
# ============================================================

def shap_analysis(X_train, X_test, y_train, feature_names, model_name='XGBoost'):
    """SHAP analysis untuk model terbaik."""
    print(f"\n{'='*60}")
    print(f"SHAP ANALYSIS — {model_name}")
    print(f"{'='*60}")

    # Train model
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, scale_pos_weight=_spw(y_train),
            eval_metric='logloss', use_label_encoder=False
        )
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42,
            n_jobs=-1
        )
    elif model_name == 'MLP':
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            random_state=42, early_stopping=True
        )
        model.fit(X_train_s, y_train)
        # For MLP, use permutation-based SHAP
        explainer = shap.LinearExplainer(model, X_test_s[:200])
        shap_values = explainer.shap_values(X_test_s[:200])

        # Save plots
        plt.figure()
        shap.summary_plot(shap_values, X_test_s[:200],
                         feature_names=feature_names, show=False)
        plt.title(f'SHAP Summary — {model_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'shap_summary_{model_name.lower().replace(" ", "_")}.png'),
                    bbox_inches='tight')
        plt.close()

        return model, shap_values

    model.fit(X_train, y_train)

    # TreeExplainer untuk tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Class 1 (tsunami)
    else:
        shap_vals = shap_values

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_vals, X_test[:500],
                     feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary — {model_name} (Class: Tsunami)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'shap_summary_{model_name.lower().replace(" ", "_")}.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  [OK] shap_summary_{model_name.lower().replace(' ', '_')}.png")

    # Bar plot (mean |SHAP|)
    plt.figure()
    shap.summary_plot(shap_vals, X_test[:500],
                     feature_names=feature_names, plot_type='bar', show=False)
    plt.title(f'SHAP Feature Importance — {model_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'shap_importance_{model_name.lower().replace(" ", "_")}.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  [OK] shap_importance_{model_name.lower().replace(' ', '_')}.png")

    # Top feature dependences
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:3]

    for idx in top_features_idx:
        fname = feature_names[idx]
        plt.figure()
        shap.dependence_plot(idx, shap_vals, X_test[:500],
                            feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence — {fname}', fontweight='bold')
        plt.tight_layout()
        safe_name = fname.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(PLOTS_DIR, f'shap_dependence_{safe_name}.png'),
                    bbox_inches='tight')
        plt.close()
        print(f"  [OK] shap_dependence_{safe_name}.png")

    # Feature importance ranking
    importance = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    print("\n  Feature Importance (mean |SHAP|):")
    for i, (name, imp) in enumerate(importance, 1):
        print(f"    {i:2d}. {name:30s} {imp:.6f}")

    return model, shap_values, importance


# ============================================================
# 2. THRESHOLD TUNING
# ============================================================

def threshold_tuning(model, X_test, y_test, model_name):
    """Kurva threshold pada test set — IN-SAMPLE, hanya indikatif.

    Ambang yang dipakai produk TIDAK diambil dari sini. Protokol final ada di
    phase 6: ambang per model di-tuning pada inner-validation data latih, lalu
    dievaluasi sekali pada test (`phase_6_deployment/model/thresholds.json`).
    """
    print(f"\n{'='*60}")
    print(f"THRESHOLD TUNING (in-sample, indikatif) — {model_name}")
    print(f"{'='*60}")

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        f1 = f1_score(y_test, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cost = fn * 100 + fp * 1

        results.append({
            'threshold': float(thresh),
            'f1': float(f1),
            'precision': float(precision),
            'recall': float(recall),
            'cost': int(cost),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })

    # Plot threshold vs metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    thresholds_arr = [r['threshold'] for r in results]
    f1_arr = [r['f1'] for r in results]
    prec_arr = [r['precision'] for r in results]
    rec_arr = [r['recall'] for r in results]
    cost_arr = [r['cost'] for r in results]

    # F1 vs Threshold
    axes[0].plot(thresholds_arr, f1_arr, 'b-', linewidth=2, label='F1-score')
    axes[0].plot(thresholds_arr, prec_arr, 'g--', linewidth=1.5, label='Precision')
    axes[0].plot(thresholds_arr, rec_arr, 'r--', linewidth=1.5, label='Recall')

    best_f1_idx = np.argmax(f1_arr)
    axes[0].axvline(thresholds_arr[best_f1_idx], color='gray', linestyle=':', alpha=0.7)
    axes[0].scatter([thresholds_arr[best_f1_idx]], [f1_arr[best_f1_idx]], color='blue', s=100, zorder=5)
    axes[0].annotate(f'Best F1={f1_arr[best_f1_idx]:.3f}\n@ threshold={thresholds_arr[best_f1_idx]:.2f}',
                     xy=(thresholds_arr[best_f1_idx], f1_arr[best_f1_idx]),
                     xytext=(thresholds_arr[best_f1_idx]+0.1, f1_arr[best_f1_idx]+0.05),
                     arrowprops=dict(arrowstyle='->', color='gray'),
                     fontsize=9)

    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Precision/Recall/F1 vs Threshold — {model_name}', fontweight='bold')
    axes[0].legend()

    # Cost vs Threshold
    axes[1].plot(thresholds_arr, cost_arr, 'r-', linewidth=2, label='Cost (FN×100 + FP×1)')
    best_cost_idx = np.argmin(cost_arr)
    axes[1].axvline(thresholds_arr[best_cost_idx], color='gray', linestyle=':', alpha=0.7)
    axes[1].scatter([thresholds_arr[best_cost_idx]], [cost_arr[best_cost_idx]], color='red', s=100, zorder=5)
    axes[1].annotate(f'Best Cost={cost_arr[best_cost_idx]:,}\n@ threshold={thresholds_arr[best_cost_idx]:.2f}',
                     xy=(thresholds_arr[best_cost_idx], cost_arr[best_cost_idx]),
                     xytext=(thresholds_arr[best_cost_idx]+0.1, cost_arr[best_cost_idx]-500),
                     arrowprops=dict(arrowstyle='->', color='gray'),
                     fontsize=9)

    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Cost')
    axes[1].set_title(f'Cost vs Threshold — {model_name}', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'threshold_tuning_{model_name.lower().replace(" ", "_")}.png'),
                bbox_inches='tight')
    plt.close()
    print(f"  [OK] threshold_tuning_{model_name.lower().replace(' ', '_')}.png")

    # Best results
    best_f1 = results[best_f1_idx]
    best_cost = results[best_cost_idx]

    print(f"\n  Best F1: threshold={best_f1['threshold']:.2f}, F1={best_f1['f1']:.4f}, "
          f"Precision={best_f1['precision']:.4f}, Recall={best_f1['recall']:.4f}")
    print(f"  Best Cost: threshold={best_cost['threshold']:.2f}, Cost={best_cost['cost']:,}, "
          f"Precision={best_cost['precision']:.4f}, Recall={best_cost['recall']:.4f}")

    return {
        'model_name': model_name,
        'best_f1': best_f1,
        'best_cost': best_cost,
        'all_thresholds': results,
    }


# ============================================================
# 3. REGIONAL ANALYSIS
# ============================================================

def regional_analysis(df_encoded, feature_names, df_original):
    """Leave-One-Region-Out analysis."""
    print(f"\n{'='*60}")
    print("REGIONAL ANALYSIS — Leave-One-Region-Out")
    print(f"{'='*60}")

    regions = df_original['tectonic_zone'].unique()
    region_results = []

    for test_region in regions:
        print(f"\n  Testing on: {test_region}")

        test_mask = df_original['tectonic_zone'] == test_region
        train_mask = ~test_mask

        X_test_r = df_encoded.loc[test_mask, feature_names].values
        y_test_r = df_encoded.loc[test_mask, TARGET_COL].values
        X_train_r = df_encoded.loc[train_mask, feature_names].values
        y_train_r = df_encoded.loc[train_mask, TARGET_COL].values

        # Skip if no positive samples in test
        if sum(y_test_r == 1) == 0:
            print(f"    Skipped (no tsunami events)")
            continue

        # Scale
        scaler = StandardScaler()
        X_train_r = scaler.fit_transform(X_train_r)
        X_test_r = scaler.transform(X_test_r)

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, scale_pos_weight=_spw(y_train_r),
            eval_metric='logloss', use_label_encoder=False
        )
        model.fit(X_train_r, y_train_r)

        # Evaluate
        y_pred = model.predict(X_test_r)
        y_prob = model.predict_proba(X_test_r)[:, 1]

        report = classification_report(y_test_r, y_pred, output_dict=True)
        auc = roc_auc_score(y_test_r, y_prob) if len(np.unique(y_test_r)) > 1 else 0
        cm = confusion_matrix(y_test_r, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        result = {
            'region': test_region,
            'train_size': int(train_mask.sum()),
            'test_size': int(test_mask.sum()),
            'test_tsunami': int(sum(y_test_r == 1)),
            'f1_1': float(report['1']['f1-score']) if '1' in report else 0,
            'recall_1': float(report['1']['recall']) if '1' in report else 0,
            'precision_1': float(report['1']['precision']) if '1' in report else 0,
            'auc': float(auc),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }
        region_results.append(result)

        print(f"    Test size: {result['test_size']:,} (tsunami: {result['test_tsunami']})")
        print(f"    F1={result['f1_1']:.4f}, AUC={result['auc']:.4f}, Recall={result['recall_1']:.4f}")

    # Plot regional comparison
    if region_results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        region_names = [r['region'] for r in region_results]
        f1_scores = [r['f1_1'] for r in region_results]
        recall_scores = [r['recall_1'] for r in region_results]
        auc_scores = [r['auc'] for r in region_results]

        x = np.arange(len(region_names))
        colors = plt.cm.Set2(np.linspace(0, 1, len(region_names)))

        axes[0].bar(x, f1_scores, color=colors, edgecolor='white')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(region_names, rotation=30, ha='right')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_title('F1-Score by Region', fontweight='bold')

        axes[1].bar(x, recall_scores, color=colors, edgecolor='white')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(region_names, rotation=30, ha='right')
        axes[1].set_ylabel('Recall')
        axes[1].set_title('Recall by Region', fontweight='bold')

        axes[2].bar(x, auc_scores, color=colors, edgecolor='white')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(region_names, rotation=30, ha='right')
        axes[2].set_ylabel('AUC-ROC')
        axes[2].set_title('AUC-ROC by Region', fontweight='bold')

        plt.suptitle('Leave-One-Region-Out Analysis (XGBoost)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'regional_analysis.png'), bbox_inches='tight')
        plt.close()
        print(f"\n  [OK] regional_analysis.png")

    return region_results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("5a: EVALUATION")
    print("SHAP + Threshold Tuning + Regional Analysis")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load data
    print("\n[1/4] Loading prepared dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Rows: {len(df)}")

    # Prepare features
    print("\n[2/4] Preparing features...")
    X, y, feature_names, df_encoded = prepare_features(df)
    X = np.nan_to_num(X, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  Features: {len(feature_names)}")

    # SHAP Analysis
    print("\n[3/4] SHAP Analysis...")
    shap_model, shap_vals, importance = shap_analysis(
        X_train_s, X_test_s, y_train, feature_names, model_name='XGBoost'
    )

    # Threshold Tuning (XGBoost)
    print("\n[4/4] Threshold Tuning + Regional Analysis...")
    threshold_results = threshold_tuning(shap_model, X_test_s, y_test, 'XGBoost')

    # Regional Analysis
    regional_results = regional_analysis(df_encoded, feature_names, df)

    # Save all results
    all_results = {
        'shap_feature_importance': [{'feature': f, 'importance': float(i)} for f, i in importance],
        'threshold_tuning': {
            'best_f1': threshold_results['best_f1'],
            'best_cost': threshold_results['best_cost'],
        },
        'regional_analysis': regional_results,
    }

    output_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")
    print(f"Plots saved to: {PLOTS_DIR}/")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
