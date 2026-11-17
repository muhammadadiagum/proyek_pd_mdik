#!/usr/bin/env python3
"""
5d — Statistical Significance Tests
McNemar's Test + 95% Confidence Intervals
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from itertools import combinations
from scipy import stats
import xgboost as xgb


def _spw(y):
    """scale_pos_weight = rasio negatif/positif pada data latih (bukan konstanta)."""
    y = np.asarray(y)
    return float((len(y) - y.sum()) / max(y.sum(), 1))


DATA_PATH = "../../00_data/processed/modeling_dataset.csv"
OUTPUT_DIR = "results"
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("5d: STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    Xs_train, Xs_test, y_train, y_test = prepare_data()

    # Train models
    models = {}
    models['XGBoost'] = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           random_state=42, scale_pos_weight=_spw(y_train),
                                           eval_metric='logloss', use_label_encoder=False)
    models['LogisticRegression'] = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    models['RandomForest'] = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42,
                                                    n_jobs=-1)
    models['MLP'] = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                  random_state=42, early_stopping=True)

    for name, model in models.items():
        model.fit(Xs_train, y_train)

    preds = {n: m.predict(Xs_test) for n, m in models.items()}

    # --- McNemar's Test ---
    print("\n--- McNemar's Test ---")
    mcnemar = {}
    for n1, n2 in combinations(models.keys(), 2):
        b = int(((preds[n1] == y_test) & (preds[n2] != y_test)).sum())
        c = int(((preds[n1] != y_test) & (preds[n2] == y_test)).sum())
        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
        p = float(1 - stats.chi2.cdf(chi2, df=1))
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        mcnemar[f"{n1} vs {n2}"] = {'chi2': round(chi2, 4), 'p': round(p, 6), 'sig': sig, 'b_only_n1': b, 'b_only_n2': c}
        print(f"  {n1:20s} vs {n2:20s}: chi2={chi2:.4f}, p={p:.6f} [{sig}]")

    # --- Confidence Intervals (Bootstrap) ---
    print("\n--- 95% Confidence Intervals (F1, 200x bootstrap) ---")
    ci = {}
    n = len(y_test)
    for name in models:
        f1s = []
        for _ in range(200):
            idx = np.random.choice(n, n, replace=True)
            f1s.append(f1_score(y_test[idx], preds[name][idx]))
        ci[name] = {
            'mean': round(float(np.mean(f1s)), 4),
            'ci_low': round(float(np.percentile(f1s, 2.5)), 4),
            'ci_high': round(float(np.percentile(f1s, 97.5)), 4),
        }
        print(f"  {name:20s}: F1={ci[name]['mean']:.4f} [{ci[name]['ci_low']:.4f}, {ci[name]['ci_high']:.4f}]")

    results = {'mcnemar': mcnemar, 'confidence_intervals': ci}
    with open(os.path.join(OUTPUT_DIR, "statistical_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/statistical_results.json")
    print("=" * 60)


if __name__ == '__main__':
    main()
