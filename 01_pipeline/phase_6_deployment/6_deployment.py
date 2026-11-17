#!/usr/bin/env python3
"""
6 — Deployment — Model Final & Pipeline Prediksi
=================================================
Menyimpan model terlatih (konfigurasi FINAL Phase 4) dan model estimasi
ordinal 3 kelas, lalu dipakai oleh `predict.py` dan `04_app/app.py`.

Dua peran yang dideploy:
1. KLASIFIKASI  : apakah gempa memicu tsunami (4 model, ambang probabilitas).
2. ESTIMASI     : bila tsunami, seberapa besar — REGRESI ORDINAL 3 kelas
                  (Waspada <= 0,5 m | Siaga 0,5–3,0 m | Awas > 3,0 m).

Catatan: tidak ada lagi prediksi tinggi air dalam meter. Regresi kontinu
sudah diganti regresi ordinal karena R² negatif di data uji (lihat
.audit/00_RINGKASAN.md §5 keputusan 1). Yang ditampilkan adalah kelas
peringatan + probabilitas tiap kelas + rentang tinggi sesuai ambang BMKG.

Konfigurasi klasifikasi (identik 1_modeling.py):
- LogisticRegression: max_iter=2000, class_weight='balanced'
- RandomForest: n_estimators=200, max_depth=15, tanpa class_weight
- XGBoost: n_estimators=200, max_depth=6, lr=0.1, scale_pos_weight=rasio data
- MLP: hidden=(128,64,32), max_iter=500, early_stopping

Output: model/ (model + scaler + feature names + metadata),
        results/deployment_results.json
"""

import os
import json
import pickle
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score, roc_auc_score,
)

import xgboost as xgb

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "../phase_4_modeling/1_estimation"))
from ordinal import OrdinalCumulative, ordinal_metrics, predict_warning  # noqa: E402

# === SCRIPT LOGGING (auto-injected) ===
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

# Paths
DATA_PATH = os.path.join(BASE_DIR, "../../00_data/processed/modeling_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# Feature config (same as modeling)
FEATURE_COLS = [
    'mag', 'depth', 'nst', 'gap', 'dmin', 'rms',
    'latitude', 'longitude', 'abs_latitude',
    'depth_mag_ratio', 'monitoring_quality',
]
CATEGORICAL_COLS = ['tectonic_zone', 'era', 'season']
TARGET_COL = 'label'

# Biaya keputusan (FN jauh lebih mahal daripada FP)
FN_COST = 100
FP_COST = 1

# Protokol ambang: tuning HANYA pada inner-validation dari data latih.
# Constraint presisi: tanpa lantai presisi, rasio biaya 100:1 mendorong ambang
# ke 0,01 (praktis semua gempa jadi alarm: ~57.000 false alarm untuk 131 hit).
# Lantai 10% = paling banyak ~9 alarm palsu per tsunami yang tertangkap.
INNER_VAL_SIZE = 0.25
INNER_SEED = 7
MIN_PRECISION = 0.10
MODEL_NAMES = ['xgboost', 'random_forest', 'logistic_regression', 'mlp']

# Estimasi ordinal — ambang BMKG/InaTEWS
THRESHOLDS = (0.5, 3.0)
WARNING_THRESHOLDS = (0.25, 0.40)
CLASS_NAMES = ["Waspada", "Siaga", "Awas"]
CLASS_RANGES = ["<= 0.5 m", "0.5 - 3.0 m", "> 3.0 m"]


def height_to_class(h):
    """Konversi tinggi air (meter) ke kelas ordinal 0/1/2."""
    lo, hi = THRESHOLDS
    return np.where(h <= lo, 0, np.where(h <= hi, 1, 2)).astype(int)


def prepare_features(df):
    """Persiapan fitur (sama seperti modeling)."""
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    all_feature_cols = FEATURE_COLS.copy()
    for col in df_encoded.columns:
        if col.startswith(('tectonic_zone_', 'era_', 'season_')):
            all_feature_cols.append(col)

    available_cols = [c for c in all_feature_cols if c in df_encoded.columns]
    X = (df_encoded[available_cols].astype(float)
         .replace([np.inf, -np.inf], 0.0).fillna(0.0).values)
    y = df_encoded[TARGET_COL].values
    return X, y, available_cols, df_encoded


def build_model(name, spw):
    """Model belum terlatih dengan konfigurasi FINAL Phase 4."""
    if name == 'xgboost':
        return xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, scale_pos_weight=spw,
            eval_metric='logloss', n_jobs=-1)
    if name == 'random_forest':
        return RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    if name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, random_state=42,
                                  class_weight='balanced')
    if name == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                             random_state=42, early_stopping=True)
    raise ValueError(f'model tidak dikenal: {name}')


def _fit_proba(name, X_tr, y_tr, X_va, spw):
    """Latih model sementara, kembalikan probabilitas pada data validasi."""
    if name in ('logistic_regression', 'mlp'):
        sc = StandardScaler().fit(X_tr)
        model = build_model(name, spw).fit(sc.transform(X_tr), y_tr)
        return model.predict_proba(sc.transform(X_va))[:, 1]
    model = build_model(name, spw).fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]


def tune_thresholds(X_train, y_train):
    """
    Protokol ambang (menyelesaikan F-06/F-11).

    Ambang dipilih pada inner-validation 25% dari data LATIH, bukan pada test
    set. Kriteria: biaya minimum (FN x100 + FP x1) **dengan lantai presisi**
    MIN_PRECISION — supaya ambang tidak runtuh ke 0,01 dan menghasilkan puluhan
    ribu alarm palsu. Ambang biaya-minimum tanpa kendala dan ambang F1-maksimum
    tetap disimpan sebagai pembanding. Test set dipakai sekali untuk melaporkan.
    """
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=INNER_VAL_SIZE,
        random_state=INNER_SEED, stratify=y_train)
    spw = float((len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1))
    print(f"  Inner-validation: train {len(y_tr):,} / val {len(y_va):,} "
          f"({int(y_va.sum())} tsunami)")

    grid = np.arange(0.01, 1.00, 0.01)
    out = {}
    probs = {}
    for name in MODEL_NAMES:
        prob = _fit_proba(name, X_tr, y_tr, X_va, spw)
        probs[name] = prob
        rows = []
        for t in grid:
            pred = (prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_va, pred).ravel()
            rows.append({
                'threshold': float(t),
                'cost': int(fn * FN_COST + fp * FP_COST),
                'f1': float(f1_score(y_va, pred, zero_division=0)),
                'recall': float(recall_score(y_va, pred, zero_division=0)),
                'precision': float(precision_score(y_va, pred, zero_division=0)),
                'fn': int(fn), 'fp': int(fp),
            })
        unconstrained = min(rows, key=lambda r: (r['cost'], -r['f1']))
        best_f1 = max(rows, key=lambda r: (r['f1'], -r['cost']))
        feasible = [r for r in rows if r['precision'] >= MIN_PRECISION]
        chosen = min(feasible, key=lambda r: (r['cost'], -r['f1'])) if feasible else best_f1
        out[name] = {
            'threshold': chosen['threshold'],
            'selection': (f'min cost dengan presisi >= {MIN_PRECISION:.2f}'
                          if feasible else 'F1 maksimum (tidak ada yang memenuhi lantai presisi)'),
            'at_threshold': chosen,
            'unconstrained_cost_threshold': unconstrained['threshold'],
            'unconstrained_cost': unconstrained['cost'],
            'f1_optimal_threshold': best_f1['threshold'],
            'f1_optimal': best_f1,
            'inner_val_samples': int(len(y_va)),
            'inner_val_positives': int(y_va.sum()),
        }
        print(f"    {name:22s}: ambang={chosen['threshold']:.2f} "
              f"(F1={chosen['f1']:.4f}, recall={chosen['recall']:.4f}, "
              f"precision={chosen['precision']:.4f}, cost={chosen['cost']:,}) "
              f"| tanpa-lantai={unconstrained['threshold']:.2f}")

    # Ensemble soft voting: tuning langsung pada rata-rata probabilitas
    ens_prob = np.mean([probs[m] for m in MODEL_NAMES], axis=0)
    rows = []
    for t in grid:
        pred = (ens_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_va, pred).ravel()
        rows.append({
            'threshold': float(t),
            'cost': int(fn * FN_COST + fp * FP_COST),
            'f1': float(f1_score(y_va, pred, zero_division=0)),
            'recall': float(recall_score(y_va, pred, zero_division=0)),
            'precision': float(precision_score(y_va, pred, zero_division=0)),
            'fn': int(fn), 'fp': int(fp),
        })
    feasible = [r for r in rows if r['precision'] >= MIN_PRECISION]
    chosen = min(feasible, key=lambda r: (r['cost'], -r['f1'])) if feasible else max(rows, key=lambda r: r['f1'])
    out['ensemble'] = {
        'threshold': chosen['threshold'],
        'selection': f'min cost dengan presisi >= {MIN_PRECISION:.2f} (soft voting)',
        'at_threshold': chosen,
        'inner_val_samples': int(len(y_va)),
        'inner_val_positives': int(y_va.sum()),
    }
    print(f"    {'ensemble':22s}: ambang={chosen['threshold']:.2f} "
          f"(F1={chosen['f1']:.4f}, recall={chosen['recall']:.4f}, "
          f"precision={chosen['precision']:.4f})")
    return out


def train_final_models(X_train, y_train, feature_names):
    """Train model final untuk deployment (konfigurasi FINAL Phase 4)."""
    models = {}

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  scale_pos_weight dihitung dari data latih: {scale_pos_weight:.1f} "
          f"({n_neg:,} negatif / {n_pos:,} positif)")

    print("  Training XGBoost...")
    models['xgboost'] = build_model('xgboost', scale_pos_weight).fit(X_train, y_train)

    print("  Training Logistic Regression (scaled)...")
    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    models['logistic_regression'] = build_model('logistic_regression', scale_pos_weight).fit(X_train_lr, y_train)

    print("  Training Random Forest (no class_weight)...")
    models['random_forest'] = build_model('random_forest', scale_pos_weight).fit(X_train, y_train)

    print("  Training MLP...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    models['mlp'] = build_model('mlp', scale_pos_weight).fit(X_train_scaled, y_train)

    return models, scaler, scaler_lr, scale_pos_weight


def train_ordinal(df_encoded, feature_names):
    """Train model regresi ordinal 3 kelas (XGBoost, konfigurasi Phase 4)."""
    sub = df_encoded[(df_encoded['label'] == 1) &
                     (df_encoded['noaa_water_height'].notna()) &
                     (df_encoded['noaa_cause_code'] == 1)].copy()

    X = (sub[feature_names].astype(float)
         .replace([np.inf, -np.inf], 0.0).fillna(0.0).values)
    y_meter = sub['noaa_water_height'].values.astype(float)
    y = height_to_class(y_meter)

    model = OrdinalCumulative(
        xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss', n_jobs=-1,
        ),
        n_classes=3,
    )
    model.fit(X, y)

    dist = {CLASS_NAMES[k]: int((y == k).sum()) for k in range(3)}
    print(f"  Ordinal dilatih pada {len(y)} tsunami gempa ber-water_height")
    print(f"  Distribusi kelas: {dist}")
    return model, X, y, sub, feature_names


def evaluate_ordinal(X, y):
    """Evaluasi ordinal pada holdout 80/20 (stratified)."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    m = OrdinalCumulative(
        xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric='logloss', n_jobs=-1,
        ),
        n_classes=3,
    )
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)
    y_pred = proba.argmax(axis=1)
    metrics = ordinal_metrics(y_te, y_pred)
    warn_pred = predict_warning(proba, *WARNING_THRESHOLDS)
    metrics['warning_rule'] = ordinal_metrics(y_te, warn_pred)
    metrics['warning_thresholds'] = {
        't_high_p_awas': WARNING_THRESHOLDS[0],
        't_mid_p_siaga_plus_awas': WARNING_THRESHOLDS[1],
    }
    return metrics


def save_models(models, scaler, scaler_lr, feature_names):
    """Simpan model, scaler, dan daftar fitur ke disk."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    for name, model in models.items():
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"  Saved: {path}")

    for fname, obj in [("scaler.pkl", scaler), ("scaler_lr.pkl", scaler_lr)]:
        path = os.path.join(MODEL_DIR, fname)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved: {path}")

    path = os.path.join(MODEL_DIR, "feature_names.json")
    with open(path, 'w') as f:
        json.dump(feature_names, f)
    print(f"  Saved: {path}")


def save_thresholds(thresholds):
    """Simpan ambang keputusan per model (dipakai predict.py dan app)."""
    path = os.path.join(MODEL_DIR, "thresholds.json")
    payload = {
        'protocol': {
            'source': 'inner-validation 25% data latih (test tidak dipakai tuning)',
            'cost': f'FN x{FN_COST} + FP x{FP_COST}',
            'min_precision': MIN_PRECISION,
            'inner_val_size': INNER_VAL_SIZE,
            'inner_seed': INNER_SEED,
        },
        'models': {k: v['threshold'] for k, v in thresholds.items()},
        'details': thresholds,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {path}")
    for k, v in thresholds.items():
        print(f"    {k:22s}: ambang={v['threshold']:.2f} ({v['selection']})")


def save_ordinal(model, feature_names):
    """Simpan model ordinal + metadata ambang."""
    path = os.path.join(MODEL_DIR, "ordinal_height.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved: {path}")

    meta = {
        'model': 'OrdinalCumulative(XGBClassifier)',
        'n_classes': 3,
        'class_names': CLASS_NAMES,
        'class_ranges_m': CLASS_RANGES,
        'thresholds_m': list(THRESHOLDS),
        'warning_thresholds': {
            't_high_p_awas': WARNING_THRESHOLDS[0],
            't_mid_p_siaga_plus_awas': WARNING_THRESHOLDS[1],
        },
        'features': feature_names,
        'target': 'noaa_water_height -> kelas ordinal',
        'note': 'bukan prediksi meter; kelas + probabilitas + rentang',
    }
    meta_path = os.path.join(MODEL_DIR, "ordinal_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")


def main():
    print("=" * 60)
    print("6: DEPLOYMENT — klasifikasi + estimasi ordinal 3 kelas")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/5] Loading modeling dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Rows: {len(df):,} | positif: {int(df['label'].sum())}")

    print("\n[2/5] Preparing features...")
    X, y, feature_names, df_encoded = prepare_features(df)
    X = np.nan_to_num(X, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

    print("\n[3/6] Tuning ambang per model (inner-validation data latih)...")
    thresholds = tune_thresholds(X_train, y_train)
    print("\n[4/6] Training final models (config FINAL Phase 4)...")
    models, scaler, scaler_lr, spw = train_final_models(
        X_train, y_train, feature_names
    )

    print("\n[4/6] Evaluasi model klasifikasi pada test set (ambang dari inner-validation):")
    deployment_results = {}
    for name, model in models.items():
        if name == 'mlp':
            X_test_eval = scaler.transform(X_test)
        elif name == 'logistic_regression':
            X_test_eval = scaler_lr.transform(X_test)
        else:
            X_test_eval = X_test

        y_prob = model.predict_proba(X_test_eval)[:, 1]
        th = thresholds[name]['threshold']

        def _metrics(pred):
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            return {
                'f1': float(f1_score(y_test, pred, zero_division=0)),
                'recall': float(recall_score(y_test, pred, zero_division=0)),
                'precision': float(precision_score(y_test, pred, zero_division=0)),
                'cost': int(fn * FN_COST + fp * FP_COST),
                'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            }

        result = _metrics((y_prob >= th).astype(int))
        result['threshold'] = float(th)
        result['auc'] = float(roc_auc_score(y_test, y_prob))
        result['at_default_0.5'] = _metrics((y_prob >= 0.5).astype(int))
        deployment_results[name] = result
        print(f"  {name:22s}: ambang={th:.2f} F1={result['f1']:.4f}, "
              f"Recall={result['recall']:.4f}, AUC={result['auc']:.4f}, "
              f"Cost={result['cost']:,}")

    # Ensemble soft voting (rata-rata probabilitas 4 model) — ikut dievaluasi
    # pada test set, bukan hanya dipakai di app (menutup F-19).
    probs_test = []
    for name in MODEL_NAMES:
        if name == 'mlp':
            X_eval = scaler.transform(X_test)
        elif name == 'logistic_regression':
            X_eval = scaler_lr.transform(X_test)
        else:
            X_eval = X_test
        probs_test.append(models[name].predict_proba(X_eval)[:, 1])
    ens_prob = np.mean(probs_test, axis=0)
    ens_th = thresholds['ensemble']['threshold']
    ens_pred = (ens_prob >= ens_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, ens_pred).ravel()
    deployment_results['ensemble'] = {
        'f1': float(f1_score(y_test, ens_pred, zero_division=0)),
        'recall': float(recall_score(y_test, ens_pred, zero_division=0)),
        'precision': float(precision_score(y_test, ens_pred, zero_division=0)),
        'cost': int(fn * FN_COST + fp * FP_COST),
        'auc': float(roc_auc_score(y_test, ens_prob)),
        'threshold': float(ens_th),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }
    e = deployment_results['ensemble']
    print(f"  {'ensemble':22s}: ambang={ens_th:.2f} F1={e['f1']:.4f}, "
          f"Recall={e['recall']:.4f}, AUC={e['auc']:.4f}, Cost={e['cost']:,}")

    print("\n[5/6] Training model estimasi ordinal (3 kelas)...")
    ordinal_model, X_ord, y_ord, sub_ord, ord_features = train_ordinal(
        df_encoded, feature_names
    )
    print("  Evaluasi holdout 80/20 (stratified):")
    ord_metrics = evaluate_ordinal(X_ord, y_ord)
    print(f"    QWK={ord_metrics['qwk']:.4f}, MAE kelas={ord_metrics['mae_class']:.4f}, "
          f"tepat={ord_metrics['exact_acc']:.4f}, +/-1={ord_metrics['adjacent_acc']:.4f}")
    wr = ord_metrics['warning_rule']
    print(f"    Aturan sadar-biaya: QWK={wr['qwk']:.4f}, "
          f"recall Awas={wr['per_class']['class_2']['recall']:.4f}")

    print("\n[6/6] Saving models...")
    save_models(models, scaler, scaler_lr, feature_names)
    save_thresholds(thresholds)
    save_ordinal(ordinal_model, ord_features)

    metadata = {
        'model_date': datetime.now().isoformat(),
        'label_definition': 'positif = label=1 & noaa_cause_code=1',
        'dataset': os.path.relpath(DATA_PATH, BASE_DIR),
        'dataset_rows': int(len(df)),
        'positives': int(y.sum()),
        'config_version': 'final-v2 (ordinal estimation, scale_pos_weight dari data)',
        'features': feature_names,
        'feature_count': len(feature_names),
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'positive_ratio': float(y.sum() / len(y)),
        'scale_pos_weight': float(spw),
        'thresholds': {k: {'threshold': v['threshold'],
                           'selection': v['selection']}
                       for k, v in thresholds.items()},
        'threshold_protocol': {
            'source': 'inner-validation 25% dari data latih (test tidak dipakai)',
            'cost': f'FN x{FN_COST} + FP x{FP_COST}',
            'min_precision': MIN_PRECISION,
            'inner_val_seed': INNER_SEED,
            'note': 'ambang runtuh ke 0,01 tanpa lantai presisi (alarm palsu massal)',
        },
        'models_trained': list(models.keys()),
        'estimation': {
            'type': 'regression ordinal (cumulative binary decomposition)',
            'algorithm': 'XGBoost x 2 model biner',
            'n_classes': 3,
            'class_names': CLASS_NAMES,
            'class_ranges_m': CLASS_RANGES,
            'thresholds_m': list(THRESHOLDS),
            'warning_thresholds': list(WARNING_THRESHOLDS),
            'subset_samples': int(len(y_ord)),
            'holdout_metrics': ord_metrics,
        },
    }

    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    payload = {'classification': deployment_results,
               'thresholds': {k: v['threshold'] for k, v in thresholds.items()},
               'estimation': ord_metrics}
    results_path = os.path.join(OUTPUT_DIR, "deployment_results.json")
    with open(results_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved: {results_path}")

    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"\nModels saved to: {MODEL_DIR}/")
    print("Prediction pipeline: predict.py (kelas + probabilitas)")
    print("\nUsage:")
    print("  python predict.py --mag 7.0 --depth 20.0 --lat -5.0 --lon 110.0")
    print("  python predict.py --mag 8.5 --depth 10.0 --lat 35.0 --lon 140.0 --model random_forest")
    print("=" * 60)


if __name__ == '__main__':
    main()
