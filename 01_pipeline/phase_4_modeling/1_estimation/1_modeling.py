#!/usr/bin/env python3
"""
1_modeling.py — Phase 4: ESTIMATION (regresi ordinal tinggi air tsunami)
=========================================================================
Menggantikan regresi kontinu pada proyek lama (MAE 0,99 m; R² 0,26; praktis
tidak lebih baik dari menebak median 0,3 m) dengan regresi ordinal 3 kelas.

Target : noaa_water_height (meter) pada tsunami akibat gempa (cause_code = 1)
          n = 610 dari 833 tsunami; event non-seismik dibuang (di luar domain).
Kelas  : 0 = Waspada  (<= 0,5 m)
         1 = Siaga     (0,5 - 3,0 m)
         2 = Awas      (> 3,0 m)
         Ambang mengikuti tahapan peringatan dini tsunami (InaTEWS/BMKG).

Model  : 4 pembelajar dasar (LR, RF, XGB, MLP) x 2 skema
         - ordinal   : cumulative binary decomposition (K-1 model biner)
         - multikelas: baseline nominal (softmax/argmax) untuk pembanding
Metrik : QWK (utama), MAE kelas, akurasi tepat, akurasi +/-1 kelas,
         presisi/recall/F1 per kelas, confusion matrix.
         Ditambah aturan keputusan sadar-biaya (P(Awas)>=0,25; P(Siaga+Awas)>=0,40)
         untuk menaikkan recall kelas Awas dengan biaya MAE kecil.

Evaluasi: 5-fold stratified CV (out-of-fold) + holdout 20% stratified.
Baseline: kelas mayoritas, dan regresi kontinu lama yang dibulatkan ke kelas.

Input : ../../../00_data/processed/modeling_dataset.csv
Output: results/estimation_ordinal_results.json
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ordinal import OrdinalCumulative, ordinal_metrics, predict_warning

# ============================================================
# LOGGING
# ============================================================

class Tee:
    def __init__(self, log_path):
        self.stdout = sys.stdout
        self.log_file = open(log_path, "w")

    def write(self, data):
        self.stdout.write(data)
        self.log_file.write(data)
        self.stdout.flush()
        self.log_file.flush()

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        sys.stdout = self.stdout


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, ".script_logs")
os.makedirs(LOG_DIR, exist_ok=True)
tee = Tee(os.path.join(LOG_DIR, os.path.basename(__file__) + ".log"))
sys.stdout = tee

# ============================================================
# KONFIGURASI
# ============================================================

DATA_PATH = os.path.join(BASE_DIR, "../../../00_data/processed/modeling_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "estimation_ordinal_results.json")

RANDOM_STATE = 42
N_CLASSES = 3
THRESHOLDS = (0.5, 3.0)          # meter: batas Waspada|Siaga dan Siaga|Awas
WARNING_THRESHOLDS = (0.25, 0.40)  # aturan sadar-biaya: P(Awas), P(Siaga+Awas)
CLASS_NAMES = ["Waspada", "Siaga", "Awas"]

CAT_COLS = ["tectonic_zone", "era", "season"]
FEATURE_COLS = ["mag", "depth", "nst", "gap", "dmin", "rms",
                "latitude", "longitude", "abs_latitude",
                "depth_mag_ratio", "monitoring_quality"]


def height_to_class(h):
    """Konversi tinggi air (meter) ke kelas ordinal 0/1/2."""
    lo, hi = THRESHOLDS
    return np.where(h <= lo, 0, np.where(h <= hi, 1, 2)).astype(int)


def model_factories():
    """Factory mengembalikan estimator BARU tiap panggilan (aman untuk CV)."""
    return {
        "LogisticRegression": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        "MLP": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                                  random_state=RANDOM_STATE, early_stopping=True)),
        ]),
    }


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    sub = df[(df["label"] == 1)
             & (df["noaa_water_height"].notna())
             & (df["noaa_cause_code"] == 1)].copy()

    dummies = pd.get_dummies(sub[CAT_COLS], drop_first=False).astype(float)
    X = pd.concat([sub[FEATURE_COLS].astype(float), dummies], axis=1)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    y_meter = sub["noaa_water_height"].values.astype(float)
    y = height_to_class(y_meter)
    return X, y, y_meter, sub


def aggregate_folds(per_fold):
    """Rata-rata dan simpangan baku antar fold."""
    keys = ["qwk", "mae_class", "exact_acc", "adjacent_acc"]
    return {
        k: {"mean": round(float(np.mean([f[k] for f in per_fold])), 4),
            "sd": round(float(np.std([f[k] for f in per_fold], ddof=1)), 4)}
        for k in keys
    }


def evaluate_cv(X, y, factory, schema):
    """5-fold stratified CV; kembalikan metrik per fold + metrik OOF gabungan."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    per_fold, oof = [], np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), N_CLASSES))

    for tr, va in skf.split(X, y):
        if schema == "ordinal":
            est = OrdinalCumulative(factory(), n_classes=N_CLASSES)
        else:
            est = factory()
        est.fit(X.iloc[tr], y[tr])
        proba = est.predict_proba(X.iloc[va])
        pred = proba.argmax(axis=1)
        oof_proba[va] = proba
        oof[va] = pred
        per_fold.append(ordinal_metrics(y[va], pred, N_CLASSES))

    return {
        "per_fold": per_fold,
        "folds_mean": aggregate_folds(per_fold),
        "pooled_oof": ordinal_metrics(y, oof, N_CLASSES),
        "warning_rule": ordinal_metrics(
            y, predict_warning(oof_proba, *WARNING_THRESHOLDS), N_CLASSES),
    }


def evaluate_holdout(X, y, factory, schema, Xtr, Xte, ytr, yte):
    if schema == "ordinal":
        est = OrdinalCumulative(factory(), n_classes=N_CLASSES)
    else:
        est = factory()
    est.fit(Xtr, ytr)
    proba = est.predict_proba(Xte)
    return {
        "argmax": ordinal_metrics(yte, proba.argmax(axis=1), N_CLASSES),
        "warning_rule": ordinal_metrics(
            yte, predict_warning(proba, *WARNING_THRESHOLDS), N_CLASSES),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    print("=" * 78)
    print("PHASE 4 — ESTIMATION (REG RESI ORDINAL TINGGI AIR TSUNAMI)")
    print("=" * 78)
    print(f"Waktu    : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Data     : {DATA_PATH}")
    print(f"Ambang   : Waspada <= {THRESHOLDS[0]} m | Siaga <= {THRESHOLDS[1]} m | Awas > {THRESHOLDS[1]} m")

    X, y, y_meter, sub = load_data()
    counts = np.bincount(y, minlength=N_CLASSES)

    print(f"\nSampel   : {len(y)} tsunami akibat gempa (cause=1) dengan water_height")
    print(f"Fitur    : {X.shape[1]} ({len(FEATURE_COLS)} numerik + {X.shape[1]-len(FEATURE_COLS)} dummy)")
    print("\nDistribusi kelas:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name:8s}: {counts[i]:4d} ({counts[i]/len(y)*100:5.1f}%)")
    print(f"\nStatistik tinggi air: median={np.median(y_meter):.2f} m, "
          f"mean={y_meter.mean():.2f} m, max={y_meter.max():.1f} m")

    # Split holdout sekali, dipakai semua model
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"Holdout  : train={len(ytr)}, test={len(yte)} (stratified 80/20)")

    factories = model_factories()
    results = {"cv": {}, "holdout": {}, "holdout_warning": {}}

    # ---------- 5-fold CV: ordinal vs multikelas ----------
    print("\n" + "=" * 78)
    print("[1/3] 5-FOLD STRATIFIED CV")
    print("=" * 78)
    print(f"{'Model':22s} {'Skema':10s} {'QWK':>7s} {'MAE-kls':>8s} {'Tepat':>7s} {'+/-1':>7s}")
    print("-" * 78)

    for name, make in factories.items():
        results["cv"][name] = {}
        for schema in ("ordinal", "multiclass"):
            t0 = time.time()
            r = evaluate_cv(X, y, make, schema)
            results["cv"][name][schema] = r
            m = r["folds_mean"]
            print(f"{name:22s} {schema:10s} "
                  f"{m['qwk']['mean']:7.4f} {m['mae_class']['mean']:8.4f} "
                  f"{m['exact_acc']['mean']:7.4f} {m['adjacent_acc']['mean']:7.4f}"
                  f"   ({time.time()-t0:.1f}s)")

    # ---------- Holdout ----------
    print("\n" + "=" * 78)
    print("[2/3] HOLDOUT 20%")
    print("=" * 78)
    print(f"{'Model':22s} {'Skema':10s} {'QWK':>7s} {'MAE-kls':>8s} {'Tepat':>7s} {'+/-1':>7s}")
    print("-" * 78)

    for name, make in factories.items():
        results["holdout"][name] = {}
        results["holdout_warning"][name] = {}
        for schema in ("ordinal", "multiclass"):
            r = evaluate_holdout(X, y, make, schema, Xtr, Xte, ytr, yte)
            results["holdout"][name][schema] = r["argmax"]
            results["holdout_warning"][name][schema] = r["warning_rule"]
            m = r["argmax"]
            print(f"{name:22s} {schema:10s} "
                  f"{m['qwk']:7.4f} {m['mae_class']:8.4f} "
                  f"{m['exact_acc']:7.4f} {m['adjacent_acc']:7.4f}")

    # ---------- Aturan keputusan sadar-biaya ----------
    print("\n" + "=" * 78)
    print(f"[2b/3] ATURAN SADAR-BIAYA  P(Awas)>={WARNING_THRESHOLDS[0]:.2f}, "
          f"P(Siaga+Awas)>={WARNING_THRESHOLDS[1]:.2f}")
    print("=" * 78)
    print(f"{'Model':22s} {'Skema':10s} {'QWK':>7s} {'MAE-kls':>8s} "
          f"{'Awas-R':>7s} {'Siaga-R':>8s} {'Prec-Awas':>10s}")
    print("-" * 78)
    for name in factories:
        for schema in ("ordinal", "multiclass"):
            d = results["holdout_warning"][name][schema]
            pc = d["per_class"]
            print(f"{name:22s} {schema:10s} {d['qwk']:7.4f} {d['mae_class']:8.4f} "
                  f"{pc['class_2']['recall']:7.2f} {pc['class_1']['recall']:8.2f} "
                  f"{pc['class_2']['precision']:10.2f}")

    # ---------- Baseline ----------
    print("\n" + "=" * 78)
    print("[3/3] BASELINE")
    print("=" * 78)

    # (a) kelas mayoritas
    majority = int(np.bincount(ytr).argmax())
    pred_maj = np.full(len(yte), majority)
    maj = ordinal_metrics(yte, pred_maj, N_CLASSES)

    # (b) regresi kontinu lama -> dibulatkan ke kelas (pendekatan proyek lama)
    reg = Pipeline([("scaler", StandardScaler()),
                    ("reg", LinearRegression())])
    reg.fit(Xtr, np.log1p(sub.loc[Xtr.index, "noaa_water_height"].values))
    pred_meter = np.clip(np.expm1(reg.predict(Xte)), 0, None)
    pred_reg = height_to_class(pred_meter)
    reg_m = ordinal_metrics(yte, pred_reg, N_CLASSES)

    print(f"{'Mayoritas (kelas '+str(majority)+')':22s} {'-':10s} "
          f"{maj['qwk']:7.4f} {maj['mae_class']:8.4f} "
          f"{maj['exact_acc']:7.4f} {maj['adjacent_acc']:7.4f}")
    print(f"{'Regresi+ambang (lama)':22s} {'-':10s} "
          f"{reg_m['qwk']:7.4f} {reg_m['mae_class']:8.4f} "
          f"{reg_m['exact_acc']:7.4f} {reg_m['adjacent_acc']:7.4f}")

    # ---------- Simpan ----------
    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": os.path.basename(__file__),
            "data": os.path.relpath(DATA_PATH, BASE_DIR),
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "feature_names": list(X.columns),
            "random_state": RANDOM_STATE,
            "cv": "StratifiedKFold(n_splits=5, shuffle=True)",
            "holdout": "80/20 stratified",
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
            },
        },
        "class_definition": {
            "names": CLASS_NAMES,
            "thresholds_m": list(THRESHOLDS),
            "counts": counts.tolist(),
            "shares": [round(float(c / len(y)), 4) for c in counts],
        },
        "cv": results["cv"],
        "holdout": results["holdout"],
        "holdout_warning": results["holdout_warning"],
        "warning_rule": {
            "t_high_p_awas": WARNING_THRESHOLDS[0],
            "t_mid_p_siaga_plus_awas": WARNING_THRESHOLDS[1],
        },
        "baselines": {
            "majority_class": {"class": majority, **maj},
            "continuous_regression_thresholded": reg_m,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 78)
    print(f"Selesai dalam {time.time()-t_start:.1f}s")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
    tee.close()
