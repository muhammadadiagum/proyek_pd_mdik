#!/usr/bin/env python3
"""
Tsunami Prediction Pipeline — kelas + probabilitas
===================================================
Dua tahap:
1. Klasifikasi : apakah gempa memicu tsunami (4 model).
2. Estimasi    : bila tsunami, kelas tinggi air (regresi ordinal 3 kelas)
                 Waspada <= 0,5 m | Siaga 0,5–3,0 m | Awas > 3,0 m.

Tidak ada prediksi tinggi air dalam meter. Yang keluar adalah kelas
peringatan, probabilitas tiap kelas, dan rentang tinggi sesuai ambang
BMKG/InaTEWS. Model dimuat dari `model/` hasil `6_deployment.py`.

Usage:
  python predict.py --mag 7.0 --depth 20.0 --lat -5.0 --lon 110.0
  python predict.py --mag 8.5 --depth 10.0 --lat 35.0 --lon 140.0 --model random_forest
  python predict.py --mag 9.1 --depth 30 --lat 3.3 --lon 95.8 --year 2004 --month 12 --json
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

# Model ordinal dipickle sebagai kelas dari modul `ordinal` (phase 4 estimation),
# jadi path-nya harus ada saat unpickle.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "../phase_4_modeling/1_estimation"))

MODEL_DIR = os.path.join(BASE_DIR, "model")

CLASS_NAMES = ["Waspada", "Siaga", "Awas"]
CLASS_RANGES = ["<= 0.5 m", "0.5 - 3.0 m", "> 3.0 m"]


def _load(name):
    with open(os.path.join(MODEL_DIR, name), 'rb') as f:
        return pickle.load(f)


def load_model(model_name="xgboost"):
    return _load(f"{model_name}.pkl")


def load_scaler(kind="mlp"):
    return _load("scaler.pkl" if kind == "mlp" else "scaler_lr.pkl")


def load_features():
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        return json.load(f)


def load_thresholds():
    """Ambang keputusan per model (hasil tuning inner-validation)."""
    with open(os.path.join(MODEL_DIR, "thresholds.json")) as f:
        return json.load(f)


def load_ordinal():
    """Model ordinal + metadata ambang."""
    model = _load("ordinal_height.pkl")
    with open(os.path.join(MODEL_DIR, "ordinal_meta.json")) as f:
        meta = json.load(f)
    return model, meta


def classify_tectonic_zone(lat, lon):
    lat, lon = float(lat), float(lon)
    if -60 < lat < 60 and ((100 < lon < 180) or (-180 < lon < -60)):
        return 'pacific_ring_of_fire'
    if 20 < lat < 50 and -10 < lon < 90:
        return 'mediterranean_asian_belt'
    if -40 < lon < 0 and -60 < lat < 60:
        return 'mid_atlantic_ridge'
    return 'other'


def classify_era(year):
    y = int(year)
    if y < 1930:
        return 'early_1900s'
    if y < 1960:
        return 'mid_1900s'
    if y < 1990:
        return 'late_1900s'
    if y < 2010:
        return 'early_2000s'
    return 'modern'


def season_of(month):
    if month in (12, 1, 2):
        return 'djf'
    if month in (3, 4, 5):
        return 'mam'
    if month in (6, 7, 8):
        return 'jja'
    return 'son'


def monitoring_quality(nst, gap, dmin, rms):
    score, count = 0.0, 0
    if nst > 0:
        score += min(nst / 100, 1.0)
        count += 1
    if gap > 0:
        score += 1.0 - min(gap / 360, 1.0)
        count += 1
    if dmin > 0:
        score += 1.0 - min(dmin / 10, 1.0)
        count += 1
    if rms > 0:
        score += 1.0 - min(rms / 2.0, 1.0)
        count += 1
    return score / count if count else 0.5


def build_features(mag, depth, lat, lon, nst=50, gap=89.7, dmin=2.6, rms=0.92,
                   year=2026, month=6):
    """Nilai fitur mentah + turunan (belum one-hot)."""
    zone = classify_tectonic_zone(lat, lon)
    return {
        'mag': mag, 'depth': depth, 'nst': nst, 'gap': gap,
        'dmin': dmin, 'rms': rms, 'latitude': lat, 'longitude': lon,
        'abs_latitude': abs(lat),
        'depth_mag_ratio': depth / mag if mag > 0 else 0.0,
        'monitoring_quality': monitoring_quality(nst, gap, dmin, rms),
        '_zone': zone,
        '_era': classify_era(year),
        '_season': season_of(month),
    }


def to_vector(values, feature_names):
    """Susun vektor sesuai urutan fitur saat training (one-hot eksplisit)."""
    row = []
    for f in feature_names:
        if f == 'tectonic_zone_' + values['_zone']:
            row.append(1.0)
        elif f == 'era_' + values['_era']:
            row.append(1.0)
        elif f == 'season_' + values['_season']:
            row.append(1.0)
        elif f.startswith(('tectonic_zone_', 'era_', 'season_')):
            row.append(0.0)
        else:
            row.append(float(values.get(f, 0.0)))
    return np.array([row], dtype=float)


def predict_single(mag, depth, lat, lon, nst=50, gap=89.7, dmin=2.6, rms=0.92,
                   year=2026, month=6, model_name="xgboost", threshold=None):
    """
    Prediksi satu gempa.

    Returns dict:
      prediction   : 'TSUNAMI' / 'NO TSUNAMI'
      probability  : probabilitas tsunami dari model klasifikasi
      warning      : kelas ordinal + probabilitas + rentang (estimasi bila tsunami)
      input        : fitur mentah + turunan yang dipakai
    """
    model = load_model(model_name)
    feature_names = load_features()
    ordinal, ord_meta = load_ordinal()

    if threshold is None:
        threshold = float(load_thresholds()['models'].get(model_name, 0.5))
        threshold_source = 'model (inner-validation)'
    else:
        threshold_source = 'override'

    values = build_features(mag, depth, lat, lon, nst, gap, dmin, rms, year, month)
    X = to_vector(values, feature_names)

    X_model = X
    if model_name == 'mlp':
        X_model = load_scaler('mlp').transform(X)
    elif model_name == 'logistic_regression':
        X_model = load_scaler('lr').transform(X)

    prob = float(model.predict_proba(X_model)[0][1])
    tsunami = prob >= threshold

    # --- Estimasi ordinal (selalu dihitung; ditandai applicable bila tsunami) ---
    ord_features = ord_meta['features']
    X_ord = to_vector(values, ord_features)
    proba = ordinal.predict_proba(X_ord)[0]
    cls = int(np.argmax(proba))

    # aturan sadar-biaya
    t_high = ord_meta['warning_thresholds']['t_high_p_awas']
    t_mid = ord_meta['warning_thresholds']['t_mid_p_siaga_plus_awas']
    if proba[2] >= t_high:
        rule_cls = 2
    elif proba[1] + proba[2] >= t_mid:
        rule_cls = 1
    else:
        rule_cls = 0

    warning = {
        'class_index': cls,
        'label': CLASS_NAMES[cls],
        'range_m': CLASS_RANGES[cls],
        'probabilities': {CLASS_NAMES[k]: round(float(proba[k]), 4) for k in range(3)},
        'rule_class_index': rule_cls,
        'rule_label': CLASS_NAMES[rule_cls],
        'applicable': bool(tsunami),
        'thresholds_m': ord_meta['thresholds_m'],
    }

    return {
        'prediction': 'TSUNAMI' if tsunami else 'NO TSUNAMI',
        'probability': prob,
        'threshold': threshold,
        'threshold_source': threshold_source,
        'model': model_name,
        'warning': warning,
        'input': {
            'mag': mag, 'depth': depth, 'lat': lat, 'lon': lon,
            'tectonic_zone': values['_zone'],
            'era': values['_era'],
            'season': values['_season'],
            'monitoring_quality': round(values['monitoring_quality'], 4),
        },
    }


def main():
    parser = argparse.ArgumentParser(description='Tsunami Prediction Pipeline')
    parser.add_argument('--mag', type=float, required=True, help='Earthquake magnitude')
    parser.add_argument('--depth', type=float, required=True, help='Earthquake depth (km)')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--nst', type=float, default=50, help='Number of stations')
    parser.add_argument('--gap', type=float, default=89.7, help='Azimuthal gap')
    parser.add_argument('--dmin', type=float, default=2.6, help='Minimum station distance')
    parser.add_argument('--rms', type=float, default=0.92, help='Root mean square')
    parser.add_argument('--year', type=int, default=2026, help='Year')
    parser.add_argument('--month', type=int, default=6, help='Month')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'logistic_regression', 'random_forest', 'mlp'],
                        help='Classification model')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override ambang (default: ambang model dari thresholds.json)')
    parser.add_argument('--json', action='store_true', help='Print raw JSON')
    args = parser.parse_args()

    result = predict_single(
        mag=args.mag, depth=args.depth, lat=args.lat, lon=args.lon,
        nst=args.nst, gap=args.gap, dmin=args.dmin, rms=args.rms,
        year=args.year, month=args.month,
        model_name=args.model, threshold=args.threshold,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    w = result['warning']
    print("=" * 54)
    print("TSUNAMI PREDICTION RESULT")
    print("=" * 54)
    print(f"Prediction   : {result['prediction']}")
    print(f"Probability  : {result['probability']:.4f} "
          f"(ambang {result['threshold']:.2f} — {result['threshold_source']})")
    print(f"Model        : {result['model']}")
    print("-" * 54)
    if w['applicable']:
        print(f"Estimasi     : {w['label']} ({w['range_m']})")
    else:
        print("Estimasi     : Tidak berlaku (prediksi NO TSUNAMI)")
    print("  Probabilitas: " + ", ".join(
        f"{k} {v:.3f}" for k, v in w['probabilities'].items()))
    print(f"  Aturan waspada: {w['rule_label']}")
    print("-" * 54)
    print("Input:")
    for k, v in result['input'].items():
        print(f"  {k}: {v}")
    print("=" * 54)


if __name__ == '__main__':
    main()
