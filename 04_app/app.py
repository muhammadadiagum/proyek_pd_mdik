#!/usr/bin/env python3
"""
Tsunami Trigger Predictor — Web Interface
==========================================
Flask web app untuk Phase 6 deployment.
Pipeline dua tahap: klasifikasi tsunami, lalu estimasi kelas tinggi air
(regresi ordinal: Waspada/Siaga/Awas) — bukan tinggi dalam meter.
Memakai logika inferensi yang sama dengan predict.py di
01_pipeline/phase_6_deployment.

Usage:
    pip install flask
    python app.py
    -> http://localhost:5000
"""

import os
import secrets
import sys

import numpy as np

# Import logika prediksi dari predict.py (phase_6_deployment)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE6_DIR = os.path.join(APP_DIR, "..", "01_pipeline", "phase_6_deployment")
sys.path.insert(0, PHASE6_DIR)
sys.path.insert(0, APP_DIR)

from flask import Flask, render_template, request, jsonify
from predict import predict_single, load_thresholds  # noqa: E402
from mdik import init_app as init_mdik  # noqa: E402

app = Flask(__name__)
# SECRET_KEY dari variabel lingkungan; kalau kosong, dibuat acak per proses.
# Konsekuensinya sesi login hilang setiap server dijalankan ulang — memadai
# untuk prototipe, tetapi isi MDIK_SECRET_KEY untuk pemakaian nyata.
app.config["SECRET_KEY"] = os.environ.get("MDIK_SECRET_KEY") or secrets.token_hex(32)
init_mdik(app)

MODELS = ['xgboost', 'random_forest', 'logistic_regression', 'mlp']
MODEL_LABELS = {
    'xgboost': 'XGBoost',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'mlp': 'MLP',
}

# Preset gempa historis nyata (dari dataset NOAA/USGS)
PRESETS = {
    'aceh2004': {
        'label': 'Aceh 2004 (M9.1)',
        'mag': 9.1, 'depth': 30.0, 'lat': 3.3, 'lon': 95.8,
        'year': 2004, 'month': 12,
    },
    'tohoku2011': {
        'label': 'Tohoku 2011 (M9.1)',
        'mag': 9.1, 'depth': 29.0, 'lat': 38.3, 'lon': 142.4,
        'year': 2011, 'month': 3,
    },
    'palu2018': {
        'label': 'Palu 2018 (M7.5, tsunami)',
        'mag': 7.5, 'depth': 10.0, 'lat': -0.256, 'lon': 119.846,
        'year': 2018, 'month': 9,
    },
    'small_no_tsunami': {
        'label': 'Gempa Kecil (M4.5, non-tsunami)',
        'mag': 4.5, 'depth': 35.0, 'lat': -5.0, 'lon': 110.0,
        'year': 2026, 'month': 6,
    },
}


def run_all_models(form):
    """Jalankan prediksi untuk model terpilih + semua model lain utk komparasi."""
    mag = float(form['mag'])
    depth = float(form['depth'])
    lat = float(form['lat'])
    lon = float(form['lon'])
    nst = float(form.get('nst', 50))
    gap = float(form.get('gap', 89.7))
    dmin = float(form.get('dmin', 2.6))
    rms = float(form.get('rms', 0.92))
    year = int(form.get('year', 2026))
    month = int(form.get('month', 6))

    # Ambang berasal dari model (tuning inner-validation), bukan slider bebas.
    threshold_cfg = load_thresholds()
    thresholds = threshold_cfg['models']

    selected = form.get('model', 'ensemble')
    if selected not in MODELS:
        selected = 'ensemble'

    results = {}
    for m in MODELS:
        r = predict_single(
            mag=mag, depth=depth, lat=lat, lon=lon,
            nst=nst, gap=gap, dmin=dmin, rms=rms,
            year=year, month=month,
            model_name=m,
        )
        results[m] = {
            'probability': r['probability'],
            'prediction': r['prediction'],
            'threshold': r['threshold'],
            'warning': r['warning'],
            'input': r['input'],
        }

    # Ensemble = soft voting (rata-rata probabilitas), ambang = rata-rata ambang model
    ens_prob = sum(results[m]['probability'] for m in MODELS) / len(MODELS)
    ens_th = float(thresholds.get('ensemble',
                                  np.mean([results[m]['threshold'] for m in MODELS])))
    ens_pred = 'TSUNAMI' if ens_prob >= ens_th else 'NO TSUNAMI'
    # Estimasi ordinal tidak bergantung pada model klasifikasi: satu model
    # ordinal dipakai untuk semua, jadi warning-nya identik.
    results['ensemble'] = {
        'probability': ens_prob,
        'prediction': ens_pred,
        'threshold': ens_th,
        'warning': results[MODELS[0]]['warning'],
    }

    main_key = selected if selected in MODELS else 'ensemble'
    return main_key, results, thresholds


@app.route('/')
def index():
    return render_template('index.html', presets=PRESETS, models=MODEL_LABELS,
                           thresholds=load_thresholds()['models'])


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Terima form (UI) maupun JSON (curl/tester)
        form = request.get_json(silent=True) or request.form
        # Validasi wajib
        for field in ('mag', 'depth', 'lat', 'lon'):
            if field not in form or form[field] == '':
                return jsonify({'error': f'Field "{field}" wajib diisi'}), 400

        main_key, results, thresholds = run_all_models(form)

        prob_main = results[main_key]['probability']
        th_main = results[main_key]['threshold']
        if prob_main >= th_main:
            risk, risk_class = 'HIGH', 'high'
        elif prob_main >= 0.5 * th_main:
            risk, risk_class = 'MODERATE', 'moderate'
        else:
            risk, risk_class = 'LOW', 'low'

        def _warn_of(m):
            """Kelas ordinal hanya bermakna bila model menyatakan TSUNAMI."""
            w = results[m]['warning']
            if not w.get('applicable', True) or results[m]['prediction'] != 'TSUNAMI':
                return {'label': 'Tidak berlaku', 'range': '—', 'rule': '—'}
            return {'label': w['label'], 'range': w['range_m'], 'rule': w['rule_label']}

        comparison = [
            {'key': m, 'name': MODEL_LABELS.get(m, 'Ensemble'),
             'probability': results[m]['probability'],
             'threshold': results[m]['threshold'],
             'prediction': results[m]['prediction'],
             'warning_label': _warn_of(m)['label'],
             'warning_range': _warn_of(m)['range'],
             'warning_rule': _warn_of(m)['rule']}
            for m in list(MODELS) + ['ensemble']
        ]

        main_warn = dict(results[main_key]['warning'])
        if results[main_key]['prediction'] != 'TSUNAMI':
            main_warn['label'] = 'Tidak berlaku'
            main_warn['range_m'] = '—'
            main_warn['applicable'] = False

        return jsonify({
            'success': True,
            'main_model': MODEL_LABELS.get(main_key, 'Ensemble'),
            'probability': prob_main,
            'prediction': results[main_key]['prediction'],
            'warning': main_warn,
            'risk_label': risk,
            'risk_class': risk_class,
            'threshold': th_main,
            'thresholds': thresholds,
            'derived': results[list(MODELS)[0]]['input'],
            'comparison': comparison,
        })
    except ValueError as e:
        return jsonify({'error': f'Input tidak valid: {e}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("TSUNAMI TRIGGER PREDICTOR — Web Interface")
    print("Buka http://localhost:5000 di browser")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
