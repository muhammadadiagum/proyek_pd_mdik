#!/usr/bin/env python3
"""Uji latensi per model klasifikasi (4 model) + ensemble + endpoint.

Satu permintaan prediksi menjalankan satu model klasifikasi dan satu model
ordinal. Skrip ini mengukur biaya tiap model secara terpisah pada payload
kasus Aceh 2004 (sama dengan kasus T01), lalu biaya ensemble (empat model
berurutan) dan biaya endpoint /predict.

Output: 05_docs/pd/latency_per_model.json + run_latency_per_model.log
Dijalankan dari root proyek dengan .venv/bin/python.
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(BASE, "..", ".."))
APP_DIR = os.path.join(ROOT, "04_app")
sys.path.insert(0, APP_DIR)
sys.path.insert(0, os.path.join(ROOT, "01_pipeline", "phase_6_deployment"))

from app import app  # noqa: E402
from predict import predict_single  # noqa: E402

LOG_PATH = os.path.join(BASE, "run_latency_per_model.log")
OUT_PATH = os.path.join(BASE, "latency_per_model.json")

PAYLOAD = {"mag": 9.1, "depth": 30, "lat": 3.3, "lon": 95.8,
           "year": 2004, "month": 12}
ARGS = (9.1, 30.0, 3.3, 95.8, 50, 89.7, 2.6, 0.92, 2004, 12)
MODELS = ["logistic_regression", "random_forest", "xgboost", "mlp"]
N_WARMUP = 2
N_REPEAT = 20
N_ENSEMBLE = 10
N_ENDPOINT = 5


class Tee:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.fh.write(text)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()


def stats(samples):
    return {
        "n": len(samples),
        "mean_s": round(statistics.mean(samples), 4),
        "median_s": round(statistics.median(samples), 4),
        "min_s": round(min(samples), 4),
        "max_s": round(max(samples), 4),
        "samples_s": [round(s, 4) for s in samples],
    }


def main():
    print(f"Mulai: {datetime.now():%Y-%m-%d %H:%M:%S}")
    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "flask": __import__("flask").__version__,
            "case": "T01 Aceh 2004 (mag 9,1 depth 30 lat 3,3 lon 95,8)",
            "n_warmup": N_WARMUP,
            "n_repeat": N_REPEAT,
        },
        "per_model": {},
        "ensemble": {},
        "endpoint": {},
    }

    print("\n== Latensi per model (prediksi tunggal, model hangat) ==")
    for model_name in MODELS:
        for _ in range(N_WARMUP):
            predict_single(*ARGS, model_name=model_name)
        samples = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            predict_single(*ARGS, model_name=model_name)
            samples.append(time.perf_counter() - t0)
        results["per_model"][model_name] = stats(samples)
        s = results["per_model"][model_name]
        print(f"  {model_name:22s} rata-rata {s['mean_s']:.4f}s "
              f"(median {s['median_s']:.4f}s, {s['min_s']:.4f}-{s['max_s']:.4f}s)")

    print("\n== Latensi ensemble (empat model berurutan) ==")
    samples = []
    for _ in range(N_ENSEMBLE):
        t0 = time.perf_counter()
        for model_name in MODELS:
            predict_single(*ARGS, model_name=model_name)
        samples.append(time.perf_counter() - t0)
    results["ensemble"] = stats(samples)
    e = results["ensemble"]
    print(f"  ensemble  rata-rata {e['mean_s']:.4f}s "
          f"(median {e['median_s']:.4f}s, {e['min_s']:.4f}-{e['max_s']:.4f}s)")

    print("\n== Latensi endpoint /predict ==")
    client = app.test_client()
    t0 = time.perf_counter()
    resp = client.post("/predict", json=PAYLOAD)
    first_s = time.perf_counter() - t0
    print(f"  request pertama (termasuk pemuatan artefak): {first_s:.3f}s "
          f"status={resp.status_code}")
    samples = []
    for _ in range(N_ENDPOINT):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=PAYLOAD)
        samples.append(time.perf_counter() - t0)
    results["endpoint"] = {
        "first_request_s": round(first_s, 4),
        "warm": stats(samples),
    }
    w = results["endpoint"]["warm"]
    print(f"  {N_ENDPOINT} request hangat: rata-rata {w['mean_s']:.4f}s "
          f"({w['min_s']:.4f}-{w['max_s']:.4f}s)")

    results["summary"] = {
        "per_model_mean_s": {m: results["per_model"][m]["mean_s"] for m in MODELS},
        "fastest_model": min(MODELS, key=lambda m: results["per_model"][m]["mean_s"]),
        "slowest_model": max(MODELS, key=lambda m: results["per_model"][m]["mean_s"]),
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSelesai: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  {OUT_PATH}")
    print(f"  {LOG_PATH}")


if __name__ == "__main__":
    sys.stdout = Tee(LOG_PATH)
    sys.stderr = sys.stdout
    main()
