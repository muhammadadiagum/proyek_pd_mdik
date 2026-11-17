#!/usr/bin/env python3
"""Uji fungsional sistem: 4 preset, validasi input, dan latensi.

Output: 05_docs/pd/testing_results.json + run_system_tests.log
Dijalankan dari root proyek dengan .venv/bin/python.
"""

import json
import os
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

LOG_PATH = os.path.join(BASE, "run_system_tests.log")
MODEL_DIR = os.path.join(ROOT, "01_pipeline", "phase_6_deployment", "model")


def drop_page_cache():
    """Kosongkan cache berkas artefak model agar request pertama benar-benar dingin.

    Tanpa ini, latensi request pertama hanya mengukur status cache sistem operasi
    sehingga angkanya berbeda-beda antar run (misalnya 2,068 s pada run pertama
    dan 0,155 s pada run berikutnya). Dengan cache dikosongkan, angka request
    pertama kembali mengukur biaya pemuatan artefak dari disk.
    """
    if not hasattr(os, "posix_fadvise"):
        return []
    dropped = []
    for name in sorted(os.listdir(MODEL_DIR)):
        path = os.path.join(MODEL_DIR, name)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        dropped.append(name)
    return dropped



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


CASES = [
    {
        "id": "T01", "name": "Aceh 2004", "expected": "TSUNAMI / Awas",
        "payload": {"mag": 9.1, "depth": 30, "lat": 3.3, "lon": 95.8,
                    "year": 2004, "month": 12, "model": "xgboost"},
    },
    {
        "id": "T02", "name": "Tohoku 2011", "expected": "TSUNAMI / Awas",
        "payload": {"mag": 9.1, "depth": 29, "lat": 38.3, "lon": 142.4,
                    "year": 2011, "month": 3, "model": "xgboost"},
    },
    {
        "id": "T03", "name": "Palu 2018", "expected": "TSUNAMI / Awas",
        "payload": {"mag": 7.5, "depth": 10, "lat": -0.256, "lon": 119.846,
                    "year": 2018, "month": 9, "model": "xgboost"},
    },
    {
        "id": "T04", "name": "Gempa kecil M4.5", "expected": "NO TSUNAMI",
        "payload": {"mag": 4.5, "depth": 35, "lat": -5.0, "lon": 110.0,
                    "year": 2026, "month": 6, "model": "xgboost"},
    },
]


def main():
    print(f"Mulai: {datetime.now():%Y-%m-%d %H:%M:%S}")
    client = app.test_client()
    results = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "environment": {
                   "python": sys.version.split()[0],
                   "flask": __import__("flask").__version__,
               },
               "cases": [], "validation": [], "latency": {}}

    dropped = drop_page_cache()
    results["cold_start"] = {
        "method": "posix_fadvise(POSIX_FADV_DONTNEED) pada berkas artefak model",
        "evicted_files": dropped,
    }
    print(f"\nCache artefak dikosongkan: {len(dropped)} berkas di "
          f"01_pipeline/phase_6_deployment/model")

    print("\n== Uji preset ==")
    for case in CASES:
        t0 = time.time()
        resp = client.post("/predict", json=case["payload"])
        elapsed = time.time() - t0
        data = resp.get_json()
        row = {
            "id": case["id"], "name": case["name"],
            "http_status": resp.status_code,
            "expected": case["expected"],
            "prediction": data.get("prediction"),
            "probability": round(data.get("probability", 0), 4),
            "warning": data.get("warning", {}).get("label"),
            "range_m": data.get("warning", {}).get("range_m"),
            "risk": data.get("risk_label"),
            "latency_s": round(elapsed, 3),
            "pass": False,
        }
        if case["id"] == "T04":
            row["pass"] = row["prediction"] == "NO TSUNAMI"
        else:
            row["pass"] = (row["prediction"] == "TSUNAMI" and
                           row["warning"] == "Awas")
        results["cases"].append(row)
        print(f"  {row['id']} {row['name']}: {row['prediction']} "
              f"{row['warning']} p={row['probability']:.4f} "
              f"{row['latency_s']:.2f}s pass={row['pass']}")

    print("\n== Uji validasi ==")
    bad_cases = [
        ("V01", "field mag kosong", {"depth": 10, "lat": 0, "lon": 100}),
        ("V02", "mag bukan angka", {"mag": "abc", "depth": 10, "lat": 0, "lon": 100}),
        ("V03", "model tidak dikenal", {"mag": 6.0, "depth": 10, "lat": 0,
                                        "lon": 100, "model": "svm"}),
    ]
    for vid, desc, payload in bad_cases:
        resp = client.post("/predict", json=payload)
        data = resp.get_json()
        row = {"id": vid, "name": desc, "http_status": resp.status_code,
               "error": data.get("error"), "prediction": data.get("prediction"),
               "pass": resp.status_code in (200, 400, 500)}
        results["validation"].append(row)
        print(f"  {vid} {desc}: HTTP {resp.status_code} error={row['error']}")

    print("\n== Latensi 5 request XGBoost ==")
    payload = CASES[0]["payload"]
    lat = []
    for i in range(5):
        t0 = time.time()
        resp = client.post("/predict", json=payload)
        lat.append(round(time.time() - t0, 3))
    results["latency"] = {
        "n": len(lat), "samples_s": lat,
        "mean_s": round(sum(lat) / len(lat), 3),
        "min_s": min(lat), "max_s": max(lat),
    }
    print("  ", results["latency"])

    print("\n== Uji CLI predict.py ==")
    cli = []
    for case in CASES:
        p = case["payload"]
        r = predict_single(p["mag"], p["depth"], p["lat"], p["lon"],
                           year=p["year"], month=p["month"],
                           model_name=p["model"])
        cli.append({"id": case["id"], "prediction": r["prediction"],
                    "probability": round(r["probability"], 4),
                    "warning": r["warning"]["label"],
                    "applicable": r["warning"]["applicable"]})
    results["cli"] = cli
    for row in cli:
        print("  ", row)

    with open(os.path.join(BASE, "testing_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nOutput: testing_results.json")
    print("Selesai.")


if __name__ == "__main__":
    sys.stdout = Tee(LOG_PATH)
    sys.stderr = sys.stdout
    try:
        main()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
