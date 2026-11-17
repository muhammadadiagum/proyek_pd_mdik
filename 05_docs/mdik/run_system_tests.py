#!/usr/bin/env python3
"""Uji fungsional modul MDIK dengan Flask test client.

Uji berjalan pada SALINAN basis data di direktori sementara, sehingga
`04_app/instance/mdik.sqlite3` tidak berubah. Hasil ditulis ke:
- 05_docs/mdik/testing_results.json
- 05_docs/mdik/run_system_tests.log (via shell redirect)

Jalankan dari root proyek:
    .venv/bin/python 05_docs/mdik/run_system_tests.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "04_app"
DOCS = ROOT / "05_docs" / "mdik"
DB_SRC = APP_DIR / "instance" / "mdik.sqlite3"
OUT_JSON = DOCS / "testing_results.json"

sys.path.insert(0, str(APP_DIR))

tmpdir = Path(tempfile.mkdtemp(prefix="mdik_test_"))
db_copy = tmpdir / "mdik.sqlite3"
shutil.copy2(DB_SRC, db_copy)
os.environ["MDIK_DB_PATH"] = str(db_copy)
os.environ["MDIK_SECRET_KEY"] = "test-only-secret"

from app import app  # noqa: E402  (harus setelah env diatur)

app.config.update(TESTING=True)
client = app.test_client()


def nf(n):
    """Format bilangan dengan pemisah ribuan gaya Indonesia (titik)."""
    return f"{n:,}".replace(",", ".")


def num(x):
    """Format pecahan empat angka penting gaya Indonesia; '-' bila kosong."""
    try:
        return f"{float(x):.4g}".replace(".", ",")
    except (TypeError, ValueError):
        return "-"


def db_one(sql, params=()):
    conn = sqlite3.connect(db_copy)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchone()
    finally:
        conn.close()


def db_all(sql, params=()):
    conn = sqlite3.connect(db_copy)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def add_case(case_id, scenario, method, path, expected, actual, status, passed,
             **extra):
    case = {
        "id": case_id,
        "scenario": scenario,
        "method": method,
        "path": path,
        "expected": expected,
        "actual": actual,
        "http_status": status,
        "pass": bool(passed),
    }
    case.update(extra)
    cases.append(case)
    print(f"[{'PASS' if passed else 'FAIL'}] {case_id} {scenario} -> {actual}")


def text_of(response):
    return response.get_data(as_text=True)


def login(username, password):
    return client.post("/mdik/login", data={"username": username, "password": password})


def logout():
    return client.post("/mdik/logout")


cases: list[dict] = []
started = time.perf_counter()

# --------------------------------------------------------------------------- #
# TC-01 dashboard publik
# --------------------------------------------------------------------------- #
total = db_one("SELECT COUNT(*) n FROM events")["n"]
rv = client.get("/mdik/")
body = text_of(rv)
add_case("TC-01", "GET dashboard tanpa login", "GET", "/mdik/",
         "HTTP 200 dan total event tampil", f"HTTP {rv.status_code}; total {nf(total)} event",
         rv.status_code, rv.status_code == 200 and nf(total) in body,
         total=total)

# --------------------------------------------------------------------------- #
# TC-02 filter dekade
# --------------------------------------------------------------------------- #
n_decade = db_one("SELECT COUNT(*) n FROM events WHERE year BETWEEN 2000 AND 2009")["n"]
rv = client.get("/mdik/events?decade=2000")
body = text_of(rv)
add_case("TC-02", "Filter dekade 2000", "GET", "/mdik/events?decade=2000",
         "HTTP 200 dan jumlah terfilter benar",
         f"HTTP {rv.status_code}; total {n_decade} event",
         rv.status_code, rv.status_code == 200 and f"{n_decade}" in body,
         total=n_decade)

# --------------------------------------------------------------------------- #
# TC-03 filter dekade + risiko
# --------------------------------------------------------------------------- #
n_decade_risk = db_one(
    "SELECT COUNT(*) n FROM events WHERE year BETWEEN 2000 AND 2009 AND risk_level='Awas'"
)["n"]
rv = client.get("/mdik/events?decade=2000&risk=Awas")
body = text_of(rv)
add_case("TC-03", "Filter dekade 2000 + risiko Awas", "GET",
         "/mdik/events?decade=2000&risk=Awas",
         "HTTP 200 dan jumlah terfilter benar",
         f"HTTP {rv.status_code}; total {n_decade_risk} event",
         rv.status_code, rv.status_code == 200 and f"{n_decade_risk}" in body,
         total=n_decade_risk)

# --------------------------------------------------------------------------- #
# TC-04 GeoJSON publik
# --------------------------------------------------------------------------- #
n_verified = db_one("SELECT COUNT(*) n FROM events WHERE status='terverifikasi'")["n"]
rv = client.get("/mdik/api/events.geojson")
payload = rv.get_json()
features = len(payload.get("features", [])) if payload else 0
add_case("TC-04", "GET GeoJSON publik", "GET", "/mdik/api/events.geojson",
         "HTTP 200; hanya event terverifikasi",
         f"HTTP {rv.status_code}; {nf(features)} fitur", rv.status_code,
         rv.status_code == 200 and features == n_verified,
         features=features, expected_features=n_verified)

# --------------------------------------------------------------------------- #
# TC-05 ekspor CSV
# --------------------------------------------------------------------------- #
rv = client.get("/mdik/export.csv")
rows = list(csv.reader(io.StringIO(text_of(rv))))
data_rows = len(rows) - 1
add_case("TC-05", "GET ekspor CSV", "GET", "/mdik/export.csv",
         "HTTP 200; header + seluruh baris",
         f"HTTP {rv.status_code}; {nf(data_rows)} baris data + header",
         rv.status_code, rv.status_code == 200 and data_rows == total,
         rows=data_rows, expected_rows=total)

# --------------------------------------------------------------------------- #
# TC-06 proteksi halaman tulis
# --------------------------------------------------------------------------- #
rv = client.get("/mdik/events/new")
add_case("TC-06", "GET form tambah tanpa login", "GET", "/mdik/events/new",
         "Dialihkan ke login", f"HTTP {rv.status_code} ke {rv.headers.get('Location', '-')}",
         rv.status_code, rv.status_code == 302 and "/mdik/login" in rv.headers.get("Location", ""))

# --------------------------------------------------------------------------- #
# TC-07 login admin
# --------------------------------------------------------------------------- #
rv = login("admin", "admin123")
add_case("TC-07", "Login admin", "POST", "/mdik/login",
         "HTTP 302 ke dashboard", f"HTTP {rv.status_code} ke {rv.headers.get('Location', '-')}",
         rv.status_code, rv.status_code == 302 and rv.headers.get("Location", "").endswith("/mdik/"))

# --------------------------------------------------------------------------- #
# TC-08 tambah event
# --------------------------------------------------------------------------- #
rv = client.post("/mdik/events/new", data={
    "year": "2026", "month": "11", "day": "4",
    "country": "Testlandia", "location": "Lab Uji",
    "latitude": "-6.2", "longitude": "106.8",
    "max_height_m": "1.2", "eq_magnitude": "7.0",
    "deaths": "0", "missing": "0", "injuries": "0",
})
new_event = db_one(
    "SELECT * FROM events WHERE country='Testlandia' ORDER BY id DESC LIMIT 1"
)
add_case("TC-08", "Tambah event valid", "POST", "/mdik/events/new",
         "HTTP 302; status perlu review; kelas Siaga",
         f"HTTP {rv.status_code}; ID {new_event['id'] if new_event else '-'}; "
         f"status {new_event['status'] if new_event else '-'}; "
         f"kelas {new_event['risk_level'] if new_event else '-'}",
         rv.status_code,
         rv.status_code == 302 and new_event is not None
         and new_event["status"] == "perlu_review" and new_event["risk_level"] == "Siaga",
         event_id=new_event["id"] if new_event else None)

event_id = new_event["id"] if new_event else None

# --------------------------------------------------------------------------- #
# TC-09 verifikasi admin
# --------------------------------------------------------------------------- #
rv = client.post(f"/mdik/events/{event_id}/verify", data={"action": "verify"})
verified = db_one("SELECT * FROM events WHERE id=?", (event_id,))
add_case("TC-09", "Verifikasi event oleh admin", "POST",
         f"/mdik/events/{event_id}/verify",
         "HTTP 302; status terverifikasi; verifikator admin",
         f"HTTP {rv.status_code}; status {verified['status']}; "
         f"verified_by {verified['verified_by']}",
         rv.status_code,
         rv.status_code == 302 and verified["status"] == "terverifikasi"
         and verified["verified_by"] == "admin",
         event_id=event_id)

# --------------------------------------------------------------------------- #
# TC-10 otorisasi operator
# --------------------------------------------------------------------------- #
rv = client.post("/mdik/events/new", data={
    "year": "2026", "country": "Testlandia Dua", "latitude": "-6.3",
    "longitude": "106.9", "max_height_m": "4.5", "deaths": "0",
})
second = db_one(
    "SELECT * FROM events WHERE country='Testlandia Dua' ORDER BY id DESC LIMIT 1"
)
second_id = second["id"] if second else None
logout()
login("operator", "operator123")
rv = client.post(f"/mdik/events/{second_id}/verify", data={"action": "verify"})
second_after = db_one("SELECT * FROM events WHERE id=?", (second_id,))
add_case("TC-10", "Verifikasi oleh operator", "POST",
         f"/mdik/events/{second_id}/verify",
         "Akses ditolak; status tidak berubah",
         f"HTTP {rv.status_code} ke {rv.headers.get('Location', '-')}; "
         f"status tetap {second_after['status']}",
         rv.status_code,
         rv.status_code == 302 and second_after["status"] == "perlu_review",
         event_id=second_id)

# --------------------------------------------------------------------------- #
# TC-11 hapus event oleh admin
# --------------------------------------------------------------------------- #
logout()
login("admin", "admin123")
rv = client.post(f"/mdik/events/{event_id}/delete")
remaining = db_one("SELECT COUNT(*) n FROM events WHERE id=?", (event_id,))["n"]
add_case("TC-11", "Hapus event oleh admin", "POST", f"/mdik/events/{event_id}/delete",
         "HTTP 302; event terhapus", f"HTTP {rv.status_code}; baris tersisa {remaining}",
         rv.status_code, rv.status_code == 302 and remaining == 0, event_id=event_id)

# --------------------------------------------------------------------------- #
# TC-12 prediksi Aceh 2004
# --------------------------------------------------------------------------- #
rv = client.post("/predict", json={
    "mag": 9.1, "depth": 30.0, "lat": 3.3, "lon": 95.8,
    "nst": 50, "gap": 89.7, "dmin": 2.6, "rms": 0.92,
    "year": 2004, "month": 12, "model": "ensemble",
})
j = rv.get_json() or {}
add_case("TC-12", "POST /predict preset Aceh 2004", "POST", "/predict",
         "TSUNAMI; risiko HIGH; kelas Awas",
         f"HTTP {rv.status_code}; p={num(j.get('probability'))}; "
         f"ambang={num(j.get('threshold'))}; {j.get('prediction')}; "
         f"{j.get('risk_label')}; {j.get('warning', {}).get('label')}",
         rv.status_code,
         rv.status_code == 200 and j.get("prediction") == "TSUNAMI"
         and j.get("risk_label") == "HIGH"
         and j.get("warning", {}).get("label") == "Awas"
         and abs(float(j.get("probability", 0)) - 0.8072) < 0.002,
         probability=j.get("probability"), threshold=j.get("threshold"),
         prediction=j.get("prediction"), warning=j.get("warning", {}).get("label"))

# --------------------------------------------------------------------------- #
# TC-13 prediksi gempa kecil
# --------------------------------------------------------------------------- #
rv = client.post("/predict", json={
    "mag": 4.5, "depth": 35.0, "lat": -5.0, "lon": 110.0,
    "nst": 50, "gap": 89.7, "dmin": 2.6, "rms": 0.92,
    "year": 2026, "month": 6, "model": "ensemble",
})
j = rv.get_json() or {}
add_case("TC-13", "POST /predict gempa kecil M4,5", "POST", "/predict",
         "NO TSUNAMI; kelas tidak berlaku",
         f"HTTP {rv.status_code}; p={num(j.get('probability'))}; "
         f"{j.get('prediction')}; {j.get('warning', {}).get('label')}",
         rv.status_code,
         rv.status_code == 200 and j.get("prediction") == "NO TSUNAMI"
         and j.get("warning", {}).get("label") == "Tidak berlaku"
         and abs(float(j.get("probability", 0)) - 0.00377) < 0.002,
         probability=j.get("probability"), threshold=j.get("threshold"),
         prediction=j.get("prediction"), warning=j.get("warning", {}).get("label"))

# --------------------------------------------------------------------------- #
# Bersihkan event uji kedua
# --------------------------------------------------------------------------- #
if second_id:
    client.post(f"/mdik/events/{second_id}/delete")
logout()

elapsed = time.perf_counter() - started
passed = sum(1 for case in cases if case["pass"])
result = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "environment": {
        "python": sys.version.split()[0],
        "flask": version("flask"),
        "database_copy": "temporary copy (removed at end)",
    },
    "summary": {
        "total": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "elapsed_s": round(elapsed, 3),
    },
    "cases": cases,
}
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
print(f"\n{passed}/{len(cases)} lulus; hasil: {OUT_JSON.relative_to(ROOT)}")
print(f"durasi: {elapsed:.2f} s")

shutil.rmtree(tmpdir, ignore_errors=True)
sys.exit(0 if passed == len(cases) else 1)
