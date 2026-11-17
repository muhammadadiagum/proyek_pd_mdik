"""Akses basis data MDIK (SQLite) + seed data NOAA.

Modul ini sengaja hanya memakai pustaka standar (sqlite3, csv) supaya aplikasi
Flask tidak bergantung pada pandas. Data mentah tetap dibaca dari
`00_data/raw/tsunami/tsunamis-2026-08-19.csv` (sumber kanonik proyek).
"""

from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from werkzeug.security import generate_password_hash

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
INSTANCE_DIR = APP_DIR / "instance"
DB_PATH = Path(os.environ.get("MDIK_DB_PATH", INSTANCE_DIR / "mdik.sqlite3"))
NOAA_CSV = PROJECT_ROOT / "00_data" / "raw" / "tsunami" / "tsunamis-2026-08-19.csv"
SCHEMA_PATH = Path(__file__).resolve().with_name("schema.sql")

# Tiga akun demo untuk mencoba sistem. Kata sandi dapat diganti lewat variabel
# lingkungan (lihat .env.example); nilai di bawah adalah bawaan demonstrasi.
DEFAULT_USERS = [
    ("admin", os.environ.get("MDIK_ADMIN_PASSWORD", "admin123"),
     "Admin BPBD", "admin"),
    ("operator", os.environ.get("MDIK_OPERATOR_PASSWORD", "operator123"),
     "Operator Data", "operator"),
    ("analis", os.environ.get("MDIK_ANALIS_PASSWORD", "analis123"),
     "Analis Kebencanaan", "analis"),
]


def get_db() -> sqlite3.Connection:
    """Koneksi SQLite per-request; row dapat diakses seperti dict."""
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _as_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None"):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _as_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "nan", "NaN", "None"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def risk_from_height(height) -> str:
    """Kelas ordinal tinggi air mengikuti ambang yang dipakai laporan PD."""
    if height is None or height <= 0:
        return "Tidak diketahui"
    if height <= 0.5:
        return "Waspada"
    if height <= 3.0:
        return "Siaga"
    return "Awas"


def _seed_users(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
        return
    for username, password, full_name, role in DEFAULT_USERS:
        conn.execute(
            "INSERT INTO users (username, password_hash, full_name, role) "
            "VALUES (?, ?, ?, ?)",
            (username, generate_password_hash(password), full_name, role),
        )
    conn.execute(
        "INSERT INTO audit_log (username, action, entity, detail) VALUES (?, ?, ?, ?)",
        ("system", "seed_users", "users", "3 akun demo dibuat"),
    )


def _seed_events(conn: sqlite3.Connection) -> int:
    """Impor event NOAA bila tabel events masih kosong."""
    if conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]:
        return 0
    if not NOAA_CSV.exists():
        return 0

    inserted = 0
    with NOAA_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row_no, row in enumerate(reader, start=2):  # baris 1 = header
            year = _as_int(row.get("Year"))
            if year is None:
                continue  # baris metadata "Search Parameters"
            lat = _as_float(row.get("Latitude"))
            lon = _as_float(row.get("Longitude"))
            height = _as_float(row.get("Maximum Water Height (m)"))
            risk = risk_from_height(height)
            complete = height is not None and lat is not None and lon is not None
            status = "terverifikasi" if complete else "perlu_review"
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """
                INSERT OR IGNORE INTO events (
                    source_id, year, month, day, country, location,
                    latitude, longitude, max_height_m, eq_magnitude,
                    cause_code, validity, deaths, missing, injuries,
                    damage_musd, houses_destroyed, houses_damaged,
                    risk_level, status, verified_by, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"noaa-{row_no:05d}",
                    year,
                    _as_int(row.get("Mo")),
                    _as_int(row.get("Dy")),
                    (row.get("Country") or "Tidak diketahui").strip(),
                    (row.get("Location Name") or "").strip() or None,
                    lat,
                    lon,
                    height,
                    _as_float(row.get("Earthquake Magnitude")),
                    _as_int(row.get("Tsunami Cause Code")),
                    _as_int(row.get("Tsunami Event Validity")),
                    _as_int(row.get("Deaths")),
                    _as_int(row.get("Missing")),
                    _as_int(row.get("Injuries")),
                    _as_float(row.get("Damage ($Mil)")),
                    _as_int(row.get("Houses Destroyed")),
                    _as_int(row.get("Houses Damaged")),
                    risk,
                    status,
                    "seed-noaa" if complete else None,
                    now if complete else None,
                ),
            )
            inserted += 1

    conn.execute(
        "INSERT INTO audit_log (username, action, entity, detail) VALUES (?, ?, ?, ?)",
        ("system", "seed_events", "events", f"{inserted} event NOAA diimpor"),
    )
    return inserted


def init_db(seed: bool = True) -> dict:
    """Buat skema + seed idempoten. Aman dipanggil setiap startup."""
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        _seed_users(conn)
        inserted = _seed_events(conn) if seed else 0
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {"inserted": inserted, "total_events": total, "db_path": str(DB_PATH)}
    finally:
        conn.close()


def log_action(username, action, entity, entity_id=None, detail=None) -> None:
    """Tulis jejak audit; dipanggil dari route yang mengubah data."""
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO audit_log (username, action, entity, entity_id, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, action, entity, entity_id, detail),
        )
        conn.commit()
    finally:
        conn.close()


def row_to_dict(row):
    return dict(row) if row is not None else None


def make_source_id(prefix: str, payload: str) -> str:
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"
