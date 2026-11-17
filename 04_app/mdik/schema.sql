-- ============================================================================
-- MDIK — Skema basis data pengelolaan data kebencanaan tsunami
-- ============================================================================
-- Prototipe akademik. Tabel pengguna menyimpan hash password (Werkzeug),
-- bukan password mentah. Jalankan `init_db()` sekali untuk membuat berkas
-- SQLite dan mengisi data awal dari NOAA.

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name     TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('admin', 'operator', 'analis', 'publik')),
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id         TEXT NOT NULL UNIQUE,
    year              INTEGER,
    month             INTEGER,
    day               INTEGER,
    country           TEXT,
    location          TEXT,
    latitude          REAL,
    longitude         REAL,
    max_height_m      REAL,
    eq_magnitude      REAL,
    cause_code        INTEGER,
    validity          INTEGER,
    deaths            INTEGER,
    missing           INTEGER,
    injuries          INTEGER,
    damage_musd       REAL,
    houses_destroyed  INTEGER,
    houses_damaged    INTEGER,
    risk_level        TEXT NOT NULL DEFAULT 'Tidak diketahui'
                      CHECK (risk_level IN ('Awas', 'Siaga', 'Waspada', 'Tidak diketahui')),
    status            TEXT NOT NULL DEFAULT 'perlu_review'
                      CHECK (status IN ('perlu_review', 'terverifikasi', 'ditolak')),
    verified_by       TEXT,
    verified_at       TEXT,
    notes             TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_year   ON events(year);
CREATE INDEX IF NOT EXISTS idx_events_country ON events(country);
CREATE INDEX IF NOT EXISTS idx_events_risk   ON events(risk_level);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);

CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL DEFAULT (datetime('now')),
    username  TEXT,
    action    TEXT NOT NULL,
    entity    TEXT NOT NULL,
    entity_id INTEGER,
    detail    TEXT
);

CREATE TABLE IF NOT EXISTS imports (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL DEFAULT (datetime('now')),
    username      TEXT,
    filename      TEXT NOT NULL,
    rows_total    INTEGER NOT NULL,
    rows_imported INTEGER NOT NULL,
    rows_skipped  INTEGER NOT NULL
);
