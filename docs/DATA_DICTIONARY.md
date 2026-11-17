# Data Dictionary — Proyek Tsunami

Semua berkas di bawah ini adalah data mentah/turunan yang dipakai pipeline.
Data mentah **tidak diedit**; kolom turunan dihasilkan oleh
`01_pipeline/phase_3_data_preparation/1_feature_engineering.py`.

---

## 1. Berkas

| Berkas | Baris × kolom | Ukuran | md5 | Sumber |
|---|---|---|---|---|
| `raw/tsunami/tsunamis-2026-08-19_21-29-48_+0800.tsv` | 3.128 × 45 | 311 KB | — | NOAA/NCEI WDS Global Historical Tsunami Database |
| `raw/tsunami/tsunamis-2026-08-19.csv` | 3.130 × 45 | 303 KB | `b98a0dba8bdb1b078d68cc808b3955ec` | konversi TSV→CSV (skrip `convert_tsv_to_csv.py`) |
| `raw/earthquake/merged_usgs_earthquakes.csv` | 311.104 × 22 | 51 MB | `8f99bc07cdcac5795bec4b0350ef49cc` | USGS ANSS Comprehensive Earthquake Catalog (ComCat) |
| `interim/merged_dataset.csv` | 310.369 × 20 | 35 MB | `2290375133d2321984b051213a9fe91b` | hasil pencocokan NOAA × USGS (phase 2) |
| `processed/prepared_dataset.csv` | 310.369 × 26 | 53 MB | `fcaf2c8c0ac49e8ca3b24c7de28386cf` | hasil feature engineering (phase 3), 833 positif |
| `processed/modeling_dataset.csv` | 310.297 × 26 | 53 MB | `d0695e96d0e31e6efa77a02cf8eeda28` | dataset modeling final: 761 positif `cause=1` |
| `processed/dropped_events.csv` | 72 × 12 | 6,6 KB | — | positif `cause ≠ 1` yang dikeluarkan (audit) |

**Aturan label (final, 3 Nov 2026).** Positif = `label = 1` **dan**
`noaa_cause_code = 1` (tsunami akibat gempa murni). Positif dengan cause lain
(3 = gempa + longsor, 4/6/7 vulkanik, 8 longsor, 9 meteorologis, NaN) **dibuang**,
tidak dilabeli 0. Negatif tetap semua `label = 0`. Hasilnya
`modeling_dataset.csv`: 310.297 baris (761 positif, 309.536 negatif).
`modeling_dataset.csv` adalah **satu-satunya input** untuk phase 4–6 dan app;
`prepared_dataset.csv` dipertahankan sebagai dataset gabungan penuh (833 positif).
Alasan: `.audit/00_RINGKASAN.md` §6.

**Sitasi NOAA (wajib):**
National Geophysical Data Center / World Data Service (NGDC/WDS): NCEI/WDS Global
Historical Tsunami Database. NOAA National Centers for Environmental Information.
doi:10.7289/V5PN93H7

**Sitasi USGS:**
U.S. Geological Survey, Earthquake Hazards Program, ANSS Comprehensive Earthquake
Catalog (ComCat), https://earthquake.usgs.gov/fdsnws/event/1/

Keduanya data publik pemerintah AS; NOAA meminta sitasi DOI di atas, USGS tidak
memiliki batasan lisensi untuk data katalog.

---

## 2. Kolom `prepared_dataset.csv` (26 kolom)

| Kolom | Tipe | Asal | Deskripsi |
|---|---|---|---|
| `time` | string | USGS | Waktu gempa (UTC, ISO-8601) |
| `year` | int | USGS | Tahun gempa |
| `month` | int | USGS | Bulan (1–12) |
| `day` | int | USGS | Tanggal (1–31) |
| `latitude` | float | USGS | Lintang episentrum (derajat) |
| `longitude` | float | USGS | Bujur episentrum (derajat) |
| `mag` | float | USGS | Magnitudo |
| `depth` | float | USGS | Kedalaman (km) |
| `nst` | float | USGS | Jumlah stasiun yang mencatat |
| `gap` | float | USGS | Azimuthal gap (derajat) |
| `dmin` | float | USGS | Jarak stasiun terdekat (derajat) |
| `rms` | float | USGS | Residual RMS waktu tiba (detik) |
| `magType` | string | USGS | Jenis magnitudo (ml, mb, mw, …) |
| `place` | string | USGS | Deskripsi lokasi dari USGS |
| `label` | int | turunan | **1** = gempa cocok dengan tsunami NOAA, **0** = tidak |
| `noaa_country` | string | NOAA | Negara terdampak — **hanya terisi saat label=1** |
| `noaa_location` | string | NOAA | Nama lokasi — **hanya terisi saat label=1** |
| `noaa_cause_code` | float | NOAA | Skema NCEI: 1 = gempa, 2 = kemungkinan gempa, 3 = gempa + longsor, 4 = vulkanik + gempa, 5 = erupsi vulkanik, 6 = vulkanik, 7 = vulkanik + longsor, 8 = longsor, 9 = meteorologis, 10 = ledakan |
| `noaa_water_height` | float | NOAA | Tinggi air maksimum (meter) — **hanya terisi saat label=1** |
| `noaa_intensity` | float | NOAA | Intensitas tsunami skala NOAA (1–6) — **hanya terisi saat label=1** |
| `depth_mag_ratio` | float | turunan | `depth / mag` — gempa dangkal-besar lebih tsunamigenik |
| `abs_latitude` | float | turunan | `abs(latitude)` — proxy lintang absolut |
| `tectonic_zone` | string | turunan | `pacific_ring_of_fire`, `mediterranean_asian_belt`, `mid_atlantic_ridge`, `other` |
| `monitoring_quality` | float | turunan | Skor 0–1 dari `nst`, `gap`, `dmin`, `rms` (semakin tinggi semakin baik) |
| `era` | string | turunan | `early_1900s` (<1930), `mid_1900s` (<1960), `late_1900s` (<1990), `early_2000s` (<2010), `modern` (≥2010) |
| `season` | string | turunan | `djf`, `mam`, `jja`, `son` dari bulan |

### Aturan turunan (dari `1_feature_engineering.py`)

- `depth_mag_ratio = depth / mag` bila `mag > 0`, selain itu kosong.
- `abs_latitude = |latitude|`.
- `monitoring_quality` = rata-rata skor yang tersedia:
  `nst/100` (dibatasi 1), `1 − gap/360`, `1 − dmin/10`, `1 − rms/2` (masing-masing dibatasi 0–1).
- `era` dan `season` adalah binning, bukan hasil model.
- Missing `nst`, `gap`, `dmin`, `rms` diisi median kolom.

---

## 3. Peringatan kebocoran (penting)

Lima kolom `noaa_*` **hanya terisi pada baris label = 1**. Memakainya sebagai fitur
akan membuat model "melihat jawaban" dan menghasilkan akurasi palsu. Kolom tersebut
hanya boleh dipakai sebagai target/analisis, bukan prediktor. Temuan ini tercatat
sebagai **F-01** di `.audit/04_TEMUAN_TEKNIS.md`; proses RapidMiner lama masih
memakainya.

## 4. Fitur model (24 kolom)

11 numerik:
`mag`, `depth`, `nst`, `gap`, `dmin`, `rms`, `latitude`, `longitude`, `abs_latitude`,
`depth_mag_ratio`, `monitoring_quality`

13 dummy (one-hot, `drop_first=False`):
`tectonic_zone` (4) + `era` (5) + `season` (4)

Kolom yang **tidak** dipakai: `time`, `year`, `month`, `day`, `magType`, `place`,
`label`, dan seluruh `noaa_*`.

## 5. Target

- **Klasifikasi:** `label` (1 = tsunami, 0 = bukan). Dataset final `modeling_dataset.csv`
  memakai **761 positif** dari 310.297 baris (0,25%), yaitu `label=1` **dan**
  `noaa_cause_code=1`. Sensitivitas `cause 1–3` (816 positif) tidak dipakai sebagai
  definisi utama; 72 baris cause lain dibuang, bukan dilabeli negatif. Dataset gabungan
  lama (833 positif / 310.369 baris) tetap tersedia di `prepared_dataset.csv`.
- **Estimasi ordinal:** `noaa_water_height` pada `label=1` & `cause_code=1` (n=610),
  dipetakan ke kelas Waspada (≤0,5 m), Siaga (0,5–3 m), Awas (>3 m).
