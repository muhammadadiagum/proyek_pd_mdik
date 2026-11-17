# Tahap 3: Persiapan Data (Data Preparation)

> Framework: CRISP-DM Phase 3
> Terakhir diperbarui: **2026-11-03** (bagian dataset modeling final ditambahkan)
> Script: `1_feature_engineering.py` (prepared) + `3_build_modeling_set.py` (modeling)
> Output: `prepared_dataset.csv`, lalu `modeling_dataset.csv` + `dropped_events.csv`

---

## 1. Ringkasan

| Properti | Nilai |
|----------|-------|
| Input | `merged_dataset.csv` (20 kolom, 310,369 baris) |
| Output | `prepared_dataset.csv` (26 kolom, 310,369 baris) |
| Fitur baru ditambahkan | 6 |
| Missing values diimputasi | ~592K values (median imputation) |

---

## 2. Penanganan Missing Values

### Strategi: Median Imputation

| Kolom | Missing Sebelum | Median | Keterangan |
|-------|----------------|--------|------------|
| nst | 61.2% | 50.0 | Jumlah stasiun |
| gap | 45.1% | 89.7 | Azimuthal gap |
| dmin | 68.7% | 2.608 | Jarak minimum stasiun |
| rms | 15.9% | 0.92 | Residual mean square |

**Total imputasi:** ~592,000 values

**Rationale:** Median dipilih karena robust terhadap outlier. Missing values pada fitur kualitas ini disebabkan oleh keterbatasan monitoring di era awal (1900-1950), sehingga median lebih representatif daripada mean.

---

## 3. Fitur Baru yang Dibuat

### 3.1 depth_mag_ratio
- **Rumus:** depth / magnitude
- **Makna:** Rasio kedalaman terhadap magnitudo
- **Insight:** Gempa dangkal + besar = lebih tsunamigenic
- **Range:** -0.69 hingga 154.67
- **Median:** 6.73

### 3.2 abs_latitude
- **Rumus:** |latitude|
- **Makna:** Absolut lintang (proxy jarak dari equator)
- **Insight:** Equatorial regions lebih aktif secara tektonik
- **Range:** 0.00 hingga 87.39
- **Median:** 21.46

### 3.3 tectonic_zone
- **Metode:** Klasifikasi manual berdasarkan lat/lon
- **Kategori:**
  - `pacific_ring_of_fire` (~78%) — Zona paling aktif
  - `other` (~11%)
  - `mediterranean_asian_belt` (~7%)
  - `mid_atlantic_ridge` (~5%)

### 3.4 monitoring_quality
- **Rumus:** Composite score (0-1) dari nst, gap, dmin, rms
- **Makna:** Kualitas monitoring gempa
- **Insight:** Score tinggi = monitoring baik
- **Range:** 0.20 hingga 0.98
- **Median:** 0.63

### 3.5 era
- **Metode:** Klasifikasi berdasarkan tahun
- **Kategori:**
  - `early_1900s` (1900-1929) — ~1%
  - `mid_1900s` (1930-1959) — ~2%
  - `late_1900s` (1960-1989) — ~23%
  - `early_2000s` (1990-2009) — ~33%
  - `modern` (2010-2026) — ~41%

### 3.6 season
- **Metode:** Kategorisasi bulan
- **Kategori:** djf, mam, jja, son
- **Distribusi:** relatif merata (~25% each)

---

## 4. Distribusi Fitur Baru

### Statistik Deskriptif (setelah imputasi)

| Fitur | Mean | Std | Min | 25% | Median | 75% | Max |
|-------|------|-----|-----|-----|--------|-----|-----|
| depth_mag_ratio | 14.16 | 23.84 | -0.69 | 2.22 | 6.73 | 12.41 | 154.67 |
| abs_latitude | 24.39 | 17.31 | 0.00 | 9.06 | 21.46 | 36.82 | 87.39 |
| monitoring_quality | 0.63 | 0.08 | 0.20 | 0.59 | 0.63 | 0.66 | 0.98 |

### Tectonic Zone Distribution (833 tsunami)

| Zone | Total | Tsunami | Ratio |
|------|-------|---------|-------|
| pacific_ring_of_fire | 240,889 | 699 | 1:344 |
| mediterranean_asian_belt | 22,304 | 96 | 1:232 |
| other | 32,933 | 30 | 1:1,097 |
| mid_atlantic_ridge | 14,243 | 8 | 1:1,780 |

**Insight:** Mediterranean-Asian Belt punya ratio tsunami lebih tinggi (1:232) dibanding Pacific Ring of Fire (1:344). Mid-Atlantic Ridge paling rendah.

### Era Distribution (tsunami, 833 total)

| Era | Tsunami | % |
|-----|---------|---|
| early_1900s | 71 | 8.5% |
| mid_1900s | 117 | 14.0% |
| late_1900s | 229 | 27.5% |
| early_2000s | 189 | 22.7% |
| modern | 227 | 27.3% |

### Season Distribution (tsunami, 833 total)

| Season | Tsunami | % |
|--------|---------|---|
| jja (Jun-Aug) | 228 | 27.4% |
| djf (Dec-Feb) | 204 | 24.5% |
| mam (Mar-May) | 201 | 24.1% |
| son (Sep-Nov) | 200 | 24.0% |

---

## 5. Validasi Fitur

### Missing Values Setelah Imputasi

| Kolom | Missing |
|-------|---------|
| depth_mag_ratio | 0 (0.0%) |
| abs_latitude | 0 (0.0%) |
| tectonic_zone | 0 (0.0%) |
| monitoring_quality | 0 (0.0%) |
| era | 0 (0.0%) |
| season | 0 (0.0%) |

**Semua fitur baru lengkap tanpa missing values.**

---

## 6. Daftar Fitur Final (26 Kolom)

### Fitur Asli (20)
1. time
2. year, month, day
3. latitude, longitude
4. mag, depth
5. nst, gap, dmin, rms
6. magType, place
7. label (target)
8. noaa_country, noaa_location
9. noaa_cause_code, noaa_water_height, noaa_intensity

### Fitur Baru (6)
21. depth_mag_ratio
22. abs_latitude
23. tectonic_zone (kategorikal)
24. monitoring_quality
25. era (kategorikal)
26. season (kategorikal)

---

## 7. Rekomendasi untuk Tahap Selanjutnya

1. **Encoding:** tectonic_zone, era, season, magType perlu di-one-hot encode
2. **Scaling:** numeric features perlu di-standardize untuk SVM dan MLP
3. **Feature Selection:** evaluasi fitur mana yang paling informatif
4. **Train-Test Split:** stratified split mempertahankan ratio imbalance — 1:371 pada
   `prepared_dataset.csv`, tetapi **1:406,6** pada `modeling_dataset.csv` yang dipakai modeling
5. **Imbalance Handling:** SMOTE, cost-sensitive hanya pada data TRAIN

---

## 8. Dataset Modeling Final (3 November 2026)

`prepared_dataset.csv` (310.369 baris, 833 positif) **bukan** dataset yang dipakai modeling.
Sebelum phase 4, dijalankan `3_build_modeling_set.py` yang membatasi positif ke gempa murni.

| Properti | Nilai |
|---|---|
| Input | `00_data/processed/prepared_dataset.csv` (310.369 × 26) |
| Aturan positif | `label = 1` **dan** `noaa_cause_code = 1` |
| Output | `00_data/processed/modeling_dataset.csv` (**310.297 × 26**) |
| Positif / negatif | **761 / 309.536** (rasio 1:406,6) |
| Positif dengan `noaa_water_height` | **610** (80,2%) |
| Dibuang | **72** baris → `00_data/processed/dropped_events.csv` |
| Manifest | `results/modeling_set_manifest.json` |
| md5 `modeling_dataset.csv` | `d0695e96d0e31e6efa77a02cf8eeda28` |

**Sebaran 72 baris yang dibuang:** cause 3 (gempa+longsor) = 55; cause 6 (vulkanik) = 6;
cause 4 (vulkanik+gempa) = 3; cause 9 (meteorologis) = 3; cause 7 = 2; cause 8 = 2;
NaN = 1.

**Alasan:** fitur gempa (magnitudo, kedalaman, kualitas monitoring) tidak dapat menjelaskan
tsunami vulkanik atau longsor, sehingga memasukkannya sebagai positif akan mengajari model
pola yang tidak tersedia saat inferensi. Baris tersebut **dibuang, bukan dilabeli 0**, agar
tidak mencemari kelas negatif. Biaya: 72/310.369 = 0,023% data.

**Distribusi kelas ordinal (subset 610):** Waspada (≤0,5 m) = 368; Siaga (0,5–3,0 m) = 158;
Awas (>3,0 m) = 84.

> Semua konsumen data (phase 4, phase 5a–5j, phase 6, app) sekarang membaca
> `modeling_dataset.csv`. `prepared_dataset.csv` disimpan hanya sebagai artefak antara.
