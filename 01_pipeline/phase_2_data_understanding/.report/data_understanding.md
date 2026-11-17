# Tahap 2: Pemahaman Data (Data Understanding)

> Framework: CRISP-DM Phase 2
> Dataset: NOAA Global Historical Tsunami Database + USGS ANSS Earthquake Catalog
> Terakhir diperbarui: **2026-11-03** (catatan dataset final ditambahkan)

> **Catatan pembaruan 3 Nov 2026.** Dokumen ini mendeskripsikan dataset **gabungan**
> (`merged_dataset.csv`, 310.369 baris, 833 positif semua penyebab). Itu tetap benar sebagai
> input phase 2–3. Dataset yang dipakai modeling sekarang lebih kecil:
> **310.297 baris, 761 positif** (positif = `label=1` & `noaa_cause_code=1`; 72 baris cause
> lain dibuang), rasio **1:406,6**. Rinciannya di
> `phase_3_data_preparation/.report/data_preparation.md` §8 dan
> `00_data/processed/modeling_dataset.csv`.

---

## 1. Ringkasan Dataset Gabungan

| Properti | Nilai |
|----------|-------|
| Total baris | 310,369 |
| Total kolom | 20 |
| Positive (tsunami) | 833 (0.27%) |
| Negative (non-tsunami) | 309,536 (99.73%) |
| Ratio imbalance | 1:371 |
| Periode waktu | 1900 – 2026 |

## 2. Sumber Data

### NOAA Global Historical Tsunami Database
- **DOI:** 10.7289/V5PN93H7
- **Jumlah event:** 3,129 (dari -2000 BC s/d 2026)
- **Event yang ter-match:** 2,642 (tahun ≥1900)
- **Match ke USGS:** 833 event (32%, tolerance ±0.25°)

### USGS ANSS Comprehensive Earthquake Catalog
- **Sumber:** earthquake.usgs.gov
- **Threshold:** M4.5+
- **Jumlah event (raw CSV):** 311,104 (1900–2026); **valid setelah filter (time/koordinat):** 310,369
- **Event di merged dataset:** 310,369 (beberapa ter-filter)

### Proses Merge
```
NOAA tsunami DB ──┐
                   ├── JOIN (tolerance: ±1 hari, ±0.25 derajat) ──→ merged_dataset.csv
USGS earthquake ──┘
```

## 3. Deskripsi Kolom

### Fitur Utama (dari USGS)

| Kolom | Tipe | Deskripsi | Missing (Total) | Missing (Tsunami) |
|-------|------|-----------|-----------------|-------------------|
| mag | float | Magnitudo gempa (Mw) | 0% | 0% |
| depth | float | Kedalaman hypocenter (km) | 0% | 0% |
| nst | int | Jumlah stasiun yang merekam | 61.2% | terimputasi |
| gap | float | Azimuthal gap (derajat) | 45.1% | terimputasi |
| dmin | float | Jarak minimum ke stasiun (derajat) | 68.7% | terimputasi |
| rms | float | Residual mean square (detik) | 15.9% | terimputasi |

### Fitur Geospasial

| Kolom | Tipe | Deskripsi | Missing (Total) | Missing (Tsunami) |
|-------|------|-----------|-----------------|-------------------|
| latitude | float | Lintang epicenter | 0% | 0% |
| longitude | float | Bujur epicenter | 0% | 0% |
| time | string | Waktu kejadian (UTC) | 0% | 0% |
| year | int | Tahun | 0% | 0% |
| month | int | Bulan | 0% | 0% |
| day | int | Hari | 0% | 0% |
| place | string | Deskripsi lokasi | 0% | 0% |
| magType | string | Tipe magnitudo | 0% | 0% |

### Fitur NOAA (hanya untuk event yang match)

| Kolom | Tipe | Deskripsi | Missing (Total) | Missing (Tsunami) |
|-------|------|-----------|-----------------|-------------------|
| label | int | Target: 0=no tsunami, 1=tsunami | 0% | 0% |
| noaa_country | string | Negara | 99.7% | **0%** |
| noaa_location | string | Lokasi deskriptif | 99.7% | **0%** |
| noaa_cause_code | float | Kode penyebab tsunami | 99.7% | **0.1%** |
| noaa_water_height | float | Ketinggian air maksimum (m) | 99.7% | **19.4%** |
| noaa_intensity | float | Intensitas tsunami | 99.7% | **68.7%** |

> **Catatan Penting:** Angka "99.7% missing" pada kolom NOAA dihitung dari
> seluruh 310K baris, termasuk 309K non-tsunami yang memang tidak punya data NOAA.
> Untuk **subset 833 event tsunami saja**, coverage jauh lebih tinggi —
> `noaa_country` dan `noaa_location` terisi 100%, `noaa_water_height` 80,6%,
> `noaa_intensity` 31,3% (lihat §5).

## 4. Statistik Deskriptif

### 4a. Distribusi Magnitudo

| Statistik | Tsunami (label=1) | Non-Tsunami (label=0) |
|-----------|-------------------|----------------------|
| Count | 833 | 309,536 |
| Mean | 6.90 | 4.93 |
| Std | 0.83 | 0.47 |
| Min | 4.50 | 3.38 |
| 25% | 6.40 | 4.60 |
| Median | 7.00 | 4.80 |
| 75% | 7.48 | 5.10 |
| Max | 9.50 | 8.80 |

**Insight:** Gempa yang memicu tsunami cenderung memiliki magnitudo jauh lebih besar (median 7.0 vs 4.8). Ini fitur paling diskriminatif.

### 4b. Distribusi Kedalaman

| Statistik | Tsunami (label=1) | Non-Tsunami (label=0) |
|-----------|-------------------|----------------------|
| Mean | 26.85 km | 68.90 km |
| Median | 21.30 km | 33.00 km |
| Min | 0.0 km | -4.0 km |
| Max | 600.0 km | 700.9 km |

**Insight:** Gempa tsunami cenderung lebih DANGKAL (median 21.3 vs 33 km). Gempa dangkal lebih efisien menggeser air.

### 4c. Distribusi Water Height (Event Tsunami Saja, n=671 pada dataset gabungan; n=610 pada dataset modeling final)

| Statistik | Nilai |
|-----------|-------|
| Mean | 4.24 m |
| Min | 0.0 m |
| Max | 524.6 m |
| Median | 0.4 m |
| Q75 | 1.8 m |

**Insight:** Mayoritas tsunami memiliki water height < 5m. Outlier ekstrem (500m+) kemungkinan dari event langka dengan runup ekstrem.

### 4d. Distribusi Intensity (Event Tsunami Saja, n=261)

| Statistik | Nilai |
|-----------|-------|
| Mean | 0.45 |
| Min | -4.14 |
| Max | 9.0 |

**Insight:** Intensity terdistribusi lebar. Nilai negatif menunjukkan tsunami yang sangat kecil.

## 5. Analisis Missing Values

### Fitur USGS (Critical untuk Semua Peran)

| Kolom | Missing % | Dampak |
|-------|-----------|--------|
| nst | 61.2% | Kualitas observasi gempa |
| gap | 45.1% | Cakupan monitoring |
| dmin | 68.7% | Jarak stasiun terdekat |
| rms | 15.9% | Akurasi lokasi gempa |

**Catatan:** Missing values pada fitur kualitas ini RELEVAN secara domain — gempa lama (1900-1950) punya lebih sedikit stasiun, sehingga kualitas data lebih rendah. Ini bisa menjadi **fitur tersendiri** (`monitoring_quality`).

### Fitur NOAA (Krusial untuk Peran Estimation & Association)

Untuk **833 event tsunami** (tolerance ±0.25°), cover data NOAA:

| Kolom | Terisi | Kosong | Coverage | Peran DM |
|-------|--------|--------|----------|----------|
| noaa_country | 833 | 0 | **100%** | Clustering, Association |
| noaa_location | 833 | 0 | **100%** | Clustering, Description |
| noaa_cause_code | 832 | 1 | **99.9%** | Association, Description |
| noaa_water_height | 671 | 162 | **80.6%** | Estimation (target) |
| noaa_intensity | 261 | 572 | **31.3%** | Estimation (alternatif) |

## 6. Kesiapan Data untuk Klasifikasi

### Target
- **Kolom:** `label` (0 = non-tsunami, 1 = tsunami)
- **Coverage:** 100% terisi, 310,369 baris
- **Tipe:** Biner — cocok untuk klasifikasi

### Fitur (20 kolom)
| Kategori | Kolom | Coverage | Keterangan |
|----------|-------|----------|------------|
| Gempa (USGS) | mag, depth | 100% | Fitur utama prediksi |
| Kualitas monitoring | nst, gap, dmin, rms | 31–84% | Missing relevan secara domain |
| Geospasial | latitude, longitude | 100% | Lokasi epicenter |
| Temporal | year, month, day | 100% | Fitur turunan (era, season) |
| NOAA (match only) | noaa_country, noaa_cause_code | 0–100%* | Metadata tsunami |

> *Kolom NOAA hanya terisi untuk 833 baris tsunami. Fitur ini TIDAK digunakan sebagai
> input model — hanya untuk interpretasi/analisis error. Tujuh kolom lain pada dataset
> gabungan (`time`, `place`, `magType`, `label`, `noaa_location`, `noaa_water_height`,
> `noaa_intensity`) adalah metadata/target, bukan fitur.

### Tantangan Utama
1. **Imbalance ekstrem (1:371):** 833 tsunami vs 309,536 non-tsunami
   → Cost-sensitive threshold tuning, bukan SMOTE
2. **Missing values pada fitur kualitas:** nst (61.2%), gap (45.1%), dmin (68.7%)
   → Median imputation + indicator feature `monitoring_quality`
3. **Distribution shift temporal:** Data 1900–1950 kurang lengkap
   → Era-based evaluation untuk mengukur robustness

### Rekomendasi untuk Phase 3–6
- **Feature engineering:** Tambah `depth_mag_ratio`, `monitoring_quality`, `tectonic_zone`, `era`, `season`
- **Modeling:** 4 model (LR, RF, XGBoost, MLP) dengan class_weight='balanced'
- **Evaluasi:** F1 primary metric (mengingat imbalance), threshold tuning, SHAP interpretability

## 7. Distribusi Temporal

### Tsunami per Dekade

| Dekade | Jumlah | Keterangan |
|--------|--------|------------|
| 1900s | 12 | Monitoring terbatas |
| 1910s | 19 | |
| 1920s | 40 | |
| 1930s | 40 | |
| 1940s | 32 | Perang dunia |
| 1950s | 45 | |
| 1960s | 73 | Peningkatan monitoring |
| 1970s | 79 | |
| 1980s | 77 | |
| 1990s | 92 | |
| 2000s | 97 | |
| 2010s | 141 | Puncak (termasuk 2004, 2011) |
| 2020s | 86 | Hingga 2026 |

**Insight:** Jumlah tsunami meningkat seiring waktu — kemungkinan karena *better monitoring*, bukan peningkatan aktivitas aktual.

## 8. Distribusi Geografis

### Top 10 Negara (Tsunami)

| Negara | Jumlah | % dari Total |
|--------|--------|-------------|
| JAPAN | 138 | 16.6% |
| INDONESIA | 76 | 9.1% |
| RUSSIA | 60 | 7.2% |
| USA | 56 | 6.7% |
| GREECE | 50 | 6.0% |
| CHILE | 43 | 5.2% |
| PAPUA NEW GUINEA | 41 | 4.9% |
| PHILIPPINES | 39 | 4.7% |
| VANUATU | 34 | 4.1% |
| SOLOMON ISLANDS | 32 | 3.8% |

**Insight:** Pacific Ring of Fire mendominasi (~70% tsunami). Indonesia dan Japan = target utama untuk *early warning*.

## 9. Kode Penyebab Tsunami

| Kode | Keterangan | Jumlah | % |
|------|-----------|--------|---|
| 1 | Earthquake | 761 | 91.4% |
| 3 | Earthquake + Landslide | 55 | 6.6% |
| 6 | Volcano | 6 | 0.7% |
| 4 | Volcano + Earthquake | 3 | 0.4% |
| 9 | Meteorological | 3 | 0.4% |
| 8 | Landslide | 2 | 0.2% |
| 7 | Volcano + Landslide | 2 | 0.2% |
| — | Tidak diketahui (NaN) | 1 | 0.1% |

**Insight:** 91% tsunami disebabkan oleh gempa murni. *Multi-hazard* (earthquake+landslide) = 6.6%.

**Konsekuensi untuk modeling (3 Nov 2026):** hanya **cause = 1** yang dipakai sebagai positif
final (761). 72 baris dengan cause 3–9 dan NaN **dibuang** — bukan dilabeli 0 — karena fitur
gempa tidak bisa menjelaskan tsunami vulkanik/ longsor. Sensitivitas cause 1–3 disiapkan
sebagai analisis tambahan.

## 10. Temuan Utama

1. **Imbalance ekstrem:** 1:371 pada dataset gabungan; **1:406,6** pada dataset modeling final — perlu cost-sensitive threshold tuning
2. **Magnitude = fitur terkuat:** Tsunami median M7.0 vs non-tsunami M4.8
3. **Depth = fitur penting:** Tsunami cenderung dangkal (median 21.3 km)
4. **Fitur kualitas missing tinggi:** nst, gap, dmin punya 45-69% missing
5. **Temporal bias:** Data 1900-1950 kurang lengkap
6. **Geographic clustering:** Pacific Ring of Fire mendominasi
7. **Water height tersedia:** 671 dari 833 tsunami gabungan (80,6%); pada dataset final 610 dari 761 — cukup untuk estimation
8. **Kode penyebab terbatas:** 91.3% earthquake — association rules perlu binning strategis

## 11. Rekomendasi untuk Tahap Selanjutnya

### Classification (tujuan utama proyek)
- **Feature Engineering:** Buat `depth_mag_ratio`, `monitoring_quality`, `tectonic_zone`, `era`, `season`
- **Imbalance Handling:** Cost-sensitive threshold tuning (SMOTE/ADASYN terbukti menurunkan F1 di ratio 1:371)
- **Missing Values:** Median imputation + indicator untuk nst, gap, dmin





## 12. Catatan Metodologis: Toleransi Matching

| Parameter | Nilai | Rationale |
|-----------|-------|-----------|
| TIME_TOLERANCE | ±1 hari | Tsunami terdeteksi dalam hitungan jam dari gempa |
| LAT_LON_TOLERANCE | **±0.25 derajat** (~28 km) | Sweet spot berdasarkan analisis manual |

### Justifikasi Toleransi Lokasi 0.25°

Analisis manual terhadap 1,486 tsunami post-1900 dengan koordinat:

| Tolerance | Estimasi Match | Keterangan |
|-----------|---------------|------------|
| 0.1° | 758 (55%) | Terlalu ketat — NOAA lokasi tsunami teramati (pantai), bukan epicenter |
| **0.25°** | **905 (66%)** | **Sweet spot estimasi — actual script: 833 (match greedy)** |
| 0.5° | 1,002 (73%) | Terlalu longgar — risiko false match dengan gempa non-penabi |
| 1.0° | 1,068 (78%) | Far-field tsunami ikut match — tidak ideal |

> **Catatan:** Angka di atas dari analisis nearest-distance (per tsunami, independent).
> Script merge menggunakan greedy matching (setiap gempa hanya match 1 tsunami),
> sehingga hasil aktual = **833 tsunami, 1:371 ratio**.

**Rationale 0.25°:**
1. Koordinat NOAA = lokasi tsunami teramati (pantai/pelabuhan), BUKAN epicenter gempa
2. Near-field tsunami (gempa → tsunami dalam <1 jam) biasanya within 20-30 km
3. Far-field tsunami (>2°) memiliki dinamika berbeda — tidak perlu di-force match
4. 833 tsunami matched — realistis dan defensible untuk laporan akademik
