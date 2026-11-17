# Tahap 1: Pemahaman Bisnis (Business Understanding)

> Framework: CRISP-DM Phase 1
> Terakhir diperbarui: **2026-11-03** (revisi keputusan final: label `cause=1`, estimasi ordinal 3 kelas)
> Status: **final** — angka di sini sudah sinkron dengan `03_results/ANGKA_FINAL.md`

---

## 1. Latar Belakang

Tsunami merupakan salah satu bencana alam paling mematikan di dunia. Dalam 120 tahun terakhir (1900–2020), lebih dari 300.000 orang meninggal akibat tsunami (Reid & Mooney, 2023). Sistem peringatan dini tsunami (TEWS) saat ini masih mengandalkan *Decision Matrix* berbasis ambang magnitudo dan kedalaman gempa, yang menghasilkan banyak alarm palsu dan menurunkan kepercayaan masyarakat terhadap sistem peringatan (*warning fatigue*).

## 2. Problem Statement

**Pusat masalah.** Saat gempa terjadi, ada dua pertanyaan kritis:

1. *"Apakah gempa ini akan memicu tsunami?"* → **klasifikasi biner** (gerbang).
2. *"Jika ya, seberapa besar peringatan yang perlu dikeluarkan?"* → **estimasi tingkat bahaya** dalam 3 kelas.

Keduanya disusun berantai (*chained*): klasifikasi bertindak sebagai gerbang; estimasi hanya bermakna bila klasifikasi menyatakan tsunami.

**Keputusan desain 3 November 2026.** Estimasi tidak lagi memprediksi ketinggian air dalam meter. Target diganti menjadi **regresi ordinal 3 kelas** yang mengikuti ambang peringatan BMKG/InaTEWS:

| Kelas | Rentang tinggi air |
|---|---|
| Waspada | ≤ 0,5 m |
| Siaga | 0,5–3,0 m |
| Awas | > 3,0 m |

Alasannya: regresi kontinu pada data ini tidak layak produk (R² 0,26; MAPE 319–403%; RF bahkan R² −0,16), sedangkan klasifikasi ordinal memberi output yang bisa langsung dipakai dan tetap mempertahankan sifat berurutan dari target. Produk akhir menampilkan **kelas + probabilitas**, bukan angka meter.

**Keterbatasan solusi saat ini:**

- **Decision Matrix (ambang Mw + depth):** banyak alarm palsu; pada data uji, aturan M≥7 menangkap 53% tsunami dengan presisi 26%.
- **Simulasi numerik:** butuh puluhan menit, terlalu lambat untuk *near-field tsunami*.
- **ML single-source:** belum ada yang menggabungkan katalog tsunami NOAA dengan katalog gempa USGS.

**Gap yang ingin diisi:** belum ada penelitian yang melakukan *data-driven tsunami trigger prediction* multi-source (NOAA + USGS) skala global dengan data historis (bukan simulasi), sekaligus memberi estimasi tingkat bahaya secara berantai setelah trigger terdeteksi.

## 3. Research Questions

| No | Research Question | Cara menjawab | Jawaban singkat (3 Nov 2026) |
|----|-------------------|---------------|------------------------------|
| RQ1 | Apakah fitur gempa USGS bisa memprediksi trigger tsunami? | Klasifikasi 4 model, AUC + cost | **Ya, untuk diskriminasi** (AUC 0,965–0,978), tetapi presisi rendah pada ambang operasional |
| RQ2 | Fitur gempa mana yang paling mempengaruhi prediksi? | SHAP pada XGBoost | Magnitudo dan kedalaman mendominasi; fitur kualitas monitoring memberi kontribusi kecil dan tidak stabil (lihat F-24) |
| RQ3 | Bagaimana performa ML pada imbalance ekstrem? | 4 model × 5 metode resampling | Rasio final **1:406,6**; resampling tidak memperbaiki F1 secara berarti (Original 0,2178; ROS 0,2439; SMOTE 0,2152; ADASYN 0,2010; RUS 0,0705) |
| RQ4 | Apakah model generalisasi ke era berbeda? | Analisis per era + time-based split | Tidak stabil: F1 random split 0,2178 vs time-based 0,5212, tetapi prevalensi test berbeda (0,245% vs 0,168%) — **bukan** bukti peningkatan |
| RQ5 | Seberapa akurat tingkat bahaya dapat diestimasi setelah trigger? | Regresi ordinal 3 kelas | QWK holdout **0,4667**; MAE kelas 0,4016; tepat 63,9%; ±1 kelas 95,9% |

## 4. Dataset

| Dataset | Sumber | Jumlah | Periode | Status |
|---------|--------|--------|---------|--------|
| Global Historical Tsunami Database | NOAA/NCEI (doi:10.7289/V5PN93H7) | 3.129 event | -2000 SM – 2026 | ✅ |
| ANSS Comprehensive Earthquake Catalog | USGS (earthquake.usgs.gov) | 311.104 event (M4.5+) | 1900–2026 | ✅ |

**Dataset gabungan (hasil join, input phase 2–3):**

- Total: **310.369** baris
- Positif (tsunami, semua penyebab): **833**
- Negatif: **309.536**
- Rasio: 1:371

**Dataset modeling final (keputusan 3 Nov 2026):**

- Positif = `label = 1` **dan** `noaa_cause_code = 1` (gempa murni) → **761**
- Negatif tetap **309.536**
- Total **310.297** baris; rasio **1:406,6** (309.536 / 761)
- 72 baris positif dengan cause lain **dibuang**, bukan dilabeli 0 (0,023% data):
  cause 3 = 55, cause 4 = 3, cause 6 = 6, cause 7 = 2, cause 8 = 2, cause 9 = 3, NaN = 1.
- Dasar keputusan lengkap: `.audit/00_RINGKASAN.md` §6. Daftar baris yang dibuang:
  `00_data/processed/dropped_events.csv`.
- Sensitivitas cause 1–3 disiapkan sebagai analisis tambahan, bukan dataset utama.

**Subset estimasi (chained):**

- Dari 761 tsunami, **610** punya nilai `noaa_water_height` (80,2%)
- Distribusi kelas: Waspada **368** · Siaga **158** · Awas **84**
- Target: kelas tinggi air (ordinal), bukan meter

## 5. Success Criteria

Target awal dipertahankan sebagai pembanding, tetapi statusnya diisi apa adanya.

**Klasifikasi (test 62.060 baris, 152 tsunami):**

| Metrik | Target awal | Hasil pada ambang operasional | Status |
|---|---|---|---|
| Recall tsunami | ≥ 80% | Ensemble **0,8289**; XGB 0,7961; LR 0,7895; RF 0,7829; MLP 0,7632 | ⚠️ tercapai hanya oleh ensemble |
| Precision | ≥ 60% | 0,070–0,121 (ambang operasional); 0,53–0,74 hanya pada ambang 0,5 | ❌ tidak tercapai bersamaan dengan recall |
| F1 | ≥ 0,45 | 0,1285–0,2102 (operasional); maksimum 0,3849 (MLP @0,5) | ❌ tidak tercapai |
| AUC-ROC | ≥ 0,85 | 0,9654–0,9785 | ✅ tercapai |
| Cost (FN×100 + FP) | ≤ baseline | Ensemble **3.774** vs default 0,5 (5.022–11.812) dan aturan M≥7 (7.422) | ✅ tercapai |

Catatan penting: presisi rendah adalah **konsekuensi pilihan biaya** (FN 100× lebih mahal), bukan bug. Pada F1, model juga tidak mengungguli aturan sederhana M≥7 & depth≤70 (0,3788) — lihat F-26 di `.audit/04_TEMUAN_TEKNIS.md`. Keunggulan model ada pada biaya dan pada estimasi ordinal.

**Estimasi ordinal (holdout 122 baris):**

| Metrik | Target | Hasil XGBoost | Status |
|---|---|---|---|
| QWK | ≥ 0,40 | **0,4667** | ✅ |
| MAE kelas | ≤ 0,50 | **0,4016** | ✅ |
| Exact accuracy | ≥ 0,60 | **0,6393** | ✅ |
| Adjacent accuracy (±1 kelas) | ≥ 0,90 | **0,9590** | ✅ |

Pembanding: aturan kelas sadar-biaya QWK 0,4320 (recall Awas 0,4118); baseline regresi lama→kelas QWK 0,275; baseline mayoritas QWK 0,000.

## 6. Literature Review (Ringkas)

| Paper | Pendekatan | Keterbatasan |
|-------|-----------|-------------|
| [23] IJRASET 2025 | Random Forest, NOAA saja | Dataset kecil, 1 sumber data |
| [24] AGU 2023 | MLP + model Okada | Data sintetis, bukan historis |
| [25] TSUSY 2026 | NOAA+USGS, simulasi numerik | Filter M6+, fokus simulasi |
| [26] Pasupuleti 2026 | Physics-informed residual | Gagal generalisasi antar-region |

**Gap utama:**

1. Belum ada *multi-source data mining* (bukan simulasi) untuk klasifikasi tsunami.
2. Belum ada analisis imbalance ekstrem **1:406,6** pada skala global.
3. Fitur kualitas monitoring gempa (`nst`, `gap`, `dmin`, `rms`) belum pernah dipakai.
4. Optimasi sadar-biaya dengan **lantai presisi** belum dieksplorasi.
5. Estimasi tingkat bahaya sebagai **regresi ordinal** berantai belum ada.

## 7. Novelty yang Ditawarkan

1. **Multi-source data fusion** — NOAA tsunami + USGS earthquake, data historis langsung.
2. **Skala global M4.5+** — lebih luas dari TSUSY 2026 (M6+).
3. **Fitur kualitas monitoring gempa** — `nst`, `gap`, `dmin`, `rms`.
4. **Optimasi sadar-biaya dengan lantai presisi** — ambang per model dari inner-validation, bukan test set.
5. **Analisis imbalance ekstrem 1:406,6** — termasuk bukti bahwa resampling tidak otomatis menolong.
6. **Chained classification → ordinal estimation** — kelas Waspada/Siaga/Awas mengikuti ambang BMKG/InaTEWS.
7. **Kejujuran benchmark** — model dibandingkan dengan aturan magnitudo sederhana, bukan hanya antar-model.

## 8. Metodologi (Overview)

```
Data Integration (NOAA + USGS, join toleransi ±0,25°)
        ↓
Feature Engineering (26 kolom → 24 fitur model: 11 numerik + 13 dummy)
        ↓
Definisi Label Final (positif = label=1 & cause=1; 72 baris cause lain dibuang)
        ↓
Split 80/20 stratified (seed 42) — dipakai sama di Python dan RapidMiner
        ↓
[Klasifikasi] 4 model (LR, RF, XGBoost, MLP) × 5 metode imbalance
        ↓
Evaluasi (F1, AUC, cost, kalibrasi, ablasi, time-based split, baseline aturan)
        ↓
Explainability (SHAP pada XGBoost)
        ↓
[Estimasi — chained] Regresi ordinal 3 kelas (4 model, cumulative binary)
        ↓
Deployment: ambang per model dari inner-validation 25% data latih
        ↓
Produk: kelas + probabilitas (predict.py + web app), tanpa output meter
```

## 9. Target Output

- Model klasifikasi trigger tsunami dari fitur gempa (gerbang).
- Benchmark 4 model × 5 metode imbalance + tuning ambang sadar-biaya.
- Analisis SHAP, per era, per zona tektonik, dan kalibrasi.
- Model regresi ordinal 3 kelas untuk tingkat bahaya (chained, hanya bila trigger).
- Ambang keputusan per model tersimpan (`thresholds.json`) dan dipakai konsisten oleh CLI + app.
- Pipeline reproducible + angka beku di `03_results/ANGKA_FINAL.md`.

## 10. Riwayat Keputusan yang Mengubah Dokumen Ini

| Tanggal | Keputusan | Alasan |
|---|---|---|
| 3 Nov 2026 | Positif dibatasi `cause = 1`; 72 baris dibuang | Menghindari model belajar pola non-seismik (letusan gunung, longsor) yang tidak bisa diprediksi dari fitur gempa |
| 3 Nov 2026 | Estimasi diganti regresi ordinal 3 kelas | Regresi meter tidak layak produk (R² 0,26; MAPE 319–403%) |
| 3 Nov 2026 | Ambang per model dari inner-validation + lantai presisi 0,10 | Menghapus konflik 0,94 vs 0,22 dan mencegah ambang runtuh ke 0,01 |
| 3 Nov 2026 | Baseline aturan magnitudo masuk evaluasi | Model tidak unggul F1; klaim harus jujur (F-26) |

> Rujukan angka: `03_results/ANGKA_FINAL.md`. Rujukan temuan: `.audit/04_TEMUAN_TEKNIS.md`.
