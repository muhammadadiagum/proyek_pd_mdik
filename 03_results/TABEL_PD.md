# Tabel Induk PD: Klasifikasi & Estimasi Tsunami

> **Dibuat:** 4 November 2026
> **Sumber angka:** JSON hasil eksekusi tiap fase.
> **Data:** `00_data/processed/modeling_dataset.csv`: **310.297 × 26**;
> **761 positif** (`label=1 & noaa_cause_code=1`); 309.536 negatif.
> **Dua split berbeda:** Python **62.060** baris test; RapidMiner **62.059**
> baris test. Keduanya punya **152 positif**, tetapi tidak row-identik.

Dokumen ini adalah tabel siap salin untuk laporan PD 14 bagian. Angka utama
tetap di `03_results/ANGKA_FINAL.md`; kalau pipeline dijalankan ulang, bekukan
ulang lewat `0_freeze_numbers.py` lalu periksa tabel ini.

---

## 1. Sumber Angka

| Bagian | JSON sumber | Isi |
|---|---|---|
| Data & label | `phase_3_data_preparation/results/modeling_set_manifest.json` | jumlah baris, fitur, md5 |
| Klasifikasi | `phase_4_modeling/0_classification/results/model_results.json` | 4 model × 5 skema |
| Estimasi ordinal | `phase_4_modeling/1_estimation/results/estimation_ordinal_results.json` | CV, holdout, aturan peringatan |
| Deployment | `phase_6_deployment/results/deployment_results.json` | ambang, metrik produk, ordinal |
| Baseline aturan | `phase_5_evaluation/results/baseline_match_results.json` | aturan M/depth |
| Ensemble | `phase_5_evaluation/results/ensemble_results.json` | soft/hard voting |
| Imbalance | `phase_5_evaluation/results/imbalance_results.json` | ROS/SMOTE/ADASYN/RUS |
| Time split & ablasi | `phase_5_evaluation/results/time_based_split_results.json`, `ablation_results.json` | evaluasi temporal |
| Kalibrasi | `phase_5_evaluation/results/calibration_results.json` | Brier/ECE |
| Uji statistik | `phase_5_evaluation/results/statistical_results.json` | McNemar, CI |
| RapidMiner | `01_pipeline/RAPID_MINER/results/rapidminer_metrics.json` | confusion, metrik, path |

---

## 2. Data dan Definisi Label

| Item | Nilai |
|---|---:|
| Baris modeling | **310.297** |
| Positif tsunami (`cause=1`) | **761** |
| Negatif | **309.536** |
| Rasio positif:negatif | **1 : 406,75** |
| Positif dengan `noaa_water_height` | **610** |
| Baris dibuang (`cause≠1`) | **72** |
| Fitur model | **24** = 11 numerik + 13 dummy |
| md5 `modeling_dataset.csv` | `d0695e96d0e31e6efa77a02cf8eeda28` |

Rasio **1:406,75** dihitung dari 309.536/761. Untuk training, `scale_pos_weight`
XGB memakai 247.628/609 = **406,61**; jangan menukar kedua angka itu.

Catatan: 72 baris cause lain tidak dipakai karena definisi positif final
dibatasi pada tsunami seismik (`cause=1`).

---

## 3. Split Data

| Item | Python (scikit-learn) | RapidMiner |
|---|---:|---:|
| Train | **248.237** | **248.238** |
| Test | **62.060** | **62.059** |
| Positif di test | **152** | **152** |
| Stratified | ya | ya |
| Seed | 42 | 42 |
| Row-identik? | - | **tidak** |

> **Penting:** jangan menulis “keduanya 62.059”. Python memakai
> `test_size=0.2` yang dibulatkan ke atas menjadi 62.060; RapidMiner
> menghasilkan 62.059. Selisih 1 baris ini konsisten dengan F-03: jumlah
> positif sama, tetapi anggota baris tidak identik.

---

## 4. Klasifikasi Phase 4: Ambang 0,5

Test Python: 62.060 baris, 152 positif. Ini **bukan** ambang produk.

| Model | F1 | Recall | Precision | AUC | TN | FP | FN | TP | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | **0.3849** | 0.3026 | 0.5287 | 0.9654 | 61.867 | 41 | 106 | 46 | 10.641 |
| RandomForest | 0.3434 | 0.2237 | **0.7391** | 0.9734 | 61.896 | 12 | 118 | 34 | 11.812 |
| XGBoost | 0.2178 | 0.6513 | 0.1308 | 0.9672 | 61.250 | 658 | 53 | 99 | 5.958 |
| LogisticRegression | 0.0681 | **0.9211** | 0.0353 | **0.9778** | 58.086 | 3.822 | 12 | 140 | **5.022** |

Interpretasi singkat:

- **F1 tertinggi:** MLP (0.3849), tetapi recall hanya 0.3026.
- **Recall tertinggi:** LogisticRegression (0.9211), tetapi precision 0.0353.
- **XGBoost** memberi kompromi recall 0.6513 dengan cost 5.958.
- Cost = `FN × 100 + FP`; biaya FN sengaja 100× FP.

---

## 5. Estimasi Ordinal: 3 Kelas

Kelas: **Waspada** ≤ 0,5 m; **Siaga** 0,5–3,0 m; **Awas** > 3,0 m.
Distribusi positif: **368 / 158 / 84** (60,3% / 25,9% / 13,8%).
Holdout ordinal: 122 baris.

| Model | QWK CV (mean ± sd) | QWK holdout | MAE kelas | Tepat | ±1 kelas |
|---|---:|---:|---:|---:|---:|
| XGBoost | **0.4098 ± 0.0845** | **0.4667** | **0.4016** | 0.6393 | **0.9590** |
| RandomForest | 0.4081 ± 0.1159 | 0.3639 | 0.4180 | **0.6639** | 0.9180 |
| LogisticRegression | 0.3943 ± 0.0169 | 0.2474 | 0.4508 | 0.6557 | 0.8934 |
| MLP | 0.2011 ± 0.1404 | 0.1830 | 0.5328 | 0.6148 | 0.8525 |

Aturan peringatan sadar-biaya (perbandingan antar model; yang dipakai produk
adalah XGB), holdout 122 baris:

| Model | QWK | MAE kelas | Recall Awas |
|---|---:|---:|---:|
| XGBoost | **0.4320** | **0.4426** | 0.4118 |
| RandomForest | 0.3897 | 0.4590 | 0.3529 |
| LogisticRegression | 0.2714 | 0.5410 | 0.2941 |
| MLP | 0.0343 | 1.2049 | **0.7647** |

Catatan: OOF terkumpul XGB ordinal QWK **0.4128**, tepat 0.6393, ±1 kelas
0.9279 (n=610). Angka QWK 0.4320 adalah aturan peringatan, bukan model murni.

---

## 6. Deployment: Ambang per Model

Protokol: inner-validation 25%, seed 7, biaya `FN×100 + FP`, lantai presisi 0.1.
Test Python 62.060; positif 152.

| Model | Ambang | F1 | Recall | Precision | AUC | Cost | TP | FP | FN | F1@0,5 | Recall@0,5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.13 | 0.1285 | 0.7961 | 0.0699 | 0.9672 | 4.710 | 121 | 1.610 | 31 | 0.2178 | 0.6513 |
| LogisticRegression | 0.85 | 0.1757 | 0.7895 | 0.0988 | 0.9778 | 4.294 | 120 | 1.094 | 32 | 0.0681 | 0.9211 |
| RandomForest | 0.02 | **0.2102** | 0.7829 | 0.1214 | 0.9734 | 4.161 | 119 | 861 | 33 | 0.3434 | 0.2237 |
| MLP | 0.02 | 0.1627 | 0.7632 | 0.0911 | 0.9654 | 4.758 | 116 | 1.158 | 36 | 0.3849 | 0.3026 |
| Ensemble (soft voting) | 0.26 | 0.1736 | **0.8289** | 0.0969 | **0.9785** | **3.774** | 126 | 1.174 | 26 | - | - |

Interpretasi:

- **Recall tertinggi + cost terendah:** ensemble (0.8289; 3.774).
- **F1 tertinggi pada ambang produk:** RandomForest (0.2102).
- **F1@0,5 tertinggi:** MLP (0.3849), tetapi recall turun ke 0.3026.
- Precision semua model rendah karena konsekuensi biaya FN 100× FP.

---

## 7. Baseline Aturan Sederhana (Test Python yang Sama)

| Aturan | F1 | Recall | Precision | Cost |
|---|---:|---:|---:|---:|
| M ≥ 6.0 | 0.0894 | 0.8816 | 0.0471 | 4.511 |
| M ≥ 6.5 | 0.2115 | 0.7237 | 0.1239 | 4.978 |
| M ≥ 7.0 | 0.3524 | 0.5263 | 0.2649 | 7.422 |
| **M ≥ 7.0 & depth ≤ 70** | **0.3788** | 0.4934 | 0.3074 | 7.869 |
| M ≥ 8.0 | 0.1609 | 0.0921 | 0.6364 | 13.808 |

**Cara menulis yang benar:**

- Pada **ambang operasional**, F1 model (0.1285–0.2102) **di bawah** aturan
  M≥7 & depth≤70 (0.3788).
- Pada **ambang 0,5**, MLP sedikit di atas baseline (0.3849 vs 0.3788), tetapi
  recall hanya 0.3026 dan biaya lebih tinggi.
- Keunggulan model ada pada **cost/recall** (ensemble recall 0.8289, cost 3.774
  vs baseline cost 7.869) dan pada estimasi ordinal, bukan pada F1.
- Jangan menulis “AI lebih akurat”.

---

## 8. Imbalance, Ensemble, dan Evaluasi Tambahan

### 8.1 XGBoost: skema imbalance (ambang 0,5)

| Skema | Train size | F1 | Recall | Cost |
|---|---:|---:|---:|---:|
| Original | 248.237 | 0.2178 | 0.6513 | 5.958 |
| Random Oversampling | 495.256 | **0.2439** | 0.6579 | 5.768 |
| SMOTE | 495.256 | 0.2152 | 0.6513 | 5.969 |
| ADASYN | 495.146 | 0.2010 | 0.6316 | 6.307 |
| Random Undersampling | 1.218 | 0.0705 | **0.9211** | 4.882 |

ROS sedikit lebih tinggi dari Original (0.2439 vs 0.2178), tetapi tidak
konsisten; produk tetap memakai Original.

### 8.2 Ensemble

| Konfigurasi | F1 | Recall | Precision | AUC | Cost |
|---|---:|---:|---:|---:|---:|
| XGB @0,36 (5c) | 0.1896 | 0.7105 | 0.1094 | 0.9672 | 5.279 |
| Soft voting @0,36 (5c) | **0.2494** | 0.6645 | 0.1535 | 0.9777 | 5.657 |
| Hard voting @0,36 (5c) | 0.2250 | 0.6513 | 0.1360 | - | 5.929 |
| Ensemble produk @0,26 | 0.1736 | **0.8289** | 0.0969 | **0.9785** | **3.774** |

Jangan mencampur angka soft voting @0,36 (0.2494) dengan ensemble produk
@0,26 (0.1736); keduanya protokol berbeda.

### 8.3 Time-based split dan ablasi

Split temporal: train ≤ 2015, test ≥ 2016 (bukan random split).

| Konfigurasi | XGB F1 | XGB @0,36 F1 | Soft voting F1 |
|---|---:|---:|---:|
| A: 24 fitur | **0.5212** | 0.5014 | **0.5399** |
| B: tanpa `monitoring_quality` (23) | 0.4972 | 0.4700 | 0.5220 |
| C: tanpa fitur kualitas (20) | 0.4594 | 0.4056 | 0.4898 |

Perbandingan split:

| Split | F1 XGB | Recall | Precision | Cost |
|---|---:|---:|---:|---:|
| Random split | 0.2178 | 0.6513 | 0.1308 | 5.958 |
| Time-based | **0.5212** | 0.6466 | 0.4365 | 4.811 |

Angka time-based tidak boleh dibandingkan langsung dengan §4 karena prevalensi
test berbeda.

### 8.4 Kalibrasi

| Model | Brier | ECE |
|---|---:|---:|
| XGBoost | 0.0016 | 0.0015 |
| LogisticRegression | 0.0121 | 0.0369 |
| RandomForest | **0.0011** | **0.0004** |
| MLP | 0.0012 | 0.0004 |

### 8.5 Uji statistik

McNemar XGB vs LR/RF/MLP semuanya `p < 0.001` (`***`). Detail:
`phase_5_evaluation/results/statistical_results.json`.

---

## 9. RapidMiner vs Python (empat model)

Sumber: `rapidminer_4model_results.json` (dihitung dari empat berkas
`rm_*_predictions.csv` run GUI 14 November 2026, dicocokkan dengan empat
`PerformanceVector_*.rmperf`). Python: `model_results.json` (ambang 0,5).

| Model | RM F1 | PY F1 | RM Recall | PY Recall | RM Precision | PY Precision | RM AUC | PY AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.3053 | 0.3434 | 0.1908 | 0.2237 | 0.7632 | 0.7391 | 0.9566 | 0.9734 |
| Logistic Regression | 0.2328 | 0.0681 | 0.1447 | 0.9211 | 0.5946 | 0.0353 | 0.9685 | 0.9778 |
| XGBoost | **0.3654** | 0.2178 | 0.2500 | 0.6513 | 0.6786 | 0.1308 | 0.9614 | 0.9672 |
| Neural Net (MLP) | 0.3309 | **0.3849** | 0.6053 | 0.3026 | 0.2277 | 0.5287 | 0.9172 | 0.9654 |

Confusion matrix RapidMiner (TN/FP/FN/TP, test 62.059 baris): RF
61.898/9/123/29 · LR 61.892/15/130/22 · XGB 61.889/18/114/38 · MLP
61.595/312/60/92.

**Cara menulis:** selisih terbesar ada pada LR dan XGB karena Python memakai
`class_weight balanced` / `scale_pos_weight` yang tidak tersedia di RapidMiner;
MLP RapidMiner dilatih pada data ter-undersample 40:1. Split tidak row-identik
dan jumlah test berbeda 1 baris, jadi perbandingan sah pada level pola dan
metodologi, bukan pada level baris atau klaim “alat mana lebih unggul”.

### 9b. Detail cabang Random Forest (`proyek_akhir.rmp`, arsip)

| Item | RapidMiner `proyek_akhir.rmp` | Python RF |
|---|---:|---:|
| Test rows | **62.059** | **62.060** |
| Test positif | **152** | **152** |
| TP / FP / FN / TN | 29 / 9 / 123 / 61.898 | 34 / 12 / 118 / 61.896 |
| Precision | 0.7632 | 0.7391 |
| Recall | 0.1908 | 0.2237 |
| F1 | 0.3053 | 0.3434 |
| AUC | 0.9566 | 0.9734 |

Angka RF di kedua proses RapidMiner identik, sehingga baris ini tetap berlaku
sebagai pembanding cabang Random Forest.

---

## 10. Angka yang Tidak Boleh Dikutip

| Angka lama | Kenapa salah | Ganti dengan |
|---|---|---|
| 833 positif / 310.369 baris | label belum dibatasi `cause=1` | 761 / 310.297 |
| Test 62.074 / 185 positif | label lama, split lama | Python 62.060/152; RM 62.059/152 |
| RF F1 0.2975 | process lama seed 1992 | 0.3434 Python; 0.3053 RM |
| XGB F1 0.2260 | `scale_pos_weight` 371 (label lama) | 0.2178 |
| Ambang 0.22 / 0.94 sebagai ambang produk | hardcode / in-sample | `model/thresholds.json` |
| MAPE 319–403%, R² 0.26/−0.16 | regresi meter sudah tidak dipakai | QWK 0.4667 |
| 92 TP / 49 FN | threshold 0.5, data sebelum filter | 5i: 89 TP / 44 FN @0,36 |

---

## 11. Daftar Plot untuk Laporan

Folder target: `03_results/plots/`. Ini **hasil kurasi**, bukan salinan semua
plot per fase. **Status 4 Nov 2026:** plot final yang dipakai laporan sudah
dibuat di `05_docs/pd/plots/` (18 PNG); tabel di bawah adalah rencana awal.

### 11.1 Sudah ada (bisa langsung dipakai/dikurasi)

| Plot | Path sumber | Status |
|---|---|---|
| Perbandingan metrik 4 model | `phase_4_modeling/0_classification/plots/model_comparison_metrics.png` | ✅ |
| AUC 4 model | `phase_4_modeling/0_classification/plots/model_comparison_auc.png` | ✅ |
| Cost 4 model | `phase_4_modeling/0_classification/plots/model_comparison_cost.png` | ✅ |
| Heatmap F1 model × skema | `phase_4_modeling/0_classification/plots/model_heatmap_f1.png` | ✅ |
| SHAP importance XGB | `phase_5_evaluation/plots/shap_importance_xgboost.png` | ✅ |
| SHAP summary XGB | `phase_5_evaluation/plots/shap_summary_xgboost.png` | ✅ |
| SHAP dependence (4 fitur) | `phase_5_evaluation/plots/shap_dependence_*.png` | ✅ |
| Threshold tuning XGB | `phase_5_evaluation/plots/threshold_tuning_xgboost.png` | ✅ |
| Kalibrasi | `phase_5_evaluation/plots/calibration_analysis.png` | ✅ |
| Imbalance | `phase_5_evaluation/plots/imbalance_analysis.png` | ✅ |
| Ablasi | `phase_5_evaluation/plots/ablation_study.png` | ✅ |
| Time-based split | `phase_5_evaluation/plots/time_based_split_comparison.png` | ✅ |
| Era | `phase_5_evaluation/plots/era_analysis.png` | ✅ |
| Error analysis | `phase_5_evaluation/plots/error_analysis.png` | ✅ |
| Regional | `phase_5_evaluation/plots/regional_analysis.png` | ✅ |

### 11.2 Perlu dibuat (prioritas P0)

| # | Plot | File target | Isi |
|---|---|---|---|
| 1 | Distribusi kelas dataset | `01_dataset_class_distribution.png` | 761 vs 309.536; skala log |
| 2 | Confusion matrix 4 model @0,5 | `02_confusion_matrix_4models.png` | 2×2 panel, TN/FP/FN/TP |
| 3 | ROC + PR curves | `03_roc_pr_curves.png` | 4 model + ensemble, AUC/AP |
| 4 | Threshold vs precision/recall/cost | `04_threshold_cost_curves.png` | 4 panel; tunjukkan titik produk |
| 5 | Confusion matrix ordinal 3×3 | `05_ordinal_confusion.png` | XGB holdout, n=122 |
| 6 | Perbandingan model ordinal | `06_ordinal_model_comparison.png` | QWK, MAE, exact, ±1 |
| 7 | RM vs Python RF | `07_rm_vs_python.png` | grouped bar + CM |
| 8 | Baseline vs model | `08_baseline_vs_model.png` | F1 dan cost |

### 11.3 Perlu dibuat/dikurasi (prioritas P1)

| # | Plot | File target | Isi |
|---|---|---|---|
| 9 | SHAP final | `09_shap_summary.png` | kurasi dari plot 5a |
| 10 | Kalibrasi final | `10_calibration_curve.png` | kurasi dari 5j |
| 11 | Ablasi final | `11_ablation.png` | A/B/C |
| 12 | Time split final | `12_time_split.png` | random vs temporal |
| 13 | Era analysis | `13_era_analysis.png` | F1 per era |
| 14 | Regional analysis | `14_regional_analysis.png` | F1 per zona |

### 11.4 Sumber visual

Diagram alur dan arsitektur proyek tersedia di `05_docs/diagrams/` dalam format
`.drawio` dan `.png`. Angka ringkas ada di `03_results/ANGKA_FINAL.md`;
perbandingan RapidMiner dan Python ada di §9.

---

## 12. Cara Memakai Dokumen Ini

1. Ambil angka dari tabel ini sebagai acuan laporan.
2. Setiap tabel di laporan menyebut sumber JSON-nya (lihat §1).
3. Selalu tulis perbedaan split 62.060 (Python) vs 62.059 (RapidMiner).
4. Untuk klaim performa, bedakan **ambang 0,5**, **ambang produk**, dan
   **aturan peringatan ordinal**.
5. Kalau pipeline berubah: jalankan ulang `0_freeze_numbers.py`, lalu audit
   ulang `ANGKA_FINAL.md` dan file ini.
