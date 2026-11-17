# ANGKA FINAL PROYEK — dari JSON hasil eksekusi

Dibuat: **03 November 2026 12:19** oleh `03_results/0_freeze_numbers.py`.
Dokumen ini dihasilkan otomatis. Jalankan ulang skripnya setelah pipeline berubah.

Sumber per bagian (timestamp berkas):
- manifest data: `phase_3_data_preparation/results/modeling_set_manifest.json` — 03 Nov 2026 08:49
- klasifikasi: `phase_4_modeling/0_classification/results/model_results.json` — 03 Nov 2026 09:53
- ordinal: `phase_4_modeling/1_estimation/results/estimation_ordinal_results.json` — 03 Nov 2026 09:53
- deployment: `phase_6_deployment/results/deployment_results.json` — 03 Nov 2026 10:27
- ambang: `phase_6_deployment/model/thresholds.json` — 03 Nov 2026 10:27
- baseline: `phase_5_evaluation/results/baseline_match_results.json` — 03 Nov 2026 10:18

## 1. Data dan definisi label

- Baris: **310.297**; positif (tsunami, cause = 1): **761**; negatif: **309.536**
- Positif dengan `noaa_water_height`: **610**
- Dibuang (cause ≠ 1): **72** baris → `00_data/processed/dropped_events.csv`
- md5 `modeling_dataset.csv`: `d0695e96d0e31e6efa77a02cf8eeda28`

## 2. Klasifikasi — phase 4 (ambang bawaan 0,5, test 20%)

| Model | F1 | Recall | Precision | AUC | TN | FP | FN | TP | Cost |
|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.3849 | 0.3026 | 0.5287 | 0.9654 | 61867 | 41 | 106 | 46 | 10.641 |
| RandomForest | 0.3434 | 0.2237 | 0.7391 | 0.9734 | 61896 | 12 | 118 | 34 | 11.812 |
| XGBoost | 0.2178 | 0.6513 | 0.1308 | 0.9672 | 61250 | 658 | 53 | 99 | 5.958 |
| LogisticRegression | 0.0681 | 0.9211 | 0.0353 | 0.9778 | 58086 | 3822 | 12 | 140 | 5.022 |

Pada 20 kombinasi phase 4, tidak ada metode resampling yang mengungguli konfigurasi Original terbaik (MLP 0,3849; ROS terbaik 0,3262). Eksperimen imbalance phase 5 pada XGB memberi ROS F1 0,2439 vs Original 0,2178 — sedikit lebih tinggi tetapi tidak konsisten, jadi produk tetap memakai Original. Tabel penuh: `model_results.json` dan `imbalance_results.json`.

## 3. Estimasi — regresi ordinal 3 kelas

Kelas: Waspada ≤ 0.5 m; Siaga 0.5–3.0 m; Awas > 3.0 m. Distribusi positif: [368, 158, 84].

| Model | QWK CV (mean ± std) | QWK holdout | MAE kelas | Tepat | ±1 kelas |
|---|---|---|---|---|---|
| XGBoost | 0.4098 ± 0.0845 | 0.4667 | 0.4016 | 0.6393 | 0.9590 |
| RandomForest | 0.4081 ± 0.1159 | 0.3639 | 0.4180 | 0.6639 | 0.9180 |
| LogisticRegression | 0.3943 ± 0.0169 | 0.2474 | 0.4508 | 0.6557 | 0.8934 |
| MLP | 0.2011 ± 0.1404 | 0.1830 | 0.5328 | 0.6148 | 0.8525 |

Aturan peringatan sadar-biaya (dipakai produk, holdout 122 baris):

| Model | QWK | MAE kelas | Recall Awas |
|---|---|---|---|
| LogisticRegression | 0.2714 | 0.5410 | 0.2941 |
| RandomForest | 0.3897 | 0.4590 | 0.3529 |
| XGBoost | 0.4320 | 0.4426 | 0.4118 |
| MLP | 0.0343 | 1.2049 | 0.7647 |

OOF terkumpul XGB ordinal: QWK 0.4128, tepat 0.6393, ±1 kelas 0.9279 (n=610).

## 4. Deployment — ambang per model (inner-validation)

Protokol: inner-validation 25%, seed 7, biaya FN×100 + FP×1, lantai presisi 0.1.

| Model | Ambang deploy | F1 | Recall | Precision | AUC | Cost | F1 @0,5 | Recall @0,5 |
|---|---|---|---|---|---|---|---|---|
| xgboost | 0.13 | 0.1285 | 0.7961 | 0.0699 | 0.9672 | 4.710 | 0.2178 | 0.6513 |
| logistic_regression | 0.85 | 0.1757 | 0.7895 | 0.0988 | 0.9778 | 4.294 | 0.0681 | 0.9211 |
| random_forest | 0.02 | 0.2102 | 0.7829 | 0.1214 | 0.9734 | 4.161 | 0.3434 | 0.2237 |
| mlp | 0.02 | 0.1627 | 0.7632 | 0.0911 | 0.9654 | 4.758 | 0.3849 | 0.3026 |
| ensemble | 0.26 | 0.1736 | 0.8289 | 0.0969 | 0.9785 | 3.774 | — | — |

Ordinal holdout (dari deployment): QWK 0.4667, MAE kelas 0.4016, tepat 0.6393, ±1 0.9590.

## 5. Baseline aturan sederhana (split test yang sama)

| Aturan | F1 | Recall | Precision | Cost |
|---|---|---|---|---|
| M >= 6.0 | 0.0894 | 0.8816 | 0.0471 | 4.511 |
| M >= 6.5 | 0.2115 | 0.7237 | 0.1239 | 4.978 |
| M >= 7.0 | 0.3524 | 0.5263 | 0.2649 | 7.422 |
| M >= 7.0 & depth <= 70 | 0.3788 | 0.4934 | 0.3074 | 7.869 |
| M >= 8.0 | 0.1609 | 0.0921 | 0.6364 | 13.808 |

**Implikasi:** model kalah pada F1 dari aturan `M ≥ 7 & depth ≤ 70`, tetapi menang pada biaya (recall tinggi, FN 100× lebih mahal). Jangan klaim akurasi lebih tinggi.

## 6. Evaluasi tambahan (phase 5)

Kalibrasi (test): XGBoost Brier 0.0016/ECE 0.0015; LogisticRegression Brier 0.0121/ECE 0.0369; RandomForest Brier 0.0011/ECE 0.0004; MLP Brier 0.0012/ECE 0.0004.

Imbalance (F1, recall): Original 0.2178/0.6513; Random Oversampling 0.2439/0.6579; SMOTE 0.2152/0.6513; ADASYN 0.2010/0.6316; Random Undersampling 0.0705/0.9211.

Ablasi fitur (**time-based split** train≤2015 / test≥2016, bukan random split — prevalensi test lebih rendah, jangan dibandingkan langsung dengan §2): A: All Features (24)|XGBoost = 0.5212; A: All Features (24)|XGB (th=0.36) = 0.5014; A: All Features (24)|Soft Voting = 0.5399; B: No monitoring_quality (23)|XGBoost = 0.4972; B: No monitoring_quality (23)|XGB (th=0.36) = 0.4700; B: No monitoring_quality (23)|Soft Voting = 0.5220; C: No Quality Features (20)|XGBoost = 0.4594; C: No Quality Features (20)|XGB (th=0.36) = 0.4056; C: No Quality Features (20)|Soft Voting = 0.4898.

- random_split XGB: F1 0.2178, recall 0.6513, precision 0.1308, cost 5.958
- time_based_split XGB: F1 0.5212, recall 0.6466, precision 0.4365, cost 4.811

Ensemble 5g (soft voting @0,36): F1 0.2494, recall 0.6645, cost 5.657.

Analisis error (test 79.150 baris, 133 tsunami): FN 44.

## 7. Angka lama yang tidak boleh dikutip

| Angka lama | Kenapa salah | Ganti dengan |
|---|---|---|
| XGB F1 0,2260 | `scale_pos_weight` 371 (label lama 833) | 0,2178 |
| Positif 833 / 310.369 baris | label belum dibatasi cause = 1 | 761 / 310.297 |
| Ambang 0,22 atau 0,94 sebagai ambang produk | 0,22 hardcode; 0,94 in-sample | `model/thresholds.json` |
| MAPE 319–403%, R² 0,26/−0,16 | regresi meter sudah tidak dipakai | QWK 0,4667 |
| 92 TP / 49 FN | threshold 0,5 dan data sebelum filter | 5i terbaru (89 TP / 44 FN @0,36) |

