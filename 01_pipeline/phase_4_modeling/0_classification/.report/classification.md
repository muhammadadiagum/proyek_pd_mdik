# Classification — Phase 4, Peran 1 (4 Models × 5 Imbalance Methods)
> ⚠️ **DOKUMEN HISTORIS.** Angka di bawah berasal dari pipeline sebelum definisi label
> final (761 positif) dan sebelum perbaikan F-05/F-11/F-25. Sumber kebenaran: `results/*.json`
> hasil eksekusi 3 Nov 2026, `03_results/ANGKA_FINAL.md`, dan
> `01_pipeline/phase_6_deployment/README.md`.


> Terakhir diperbarui: 2026-09-04
> Script: `0_classification/1_modeling.py`, `0_classification/2_visualization.py`
> Output: `0_classification/results/model_results.json` (ditimpa run 3 Nov; angka dokumen ini run 4 Sep)
> Eksekusi: WSL2 (12GB RAM) via network drive — VM lokal hanya 5.2GB
> Struktur: phase_4_modeling kini memiliki 2 subfolder peran — `0_classification/` (file ini) dan `1_estimation/`.

---

## 1. Ringkasan

BAGIAN INI menunjukkan peran **Klasifikasi** (trigger tsunami). Perbandingan **4 models dari 4 algorithm family** dengan **5 imbalance handling methods** (20 kombinasi). Peran estimation (ketinggian air) diadaptasi di `1_estimation/`.

**Keputusan model final** (setelah eksperimen v2 algoritma modern):

| Family | Model Terpilih | Alasan |
|--------|---------------|--------|
| Linear | Logistic Regression | Baseline interpretable |
| Bagging | Random Forest | Unggul di Original data |
| Boosting | XGBoost | Best cost-sensitive |
| Neural Net | MLP | Best F1 keseluruhan |

SVM diuji empiris lalu dieksklusi (lihat §6).

---

## 2. Konfigurasi Models

| Model | Parameter Kunci |
|-------|-----------------|
| LogisticRegression | max_iter=2000, class_weight='balanced', C=1.0, L2 |
| RandomForest | n_estimators=200, max_depth=15, n_jobs=-1 |
| XGBoost | n_estimators=200, max_depth=6, lr=0.1, scale_pos_weight=371 |
| MLP | hidden=(128,64,32), max_iter=500, early_stopping |

Scaling (StandardScaler) diterapkan pada LR dan MLP.

---

## 3. Konfigurasi Imbalance Methods

| Method | Rows Train | Mekanisme |
|--------|-----------|-----------|
| Original | 248,295 | Data asli 1:371 + class_weight di model |
| Random Oversampling | 495,114 | Duplikasi minority ×267 |
| SMOTE | 495,114 | Interpolasi synthetic minority |
| ADASYN | 495,114 | Synthetic fokus di region sulit |
| Random Undersampling | 979 | Buang majority sampai seimbang |

Split: train_test_split 80/20 stratified, random_state=42.
Test set: 62,074 rows (167 tsunami).

---

## 4. Hasil Lengkap — F1-Score

| Method | LR | RF | XGBoost | MLP |
|--------|-----|-----|---------|-----|
| **Original** | 0.0699 | 0.2967 | **0.2189** | **0.3937** ⭐ |
| Rand Oversampling | 0.0700 | 0.1924 | 0.1120 | 0.3139 |
| SMOTE | 0.0844 | 0.2870 | 0.1419 | 0.2659 |
| ADASYN | 0.0824 | 0.2768 | 0.1438 | 0.2734 |
| Rand Undersampling | 0.0671 | 0.0643 | 0.0285 | 0.0573 |

### Detail per Model (Original)

| Model | F1 | Recall | Precision | AUC | Cost |
|-------|-----|--------|-----------|-----|------|
| LogisticRegression | 0.0699 | 92.8% | 3.6% | 0.9763 | 5,313 |
| RandomForest | 0.2967 | 18.6% | 73.8% | 0.9745 | 13,611 |
| XGBoost | 0.2189 | 75.5% | 12.8% | 0.9682 | 4,958 |
| MLP | 0.3937 | 29.9% | 57.5% | 0.9726 | 11,737 |

---

## 5. Temuan Utama

1. **Original menang untuk 3 dari 4 models** (RF, XGB, MLP).
   Untuk imbalance ekstrem 1:371, resampling TIDAK membantu;
   class_weight/scale_pos_weight sudah cukup.

2. **Undersampling menghancurkan semua models**
   (F1 turun 50-85%). Membuang 99.7% majority = kehilangan
   informasi kritis.

3. **SMOTE/ADASYN hanya membantu LR** (satu-satunya model yang
   F1-nya naik dengan resampling: 0.070 → 0.084). Efeknya kecil
   dan tidak mengubah ranking.

4. **Trade-off Recall vs Precision sangat tajam:**
   - LR: recall 92.8% tapi precision 3.6% (5,081 false alarm)
   - MLP: precision 57.5% + recall 29.9% (balanced); RF: precision 73.8% tapi recall 18.6%
   - XGBoost: cost terendah (4,958) dengan recall 75.5% — best cost-sensitive

5. **AUC semua > 0.95** — ranking probabilistik semua models baik;
   tantangan utamanya di threshold/calibration, bukan diskriminasi.

---

## 6. Eksklusi SVM (Justifikasi Empiris)

SVM (RBF, probability=True) diuji terbatas sebelum dieksklusi:

| Kondisi | Rows | Waktu | F1 |
|---------|------|-------|-----|
| Original | 248k | 44.7 menit | 0.1060 |
| Random Oversampling | 495k | >3 jam (dihentikan) | — |

Kompleksitas O(n²)-O(n³) membuat SVM tidak praktis di dataset ini.
Hasil SVM F1=0.106 tidak memenuhi threshold minimum.

---

## 7. Riwayat Eksperimen

| Eksperimen | Hasil | Keputusan |
|------------|-------|-----------|
| SVM (RBF) | F1=0.106, 44+ min/fit | Dieksklusi |
| ElasticNet | ≈ LR (tidak signifikan) | Tidak dipakai |
| BalancedRF | -52% F1 vs RF | Tidak dipakai |
| CatBoost | -25% F1 vs XGBoost | Tidak dipakai |

Semua algoritma "modern" tidak mengungguli baseline klasik (LR, RF, XGBoost, MLP).

---

## 8. Plots

| Plot | Deskripsi |
|------|-----------|
| model_comparison_metrics.png | F1 / Recall / Precision per model (Original) |
| model_comparison_auc.png | AUC-ROC per model |
| model_comparison_cost.png | Cost per model (FN×100 + FP×1) |
| model_heatmap_f1.png | Heatmap F1: 4 models × 5 methods |

---

## 9. Struktur File

```
phase_4_modeling/
├── README.md                        ← deskripsi 2 peran
├── 0_classification/                ← PERAN 1: klasifikasi (file ini)
│   ├── 1_modeling.py                ← Training script (4 models, tanpa SVM)
│   ├── 2_visualization.py           ← Plot generator (PNG + SVG)
│   ├── results/
│   │   └── model_results.json       ← HASIL FINAL (4×5)
│   ├── plots/
│   │   ├── model_comparison_metrics.png + .svg
│   │   ├── model_comparison_auc.png + .svg
│   │   ├── model_comparison_cost.png + .svg
│   │   └── model_heatmap_f1.png + .svg
│   ├── .report/
│   │   └── classification.md        ← File ini (renamed dari modeling.md)
│   └── .script_logs/
│       ├── 1_modeling.py.log
│       └── 2_visualization.py.log
└── 1_estimation/                    ← PERAN 2: estimasi ketinggian air (lihat .report/estimation.md)
```

> **Catatan konsistensi:** struktur lama `phase_4_modeling/1_modeling.py` (root) + `results/` dipindah ke `0_classification/` saat adaptasi peran estimation. Path `DATA_PATH` di `0_classification/1_modeling.py` = `../../phase_3_data_preparation/prepared_dataset.csv`.

---

## 10. Rekomendasi untuk Phase 5

- Gunakan **Original method** sebagai konfigurasi evaluasi utama
- XGBoost = kandidat deployment (cost terendah + recall seimbang)
- MLP = kandidat utama (F1 tertinggi 0.3937, precision 57.5%, recall 29.9%)
- Threshold tuning (Phase 5) penting: AUC tinggi + F1 rendah
  menandakan threshold default 0.5 sub-optimal

## 11. Hubungan dengan Peran Estimation (Chained)

Classification bertindak sebagai **gerbang** pada pipeline chained:
1. Klasifikasi menentukan apakah suatu gempa memicu tsunami (label 0/1)
2. Jika label=1, model estimation (peran 2, di `1_estimation/`) memperkirakan ketinggian air
3. Hasil classification di sini menjadi input gate bagi estimation di phase_6 deployment