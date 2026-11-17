# Evaluation — Phase 5
> ⚠️ **DOKUMEN HISTORIS.** Angka di bawah berasal dari pipeline sebelum definisi label
> final (761 positif) dan sebelum perbaikan F-05/F-11/F-25. Sumber kebenaran: `results/*.json`
> hasil eksekusi 3 Nov 2026, `03_results/ANGKA_FINAL.md`, dan
> `01_pipeline/phase_6_deployment/README.md`.


> Terakhir diperbarui: 2026-09-04
> Dataset: 833 positive tsunami, 1:371 imbalance (tolerance ±0.25°)
> Model: konfigurasi final Phase 4 (LR max_iter=2000, RF tanpa class_weight, XGB spw=371, MLP 128-64-32)
> Split: random 80/20 stratified (train 248,295 / test 62,074; test tsunami=167)

---

## Daftar Skrip & Output

| # | Skrip | Analisis | Output JSON | Plots |
|---|-------|----------|-------------|-------|
| 1 | 5a_evaluation.py | SHAP + Threshold + Regional | evaluation_results.json | shap ×5, threshold, regional |
| 2 | 5b_baseline_match.py | Baseline aturan + match rate | baseline_match_results.json | — |
| 3 | 5c_ensemble.py | Voting ensemble | ensemble_results.json | — |
| 4 | 5d_statistical_tests.py | McNemar + bootstrap CI | statistical_results.json | — |
| 5 | 5e_imbalance.py | Perbandingan imbalance method (XGB) | imbalance_results.json | imbalance_analysis.png |
| 6 | 5f_time_based_split.py | Random vs time-based split | time_based_split_results.json | time_based_split_comparison.png |
| 7 | 5g_ablation_study.py | Ablation fitur quality | ablation_results.json | ablation_study.png |
| 8 | 5h_era_analysis.py | Performa per era | era_analysis_results.json | era_analysis.png |
| 9 | 5i_error_analysis.py | False Negative analysis | error_analysis_results.json | error_analysis.png |
| 10 | 5j_calibration.py | Brier + ECE | calibration_results.json | calibration_analysis.png |

---

## 1. SHAP Feature Importance (XGBoost)

| Rank | Fitur | mean \|SHAP\| |
|------|-------|--------------|
| 1 | mag | 3.1719 |
| 2 | depth_mag_ratio | 1.3113 |
| 3 | abs_latitude | 0.5557 |
| 4 | longitude | 0.5261 |
| 5 | latitude | 0.4235 |
| 6 | gap | 0.3847 |
| 7 | monitoring_quality | 0.3721 |
| 8 | rms | 0.3508 |
| 9 | depth | 0.2829 |
| 10 | dmin | 0.1906 |

**Interpretasi:** Magnitude dominan absolut (2.4x fitur kedua).
`depth_mag_ratio` tetap fitur engineered terkuat kedua — kontribusi
novelty feature engineering. Fitur monitoring individual (gap, rms,
dmin) masuk 10 besar; `abs_latitude` (fitur baru pengganti
`distance_to_coast_proxy`) masuk top-3. monitoring_quality di
posisi 7 — meskipun ablation menunjukkan noise (§8).

---

## 2. Threshold Tuning (XGBoost)

| Target | Threshold | F1 | Recall | Cost |
|--------|-----------|-----|--------|------|
| Best F1 | 0.94 | 0.4310 | 53.3% | 7,957 |
| Best Cost | 0.22 | 0.1389 | 84.4% | 4,322 |

Default t=0.5 bukan optimum: turunkan ke ~0.22 untuk operasional
cost-sensitive, atau naikkan ke ~0.94 untuk precision-oriented.

> **Klarifikasi threshold (2026-09-04):** Keputusan produksi/deployment =
> **0.22** (Best Cost, recall 84.4%, cost 4,322) — dipakai phase_6
> (`predict.py`, metadata, deployment report) & WEB_APP (slider default 0.22).
> Nilai **0.36** yang muncul di beberapa skrip phase_5 (5g_ablation,
> 5h_era, 5i_error, dan tabel ensemble di file ini) adalah **varian
> eksperimen untuk studi pembanding**, BUKAN konfigurasi produksi —
> bukan threshold yang di-deploy. Begitu juga 0.94 (Best F1) hanya acuan
> precision-oriented. Jika melihat `XGB (th=0.36)` pada hasil studi
> ablation/era/error, itu adalah comparator, bukan batas keputusan operasional.

---

## 3. Baseline Aturan Sederhana + Match Rate

| Baseline | F1 | Recall | Precision | Cost |
|----------|-----|--------|-----------|------|
| M ≥ 6.0 | 0.0955 | 87.4% | 5.1% | 24,182 |
| M ≥ 7.0 | 0.3508 | 50.9% | 26.8% | 42,060 |
| M ≥ 7.0 & depth ≤ 70 km | 0.3888 | 49.2% | 32.1% | 43,166 |

Match rate NOAA↔USGS: 833/1,660 event modern (50.2%); 346 tsunami
non-seismic, 1,311 seismic.

**Catatan penting:** baseline M≥7&shallow punya F1 tinggi (0.389)
TAPI cost-nya 3.7x model ML (43,166 vs 11,737). ML unggul pada
efisiensi false alarm, bukan sekadar F1.

---

## 4. Ensemble (Random Split)

| Model | F1 | Recall | Cost |
|-------|-----|--------|------|
| MLP (single best) | **0.3937** ⭐ | 29.9% | 11,737 |
| Soft Voting (LR+XGB+RF+MLP) | 0.2585 | 74.9% | 4,875 |
| Hard Voting | 0.2258 | 75.5% | 4,923 |
| XGB th=0.36 | 0.1748 | 80.2% | 4,532 |
| RandomForest | 0.2967 | 18.6% | 13,611 |
| XGBoost | 0.2189 | 75.5% | 4,958 |
| LogisticRegression | 0.0699 | 92.8% | 5,313 |

MLP single > semua ensemble. Soft voting menaikkan F1 vs XGB
tunggal (+0.04) dengan cost serupa, namun tetap di bawah MLP
single-model. MLP unggul di F1 karena precision tinggi (57.5%).

---

## 5. Statistical Tests

### McNemar (semua pasangan, random split)

| Pasangan | χ² | p | Signifikan |
|----------|-----|---|------------|
| XGB vs LR | 3,101.0 | <0.001 | *** |
| XGB vs RF | 598.7 | <0.001 | *** |
| XGB vs MLP | 613.0 | <0.001 | *** |
| LR vs RF | 3,742.7 | <0.001 | *** |
| LR vs MLP | 3,769.6 | <0.001 | *** |
| RF vs MLP | 0.7 | 0.4098 | ns |

Semua perbedaan antar-model signifikan kecuali **RF vs MLP**
(p=0.41, tidak signifikan). Keduanya model precision-oriented
dengan perilaku prediksi mirip.

### Bootstrap 95% CI (F1, n=100)

| Model | Mean | CI Low | CI High |
|-------|------|--------|---------|
| MLP | 0.3942 | 0.3124 | 0.4586 |
| RandomForest | 0.2921 | 0.2030 | 0.3603 |
| XGBoost | 0.2187 | 0.1835 | 0.2516 |
| LogisticRegression | 0.0707 | 0.0606 | 0.0821 |

> CI bersifat stokastik (bootstrap resampling, n=100) — nilai di atas dari
> run 2026-09-04 dan dapat sedikit bervariasi antar run (Δ ≤ ~0.01) akibat
> XGB versi library. CI MLP dan RF saling tumpang tindih → konsisten dengan McNemar ns.

---

## 6. Imbalance Methods (XGBoost)

| Method | Train Size | F1 | Recall | Cost |
|--------|-----------|-----|--------|------|
| **Original** | 248,295 | **0.2189** ⭐ | 75.5% | 4,958 |
| Random Oversampling | 495,114 | 0.1154 | 85.6% | 4,569 |
| SMOTE | 495,114 | 0.0783 | 84.4% | 5,893 |
| ADASYN | 495,114 | 0.0804 | 85.0% | 5,725 |
| Random Undersampling | 979 | 0.0285 | 96.4% | 11,584 |

**Kesimpulan:** untuk imbalance ekstrem 1:371, resampling TIDAK
membantu XGBoost — `scale_pos_weight=371` pada data original sudah
optimal. Undersampling menghancurkan performa (buang 99.7% informasi).

---

## 7. Time-Based Split Validation ⭐

Train ≤2015 (231,211 rows), Test ≥2016 (79,158 rows / 141 tsunami).

| Model | F1 Random | F1 Time-Based | Δ |
|-------|-----------|---------------|---|
| MLP | 0.3937 | **0.4549** ⭐ | +0.061 |
| Soft Voting | 0.2256 | 0.4545 | +0.229 |
| XGBoost | 0.2189 | 0.4533 | +0.234 |
| XGB (th=0.36) | 0.1748 | 0.4423 | +0.268 |
| LogisticRegression | 0.0695 | 0.1733 | +0.104 |
| RandomForest | 0.2967 | 0.2073 | −0.089 |

**Interpretasi:** F1 naik signifikan untuk XGB (+107%) dan Soft
Voting (+101%) di temporal split. MLP naik lebih moderat (+15%)
karena sudah unggul di random split. RF satu-satunya yang turun
(−30%) — precision-oriented-nya tidak adaptif terhadap distribusi
test temporal. XGB, MLP, dan Soft Voting konvergen ke F1 ~0.45
di temporal split.

---

## 8. Ablation Study (Time-Based Split)

| Varian | XGB F1 | XGB th=0.36 | SoftVoting F1 |
|--------|--------|-------------|----------------|
| A: All features (24) | 0.4533 | 0.4423 | 0.4840 |
| B: −monitoring_quality (23) | **0.4899** (+0.037) | 0.4658 | **0.5321** (+0.048) |
| C: −nst/gap/dmin/rms (20) | 0.4113 (−0.042) | 0.3589 | 0.4706 |

**Kesimpulan:**
- Composite score `monitoring_quality` = NOISE (F1 naik saat dihapus)
- Fitur monitoring INDIVIDUAL (nst, gap, dmin, rms) berguna (F1 turun saat dihapus)
- Revisi novelty claim: *"individual earthquake monitoring features"*
  (bukan composite score)

---

## 9. Temporal Era Analysis

| Era | n | Tsunami | XGB F1 | RF F1 | MLP F1 |
|-----|---|---------|--------|-------|--------|
| 1900-1930 | 3,336 | 74 | 0.175 | 0.577 | 0.217 |
| 1931-1960 | 7,233 | 120 | 0.174 | 0.588 | 0.331 |
| 1961-1990 | 74,684 | 232 | 0.305 | 0.619 | 0.361 |
| 1991-2015 | 145,958 | 266 | 0.504 | 0.836 | 0.594 |
| 2016-2026 (test) | 79,158 | 141 | 0.453 | 0.207 | 0.455 |

**Temuan:**
- XGB & MLP: F1 meningkat gradual dari era lama ke modern
  (0.17 → 0.50) — kualitas katalog modern membantu.
- **RF konsisten tinggi di era 1900-2015 (F1 0.58-0.84) tapi
  AMBRUK di 2016+ (0.836 → 0.207)** — profil overfit ke era
  lama, tidak generalis ke data modern terkini.
- MLP & XGB konvergen ke ~0.45 di 2016+ — model yang lebih
  adaptif terhadap distribusi data terbaru.

---

## 10. Error Analysis — False Negatives (Test 2016+)

Dari 141 tsunami di test set: **92 TP, 49 FN** (threshold default 0.5).

| Aspek | FN | TP |
|-------|-----|-----|
| Magnitude mean | 5.88 | 7.18 |
| Magnitude median | 5.90 | 7.20 |
| Depth mean | 32.8 km | 26.1 km |
| Monitoring quality | 0.702 | 0.752 |

Prediction probability: FN mean=0.073, median=0.012;
**36/49 FN (73%) prob < 0.05** — model confident salah pada
sebagian besar FN.

**Interpretasi:** FN didominasi gempa KECIL (M4.5-6.0) yang memicu
tsunami non-karakteristik. Magnitude adalah pembeda utama (FN mean
5.88 vs TP mean 7.18). Depth & monitoring quality bukan pembeda
utama. Ini limitasi fundamental fitur seismik terhadap tsunami
small-magnitude — layak dibahas sebagai limitation di paper.

---

## 11. Calibration

| Model | Brier ↓ | ECE ↓ | Pred. mean | Aktual |
|-------|---------|-------|------------|--------|
| RandomForest | **0.0012** | **0.0005** ⭐ | 0.15% | 0.18% |
| MLP | **0.0012** | **0.0005** ⭐ | 0.17% | 0.18% |
| XGBoost | 0.0020 | 0.0025 | 0.43% | 0.18% |
| LogisticRegression | 0.0124 | 0.0413 | 4.30% | 0.18% |

RF & MLP kalibrasi sempurna (ECE=0.0005). XGBoost sedikit
over-predict (0.43% vs 0.18%). LR sangat overconfident
~23x (prediksi 4.30% vs aktual 0.18%).

---

## 12. Regional Analysis (XGBoost)

| Region | Train Size | Test Tsunami | F1 | Recall | AUC |
|--------|-----------|-------------|-----|--------|-----|
| Pacific Ring of Fire | 240,889 | 699 | 0.2383 | 19.3% | 0.9138 |
| Mediterranean-Asian Belt | 22,304 | 96 | 0.2199 | 43.8% | — |
| Other | 32,933 | 30 | 0.0627 | 40.0% | — |
| Mid-Atlantic Ridge | 14,243 | 8 | 0.0437 | 62.5% | — |

F1 turun drastis di region minoritas — generalisasi antar-region
adalah kelemahan utama, konsisten sebagai motivation untuk
multi-source data.

---

## 13. Pemetaan Novelty → Bukti Empiris

| Novelty Claim | Didukung oleh |
|---------------|---------------|
| #1 Multi-source integration (NOAA+USGS) | §3 match rate 50.2%; seluruh pipeline |
| #2 Feature engineering (depth_mag_ratio dll.) | §1 SHAP rank-2 (1.31); §8 ablation C |
| #3 Imbalance ekstrem 1:371 handling | §6 Original > semua resampling; §4 Phase 4 |
| #4 Explainable ML (SHAP) | §1 ranking fitur fisik-consistent |
| RQ4 Generalisasi | §7 time-based +2x; §12 regional drop; §9 era |

---

## 14. Kesimpulan Evaluasi

1. **Konfigurasi final valid**: semua temuan inti reproducible
   dengan dataset 833/1:371 (tolerance 0.25°).
2. **Model rekomendasi deployment**: MLP (F1 tertinggi 0.394
   random / 0.455 temporal, Brier 0.0012, ECE 0.0005) atau
   XGBoost cost-sensitive (cost terendah 4,958, recall 75%).
3. **Temporal split reveals true performance**: F1 ~0.45 untuk
   XGB/MLP/SoftVoting di test 2016+ — lebih realistis daripada
   random split.
4. **monitoring_quality = noise**: composite score tidak membantu;
   fitur monitoring individual (nst, gap, dmin, rms) tetap berguna.
5. **RF overfit era lama**: F1 0.84 di 1991-2015 tapi ambruk
   ke 0.21 di 2016+ — tidak recommended untuk deployment.
6. **Limitasi jujur untuk paper**: FN small-magnitude (§10),
   regional drop (§12), LR overconfident (§11).