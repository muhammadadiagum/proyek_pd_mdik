# Phase 4 — Estimation (Regresi Ordinal Tinggi Air Tsunami)

**Status:** diimplementasikan dan dieksekusi · 3 November 2026 · runtime 16,0 detik
**Keputusan:** peran estimasi dipertahankan, tetapi diubah dari regresi kontinu menjadi
**regresi ordinal 3 kelas** (opsi 2 pada diskusi 3 Nov 2026).
**Sumber angka:** `results/estimation_ordinal_results.json`.

---

## 1. Mengapa regresi kontinu diganti

Regresi kontinu pada proyek lama menghasilkan (holdout 122 baris, `estimation_results.json`):

| Model | MAE | R² |
|---|---|---|
| LinearRegression | 0,99 m | 0,2576 |
| MLP | 1,06 m | 0,1605 |
| RandomForest | 1,07 m | **−0,1646** |
| XGBoost | 1,22 m | −1,2232 |

Baseline "tebak 0,3 m untuk semua gempa" pada split yang sama: **MAE 1,019 m**.
Artinya LR hanya ~2% lebih baik dari konstanta, dan RF yang dipakai di deployment
justru lebih buruk. Penyebabnya bukan algoritma, melainkan target: tinggi air diukur
di pantai/stasiun, dipengaruhi batimetri dan jarak ke pantai — informasi yang tidak ada
pada fitur gempa.

Karena itu target diubah menjadi kelas berurutan yang punya makna operasional.

---

## 2. Definisi kelas

Ambang mengikuti tahapan peringatan dini tsunami BMKG/InaTEWS
(AWAS > 3 m; SIAGA 0,5–3 m; WASPADA < 0,5 m):

| Kelas | Nama | Aturan | n | Porsi |
|---|---|---|---|---|
| 0 | Waspada | tinggi ≤ 0,5 m | 368 | 60,3% |
| 1 | Siaga | 0,5 < tinggi ≤ 3,0 m | 158 | 25,9% |
| 2 | Awas | tinggi > 3,0 m | 84 | 13,8% |

Sampel: 610 tsunami akibat gempa (`cause_code = 1`) yang memiliki `noaa_water_height`,
dari **761 tsunami cause=1** pada dataset final (833 sebelum filter cause; 72 event cause
lain dibuang). Event non-seismik (longsor, vulkanik, meteorologis) berada di luar domain
"tsunami dipicu gempa". Median tinggi air 0,3 m; maksimum 50,9 m.

---

## 3. Metode

**Regresi ordinal via dekomposisi biner kumulatif** (`ordinal.py`), K−1 = 2 model biner:

```
P(Y > 0) = P(Siaga atau Awas)   -> model biner #1
P(Y > 1) = P(Awas)              -> model biner #2

P(Y = 0) = 1 − P(Y > 0)
P(Y = 1) = P(Y > 0) − P(Y > 1)
P(Y = 2) = P(Y > 1)
```

Dua model dilatih terpisah sehingga monotonisitas `P(Y>0) ≥ P(Y>1)` bisa dilanggar;
probabilitas kumulatif dipaksa turun monoton lewat *cumulative minimum* sebelum diubah
menjadi probabilitas kelas.

Mengapa bukan klasifikasi multikelas: multikelas membuang urutan kelas dan memberi
penalti yang sama untuk kesalahan "Awas → Siaga" dan "Awas → Waspada". Regresi ordinal
memakai urutan itu langsung, dan setiap model biner dilatih pada seluruh 610 baris
(bukan hanya subset kelas), sehingga lebih stabil pada data kecil.

Pembelajar dasar: **LogisticRegression, RandomForest, XGBoost, MLP**
(4 model, sama dengan fase klasifikasi). Pembanding: skema **multikelas** dengan
pembelajar yang sama, untuk menguji apakah ordinalitas benar-benar membantu.

Fitur: 24 kolom — 11 numerik (`mag`, `depth`, `nst`, `gap`, `dmin`, `rms`, `latitude`,
`longitude`, `abs_latitude`, `depth_mag_ratio`, `monitoring_quality`) + 13 dummy
(`tectonic_zone`, `era`, `season`). Tidak ada kolom `noaa_*` (tidak bocor).

---

## 4. Protokol evaluasi

- **5-fold stratified CV** (shuffle, seed 42), metrik dihitung pada out-of-fold.
- **Holdout 20% stratified** (122 baris) — hanya untuk konfirmasi; pemilihan model
  berdasarkan CV.
- Metrik: **QWK** (quadratic weighted kappa, metrik utama ordinal), **MAE kelas**
  (jarak 0/1/2), akurasi tepat, akurasi ±1 kelas, presisi/recall/F1 per kelas,
  confusion matrix.
- Baseline: kelas mayoritas dan regresi kontinu lama yang diambangkan ke kelas.

---

## 5. Hasil 5-fold CV (mean ± sd antar fold)

| Model | Skema | QWK | MAE kelas | Tepat | ±1 kelas |
|---|---|---|---|---|---|
| LogisticRegression | **ordinal** | 0,394 ± 0,017 | 0,418 ± 0,030 | 0,659 | 0,923 |
| LogisticRegression | multikelas | 0,382 ± 0,027 | 0,428 ± 0,034 | 0,654 | 0,918 |
| RandomForest | **ordinal** | 0,408 ± 0,116 | **0,402 ± 0,032** | 0,674 | 0,925 |
| RandomForest | multikelas | 0,394 ± 0,108 | 0,416 ± 0,029 | 0,659 | 0,925 |
| XGBoost | **ordinal** | 0,410 ± 0,085 | 0,433 ± 0,031 | 0,639 | 0,928 |
| XGBoost | multikelas | **0,427 ± 0,046** | 0,420 ± 0,039 | 0,657 | 0,923 |
| MLP | **ordinal** | 0,201 ± 0,140 | 0,500 ± 0,042 | 0,616 | 0,884 |
| MLP | multikelas | 0,220 ± 0,086 | 0,482 ± 0,039 | 0,621 | 0,897 |

## 6. Hasil holdout 20% (n = 122)

| Model | Skema | QWK | MAE kelas | Tepat | ±1 kelas |
|---|---|---|---|---|---|
| LogisticRegression | ordinal | 0,247 | 0,451 | 0,656 | 0,893 |
| RandomForest | ordinal | 0,364 | 0,418 | 0,664 | 0,918 |
| **XGBoost** | **ordinal** | **0,467** | **0,402** | 0,639 | **0,959** |
| XGBoost | multikelas | 0,321 | 0,475 | 0,607 | 0,918 |
| MLP | ordinal | 0,183 | 0,533 | 0,615 | 0,853 |

## 7. Baseline (holdout)

| Baseline | QWK | MAE kelas | Tepat | ±1 kelas |
|---|---|---|---|---|
| Kelas mayoritas (Waspada) | 0,000 | 0,541 | 0,598 | 0,861 |
| Regresi kontinu lama + ambang | 0,275 | 0,533 | 0,484 | **0,984** |
| **XGBoost ordinal** | **0,467** | **0,402** | **0,639** | 0,959 |

Regresi lama hampir tidak pernah salah dua kelas (±1 = 98,4%) tetapi sering salah satu
kelas (tepat 48,4%) karena hampir selalu memprediksi Siaga. QWK-nya 0,275. Regresi
ordinal menaikkan QWK ke 0,41 (CV) dan 0,47 (holdout), dengan akurasi tepat 64–67%.

---

## 8. Per kelas — RandomForest ordinal, out-of-fold gabungan (n = 610)

| Kelas | Presisi | Recall | F1 | Support |
|---|---|---|---|---|
| Waspada | 0,71 | **0,90** | 0,79 | 368 |
| Siaga | 0,53 | 0,31 | 0,39 | 158 |
| Awas | 0,62 | 0,35 | 0,44 | 84 |

Confusion matrix (baris = aktual, kolom = prediksi):

```
              Waspada  Siaga  Awas
Waspada   [   333      25     10 ]
Siaga     [   101      49      8 ]
Awas      [    36      19     29 ]
```

Kesalahan dominan: **36 dari 84 Awas (43%) diprediksi Waspada** — dua tingkat terlalu
rendah. Ini kelemahan paling serius dan tidak boleh disembunyikan.

---

## 9. Aturan keputusan sadar-biaya

`argmax` memilih probabilitas tertinggi, padahal melewatkan Awas jauh lebih mahal
daripada alarm palsu. Aturan peringatan (`ordinal.predict_warning`) menurunkan ambang:

```
Awas    jika P(Awas) ≥ 0,25
Siaga   jika P(Siaga) + P(Awas) ≥ 0,40
Waspada selainnya
```

Ambang dipilih dari out-of-fold CV, lalu diuji pada holdout:

| Model | Skema | QWK | MAE | Recall Awas | Recall Siaga | Presisi Awas |
|---|---|---|---|---|---|---|
| RandomForest | ordinal | 0,390 | 0,459 | 0,35 | 0,41 | 0,46 |
| XGBoost | ordinal | **0,432** | 0,443 | **0,41** | 0,38 | 0,50 |
| XGBoost | multikelas | 0,391 | 0,451 | 0,35 | 0,34 | 0,50 |
| MLP | ordinal | 0,034 | 1,205 | 0,76 | 0,06 | 0,14 |

Recall Awas naik (XGB 0,29 → 0,41; RF 0,24 → 0,35) dengan biaya MAE ~0,04 kelas.
MLP ordinal pada aturan ini degenerasi (memprediksi hampir semua Awas) — model itu
tidak dipakai.

---

## 10. Interpretasi

1. **Ordinalitas membantu, tapi tidak seragam.** Untuk LR dan RF, skema ordinal
   konsisten lebih baik dari multikelas (+0,012 dan +0,015 QWK di CV). Untuk XGB
   multikelas sedikit lebih baik di CV (−0,017), tetapi ordinal jauh lebih baik di
   holdout (+0,146). Untuk MLP keduanya buruk.
2. **Selisih antar model sebagian besar noise.** SD antar fold RF ±0,116 dan XGB ±0,085,
   sedangkan selisih rata-ratanya 0,01–0,02. Hanya LR yang stabil (±0,017), tetapi
   performanya paling rendah di holdout (0,247).
3. **QWK 0,39–0,43 tergolong "moderate"** (Landis & Koch: 0,41–0,60). Model menangkap
   sinyal nyata — jauh di atas mayoritas (0,000) dan regresi lama (0,275) — tetapi
   tidak cukup untuk dipakai sebagai satu-satunya dasar peringatan.
4. **Kesalahan umumnya satu tingkat** (akurasi ±1 = 92–96%), kecuali 43% Awas yang
   turun dua tingkat. Ini pola khas target dengan ekor panjang: model kekurangan
   informasi untuk membedakan kejadian ekstrem.
5. **Batas atas ditentukan fitur, bukan model.** Tinggi air di pantai dipengaruhi
   batimetri lokal, jarak ke garis pantai, orientasi teluk, dan pasang surut — tidak
   satu pun tersedia. Menambah model yang lebih rumit tidak akan menembus batas ini.

---

## 11. Implikasi untuk sistem

- Web/dashboard **tidak menampilkan satu angka meter**. Yang ditampilkan: kelas
  (Waspada/Siaga/Awas), probabilitas tiap kelas, dan kalimat ketidakpastian.
- Keputusan kelas memakai **aturan sadar-biaya**, bukan `argmax`, karena biaya
  melewatkan Awas tidak simetris.
- Model yang direkomendasikan: **XGBoost ordinal** (QWK holdout 0,467; aturan
  peringatan 0,432). Alternatif stabil: RandomForest ordinal (MAE CV terbaik 0,402).
- Model **wajib disertai disclaimer**: "perkiraan berbasis parameter gempa; tidak
  memperhitungkan batimetri dan jarak pantai".

---

## 12. Keterbatasan

1. n = 610; holdout hanya 122 baris, sehingga QWK holdout punya interval kepercayaan lebar.
2. Ambang kelas (0,5 m dan 3 m) mengikuti BMKG, bukan dioptimalkan dari data — memang
   disengaja agar kelas punya makna operasional.
3. Aturan sadar-biaya memakai ambang yang dipilih pada OOF CV; perlu validasi pada
   sampel lebih besar sebelum dipakai nyata.
4. Pembobotan kelas seimbang sempat diuji: QWK RF naik tipis (0,414 → 0,422) tetapi
   MAE memburuk (0,402 → 0,443), jadi tidak dipakai.
5. Split acak, bukan berbasis waktu. Evaluasi temporal ada di fase 5 proyek
   (leave-one-region-out / time-based) dan belum diulang untuk ordinal.

---

## 13. Reproduksi

```bash
cd 01_pipeline/phase_4_modeling/1_estimation
../../../.venv/bin/python 1_modeling.py
```

- Runtime: 16,0 detik (4 model × 2 skema × 5 fold + holdout + baseline)
- numpy 2.4.6 · pandas 3.0.5 · scikit-learn 1.9.0 · xgboost 3.2.0
- Input: `00_data/processed/modeling_dataset.csv` (310.297 × 26, md5 `d0695e96d0e31e6efa77a02cf8eeda28`)
- Output: `results/estimation_ordinal_results.json`
- Log: `.script_logs/1_modeling.py.log`
