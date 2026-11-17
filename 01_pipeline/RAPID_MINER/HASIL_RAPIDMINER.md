# RapidMiner Project — Hasil Eksperimen Final (arsip versi 1 model)

> ⚠️ **Arsip.** Dokumen ini merekam run `proyek_akhir.rmp` (Random Forest saja,
> 3–4 November 2026). Sejak 14 November 2026 angka laporan memakai
> `proyek_4model.rmp` (RF + LR + XGBoost + MLP); lihat README fase ini bagian
> *Process 4 Model* dan `03_results/rapidminer_4model_results.json`.

> **Run final: 3 November 2026; verifikasi ulang 4 November 2026** · `proyek_akhir.rmp`
> AI Studio 2026.1.1 (Windows host) · batch sukses dalam **4 menit 51 detik**
> (run verifikasi layout 4 Nov 2026)
> Dataset: `00_data/processed/modeling_dataset.csv` — **310.297 baris × 26 kolom**,
> **761 positif** (`label=1 & noaa_cause_code=1`), 309.536 negatif (1:406,6)
> Sumber angka: `results/rapidminer_predictions.csv` +
> `results/rapidminer_metrics.json` (dihitung dari output RapidMiner)
>
> Salinan Windows: `RAPIDMINER_LOCAL_REPO\`
> (process, dataset, output, dan log dalam satu folder).

---

## Arsitektur Process

```
Read CSV
  → Select Attributes (11 numerik + tectonic_zone/era/season + label)
  → Nominal to Numerical (dummy coding → 13 fitur)
  → Numerical to Polynominal (label)
  → Set Role (label)
  → Split Data (stratified 80/20, seed 42)
       ├── 80% → Random Forest (200 trees, gini, depth 15, no prepruning)
       └── 20% → Apply Model → Write CSV → Performance (Classification) → Result
```

- **24 fitur** (11 numerik + 13 dummy), sama dengan model Python.
- Kolom `time`, `magType`, `place`, dan semua `noaa_*` **tidak dipakai**.
- Tidak ada sampling/resampling: distribusi test tetap asli.
- 200 trees, `max_depth=15`, `minimal_leaf_size=1`, `minimal_gain=0.0`,
  `apply_prepruning=false`, seed 42.

---

## Hasil Test Set (62.059 baris; 152 tsunami)

| | Pred = 0 | Pred = 1 |
|---|---:|---:|
| **Aktual = 0** | 61.898 (TN) | 9 (FP) |
| **Aktual = 1** | 123 (FN) | 29 (TP) |

| Metrik | Nilai |
|--------|------:|
| Accuracy | 0,997873 (99,79%) |
| Precision kelas 1 | 0,7632 (76,32%) |
| Recall kelas 1 | 0,1908 (19,08%) |
| F1 kelas 1 | 0,3053 |
| AUC | 0,9566 |
| Kappa | 0,305 |

Interpretasi: model sangat konservatif. Dari 152 tsunami di test, 29 terdeteksi
dan 123 lolos; tetapi dari 38 prediksi positif, 29 benar (precision 76%).
Accuracy 99,79% hampir sama dengan baseline "selalu negatif" (99,76%), jadi
**accuracy bukan metrik utama** pada data ini.

---

## Perbandingan dengan Python (Random Forest)

| Metrik | RapidMiner `proyek_akhir` | Python RF (`model_results.json`) |
|--------|--------------------------:|---------------------------------:|
| TP / FP / FN / TN | 29 / 9 / 123 / 61.898 | 34 / 12 / 118 / 61.896 |
| F1 | 0,3053 | 0,3434 |
| Precision | 0,7632 | 0,7391 |
| Recall | 0,1908 | 0,2237 |
| AUC | 0,9566 | 0,9734 |
| Accuracy | 0,997873 | 0,997868 |

Kedua implementasi memberi pola yang sama (precision tinggi, recall rendah).
Selisih wajar karena:

1. implementasi Random Forest berbeda (RapidMiner vs scikit-learn);
2. split stratified seed 42 tidak menghasilkan baris yang identik;
3. scikit-learn memakai `max_features='sqrt'`, RapidMiner memakai
   `guess_subset_ratio` (≈ log2(24)+1);
4. penanganan nilai nominal berbeda (dummy coding eksplisit vs native).

Yang dinilai di laporan adalah **metodologi, evaluasi yang sah, dan analisis**,
bukan kesamaan angka antartool.

---

## Catatan Teknis / Jejak Debugging

1. `performance_binominal_classification` gagal connect port di environment ini;
   diganti `performance_classification` (accuracy, kappa, weighted
   precision/recall, confusion matrix). Metrik F1/AUC dihitung dari
   `rapidminer_predictions.csv`.
2. `Write CSV` harus memakai port input `input`, bukan `example set input`.
3. Satu output tidak boleh bercabang ke dua operator; alur dibuat serial:
   `Apply Model → Write CSV → Performance` (port `through`).
4. `use_weights` tidak dikenal oleh operator Random Forest di AI Studio
   2026.1.1; karena itu tidak ada class weighting. Model memakai distribusi
   asli dan prepruning dimatikan agar tetap bisa menemukan pola minoritas.
5. Batch memunculkan peringatan lisensi Altair (license server lokal tidak
   aktif), tetapi proses tetap berjalan sampai `Process finished successfully`.
   Untuk screenshot, jalankan lewat GUI.
6. **Filter nominal diperbaiki 3 Nov 2026:** `Nominal to Numerical` memakai
   format kanonik `|tectonic_zone|era|season`. Re-run pukul 19:09–19:15
   menghasilkan prediksi identik (md5 sama, F1 0,3053).
7. **Metadata Read CSV ditambahkan 3 Nov 2026 pukul 20:28:** 26 kolom
   didefinisikan di `data_set_meta_data_information` agar GUI mengenali nama
   dan tipe atribut sebelum Run. Batch ulang berhasil (run metadata 6 menit
   14 detik) dan
   prediksi tetap identik. **✅ Diverifikasi 3 Nov 2026:** warning “attribute
   filter does not match any attributes” / `label missing` hilang setelah
   process dibuka ulang; Performance Vector tetap sama.
8. **Process dirapikan di GUI dan disimpan ulang 4 Nov 2026** (md5
   `75527d4167c965d72025b57053e13f88`). Verifikasi: koneksi dan parameter
   substantif sama; batch berhasil dalam **4 menit 51 detik**, prediksi tetap
   identik (md5 `3980490417bb…`), TP/FP/FN/TN 29/9/123/61.898.

---

## Arsip Run Lama (jangan dikutip)

`UNNAMED-v1.rmp` (21 Agustus 2026) memakai `prepared_dataset.csv`, 833 positif
(termasuk tsunami non-seismik), split seed 1992, dan kolom `noaa_*`. Run 1
tanpa balancing menghasilkan 0 TP; run 2 dengan balanced sampling 750/700
menghasilkan recall 92,97% tetapi precision 3,97% (F1 0,0762). Angka itu
**historis** dan tidak sebanding dengan run final di atas.

---

## Checklist Bukti

- [x] Process final `proyek_akhir.rmp` — berhasil dijalankan via batch
- [x] Prediction output `results/rapidminer_predictions.csv`
- [x] Confusion matrix + metrik `results/rapidminer_metrics.json`
- [x] Screenshot canvas — `results/screenshots/01_canvas_clean.png`
- [x] Screenshot Performance Vector + Confusion Matrix —
      `02_performance_table.png`, `03_performance_text.png`, `11a`–`11e`
- [x] Screenshot parameter kunci — `04`–`10` (Read CSV, Select Attributes,
      Nominal to Numerical, Set Role, Split Data, Random Forest, Write CSV)

> **Arah tabel:** baris = prediksi, kolom = aktual. TN 61.898 · FP 9 ·
> FN 123 · TP 29. `weighted_mean_recall 59,53%` / `weighted_mean_precision
> 88,06%` memakai weights 1,1 (rata-rata makro), bukan support-weighted.
