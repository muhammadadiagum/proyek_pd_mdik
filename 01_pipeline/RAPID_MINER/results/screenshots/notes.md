# Catatan Screenshot Parameter Operator

> Dibuat: 4 November 2026
> Folder: `01_pipeline/RAPID_MINER/results/screenshots/`

## Status

Screenshot parameter untuk operator utama sudah lengkap. Ada **3 operator yang
panel parameternya belum di-screenshot**:

| # | Operator | Status bukti | Prioritas |
|---|---|---|---|
| 1 | `Numerical to Polynominal` | Panel parameter belum ada | Rendah |
| 2 | `Apply Model` | Panel parameter belum ada | Rendah |
| 3 | `Performance (Classification)` | Panel konfigurasi belum ada; **hasil evaluasi sudah ada** di `11a`–`11e` | Rendah |

Yang **sudah** ada untuk ketiganya:

- `Numerical to Polynominal` — perannya terlihat di `01_canvas_clean.png` dan
  XML process: `attribute = label`, tipe `numeric` → nominal.
- `Apply Model` — terlihat di `01_canvas_clean.png`; di process ini tidak ada
  parameter yang diisi (`application_parameters` kosong).
- `Performance (Classification)` — hasil lengkap ada di:
  - `11a_performance_accuracy.png`
  - `11b_performance_kappa.png`
  - `11c_performance_weighted_recall.png`
  - `11d_performance_weighted_precision.png`
  - `11e_performance_text.png`

## Kenapa belum di-screenshot

Screenshot diprioritaskan untuk operator yang parameternya dapat mengubah
validitas hasil: `Read CSV`, `Select Attributes`, `Nominal to Numerical`,
`Set Role`, `Split Data`, dan `Random Forest`.

Tiga operator di atas tidak masuk prioritas karena:

1. `Numerical to Polynominal` hanya mengubah tipe `label`; tidak ada tuning.
2. `Apply Model` tidak punya parameter yang diisi pada process ini.
3. `Performance (Classification)` sudah dibuktikan lewat hasil metrik dan
   confusion matrix, bukan hanya panel konfigurasinya.

Ini **bukan blocker** untuk laporan. Catatan ini dibuat agar tidak ada klaim
bahwa semua panel parameter operator sudah didokumentasikan.

## Kalau ingin dilengkapi

Silakan screenshot panel parameter ketiga operator tersebut, lalu simpan
dengan nama:

- `12_numerical_to_polynominal.png`
- `13_apply_model.png`
- `14_performance_params.png`

Setelah ditambahkan, perbarui checklist di:

- `01_pipeline/RAPID_MINER/README.md`
- `01_pipeline/RAPID_MINER/HASIL_RAPIDMINER.md`

## Bukti yang sudah tersedia

- Process final: `01_pipeline/RAPID_MINER/proyek_akhir.rmp`
  (md5 `75527d4167c965d72025b57053e13f88`)
- Output: `results/rapidminer_predictions.csv`
  (md5 `3980490417bb7015ec0c331adcd0db4f`)
- Metrik: `results/rapidminer_metrics.json`
- Screenshot: `01_canvas_clean.png`, `04`–`11e`
