# Deployment — Phase 6 (Pipeline Chained: Klasifikasi → Estimasi Ketinggian Air)
> ⚠️ **DOKUMEN HISTORIS.** Angka di bawah berasal dari pipeline sebelum definisi label
> final (761 positif) dan sebelum perbaikan F-05/F-11/F-25. Sumber kebenaran: `results/*.json`
> hasil eksekusi 3 Nov 2026, `03_results/ANGKA_FINAL.md`, dan
> `01_pipeline/phase_6_deployment/README.md`.


> Terakhir diperbarui: 2026-09-04
> Script: `6_deployment.py` (re-deploy), `predict.py` (di-generate ulang)
> Model: konfigurasi FINAL Phase 4 — identik dengan yang dievaluasi di Phase 5

---

## 1. Ringkasan

Phase 6 menyimpan model final untuk deployment + pipeline prediksi
CLI (`predict.py`). Sesuai target proyek (prediksi tsunami **dan** ketinggian airnya),
deployment kini berupa **pipeline chained**: klasifikasi tsunami sebagai gerbang,
lalu estimasi ketinggian air (meter) hanya bila prediksi = TSUNAMI.
Re-deploy dilakukan dengan spw=371 untuk konsistensi dengan dataset 833 positive (1:371, tolerance ±0.25°).

---

## 2. Konfigurasi Model Tersimpan

| Model | Parameter | Peran |
|-------|-----------|-------|
| xgboost.pkl | n_est=200, depth=6, lr=0.1, spw=371 | Default deployment (cost-sensitive) |
| logistic_regression.pkl | max_iter=2000, class_weight='balanced' | Recall-oriented |
| random_forest.pkl | n_est=200, depth=15, **tanpa** class_weight | Precision-oriented |
| mlp.pkl | (128,64,32), max_iter=500, early_stopping | Precision-oriented |
| scaler.pkl | StandardScaler (fit train) | Wajib untuk MLP |
| scaler_lr.pkl | StandardScaler (fit train) | Wajib untuk Logistic Regression |
| **estimator_rf.pkl** | RandomForestRegressor (200, depth 15) | **Estimasi ketinggian air (meter)** |
| feature_names.json | 24 fitur | Urutan fitur pipeline |
| metadata.json | versi config, threshold, dsb. | Dokumentasi model |

### Estimator ketinggian air (chained estimation)

- **Algoritma**: `RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)` — model estimation terbaik dari Phase 4
- **Target**: `noaa_water_height` (meter), **dilatih pada log1p(target)** lalu di-`expm1` balik ke meter saat inferensi (mengurangi skew)
- **Subset training**: **610** gempa tsunami yang memiliki `noaa_water_height` dan **cause_code=1 (earthquake)** (73.2% dari 833 tsunami) — event non-seismik (landslide 55, dam/misc dll.) dibuang karena di luar domain
- **Fitur**: 24 fitur gempa yang SAMA dengan klasifikasi — tidak ada data leak (`noaa_water_height`/`label` tidak masuk fitur)
- **Verifikasi (split 80/20, seed 42, metrik di skala meter)**: MAE **1.0720**, R² **-0.1646** — konsisten persis dengan Phase 4 estimation re-run
- **Peran di pipeline**: hanya dipanggil bila klasifikasi = TSUNAMI (gerbang chained)

---

## 3. Verifikasi Model Baru (Test Set Random Split)

| Model | F1 | Recall | AUC | Cost |
|-------|-----|--------|-----|------|
| xgboost | 0.2189 | 75.5% | 0.9682 | 4,958 |
| logistic_regression | 0.0699 | 92.8% | 0.9763 | 5,313 |
| random_forest | 0.2967 | 18.6% | 0.9745 | 13,611 |
| mlp | 0.3937 | 29.9% | 0.9726 | 11,737 |

Angka IDENTIK dengan Phase 4 → konsistensi lintas fase terjaga ✅

**Catatan:** LR dilatih dengan StandardScaler (scaler_lr.pkl) — konsisten
dengan Phase 4/5 yang menskalakan LR. Sebelumnya LR dilatih unscaled
(F1=0.0695); sekarang scaled (F1=0.0699) = match persis dengan Phase 4.

---

## 4. Threshold Deployment

Default `predict.py`: **threshold=0.22** (cost-optimized XGB dari Phase 5 §2).

| Threshold | Kapan dipakai |
|-----------|---------------|
| 0.22 | Operasional cost-sensitive (recall 84.4%, cost 4,322) |
| 0.36 | Alternatif seimbang (XGB tuning phase 5, cost 4,532) |
| 0.94 | Precision-oriented (F1 maksimum 0.43) |

---

## 4.5 Alur Pipeline Chained

```
Input gempa (mag, depth, lat, lon, ...)
        │
        ▼
┌───────────────────┐
│  Klasifikasi      │
│  (XGBoost default,│
│   threshold 0.22) │
└───────────────────┘
        │
    ┌───┴────┐
    ▼        ▼
 TSUNAMI   NO TSUNAMI
    │           │
    ▼           ▼
 Estimasi    Selesai
 ketinggian  (tanpa
 air (RF)    estimasi)
    │
    ▼
 Output: TSUNAMI (prob) + Estimated water height (m)
```

Estimasi hanya berjalan bila gerbang klasifikasi menghasilkan TSUNAMI —
sesuai target proyek "prediksi tsunami **dan** ketinggian airnya".

---

## 5. Smoke Test Pipeline (Chained)

```python
# Kasus 1 — gempa kuat → TSUNAMI → estimasi ketinggian air
$ python predict.py --mag 8.5 --depth 10.0 --lat 35.0 --lon 140.0
Prediction          : TSUNAMI      (prob 0.9977)
Estimated water height: 7.25 m    ✅  (estimasi aktif saat tsunami)

# Kasus 2 — gempa sangat kuat M9.1 (Aceh) → TSUNAMI → estimasi tinggi
$ python predict.py --mag 9.1 --depth 30.0 --lat 3.3 --lon 95.8 --model random_forest
Prediction          : TSUNAMI      (prob 0.5231)
Estimated water height: 21.87 m   (ekstrapolasi ekor kanan — realistis utk M9+)

# Kasus 3 — gempa kecil → NO TSUNAMI → tanpa estimasi
$ python predict.py --mag 3.5 --depth 15.0 --lat -6.0 --lon 107.0
Prediction          : NO TSUNAMI   (prob 0.0081)  ✅  (tanpa estimasi)
```

Perilaku sesuai ekspektasi fisik & desain chained: gempa besar shallow → tsunami
dengan estimasi ketinggian; gempa kecil → tidak tsunami tanpa estimasi.
Estimasi ketinggian hanya dihitung bila klasifikasi = TSUNAMI.
**Catatan:** nilai estimasi lebih realistis daripada versi pra-pembersihan
(M8.5 14.80→7.25m) karena outlier non-seismik (524m Lituya, dll.) telah dibuang.

**Batas interpretasi estimasi:** keluaran `Estimated water height` adalah
**water height** (agregat NOAA per event), **bukan run-up maksimum** di titik
pantai terparah. Contoh: Tohoku 2011 model ensemble ~13.6m (XGB/RF individual
~22.9m) vs run-up aktual ~40m di Miyako — perbedaan karena (a) definisi metrik,
(b) run-up lokal diperkuat morfologi pantai yang tidak ada di fitur gempa, dan
(c) ekor M8+ adalah ekstrapolasi (median data 0.3m, R² -0.16) sehingga model
cenderung konservatif. Interpretasi disarankan sebagai skala besaran, bukan run-up
presisi. (Rincian di `WEB_APP/README.md` §"Tentang Estimasi Ketinggian Air".)

---

## 6. Struktur File

```
phase_6_deployment/
├── 6_deployment.py          ← Re-train + save models (config FINAL, incl. estimator)
├── predict.py                ← CLI prediction pipeline (di-generate, chained)
├── model/
│   ├── xgboost.pkl           ← BARU (2026-09-04)
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl     ← tanpa class_weight
│   ├── mlp.pkl
│   ├── estimator_rf.pkl      ← BARU: RandomForestRegressor (ketinggian air)
│   ├── scaler.pkl
│   ├── scaler_lr.pkl
│   ├── feature_names.json
│   └── metadata.json         ← config_version: final-v1 (+ info estimator)
├── results/
│   └── deployment_results.json
├── .script_logs/
│   └── 6_deployment.py.log
└── .report/
    └── deployment.md         ← file ini
```

---

## 7. Status Proyek Keseluruhan

| Phase | Status |
|-------|--------|
| 0-3 | ✅ Selesai |
| 4 Modeling | ✅ FINAL (4 models × 5 methods) |
| 5 Evaluation | ✅ RE-RUN selesai (model final) |
| 6 Deployment | ✅ CHAINED (klasifikasi + estimasi ketinggian air) |

Seluruh pipeline CRISP-DM kini konsisten pada satu konfigurasi model.