# Klasifikasi dan Estimasi Tingkat Bahaya Tsunami dari Parameter Gempa

Prototipe sistem prediksi dini dua tahap: **tahap 1** klasifikasi biner (gempa
berpotensi tsunami atau tidak), **tahap 2** estimasi ordinal tiga kelas tingkat
bahaya (**Waspada / Siaga / Awas**) untuk kandidat positif. Dibangun dari dua basis
data publik — NCEI/WDS Global Historical Tsunami Database (NOAA) dan ANSS
Comprehensive Earthquake Catalog (USGS) — lalu diimplementasikan di Python dan
RapidMiner, serta dibungkus aplikasi Flask.

Repositori ini memuat kode pipeline, hasil evaluasi, diagram perancangan, dan
aplikasi web-nya. Data mentah dan model terlatih tersedia terpisah lewat tautan
di bagian [Data dan model](#data-dan-model).

## Hasil utama

| Aspek | Angka |
|---|---|
| Dataset final | 310.297 gempa; 761 positif (tsunami), 309.536 negatif; rasio 1 : 406,75 |
| Fitur model | 24 fitur (11 numerik + 13 dummy nominal) |
| Label ordinal | 610 kejadian positif yang punya catatan tinggi air laut |
| Klasifikasi @ ambang 0,5 | F1 terbaik MLP 0,3849; presisi terbaik RF 0,7391; recall terbaik LR 0,9211 |
| Ambang operasional | Ensemble recall 0,8289 (biaya 3.774); XGBoost recall 0,7961 @0,13 (biaya 4.710) |
| Pembanding aturan | M ≥ 7,0 dan kedalaman ≤ 70 km: F1 0,3788, recall 0,4934, biaya 7.869 |
| Estimasi ordinal (XGBoost) | QWK 0,4667; MAE kelas 0,4016; tepat kelas 0,6393; toleransi satu kelas 0,9590 |
| RapidMiner (4 model) | F1 terbaik XGBoost 0,3654; AUC 0,9172–0,9685 |
| Aplikasi | 4 preset gempa, peta episentrum, perbandingan 5 model, latensi ±0,16 detik |

Catatan pembacaan: pada rasio 1 : 406,75, akurasi selalu-menebak-negatif sudah 99,75%,
jadi metrik yang dipakai adalah F1, recall, presisi, dan biaya kesalahan — bukan akurasi.

## Struktur repositori

```
01_pipeline/        skrip per fase CRISP-DM (integrasi data, feature engineering,
                    pemodelan, evaluasi, deployment) + hasil JSON + plot
   RAPID_MINER/     proyek RapidMiner 4 model (.rmp) dan hasilnya
02_models/          feature_names.json + metadata.json (model .pkl terpisah)
03_results/         ANGKA_FINAL.md, TABEL_PD.md, JSON hasil RapidMiner
04_app/             aplikasi Flask: modul PD (/) dan MDIK (/mdik)
05_docs/            diagram .drawio + .png, skrip dan hasil pengujian
06_papers/          daftar 20 literatur
docs/               kamus data
```

## Cara menjalankan

Prasyarat: Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Dapatkan data** — unduh empat arsip di bagian [Data dan model](#data-dan-model),
   ekstrak di akar repositori sehingga terbentuk `00_data/raw/`, `00_data/processed/`,
   `02_models/*.pkl`, dan `01_pipeline/phase_6_deployment/model/*.pkl`.
2. **Jalankan pipeline** (urutan fase):
   ```bash
   python 01_pipeline/phase_2_data_understanding/1_data_integration.py
   python 01_pipeline/phase_3_data_preparation/1_feature_engineering.py
   python 01_pipeline/phase_3_data_preparation/3_build_modeling_set.py
   python 01_pipeline/phase_4_modeling/0_classification/1_modeling.py
   python 01_pipeline/phase_4_modeling/1_estimation/1_modeling.py
   python 01_pipeline/phase_5_evaluation/5a_evaluation.py
   python 01_pipeline/phase_6_deployment/6_deployment.py
   ```
   Semua hasil numerik tercatat di `03_results/ANGKA_FINAL.md` dengan seed 42.
3. **Jalankan aplikasi**:
   ```bash
   python 04_app/app.py     # http://localhost:5000  (PD)  dan  /mdik  (MDIK)
   ```
   Basis data SQLite MDIK dibuat otomatis saat pertama dijalankan. Tiga akun demo
   (`admin`, `operator`, `analis`) dibuat untuk keperluan demonstrasi — **ganti
   kata sandinya** lewat variabel lingkungan sebelum dipakai di luar laptop sendiri
   (lihat `.env.example`).
4. **RapidMiner (opsional)** — buka `01_pipeline/RAPID_MINER/proyek_4model.rmp` di
   Altair AI Studio (RapidMiner) 2026.1.1 atau lebih baru. Path berkas di dalam
   `.rmp` memakai token `RAPIDMINER_LOCAL_REPO` sebagai pengganti folder
   *Local Repository* milik penulis; sesuaikan dengan folder Anda sendiri.

## Data dan model

Satu folder Drive berisi empat arsip (data mentah, data olahan, data interim,
dan model terlatih):

> https://drive.google.com/drive/folders/19zQOJrO0MWkQNc4F1NdHH3qhYo6ZNPWS?usp=sharing

| Arsip | Isi |
|---|---|
| `pd_mdik_data_raw.zip` | unduhan NOAA NCEI + katalog USGS |
| `pd_mdik_data_processed.zip` | `modeling_dataset.csv`, `prepared_dataset.csv`, `dropped_events.csv`, kamus data |
| `pd_mdik_data_interim.zip` | dataset gabungan tahap antara |
| `pd_mdik_models.zip` | 4 model klasifikasi + estimator ordinal + scaler + metadata |

Sumber data: NOAA NCEI/WDS Global Historical Tsunami Database dan USGS ANSS
Comprehensive Earthquake Catalog. Keduanya data publik; hak dan ketentuan penggunaan
mengikuti sumber aslinya. Data olahan di repositori ini adalah turunan hasil
pencocokan spasiotemporal kedua sumber tersebut.

## Bukti pengujian

Skrip dan hasil eksekusinya tersedia di repositori ini:

- `05_docs/pd/run_system_tests.py` — pengujian fungsional dan latensi aplikasi PD;
  hasilnya di `05_docs/pd/testing_results.json` dan `05_docs/pd/latency_per_model.json`.
- `05_docs/mdik/run_system_tests.py` — kasus TC-01…TC-13 untuk modul MDIK;
  hasilnya di `05_docs/mdik/testing_results.json`.
- `01_pipeline/phase_5_evaluation/` — metrik dan plot evaluasi tiap model.

## Lingkungan

Python 3.11, versi paket dipin di `requirements.txt` (`pandas 3.0.5`,
`scikit-learn 1.9.0`, `xgboost 3.2.0`, `flask 3.1.2`, dan seterusnya).
Implementasi kedua memakai Altair AI Studio (RapidMiner) 2026.1.1 di Windows.

<!--
## Sitasi

```bibtex
@misc{febrilistyas2026tsunami,
  title  = {Klasifikasi dan Estimasi Tingkat Bahaya Tsunami dari Parameter Gempa},
  author = {Febrilistyas, Dwie Fajar and Purnamasari, Ni Made Gita Dwi and Agum, Muhammad Adi},
  year   = {2026},
  note   = {Proyek Akhir Mata Kuliah Penambangan Data, Magister Teknologi Informasi},
  url    = {https://github.com/muhammadadiagum/proyek_pd_mdik}
}
```
-->

## Lisensi

- Kode (`01_pipeline/`, `02_models/`, `04_app/`, skrip di `05_docs/`): **MIT** — lihat `LICENSE`.
- Dokumentasi, diagram, dan hasil pengujian: **CC BY 4.0** — lihat `LICENSE-REPORT`.
- Data: mengikuti ketentuan NOAA NCEI dan USGS.
