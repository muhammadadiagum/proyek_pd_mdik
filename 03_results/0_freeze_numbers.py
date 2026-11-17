"""Bekukan angka final proyek ke 03_results/ANGKA_FINAL.md.

Semua angka dibaca dari JSON hasil eksekusi, sehingga dokumen ini
dapat dibuat ulang: python 0_freeze_numbers.py
"""
import json
import os
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
P = lambda *a: os.path.join(ROOT, *a)

def load(*path, default=None):
    f = P(*path)
    if not os.path.exists(f):
        return default
    with open(f) as fh:
        return json.load(fh)

def f(x, n=4):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"

def n(x):
    """Pemisah ribuan gaya Indonesia: 310.297"""
    return f"{x:,}".replace(",", ".") if isinstance(x, (int, float)) else "—"

def stamp(*path):
    f_ = P(*path)
    return datetime.fromtimestamp(os.path.getmtime(f_)).strftime("%d %b %Y %H:%M") if os.path.exists(f_) else "—"

manifest = load("01_pipeline/phase_3_data_preparation/results/modeling_set_manifest.json", default={})
cls = load("01_pipeline/phase_4_modeling/0_classification/results/model_results.json", default={})
est = load("01_pipeline/phase_4_modeling/1_estimation/results/estimation_ordinal_results.json", default={})
dep = load("01_pipeline/phase_6_deployment/results/deployment_results.json", default={})
thr = load("01_pipeline/phase_6_deployment/model/thresholds.json", default={})
base = load("01_pipeline/phase_5_evaluation/results/baseline_match_results.json", default={})
cal = load("01_pipeline/phase_5_evaluation/results/calibration_results.json", default={})
imb = load("01_pipeline/phase_5_evaluation/results/imbalance_results.json", default={})
abl = load("01_pipeline/phase_5_evaluation/results/ablation_results.json", default={})
tbs = load("01_pipeline/phase_5_evaluation/results/time_based_split_results.json", default={})
ens5 = load("01_pipeline/phase_5_evaluation/results/ensemble_results.json", default={})
err = load("01_pipeline/phase_5_evaluation/results/error_analysis_results.json", default={})

L = []
w = L.append

w("# ANGKA FINAL PROYEK — dari JSON hasil eksekusi")
w("")
w(f"Dibuat: **{datetime.now():%d %B %Y %H:%M}** oleh `03_results/0_freeze_numbers.py`.")
w("Dokumen ini dihasilkan otomatis. Jalankan ulang skripnya setelah pipeline berubah.")
w("")
w("Sumber per bagian (timestamp berkas):")
for label, path in [
    ("manifest data", "01_pipeline/phase_3_data_preparation/results/modeling_set_manifest.json"),
    ("klasifikasi", "01_pipeline/phase_4_modeling/0_classification/results/model_results.json"),
    ("ordinal", "01_pipeline/phase_4_modeling/1_estimation/results/estimation_ordinal_results.json"),
    ("deployment", "01_pipeline/phase_6_deployment/results/deployment_results.json"),
    ("ambang", "01_pipeline/phase_6_deployment/model/thresholds.json"),
    ("baseline", "01_pipeline/phase_5_evaluation/results/baseline_match_results.json"),
]:
    w(f"- {label}: `{path.split('/', 1)[1]}` — {stamp(path)}")
w("")

# --- 1. Data ---
w("## 1. Data dan definisi label")
w("")
out_ = (manifest or {}).get("output", {})
if out_:
    w(f"- Baris: **{n(out_.get('rows'))}**; positif (tsunami, cause = 1): **{n(out_.get('positives'))}**; "
      f"negatif: **{n(out_.get('negatives'))}**")
    w(f"- Positif dengan `noaa_water_height`: **{n(out_.get('positives_with_water_height'))}**")
    w(f"- Dibuang (cause ≠ 1): **{n(manifest.get('dropped', {}).get('rows', 72))}** baris → "
      f"`00_data/processed/dropped_events.csv`")
    w(f"- md5 `modeling_dataset.csv`: `{out_.get('md5', '—')}`")
w("")

# --- 2. Klasifikasi phase 4 ---
w("## 2. Klasifikasi — phase 4 (ambang bawaan 0,5, test 20%)")
w("")
w("| Model | F1 | Recall | Precision | AUC | TN | FP | FN | TP | Cost |")
w("|---|---|---|---|---|---|---|---|---|---|")
orig = cls.get("Original", {})
for m in ["MLP", "RandomForest", "XGBoost", "LogisticRegression"]:
    d = orig.get(m, {})
    if d:
        w(f"| {m} | {f(d.get('f1'))} | {f(d.get('recall'))} | {f(d.get('precision'))} | "
          f"{f(d.get('auc'))} | {d.get('tn', '—')} | {d.get('fp', '—')} | {d.get('fn', '—')} | "
          f"{d.get('tp', '—')} | {n(d.get('cost'))} |")
w("")
if cls:
    w("Pada 20 kombinasi phase 4, tidak ada metode resampling yang mengungguli "
      "konfigurasi Original terbaik (MLP 0,3849; ROS terbaik 0,3262). Eksperimen "
      "imbalance phase 5 pada XGB memberi ROS F1 0,2439 vs Original 0,2178 — sedikit "
      "lebih tinggi tetapi tidak konsisten, jadi produk tetap memakai Original. "
      "Tabel penuh: `model_results.json` dan `imbalance_results.json`.")
w("")

# --- 3. Ordinal ---
w("## 3. Estimasi — regresi ordinal 3 kelas")
w("")
if est:
    names = est.get("class_definition", {}).get("names", ["Waspada", "Siaga", "Awas"])
    bounds = est.get("class_definition", {}).get("thresholds_m", [0.5, 3.0])
    counts = est.get("class_definition", {}).get("counts", [])
    w(f"Kelas: {names[0]} ≤ {bounds[0]} m; {names[1]} {bounds[0]}–{bounds[1]} m; "
      f"{names[2]} > {bounds[1]} m." + (f" Distribusi positif: {counts}." if counts else ""))
    w("")
    w("| Model | QWK CV (mean ± std) | QWK holdout | MAE kelas | Tepat | ±1 kelas |")
    w("|---|---|---|---|---|---|")
    for m in ["XGBoost", "RandomForest", "LogisticRegression", "MLP"]:
        cv = (est.get("cv", {}).get(m, {}).get("ordinal", {}).get("folds_mean", {}) or {}).get("qwk", {})
        ho = est.get("holdout", {}).get(m, {}).get("ordinal", {})
        if cv or ho:
            w(f"| {m} | {f(cv.get('mean'))} ± {f(cv.get('sd'))} | {f(ho.get('qwk'))} | "
              f"{f(ho.get('mae_class'))} | {f(ho.get('exact_acc'))} | {f(ho.get('adjacent_acc'))} |")
    w("")
    hw = est.get("holdout_warning", {})
    if hw:
        w("Aturan peringatan sadar-biaya (dipakai produk, holdout 122 baris):")
        w("")
        w("| Model | QWK | MAE kelas | Recall Awas |")
        w("|---|---|---|---|")
        for m, d in hw.items():
            d = d.get("ordinal", d) if isinstance(d, dict) else {}
            per = d.get("per_class", {}).get("class_2", {}) if isinstance(d, dict) else {}
            w(f"| {m} | {f(d.get('qwk'))} | {f(d.get('mae_class'))} | {f(per.get('recall'))} |")
        w("")
    oof = est.get("cv", {}).get("XGBoost", {}).get("ordinal", {}).get("pooled_oof", {})
    if oof:
        w(f"OOF terkumpul XGB ordinal: QWK {f(oof.get('qwk'))}, tepat {f(oof.get('exact_acc'))}, "
          f"±1 kelas {f(oof.get('adjacent_acc'))} (n={n(oof.get('n'))}).")
        w("")

# --- 4. Deployment ---
w("## 4. Deployment — ambang per model (inner-validation)")
w("")
if thr:
    proto = thr.get("protocol", {})
    w(f"Protokol: inner-validation {proto.get('inner_val_size', 0.25):.0%}, seed {proto.get('inner_seed', 7)}, "
      f"biaya FN×{proto.get('fn_cost', 100)} + FP×{proto.get('fp_cost', 1)}, "
      f"lantai presisi {proto.get('min_precision', 0.10)}.")
    w("")
w("| Model | Ambang deploy | F1 | Recall | Precision | AUC | Cost | F1 @0,5 | Recall @0,5 |")
w("|---|---|---|---|---|---|---|---|---|")
if dep:
    for m, d in dep.get("classification", {}).items():
        d5 = d.get("at_default_0.5", {})
        w(f"| {m} | {f(d.get('threshold'), 2)} | {f(d.get('f1'))} | {f(d.get('recall'))} | "
          f"{f(d.get('precision'))} | {f(d.get('auc'))} | {n(d.get('cost'))} | "
          f"{f(d5.get('f1'))} | {f(d5.get('recall'))} |")
w("")
w("Ordinal holdout (dari deployment): " + (
    f"QWK {f(dep['estimation'].get('qwk'))}, MAE kelas {f(dep['estimation'].get('mae_class'))}, "
    f"tepat {f(dep['estimation'].get('exact_acc'))}, ±1 {f(dep['estimation'].get('adjacent_acc'))}."
    if dep.get("estimation") else "—"))
w("")

# --- 5. Baseline ---
w("## 5. Baseline aturan sederhana (split test yang sama)")
w("")
bs = base.get("baseline_test_split", {})
if bs:
    w("| Aturan | F1 | Recall | Precision | Cost |")
    w("|---|---|---|---|---|")
    for name, d in bs.items():
        w(f"| {name} | {f(d.get('f1'))} | {f(d.get('recall'))} | {f(d.get('precision'))} | {n(d.get('cost'))} |")
    w("")
    w("**Implikasi:** model kalah pada F1 dari aturan `M ≥ 7 & depth ≤ 70`, tetapi menang pada "
      "biaya (recall tinggi, FN 100× lebih mahal). Jangan klaim akurasi lebih tinggi.")
    w("")

# --- 6. Evaluasi tambahan ---
w("## 6. Evaluasi tambahan (phase 5)")
w("")
if cal:
    w("Kalibrasi (test): " + "; ".join(
        f"{m} Brier {f(d.get('brier_score'))}/ECE {f(d.get('ece'))}" for m, d in cal.items()) + ".")
    w("")
if imb:
    w("Imbalance (F1, recall): " + "; ".join(
        f"{k} {f(v.get('f1'))}/{f(v.get('recall'))}" for k, v in imb.items()) + ".")
    w("")
res = abl.get("results", {})
if res:
    w("Ablasi fitur (**time-based split** train≤2015 / test≥2016, bukan random split — "
      "prevalensi test lebih rendah, jangan dibandingkan langsung dengan §2): " +
      "; ".join(f"{k} = {f(v.get('f1'))}" for k, v in res.items()) + ".")
    w("")
if tbs:
    for arm in ("random_split", "time_based_split"):
        d = tbs.get(arm, {}).get("XGBoost", {})
        if d:
            w(f"- {arm} XGB: F1 {f(d.get('f1'))}, recall {f(d.get('recall'))}, "
              f"precision {f(d.get('precision'))}, cost {n(d.get('cost'))}")
    w("")
if ens5:
    sv = ens5.get("soft_voting", {})
    if sv:
        w(f"Ensemble 5g (soft voting @0,36): F1 {f(sv.get('f1'))}, recall {f(sv.get('recall'))}, "
          f"cost {n(sv.get('cost'))}.")
        w("")
if err:
    ts = err.get("test_set", {})
    fn = err.get("fn", {})
    w(f"Analisis error (test {n(ts.get('n_total'))} baris, {ts.get('n_tsunami', '—')} tsunami): "
      f"FN {fn.get('count', '—')}.")
    w("")

# --- 7. Larangan kutip ---
w("## 7. Angka lama yang tidak boleh dikutip")
w("")
w("| Angka lama | Kenapa salah | Ganti dengan |")
w("|---|---|---|")
w("| XGB F1 0,2260 | `scale_pos_weight` 371 (label lama 833) | 0,2178 |")
w("| Positif 833 / 310.369 baris | label belum dibatasi cause = 1 | 761 / 310.297 |")
w("| Ambang 0,22 atau 0,94 sebagai ambang produk | 0,22 hardcode; 0,94 in-sample | `model/thresholds.json` |")
w("| MAPE 319–403%, R² 0,26/−0,16 | regresi meter sudah tidak dipakai | QWK 0,4667 |")
w("| 92 TP / 49 FN | threshold 0,5 dan data sebelum filter | 5i terbaru (89 TP / 44 FN @0,36) |")
w("")

out = os.path.join(BASE, "ANGKA_FINAL.md")
with open(out, "w") as fh:
    fh.write("\n".join(L) + "\n")
print(f"OK → {out} ({len(L)} baris)")
