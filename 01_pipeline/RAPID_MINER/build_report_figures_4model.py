"""Bangun metrik + gambar laporan untuk proses RapidMiner 4 model.

Masukan : rm_{rf,lr,xgb,mlp}_predictions.csv pada Local Repository AI Studio
Keluaran:
  - 03_results/rapidminer_4model_results.json   (metrik ambang 0,5)
  - 05_docs/pd/plots/14_rapidminer_4model_canvas.png
  - 05_docs/pd/plots/15_rapidminer_4model_performance.png
  - 05_docs/pd/plots/16_rapidminer_mlp_setup.png
  - 05_docs/pd/plots/17_rapidminer_lr_xgb_params.png

Jalankan: .venv/bin/python 01_pipeline/RAPID_MINER/build_report_figures_4model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DST = Path(
    "RAPIDMINER_LOCAL_REPO"
)
SNAP = ROOT / "01_pipeline/RAPID_MINER/results/screenshots/4model"
PLOTS = ROOT / "05_docs/pd/plots"
OUT_JSON = ROOT / "03_results/rapidminer_4model_results.json"

MODELS = {"rf": "RandomForest", "lr": "LogisticRegression", "xgb": "XGBoost", "mlp": "NeuralNet"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def metrics() -> dict:
    out = {}
    for key, label in MODELS.items():
        df = pd.read_csv(DST / f"rm_{key}_predictions.csv")
        df.columns = [c.strip('" ') for c in df.columns]
        y = (df["label"].astype(str).str.strip('"').str.lower() == "true").astype(int)
        p = (df["prediction(label)"].astype(str).str.strip('"').str.lower() == "true").astype(int)
        conf = df["confidence(true)"].astype(float)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        out[label] = {
            "n_test": int(len(df)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "accuracy": round((tp + tn) / len(df), 6),
            "precision": round(tp / max(tp + fp, 1), 4),
            "recall": round(tp / max(tp + fn, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "auc": round(float(roc_auc_score(y, conf)), 4),
        }
    return out


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def _load(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _stack(images: list[Image.Image], labels: list[str], pad: int = 18) -> Image.Image:
    """Tumpuk gambar vertikal, masing-masing diawali label."""
    width = max(im.width for im in images) + 2 * pad
    fh = int(_font(30).getbbox("Ag")[3])
    height = sum(im.height + fh + 2 * pad for im in images) + pad
    canvas = Image.new("RGB", (int(width), int(height)), "white")
    draw = ImageDraw.Draw(canvas)
    y = pad
    for im, text in zip(images, labels):
        draw.text((pad, y), text, fill=(40, 40, 40), font=_font(30))
        y += fh + pad // 2
        canvas.paste(im, (pad, y))
        draw.rectangle([pad - 1, y - 1, pad + im.width, y + im.height], outline=(200, 200, 200))
        y += im.height + pad
    return canvas


def _row(images: list[Image.Image], labels: list[str], pad: int = 18, gap: int = 24) -> Image.Image:
    """Sejajarkan gambar horizontal, tinggi disamakan dengan latar putih."""
    h = max(im.height for im in images)
    fh = int(_font(30).getbbox("Ag")[3])
    width = sum(im.width for im in images) + gap * (len(images) - 1) + 2 * pad
    canvas = Image.new("RGB", (int(width), int(h + fh + 3 * pad)), "white")
    draw = ImageDraw.Draw(canvas)
    x = pad
    for im, text in zip(images, labels):
        draw.text((x, pad), text, fill=(40, 40, 40), font=_font(30))
        canvas.paste(im, (x, pad * 2 + fh))
        draw.rectangle(
            [x - 1, pad * 2 + fh - 1, x + im.width, pad * 2 + fh + im.height],
            outline=(200, 200, 200),
        )
        x += im.width + gap
    return canvas


def figures() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    canvas = _load(SNAP / "01_canvas_4model.png")
    scale = 1600 / canvas.width
    canvas.resize((1600, round(canvas.height * scale)), Image.Resampling.LANCZOS).save(
        PLOTS / "14_rapidminer_4model_canvas.png", optimize=True
    )

    perf = [
        _load(SNAP / f"{idx}-1_performance_{key}_accuracy.png")
        for idx, key in zip(["02", "03", "04", "05"], ["rf", "lr", "xgb", "mlp"])
    ]
    _stack(
        perf,
        [
            "(a) Random Forest",
            "(b) Logistic Regression",
            "(c) XGBoost",
            "(d) Neural Net (MLP, data latih di-undersample 40:1)",
        ],
    ).save(PLOTS / "15_rapidminer_4model_performance.png", optimize=True)

    _row(
        [_load(SNAP / "07_sample_mlp_balance.png"), _load(SNAP / "08_neural_net_params.png.png")],
        ["(a) Sample: undersampling kelas negatif 10 %", "(b) Neural Net: 32-16, 200 siklus"],
    ).save(PLOTS / "16_rapidminer_mlp_setup.png", optimize=True)

    _row(
        [_load(SNAP / "09_logistic_params.png"), _load(SNAP / "10_xgboost_params.png")],
        ["(a) Logistic Regression (kernel dot, 100.000 iterasi)", "(b) XGBoost (200 rounds, depth 6)"],
    ).save(PLOTS / "17_rapidminer_lr_xgb_params.png", optimize=True)


if __name__ == "__main__":
    OUT_JSON.write_text(json.dumps(metrics(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figures()
    for name, m in metrics().items():
        print(f"{name:18s} TP{m['tp']:4d} FP{m['fp']:5d} FN{m['fn']:4d} TN{m['tn']:6d} "
              f"F1 {m['f1']:.4f} R {m['recall']:.4f} P {m['precision']:.4f} AUC {m['auc']:.4f}")
    for p in sorted(PLOTS.glob("1[4-7]_rapidminer*.png")):
        print(p.name, Image.open(p).size)
