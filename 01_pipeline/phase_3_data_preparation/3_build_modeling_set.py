#!/usr/bin/env python3
"""
3_build_modeling_set.py — Bangun dataset modeling dengan definisi label final
==============================================================================
Keputusan 3 Nov 2026 (lihat .audit/00_RINGKASAN.md §5.2):

  Positif  : label = 1 DAN noaa_cause_code = 1 (tsunami akibat gempa murni)
  Negatif  : label = 0 (gempa tanpa tsunami tercatat)
  Dibuang  : label = 1 dengan cause != 1 atau cause kosong (72 baris)
             -> TIDAK dilabeli 0, karena itu akan menjadi false negative
                yang sengaja dibuat.

Alasan: judul proyek adalah "tsunami yang dipicu gempa". Cause 3 berarti
"earthquake and landslide" (mekanisme longsor ikut berperan), cause >=4
vulkanik/meteorologis/longsor murni. Fitur gempa tidak dapat menjelaskan
mekanisme tersebut.

Input : ../../../00_data/processed/prepared_dataset.csv  (310.369 baris)
Output: ../../../00_data/processed/modeling_dataset.csv  (310.297 baris)
        ../../../00_data/processed/dropped_events.csv   (72 baris)
        results/modeling_set_manifest.json
"""

import os
import sys
import json
import hashlib
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../00_data/processed")
INPUT_PATH = os.path.join(DATA_DIR, "prepared_dataset.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "modeling_dataset.csv")
DROPPED_PATH = os.path.join(DATA_DIR, "dropped_events.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "modeling_set_manifest.json")
LOG_DIR = os.path.join(BASE_DIR, ".script_logs")

CAUSE_LABELS = {
    1: "earthquake",
    2: "questionable earthquake",
    3: "earthquake and landslide",
    4: "volcano and earthquake",
    5: "volcano/earthquake/landslide",
    6: "volcano",
    7: "volcano and landslide",
    8: "landslide",
    9: "meteorological",
    10: "explosion",
    11: "astronomical tide",
}


class Tee:
    def __init__(self, path):
        self.stdout = sys.stdout
        self.f = open(path, "w")

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)
        self.stdout.flush()
        self.f.flush()

    def flush(self):
        self.stdout.flush()
        self.f.flush()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sys.stdout = Tee(os.path.join(LOG_DIR, os.path.basename(__file__) + ".log"))

    print("=" * 70)
    print("BUILD MODELING SET — definisi label cause_code = 1")
    print("=" * 70)
    print(f"Waktu : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Input : {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    n_in = len(df)
    pos_in = int(df["label"].sum())
    print(f"Baris : {n_in:,} | positif label=1: {pos_in}")

    is_pos = df["label"] == 1
    seismic = df["noaa_cause_code"] == 1          # gempa murni
    keep = (~is_pos) | (is_pos & seismic)
    dropped = df[~keep].copy()
    out = df[keep].copy()

    print("\nRincian cause pada label=1:")
    counts = df.loc[is_pos, "noaa_cause_code"].value_counts(dropna=False).sort_index()
    for cause, n in counts.items():
        name = "TIDAK ADA (NaN)" if pd.isna(cause) else CAUSE_LABELS.get(int(cause), "?")
        status = "DIPAKAI" if cause == 1 else "DIBUANG"
        print(f"  cause {str(cause):>4s}  {name:32s} {n:4d}  -> {status}")

    n_out = len(out)
    pos_out = int(out["label"].sum())
    neg_out = n_out - pos_out
    dropped_pos = int(dropped["label"].sum())
    dropped_neg = len(dropped) - dropped_pos
    wh_out = int(out.loc[out["label"] == 1, "noaa_water_height"].notna().sum())

    print(f"\nDibuang : {len(dropped)} baris "
          f"(positif {dropped_pos}, negatif {dropped_neg})")
    print(f"Output  : {n_out:,} baris | positif {pos_out} | negatif {neg_out:,}")
    print(f"Positif dengan water_height: {wh_out}")
    assert dropped_neg == 0, "filter tidak boleh membuang baris negatif"
    assert pos_out + dropped_pos == pos_in, "cek keseimbangan positif gagal"
    assert neg_out == (n_in - pos_in), "cek keseimbangan negatif gagal"

    out.to_csv(OUTPUT_PATH, index=False)
    dropped[["time", "year", "latitude", "longitude", "mag", "depth",
             "noaa_country", "noaa_location", "noaa_cause_code",
             "noaa_water_height", "noaa_intensity", "label"]].to_csv(
        DROPPED_PATH, index=False)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "decision": "positif = label=1 & noaa_cause_code=1; cause lain dibuang, bukan dilabeli 0",
        "source": {
            "path": os.path.relpath(INPUT_PATH, BASE_DIR),
            "rows": n_in,
            "positives": pos_in,
            "md5": md5(INPUT_PATH),
        },
        "output": {
            "path": os.path.relpath(OUTPUT_PATH, BASE_DIR),
            "rows": n_out,
            "positives": pos_out,
            "negatives": neg_out,
            "positives_with_water_height": wh_out,
            "md5": md5(OUTPUT_PATH),
        },
        "dropped": {
            "path": os.path.relpath(DROPPED_PATH, BASE_DIR),
            "rows": len(dropped),
            "positives": dropped_pos,
            "negatives": dropped_neg,
            "by_cause": {str(int(k)) if not pd.isna(k) else "NaN": int(v)
                         for k, v in counts.items() if k != 1},
        },
        "class_ratio_positive": round(pos_out / n_out, 6),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTersimpan: {OUTPUT_PATH}")
    print(f"Tersimpan: {DROPPED_PATH}")
    print(f"Manifest : {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
