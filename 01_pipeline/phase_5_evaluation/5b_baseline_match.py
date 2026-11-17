#!/usr/bin/env python3
"""5b — Baseline Match + Match Rate Analysis"""

import os
import json
import warnings
from datetime import datetime

# === SCRIPT LOGGING (auto-injected) ===
import sys

class _TeeLogger:
    """Write stdout to both terminal and .script_logs/<script>.log"""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')
        self.log.write(f"\n{'='*70}\n")
        self.log.write(f"RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.write(f"{'='*70}\n")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.write(f"\nRUN FINISHED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log.close()

_LOG_PATH = os.path.join(".script_logs", os.path.basename(__file__) + ".log")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
sys.stdout = _TeeLogger(_LOG_PATH)
sys.stderr = sys.stdout
# === END SCRIPT LOGGING ===

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

MERGED_PATH = "../../00_data/interim/merged_dataset.csv"
MODELING_PATH = "../../00_data/processed/modeling_dataset.csv"
NOAA_CSV = "../../00_data/raw/tsunami/tsunamis-2026-08-19.csv"
OUTPUT_DIR = "results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("5b: BASELINE + MATCH RATE")
    print("=" * 60)

    df = pd.read_csv(MERGED_PATH)
    total_events = len(df)
    tsunami_events = int(df['label'].sum())

    # --- Baseline Comparison ---
    print("\n--- Baseline Comparison ---")
    baselines = {}

    for name, condition in [
        ("M >= 6.0", df['mag'] >= 6.0),
        ("M >= 7.0", df['mag'] >= 7.0),
        ("M >= 7.0 & depth <= 70", (df['mag'] >= 7.0) & (df['depth'] <= 70)),
    ]:
        pred = condition.astype(int)
        cm = confusion_matrix(df['label'], pred)
        tn, fp, fn, tp = cm.ravel()
        report = classification_report(df['label'], pred, output_dict=True)
        cost = fn * 100 + fp * 1
        baselines[name] = {
            'f1': round(float(report['1']['f1-score']), 4),
            'recall': round(float(report['1']['recall']), 4),
            'precision': round(float(report['1']['precision']), 4),
            'cost': int(cost),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }
        print(f"  {name:30s}: F1={baselines[name]['f1']:.4f}, "
              f"Recall={baselines[name]['recall']:.4f}, Cost={cost:,}")

    # --- Match Rate ---
    print("\n--- Match Rate Analysis ---")
    df_noaa = pd.read_csv(NOAA_CSV)
    # Clean quoted column names
    df_noaa.columns = [c.strip('"').strip("'") for c in df_noaa.columns]

    year_col = [c for c in df_noaa.columns if c.lower() == 'year'][0]
    cause_col = [c for c in df_noaa.columns if 'cause' in c.lower()][0]
    lat_col = [c for c in df_noaa.columns if c.lower() == 'latitude'][0]
    lon_col = [c for c in df_noaa.columns if c.lower() == 'longitude'][0]

    df_noaa[year_col] = pd.to_numeric(df_noaa[year_col], errors='coerce')
    df_noaa[cause_col] = pd.to_numeric(df_noaa[cause_col], errors='coerce')

    total_noaa = len(df_noaa)
    modern = len(df_noaa[df_noaa[year_col] >= 1900])
    pre_1900 = total_noaa - modern
    matched = tsunami_events

    print(f"  Total NOAA events: {total_noaa}")
    print(f"  Pre-1900 (not in USGS): {pre_1900}")
    print(f"  Modern (>=1900): {modern}")
    print(f"  Matched to USGS: {matched}")
    if modern > 0:
        print(f"  Match rate: {matched/modern*100:.1f}%")

    # Cause analysis (modern only)
    df_modern = df_noaa[df_noaa[year_col] >= 1900].copy()
    # cause 0 = unknown, bukan seismik (perbaikan F-10)
    seismic = len(df_modern[(df_modern[cause_col] >= 1) & (df_modern[cause_col] <= 3)])
    non_seismic = len(df_modern[(df_modern[cause_col] >= 4) & (df_modern[cause_col] <= 9)])
    # cause 0 = unknown, sama seperti NaN
    unknown = len(df_modern[df_modern[cause_col].isna() | (df_modern[cause_col] == 0)])

    print(f"\n  Cause distribution (modern, n={len(df_modern)}):")
    print(f"    Seismic (1-3): {seismic} ({seismic/len(df_modern)*100:.1f}%)")
    print(f"    Non-seismic (4-9): {non_seismic} ({non_seismic/len(df_modern)*100:.1f}%)")
    print(f"    Unknown: {unknown} ({unknown/len(df_modern)*100:.1f}%)")

    unmatched = modern - matched
    print(f"\n  Unmatched: {unmatched} ({unmatched/modern*100:.1f}% if modern>0)")
    print(f"    Reason estimates:")
    print(f"      Non-seismic tsunami (no earthquake): ~{non_seismic}")
    print(f"      Earthquake M<4.5 (below USGS filter): significant portion")
    print(f"      Location mismatch (±0.5° tolerance): unknown")

    # --- Baseline pada split yang sama dengan phase 4 (perbandingan adil) ---
    print("\n--- Baseline pada split test phase 4 (80/20, seed 42) ---")
    dfm = pd.read_csv(MODELING_PATH, low_memory=False)
    tr, te = train_test_split(dfm, test_size=0.2, random_state=42,
                              stratify=dfm['label'])
    print(f"  Test: {len(te):,} baris, {int(te['label'].sum())} tsunami")
    baseline_split = {}
    for name, cond in [
        ("M >= 6.0", te['mag'] >= 6.0),
        ("M >= 6.5", te['mag'] >= 6.5),
        ("M >= 7.0", te['mag'] >= 7.0),
        ("M >= 7.0 & depth <= 70", (te['mag'] >= 7.0) & (te['depth'] <= 70)),
        ("M >= 8.0", te['mag'] >= 8.0),
    ]:
        pred = cond.astype(int)
        tn, fp, fn, tp = confusion_matrix(te['label'], pred).ravel()
        rep = classification_report(te['label'], pred, output_dict=True)
        baseline_split[name] = {
            'f1': round(float(rep['1']['f1-score']), 4),
            'recall': round(float(rep['1']['recall']), 4),
            'precision': round(float(rep['1']['precision']), 4),
            'cost': int(fn * 100 + fp),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        }
        b = baseline_split[name]
        print(f"  {name:24s}: F1={b['f1']:.4f}, Recall={b['recall']:.4f}, "
              f"Precision={b['precision']:.4f}, Cost={b['cost']:,}")

    results = {
        'baselines': baselines,
        'baseline_test_split': baseline_split,
        'match_rate': {
            'total_noaa': int(total_noaa),
            'pre_1900': int(pre_1900),
            'modern_noaa': int(modern),
            'matched': int(matched),
            'match_rate_pct': round(matched / modern * 100, 1) if modern > 0 else 0,
            'cause_seismic': int(seismic),
            'cause_non_seismic': int(non_seismic),
            'cause_unknown': int(unknown),
        }
    }

    with open(os.path.join(OUTPUT_DIR, "baseline_match_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/baseline_match_results.json")


if __name__ == '__main__':
    main()
