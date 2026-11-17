#!/usr/bin/env python3
"""
Step 1: Feature Engineering — Persiapan Data untuk Modeling
============================================================
Membuat fitur baru, menangani missing values, dan
menyiapkan dataset untuk tahap pemodelan.

Input: merged_dataset.csv (Phase 2)
Output: prepared_dataset.csv (siap untuk modeling)

Adaptasi dari .UNNAMED/ dengan penyesuaian:
- Logging via Tee (stdout + file)
- Path disesuaikan ke Tugas_2
"""

import csv
import math
import os
import sys
from collections import defaultdict

# ============================================================
# LOGGING: Tee stdout -> console + file
# ============================================================

class Tee:
    def __init__(self, log_path):
        self.stdout = sys.stdout
        self.log_file = open(log_path, 'w')
    def write(self, data):
        self.stdout.write(data)
        self.log_file.write(data)
        self.stdout.flush()
        self.log_file.flush()
    def flush(self):
        self.stdout.flush()
        self.log_file.flush()
    def close(self):
        self.log_file.close()
        sys.stdout = self.stdout

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.script_logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, os.path.basename(__file__) + '.log')
tee = Tee(LOG_PATH)
sys.stdout = tee

# ============================================================
# PATHS
# ============================================================

INPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '../../00_data/interim/merged_dataset.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '../../00_data/processed/prepared_dataset.csv')

# ============================================================
# FITUR TURUNAN
# ============================================================

def add_depth_mag_ratio(row):
    """Rasio kedalaman terhadap magnitudo.
    Gempa dangkal + besar = lebih tsunamigenic."""
    try:
        depth = float(row['depth'])
        mag = float(row['mag'])
        if mag > 0:
            return round(depth / mag, 4)
    except:
        pass
    return None

def add_abs_latitude(row):
    """Absolut lintang (proxy jarak dari equator).
    Equatorial regions lebih aktif secara tektonik."""
    try:
        lat = abs(float(row['latitude']))
        return round(lat, 4)
    except:
        return None

def classify_tectonic_zone(lat, lon):
    """Klasifikasi zone tektonik berdasarkan lat/lon.
    Berdasarkan Pacific Ring of Fire dan tectonic boundaries."""
    try:
        lat = float(lat)
        lon = float(lon)
    except:
        return 'unknown'

    # Pacific Ring of Fire (approximate boundaries)
    if ((lat > -60 and lat < 60) and
        ((lon > 100 and lon < 180) or (lon > -180 and lon < -60))):
        return 'pacific_ring_of_fire'

    # Mediterranean-Asian Belt (Alpide)
    elif (lat > 20 and lat < 50 and lon > -10 and lon < 90):
        return 'mediterranean_asian_belt'

    # Mid-Atlantic Ridge
    elif (lon > -40 and lon < 0 and lat > -60 and lat < 60):
        return 'mid_atlantic_ridge'

    # Other
    else:
        return 'other'

def calc_monitoring_quality(row):
    """Composite score (0-1) untuk kualitas monitoring gempa.
    Score tinggi = monitoring baik."""
    score = 0
    count = 0

    # nst (number of stations) - higher is better
    try:
        nst = float(row['nst'])
        if nst > 0:
            nst_score = min(nst / 100, 1.0)
            score += nst_score
            count += 1
    except:
        pass

    # gap (azimuthal gap) - lower is better
    try:
        gap = float(row['gap'])
        if gap > 0:
            gap_score = 1.0 - min(gap / 360, 1.0)
            score += gap_score
            count += 1
    except:
        pass

    # dmin (minimum distance) - lower is better
    try:
        dmin = float(row['dmin'])
        if dmin > 0:
            dmin_score = 1.0 - min(dmin / 10, 1.0)
            score += dmin_score
            count += 1
    except:
        pass

    # rms (residual) - lower is better
    try:
        rms = float(row['rms'])
        if rms > 0:
            rms_score = 1.0 - min(rms / 2.0, 1.0)
            score += rms_score
            count += 1
    except:
        pass

    if count > 0:
        return round(score / count, 4)
    return None

def classify_era(year):
    """Klasifikasi era berdasarkan tahun.
    Mencerminkan kualitas monitoring sepanjang waktu."""
    try:
        y = int(year)
    except:
        return 'unknown'

    if y < 1930:
        return 'early_1900s'   # Monitoring terbatas
    elif y < 1960:
        return 'mid_1900s'    # Perang + recovery
    elif y < 1990:
        return 'late_1900s'   # Modernisasi
    elif y < 2010:
        return 'early_2000s'  # Digital era
    else:
        return 'modern'       # Full digital + internet

def add_month_category(month):
    """Kategorisasi bulan: musim."""
    try:
        m = int(month)
        if m in [12, 1, 2]:
            return 'djf'   # December-January-February
        elif m in [3, 4, 5]:
            return 'mam'   # March-April-May
        elif m in [6, 7, 8]:
            return 'jja'   # June-July-August
        else:
            return 'son'   # September-October-November
    except:
        return 'unknown'

# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def handle_missing_values(rows):
    """Median imputation untuk fitur numerik."""
    numeric_cols = ['nst', 'gap', 'dmin', 'rms']
    medians = {}

    for col in numeric_cols:
        values = []
        for row in rows:
            try:
                v = float(row[col])
                values.append(v)
            except:
                pass
        if values:
            values.sort()
            mid = len(values) // 2
            medians[col] = values[mid] if len(values) % 2 == 0 else values[mid]
        else:
            medians[col] = 0

    print(f"Medians for imputation: {medians}")

    # Imputasi
    imputed_count = 0
    for row in rows:
        for col in numeric_cols:
            if row[col] == '' or row[col] is None:
                row[col] = medians[col]
                imputed_count += 1

    print(f"Imputed {imputed_count} missing values")
    return rows

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("STEP 1: FEATURE ENGINEERING")
    print("=" * 60)

    # Read input
    print("\n[1/5] Reading merged dataset...")
    with open(INPUT_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows")

    # Handle missing values
    print("\n[2/5] Handling missing values (median imputation)...")
    rows = handle_missing_values(rows)

    # Add new features
    print("\n[3/5] Creating new features...")
    for i, row in enumerate(rows):
        row['depth_mag_ratio'] = add_depth_mag_ratio(row)
        row['abs_latitude'] = add_abs_latitude(row)
        row['tectonic_zone'] = classify_tectonic_zone(row['latitude'], row['longitude'])
        row['monitoring_quality'] = calc_monitoring_quality(row)
        row['era'] = classify_era(row['year'])
        row['season'] = add_month_category(row['month'])

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{len(rows)} rows")

    print(f"  Processed {len(rows)}/{len(rows)} rows")

    # Validate new features
    print("\n[4/5] Validating new features...")
    new_features = ['depth_mag_ratio', 'abs_latitude', 'tectonic_zone',
                    'monitoring_quality', 'era', 'season']
    for col in new_features:
        missing = sum(1 for row in rows if row[col] is None or row[col] == '')
        print(f"  {col}: {missing} missing ({missing/len(rows)*100:.1f}%)")

    # Feature columns
    feature_cols = [
        'time', 'year', 'month', 'day',
        'latitude', 'longitude', 'mag', 'depth',
        'nst', 'gap', 'dmin', 'rms',
        'magType', 'place',
        'label',
        'noaa_country', 'noaa_location',
        'noaa_cause_code', 'noaa_water_height', 'noaa_intensity',
        # New features
        'depth_mag_ratio', 'abs_latitude',
        'tectonic_zone', 'monitoring_quality',
        'era', 'season',
    ]

    # Write output
    print("\n[5/5] Writing prepared dataset...")
    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=feature_cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Total rows: {len(rows)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"New features: {', '.join(new_features)}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Log: {LOG_PATH}")

    print("\nFeature columns:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    print("=" * 60)

    tee.close()

if __name__ == '__main__':
    main()
