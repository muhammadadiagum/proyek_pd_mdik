#!/usr/bin/env python3
"""
Step 1: Data Integration — Join NOAA Tsunami + USGS Earthquake
================================================================
Menggabungkan dataset tsunami NOAA dengan dataset gempa USGS
berdasarkan waktu dan lokasi.

Output: merged_dataset.csv (ready untuk feature engineering)
"""

import csv
import math
import os
import sys
from datetime import datetime, timedelta

class Tee:
    """Tulis stdout ke terminal DAN file log secara bersamaan."""
    def __init__(self, log_path, stream=sys.stdout):
        self.log = open(log_path, 'w')
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.log.write(data)
        self.log.flush()
    def flush(self):
        self.stream.flush()
        self.log.flush()
    def close(self):
        self.log.close()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, '../../00_data')
NOAA_PATH = os.path.join(DATA_ROOT, 'raw/tsunami/tsunamis-2026-08-19_21-29-48_+0800.tsv')
USGS_PATH = os.path.join(DATA_ROOT, 'raw/earthquake/merged_usgs_earthquakes.csv')
OUTPUT_PATH = os.path.join(DATA_ROOT, 'interim/merged_dataset.csv')

# Matching tolerances
TIME_TOLERANCE_DAYS = 1  # ±1 hari
LAT_LON_TOLERANCE = 0.25  # ±0.25 derajat (~28 km) — actual: 833 match, 1:371 ratio

def haversine_km(lat1, lon1, lat2, lon2):
    """Hitung jarak antara 2 titik koordinat (km)."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

def parse_noaa_date(year, month, day):
    """Parse tanggal NOAA ke datetime. Handle missing values."""
    try:
        y = int(float(year)) if year else None
        m = int(float(month)) if month else 1
        d = int(float(day)) if day else 1
        if y and y > 0:
            return datetime(y, max(1, min(12, m)), max(1, min(28, d)))
    except:
        pass
    return None

def parse_usgs_time(time_str):
    """Parse waktu USGS ISO format ke datetime (timezone-naive)."""
    try:
        # Format: 2023-01-01T07:03:01.860Z
        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        return dt.replace(tzinfo=None)  # strip timezone
    except:
        return None

def read_noaa(path):
    """Baca dataset NOAA tsunami."""
    events = []
    with open(path, 'r') as f:
        lines = f.readlines()

    # Skip header baris 1 dan search params baris 2
    header = [c.strip('"') for c in lines[0].strip().split('\t')]
    col_idx = {name: i for i, name in enumerate(header)}

    for line in lines[2:]:  # skip header + search params
        parts = line.split('\t')
        if len(parts) < 10:
            continue

        def get(col):
            idx = col_idx.get(col)
            if idx is not None and idx < len(parts):
                return parts[idx].strip().strip('"')
            return ''

        year = get('Year')
        month = get('Mo')
        day = get('Dy')

        dt = parse_noaa_date(year, month, day)
        lat = get('Latitude')
        lon = get('Longitude')

        if not dt or not lat or not lon:
            continue

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except:
            continue

        events.append({
            'datetime': dt,
            'year': int(float(year)) if year else None,
            'month': int(float(month)) if month else None,
            'day': int(float(day)) if day else None,
            'latitude': lat_f,
            'longitude': lon_f,
            'country': get('Country'),
            'location': get('Location Name'),
            'cause_code': get('Tsunami Cause Code'),
            'water_height': get('Maximum Water Height (m)'),
            'intensity': get('Tsunami Intensity'),
            'deaths': get('Deaths'),
            'damage': get('Damage ($Mil)'),
            'total_deaths': get('Total Deaths'),
            'total_damage': get('Total Damage ($Mil)'),
            'event_validity': get('Tsunami Event Validity'),
        })

    print(f"NOAA: {len(events)} tsunami events loaded")
    return events

def read_usgs(path):
    """Baca dataset USGS earthquake."""
    events = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = parse_usgs_time(row.get('time', ''))
            if not dt:
                continue

            try:
                lat = float(row.get('latitude', ''))
                lon = float(row.get('longitude', ''))
                mag = float(row.get('mag', ''))
                depth = float(row.get('depth', ''))
            except:
                continue

            events.append({
                'datetime': dt,
                'latitude': lat,
                'longitude': lon,
                'mag': mag,
                'depth': depth,
                'nst': row.get('nst', ''),
                'gap': row.get('gap', ''),
                'dmin': row.get('dmin', ''),
                'rms': row.get('rms', ''),
                'place': row.get('place', ''),
                'magType': row.get('magType', ''),
                'type': row.get('type', ''),
            })

    print(f"USGS: {len(events)} earthquakes loaded")
    return events

def match_events(noaa_events, usgs_events):
    """
    Match tsunami events ke earthquake events.
    Strategy: untuk setiap tsunami event, cari earthquake terdekat
    dalam tolerance waktu dan lokasi.
    """
    matched = 0
    matched_indices = set()

    for i, tsunami in enumerate(noaa_events):
        best_match = None
        best_distance = float('inf')

        for j, quake in enumerate(usgs_events):
            # Skip yang sudah di-match
            if j in matched_indices:
                continue

            # Time check
            time_diff = abs((tsunami['datetime'] - quake['datetime']).total_seconds())
            if time_diff > TIME_TOLERANCE_DAYS * 86400:
                continue

            # Location check
            dist = haversine_km(
                tsunami['latitude'], tsunami['longitude'],
                quake['latitude'], quake['longitude']
            )
            if dist > LAT_LON_TOLERANCE * 111:  # convert derajat ke km approx
                continue

            if dist < best_distance:
                best_distance = dist
                best_match = j

        if best_match is not None:
            matched_indices.add(best_match)
            matched += 1

    print(f"Matched: {matched}/{len(noaa_events)} tsunami events found in USGS")
    return matched_indices

def create_dataset(noaa_events, usgs_events, matched_indices):
    """
    Buat dataset gabungan:
    - Semua earthquake dari USGS
    - Label 1 jika match tsunami, 0 jika tidak
    - Fitur dari USGS + fitur dari NOAA (jika match)
    """
    rows = []

    # Header
    feature_cols = [
        'time', 'year', 'month', 'day',
        'latitude', 'longitude', 'mag', 'depth',
        'nst', 'gap', 'dmin', 'rms',
        'magType', 'place',
        'label',  # 0/1 tsunami
        'noaa_country', 'noaa_location', 'noaa_cause_code',
        'noaa_water_height', 'noaa_intensity',
    ]

    for j, quake in enumerate(usgs_events):
        row = {
            'time': quake['datetime'].isoformat(),
            'year': quake['datetime'].year,
            'month': quake['datetime'].month,
            'day': quake['datetime'].day,
            'latitude': quake['latitude'],
            'longitude': quake['longitude'],
            'mag': quake['mag'],
            'depth': quake['depth'],
            'nst': quake['nst'],
            'gap': quake['gap'],
            'dmin': quake['dmin'],
            'rms': quake['rms'],
            'magType': quake['magType'],
            'place': quake['place'],
            'label': 1 if j in matched_indices else 0,
            'noaa_country': '',
            'noaa_location': '',
            'noaa_cause_code': '',
            'noaa_water_height': '',
            'noaa_intensity': '',
        }

        # Add NOAA features if matched
        if j in matched_indices:
            # Find matching tsunami event
            for tsunami in noaa_events:
                time_diff = abs((tsunami['datetime'] - quake['datetime']).total_seconds())
                if time_diff <= TIME_TOLERANCE_DAYS * 86400:
                    dist = haversine_km(
                        tsunami['latitude'], tsunami['longitude'],
                        quake['latitude'], quake['longitude']
                    )
                    if dist <= LAT_LON_TOLERANCE * 111:
                        row['noaa_country'] = tsunami['country']
                        row['noaa_location'] = tsunami['location']
                        row['noaa_cause_code'] = tsunami['cause_code']
                        row['noaa_water_height'] = tsunami['water_height']
                        row['noaa_intensity'] = tsunami['intensity']
                        break

        rows.append(row)

    return feature_cols, rows

def main():
    # Setup logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, '.script_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, '1_data_integration.py.log')
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print("=" * 60)
    print("STEP 1: DATA INTEGRATION")
    print("NOAA Tsunami + USGS Earthquake")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading NOAA tsunami data...")
    noaa_events = read_noaa(NOAA_PATH)

    print("\n[2/4] Loading USGS earthquake data...")
    usgs_events = read_usgs(USGS_PATH)

    # Sort USGS by time for faster matching
    usgs_events.sort(key=lambda x: x['datetime'])

    print("\n[3/4] Matching events...")
    matched_indices = match_events(noaa_events, usgs_events)

    print("\n[4/4] Creating merged dataset...")
    feature_cols, rows = create_dataset(noaa_events, usgs_events, matched_indices)

    # Write output
    with open(OUTPUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=feature_cols)
        writer.writeheader()
        writer.writerows(rows)

    # Statistics
    total = len(rows)
    positive = sum(1 for r in rows if r['label'] == 1)
    negative = total - positive

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Total rows: {total}")
    print(f"Positive (tsunami): {positive}")
    print(f"Negative (no tsunami): {negative}")
    print(f"Ratio: 1:{negative//positive if positive else 'N/A'}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

if __name__ == '__main__':
    main()
