#!/usr/bin/env python3
"""
Phase 3 Visualization — Data Preparation Plots
================================================
Membuat visualisasi untuk fitur baru dan distribusi.

Input: prepared_dataset.csv (dari Step 1)
Output: plots/*.png + plots/*.svg
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../00_data/processed/prepared_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots')

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
    'figure.dpi': 150,
})

COLORS = {
    'tsunami': '#e74c3c',
    'no_tsunami': '#3498db',
    'accent': '#2ecc71',
}

# ============================================================
# SAVE HELPER
# ============================================================

def save_fig(fig, name):
    """Simpan figure sebagai PNG + SVG."""
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.png'),
                bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(OUTPUT_DIR, f'{name}.svg'),
                bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"  [OK] {name}.png + {name}.svg")

# ============================================================
# PLOTS
# ============================================================

def plot_new_feature_distributions(df):
    """Distribusi fitur baru: depth_mag_ratio, abs_latitude, monitoring_quality."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    features = ['depth_mag_ratio', 'abs_latitude', 'monitoring_quality']
    titles = ['Depth/Magnitude Ratio', 'Absolute Latitude', 'Monitoring Quality']

    for ax, feat, title in zip(axes, features, titles):
        tsunami = df[df['label'] == 1][feat].dropna()
        no_tsunami = df[df['label'] == 0][feat].dropna()

        # Clip outliers for better visualization
        q99 = df[feat].quantile(0.99)
        q01 = df[feat].quantile(0.01)

        ax.hist(no_tsunami.clip(q01, q99), bins=50, alpha=0.6, color=COLORS['no_tsunami'],
                label='Non-Tsunami', density=True, edgecolor='white')
        ax.hist(tsunami.clip(q01, q99), bins=30, alpha=0.8, color=COLORS['tsunami'],
                label='Tsunami', density=True, edgecolor='white')

        ax.axvline(tsunami.median(), color=COLORS['tsunami'], linestyle='--', linewidth=2)
        ax.axvline(no_tsunami.median(), color=COLORS['no_tsunami'], linestyle='--', linewidth=2)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(feat)
        ax.legend(fontsize=9)

    plt.suptitle('New Feature Distributions: Tsunami vs Non-Tsunami', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'new_feature_distributions')


def plot_tectonic_zone_pie(df):
    """Pie chart distribusi tectonic zone."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # All data
    zone_all = df['tectonic_zone'].value_counts()
    colors_all = plt.cm.Set2(np.linspace(0, 1, len(zone_all)))
    wedges, texts, autotexts = axes[0].pie(
        zone_all.values, labels=zone_all.index, autopct='%1.1f%%',
        colors=colors_all, startangle=90, textprops={'fontsize': 9}
    )
    axes[0].set_title('All Earthquakes', fontweight='bold')

    # Tsunami only
    tsunami = df[df['label'] == 1]
    zone_tsunami = tsunami['tectonic_zone'].value_counts()
    colors_ts = plt.cm.Set2(np.linspace(0, 1, len(zone_tsunami)))
    wedges2, texts2, autotexts2 = axes[1].pie(
        zone_tsunami.values, labels=zone_tsunami.index, autopct='%1.1f%%',
        colors=colors_ts, startangle=90, textprops={'fontsize': 9}
    )
    axes[1].set_title('Tsunami Events Only', fontweight='bold')

    plt.suptitle('Tectonic Zone Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'tectonic_zone_pie')


def plot_monitoring_quality_boxplot(df):
    """Boxplot monitoring quality per tectonic zone."""
    fig, ax = plt.subplots(figsize=(10, 5))

    zones = df['tectonic_zone'].unique()
    data_zones = [df[df['tectonic_zone'] == z]['monitoring_quality'].dropna() for z in zones]

    bp = ax.boxplot(data_zones, patch_artist=True)
    ax.set_xticklabels(zones, rotation=30, ha='right')

    colors = plt.cm.Set2(np.linspace(0, 1, len(zones)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel('Tectonic Zone')
    ax.set_ylabel('Monitoring Quality Score')
    ax.set_title('Monitoring Quality by Tectonic Zone', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'monitoring_quality_boxplot')


def plot_era_distribution(df):
    """Bar chart distribusi era (tsunami vs non-tsunami)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    era_order = ['early_1900s', 'mid_1900s', 'late_1900s', 'early_2000s', 'modern']

    tsunami = df[df['label'] == 1]['era'].value_counts()
    no_tsunami = df[df['label'] == 0]['era'].value_counts()

    x = np.arange(len(era_order))
    width = 0.35

    ts_vals = [tsunami.get(e, 0) for e in era_order]
    nts_vals = [no_tsunami.get(e, 0) for e in era_order]

    bars1 = ax.bar(x - width/2, ts_vals, width, label='Tsunami', color=COLORS['tsunami'], edgecolor='white')
    bars2 = ax.bar(x + width/2, nts_vals, width, label='Non-Tsunami', color=COLORS['no_tsunami'], edgecolor='white')

    ax.set_xlabel('Era')
    ax.set_ylabel('Count')
    ax.set_title('Event Distribution by Era', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(era_order, rotation=30, ha='right')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    save_fig(fig, 'era_distribution')


def plot_season_distribution(df):
    """Bar chart distribusi musim."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    season_order = ['djf', 'mam', 'jja', 'son']
    season_labels = ['DJF (Dec-Feb)', 'MAM (Mar-May)', 'JJA (Jun-Aug)', 'SON (Sep-Nov)']

    # All data
    season_all = df['season'].value_counts()
    vals_all = [season_all.get(s, 0) for s in season_order]
    axes[0].bar(season_labels, vals_all, color=plt.cm.Pastel1(np.linspace(0, 1, 4)), edgecolor='white')
    axes[0].set_title('All Earthquakes', fontweight='bold')
    axes[0].set_ylabel('Count')

    # Tsunami only
    tsunami = df[df['label'] == 1]
    season_ts = tsunami['season'].value_counts()
    vals_ts = [season_ts.get(s, 0) for s in season_order]
    axes[1].bar(season_labels, vals_ts, color=[COLORS['tsunami']]*4, alpha=0.7, edgecolor='white')
    axes[1].set_title('Tsunami Events Only', fontweight='bold')
    axes[1].set_ylabel('Count')

    plt.suptitle('Seasonal Distribution', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'season_distribution')


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 3: DATA PREPARATION — VISUALIZATION")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    print("\nGenerating plots...")
    plot_new_feature_distributions(df)
    plot_tectonic_zone_pie(df)
    plot_monitoring_quality_boxplot(df)
    plot_era_distribution(df)
    plot_season_distribution(df)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR}/ (*.png + *.svg)")
    print(f"Log: {LOG_PATH}")
    print("=" * 60)

    tee.close()

if __name__ == '__main__':
    main()
