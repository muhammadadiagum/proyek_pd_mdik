#!/usr/bin/env python3
"""
Phase 2 Visualization — Data Understanding Plots
==================================================
Membuat visualisasi untuk memahami distribusi data.

Output: plots/*.png + plots/*.svg (vektor)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

# Paths (absolut terhadap lokasi skrip, bukan CWD pemanggil)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', '..', '00_data', 'interim', 'merged_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots')

# Style
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
    'neutral': '#95a5a6',
}


def save_fig(fig, name):
    """Simpan figure sebagai PNG + SVG. SVG tetap vektor (bukan rasterize)."""
    base = os.path.join(OUTPUT_DIR, name)
    fig.savefig(f'{base}.png', bbox_inches='tight')
    fig.savefig(f'{base}.svg', bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  [OK] {name}.png + {name}.svg")


def plot_label_distribution(df):
    """Bar chart distribusi label (tsunami vs non-tsunami)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    counts = df['label'].value_counts()
    labels = ['Non-Tsunami (0)', 'Tsunami (1)']
    colors = [COLORS['no_tsunami'], COLORS['tsunami']]
    bars = axes[0].bar(labels, [counts[0], counts[1]], color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Label Distribution (Linear)', fontweight='bold')
    axes[0].set_ylabel('Count')
    max_val = max(counts[0], counts[1])
    for bar, count in zip(bars, [counts[0], counts[1]]):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max_val * 0.02,
                     f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Log scale
    bars2 = axes[1].bar(labels, [counts[0], counts[1]], color=colors, edgecolor='white', linewidth=1.5)
    axes[1].set_yscale('log')
    axes[1].set_title('Label Distribution (Log Scale)', fontweight='bold')
    axes[1].set_ylabel('Count (log)')
    for bar, count in zip(bars2, [counts[0], counts[1]]):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.4,
                     f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ratio = counts[0] // counts[1]
    fig.suptitle(f'Imbalance Ratio: 1:{ratio}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'label_distribution')


def plot_magnitude_distribution(df):
    """Histogram distribusi magnitudo tsunami vs non-tsunami."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tsunami = df[df['label'] == 1]['mag']
    no_tsunami = df[df['label'] == 0]['mag']

    ax.hist(no_tsunami, bins=50, alpha=0.6, color=COLORS['no_tsunami'],
            label=f'Non-Tsunami (n={len(no_tsunami):,})', density=True, edgecolor='white')
    ax.hist(tsunami, bins=30, alpha=0.8, color=COLORS['tsunami'],
            label=f'Tsunami (n={len(tsunami):,})', density=True, edgecolor='white')

    ax.axvline(tsunami.median(), color=COLORS['tsunami'], linestyle='--', linewidth=2,
               label=f'Tsunami median: {tsunami.median():.1f}')
    ax.axvline(no_tsunami.median(), color=COLORS['no_tsunami'], linestyle='--', linewidth=2,
               label=f'Non-Tsunami median: {no_tsunami.median():.1f}')

    ax.set_xlabel('Magnitude (Mw)')
    ax.set_ylabel('Density')
    ax.set_title('Magnitude Distribution: Tsunami vs Non-Tsunami', fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    save_fig(fig, 'magnitude_distribution')


def plot_depth_distribution(df):
    """Histogram distribusi kedalaman."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tsunami = df[df['label'] == 1]['depth']
    no_tsunami = df[df['label'] == 0]['depth']

    ax.hist(no_tsunami, bins=60, alpha=0.6, color=COLORS['no_tsunami'],
            label=f'Non-Tsunami (n={len(no_tsunami):,})', density=True, edgecolor='white')
    ax.hist(tsunami, bins=40, alpha=0.8, color=COLORS['tsunami'],
            label=f'Tsunami (n={len(tsunami):,})', density=True, edgecolor='white')

    ax.axvline(tsunami.median(), color=COLORS['tsunami'], linestyle='--', linewidth=2,
               label=f'Tsunami median: {tsunami.median():.1f} km')
    ax.axvline(no_tsunami.median(), color=COLORS['no_tsunami'], linestyle='--', linewidth=2,
               label=f'Non-Tsunami median: {no_tsunami.median():.1f} km')

    ax.set_xlabel('Depth (km)')
    ax.set_ylabel('Density')
    ax.set_title('Earthquake Depth Distribution: Tsunami vs Non-Tsunami', fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    save_fig(fig, 'depth_distribution')


def plot_geographic_scatter(df):
    """Scatter plot lokasi gempa di peta."""
    fig, ax = plt.subplots(figsize=(14, 7))

    no_tsunami = df[df['label'] == 0]
    tsunami = df[df['label'] == 1]

    ax.scatter(no_tsunami['longitude'], no_tsunami['latitude'],
               s=1, alpha=0.1, color=COLORS['no_tsunami'], label='Non-Tsunami')
    ax.scatter(tsunami['longitude'], tsunami['latitude'],
               s=15, alpha=0.7, color=COLORS['tsunami'], label='Tsunami', zorder=5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geographic Distribution of Earthquakes', fontweight='bold')
    ax.legend(loc='lower left', markerscale=3, framealpha=0.9)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    save_fig(fig, 'geographic_scatter')


def plot_missing_values(df):
    """Heatmap missing values per kolom."""
    fig, ax = plt.subplots(figsize=(10, 6))

    missing_pct = df.isnull().sum() / len(df) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)

    if len(missing_pct) == 0:
        ax.text(0.5, 0.5, 'No Missing Values', ha='center', va='center',
                fontsize=16, fontweight='bold')
    else:
        colors = ['#e74c3c' if v > 50 else '#f39c12' if v > 10 else '#2ecc71'
                  for v in missing_pct.values]
        bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor='white')
        for bar, val in zip(bars, missing_pct.values):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height()/2.,
                    f'{val:.1f}%', va='center', fontweight='bold')
        ax.set_xlabel('Missing (%)')
        ax.set_title('Missing Values per Column', fontweight='bold')
        ax.set_xlim(0, 100)

    plt.tight_layout()
    save_fig(fig, 'missing_values_heatmap')


def plot_tsunami_per_decade(df):
    """Bar chart tsunami per dekade."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tsunami = df[df['label'] == 1].copy()
    tsunami['decade'] = (tsunami['year'] // 10) * 10
    decade_counts = tsunami['decade'].value_counts().sort_index()

    bars = ax.bar(decade_counts.index.astype(str), decade_counts.values,
                  color=COLORS['tsunami'], edgecolor='white', linewidth=1.5)

    for bar, count in zip(bars, decade_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.02,
                str(count), ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_xlabel('Decade')
    ax.set_ylabel('Number of Tsunami Events')
    ax.set_title('Tsunami Events per Decade (1900-2026)', fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_fig(fig, 'tsunami_per_decade')


def plot_top_countries(df):
    """Horizontal bar chart top 10 negara."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tsunami = df[df['label'] == 1]
    countries = tsunami['noaa_country'].dropna()
    top10 = countries.value_counts().head(10).sort_values(ascending=True)

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top10)))
    bars = ax.barh(top10.index, top10.values, color=colors, edgecolor='white')

    for bar, count in zip(bars, top10.values):
        ax.text(bar.get_width() + max(top10.values) * 0.01, bar.get_y() + bar.get_height()/2.,
                str(count), ha='left', va='center', fontweight='bold')

    ax.set_xlabel('Number of Tsunami Events')
    ax.set_title('Top 10 Countries by Tsunami Frequency', fontweight='bold')
    ax.set_xlim(0, max(top10.values) * 1.15)  # padding untuk label teks
    plt.tight_layout()
    save_fig(fig, 'top_countries')


def plot_correlation_heatmap(df):
    """Heatmap korelasi fitur numerik."""
    fig, ax = plt.subplots(figsize=(10, 8))

    numeric_cols = ['mag', 'depth', 'nst', 'gap', 'dmin', 'rms',
                    'latitude', 'longitude', 'label']
    corr = df[numeric_cols].corr()

    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
    ax.set_yticklabels(numeric_cols)

    # Annotate
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title('Feature Correlation Heatmap', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'correlation_heatmap')


def main():
    # Setup logging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, '.script_logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, '2_visualization.py.log')
    tee = Tee(log_path)
    sys.stdout = tee
    sys.stderr = tee

    print("=" * 60)
    print("PHASE 2: DATA UNDERSTANDING — VISUALIZATION")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    print("\nGenerating plots...")
    plot_label_distribution(df)
    plot_magnitude_distribution(df)
    plot_depth_distribution(df)
    plot_geographic_scatter(df)
    plot_missing_values(df)
    plot_tsunami_per_decade(df)
    plot_top_countries(df)
    plot_correlation_heatmap(df)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR}/ (*.png + *.svg)")
    print("=" * 60)


if __name__ == '__main__':
    main()
