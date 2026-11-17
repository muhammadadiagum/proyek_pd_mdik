#!/usr/bin/env python3
"""
2_visualization.py — Phase 4: Model Comparison Plots
"""

import os, json, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# LOGGING
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, '.script_logs')
os.makedirs(LOG_DIR, exist_ok=True)
tee = Tee(os.path.join(LOG_DIR, '2_visualization.py.log'))
sys.stdout = tee


plt.rcParams.update({
    'figure.facecolor': '#f8f9fa', 'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#cccccc', 'axes.grid': True, 'grid.alpha': 0.3,
    'font.size': 11, 'figure.dpi': 150,
})

RESULTS_PATH = "results/model_results.json"
PLOTS_DIR = "plots"


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    # Use Original method for best per-model comparison
    method = 'Original'
    data = results[method]
    model_names = list(data.keys())
    x = np.arange(len(model_names))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    f1s = [data[n]['f1'] for n in model_names]
    recs = [data[n]['recall'] for n in model_names]
    precs = [data[n]['precision'] for n in model_names]
    aucs = [data[n].get('auc', 0) for n in model_names]
    costs = [data[n]['cost'] for n in model_names]

    # --- Plot 1: F1, Recall, Precision ---
    fig, ax = plt.subplots(figsize=(12, 6))
    w = 0.25
    ax.bar(x - w, f1s, w, label='F1-Score', color='#3498db', edgecolor='white')
    ax.bar(x, recs, w, label='Recall', color='#2ecc71', edgecolor='white')
    ax.bar(x + w, precs, w, label='Precision', color='#e74c3c', edgecolor='white')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: F1 / Recall / Precision (Original Data)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_metrics.png'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_metrics.svg'), bbox_inches='tight')
    plt.close()
    print("  [OK] plots/model_comparison_metrics.png + .svg")

    # --- Plot 2: AUC ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, aucs, color=colors, edgecolor='white', width=0.5)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('AUC-ROC Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.85, 1.0)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_auc.png'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_auc.svg'), bbox_inches='tight')
    plt.close()
    print("  [OK] plots/model_comparison_auc.png + .svg")

    # --- Plot 3: Cost ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, costs, color=colors, edgecolor='white', width=0.5)
    ax.set_ylabel('Cost (FN×100 + FP×1)')
    ax.set_title('Cost Comparison (lower = better)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    for bar, v in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{v:,}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_cost.png'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_cost.svg'), bbox_inches='tight')
    plt.close()
    print("  [OK] plots/model_comparison_cost.png + .svg")

    # --- Plot 4: All methods heatmap ---
    methods = list(results.keys())
    heatmap_data = []
    for met in methods:
        row = [results[met][n]['f1'] for n in model_names]
        heatmap_data.append(row)
    heatmap_arr = np.array(heatmap_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heatmap_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.5)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(len(methods)):
        for j in range(len(model_names)):
            ax.text(j, i, f'{heatmap_arr[i,j]:.4f}', ha='center', va='center', fontsize=9,
                    color='white' if heatmap_arr[i,j] < 0.1 else 'black')
    plt.colorbar(im, ax=ax, label='F1-Score')
    ax.set_title('F1-Score Heatmap: Models × Imbalance Methods', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_heatmap_f1.png'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOTS_DIR, 'model_heatmap_f1.svg'), bbox_inches='tight')
    plt.close()
    print("  [OK] plots/model_heatmap_f1.png + .svg")

    print(f"\n  Total: 4 plots (PNG + SVG each)")


if __name__ == '__main__':
    main()
