import os
"""
generate_figure6_histogram.py
------------------------------
Generates Figure 6 (Appendix): temporal histogram comparing personality survey
samples (>=2010) vs Street View imagery counts by year.

Grayscale-compatible for print production:
  - Personality samples: solid gray bars (#888888)
  - Street View images:  white bars with /// hatch pattern

No title embedded (caption goes in the manuscript).

Usage:
    conda run -n gisenv python personality_svi/revision2/generate_figure6_histogram.py

Output:
    reports/figures/appendix_maps/KangFigure6_temporal_histogram.png  (600 dpi, <=7 in wide)
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("PERSONALITY_SVI_ROOT", Path(__file__).resolve().parents[2]))
IN_CSV  = PROJECT_ROOT / "data/interim/_processed/year_personality_vs_streetview.csv"
OUT_DIR = PROJECT_ROOT / "reports/figures/appendix_maps"
OUT_FIG = OUT_DIR / "KangFigure6_temporal_histogram.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading: {IN_CSV.name}")
df = pd.read_csv(IN_CSV)
print(f"  Years: {df['year'].min()}–{df['year'].max()},  rows: {len(df)}")

# Personality data filtered to >=2010 in the paper; zero out earlier years
df_plot = df.copy()
df_plot.loc[df_plot["year"] < 2010, "n_personality_samples"] = 0

years  = df_plot["year"].to_numpy()
survey = df_plot["n_personality_samples"].to_numpy()
svi    = df_plot["n_streetview_images"].to_numpy()

# ---------------------------------------------------------------------------
# Plot: side-by-side dual-axis bar chart
# ---------------------------------------------------------------------------
bar_width = 0.38
x = np.arange(len(years))

fig, ax1 = plt.subplots(figsize=(7, 3.5))
fig.patch.set_facecolor("white")
ax2 = ax1.twinx()

# Survey bars: solid gray on left y-axis
ax1.bar(
    x - bar_width / 2, survey, width=bar_width,
    color="#888888", edgecolor="black", linewidth=0.4, zorder=3,
)

# SVI bars: white with /// hatch on right y-axis
ax2.bar(
    x + bar_width / 2, svi, width=bar_width,
    facecolor="white", edgecolor="black", linewidth=0.4, hatch="///", zorder=3,
)

# x-axis
ax1.set_xticks(x)
ax1.set_xticklabels(years, rotation=45, ha="right", fontsize=7)
ax1.set_xlabel("Year", fontsize=8)
ax1.set_xlim(-0.7, len(years) - 0.3)

# y-axes (both in black for grayscale)
ax1.set_ylabel("Number of Personality Samples", fontsize=8, color="black")
ax2.set_ylabel("Number of Street View Images",  fontsize=8, color="black")
ax1.tick_params(axis="y", labelsize=7, colors="black")
ax2.tick_params(axis="y", labelsize=7, colors="black")

# Subtle horizontal gridlines on primary axis only
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.5, zorder=0)
ax1.set_axisbelow(True)

# Legend: match fill/hatch encoding to the bars
patch_survey = mpatches.Patch(
    facecolor="#888888", edgecolor="black", linewidth=0.5,
    label="Personality samples (≥2010)",
)
patch_svi = mpatches.Patch(
    facecolor="white", edgecolor="black", linewidth=0.5,
    hatch="///", label="Street View images",
)
ax1.legend(
    handles=[patch_survey, patch_svi],
    loc="upper left", fontsize=7,
    framealpha=0.9, edgecolor="gray",
)

# Explicit margins so the saved file is exactly 7 in wide (no bbox trim)
fig.subplots_adjust(left=0.10, right=0.92, top=0.97, bottom=0.22)
plt.savefig(OUT_FIG, dpi=600, facecolor="white")
plt.close(fig)
print(f"Saved: {OUT_FIG}")
