import os
"""
generate_alignment_figures.py
------------------------------
Generates exactly two figures for Koichi's alignment request:

  1. reports/figures/city_maps/all_cities_features_matrix_zscore.png
     - 5×4 grid: the 5 features that match the percentile grid
     - uses V2 post-2010 dataset

  2. reports/figures/appendix_maps/zipcode_model_inclusion_mask.png
     - 1×4 grid: all ZIP codes coloured by model inclusion status
     - green  = included (participant_count>=20 AND grid_coverage_mean>=0.03)
     - red    = excluded by participant count only
     - orange = excluded by grid coverage only
     - purple = excluded by both filters
     - uses V2 post-2010 dataset (same thresholds as the paper)

Does NOT regenerate individual city maps or any other figure.

Usage:
    conda run -n gisenv python personality_svi/revision2/generate_alignment_figures.py
"""

from pathlib import Path

import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from shapely import wkt

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("PERSONALITY_SVI_ROOT", Path(__file__).resolve().parents[2]))

# V2: post-2010 personality data only
IN_PARQUET = (
    PROJECT_ROOT
    / "data/processed/model"
    / "final_dataset_zipcode_spatial2_env8_medianincome_racialdiversity_post2010.parquet"
)

OUT_MATRIX   = PROJECT_ROOT / "reports/figures/city_maps"
OUT_APPENDIX = PROJECT_ROOT / "reports/figures/appendix_maps"

OUT_MATRIX.mkdir(parents=True, exist_ok=True)
OUT_APPENDIX.mkdir(parents=True, exist_ok=True)

CITIES = ["austin_texas", "dallas_texas", "houston_texas", "san_antonio_texas"]
CITY_LABELS = {
    "austin_texas":      "Austin",
    "dallas_texas":      "Dallas",
    "houston_texas":     "Houston",
    "san_antonio_texas": "San Antonio",
}

# 5 features — matches rebuild_grid.py FEATURES_ORDER exactly
MATRIX_FEATURES = [
    "env_building",
    "env_greenery",
    "env_open_space",
    "env_road",
    "env_vehicle_presence",
]
FEATURE_LABELS = {
    "env_building":       "Building",
    "env_greenery":       "Greenery",
    "env_open_space":     "Open Space",
    "env_road":           "Road",
    "env_vehicle_presence": "Vehicle Presence",
}

Z_VMIN, Z_VMAX = -2, 2
CMAP = plt.cm.RdBu_r

# V2 model filter thresholds
MIN_PARTICIPANT   = 20
MIN_GRID_COVERAGE = 0.03

# ---------------------------------------------------------------------------
# Load V2 parquet and build GeoDataFrame (all ZIPs in 4 cities, no model filter)
# ---------------------------------------------------------------------------
print(f"Loading: {IN_PARQUET.name}")
df = pd.read_parquet(IN_PARQUET)
df.columns = df.columns.str.strip()

sample = df["geometry"].dropna().iloc[0]
if isinstance(sample, str):
    df["geometry"] = df["geometry"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) else x
    )

gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf = gdf[gdf["city"].isin(CITIES)].copy()
gdf = gdf.to_crs(epsg=3857)

print(f"  Loaded {len(gdf)} ZIP codes across {gdf['city'].nunique()} cities")

# Precompute city extents from all ZIPs (not model-filtered subset)
city_info = {}
for city in CITIES:
    cg = gdf[gdf["city"] == city]
    bounds = cg.total_bounds
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    radius = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / 2 + 2000
    city_info[city] = dict(center=(cx, cy), radius=radius)

# ---------------------------------------------------------------------------
# Figure 1: 5×4 feature matrix
# ---------------------------------------------------------------------------
print("\n--- Figure 1: 5×4 feature matrix ---")

# Compute global z-scores for the 5 features
for feat in MATRIX_FEATURES:
    zcol = f"{feat}__z"
    vals = gdf[feat].to_numpy(dtype=float)
    gdf[zcol] = stats.zscore(vals, nan_policy="omit")

n_rows = len(MATRIX_FEATURES)
n_cols = len(CITIES)

fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows + 0.8))
fig.patch.set_facecolor("white")

gs = mgridspec.GridSpec(
    n_rows + 1, n_cols,
    height_ratios=[4] * n_rows + [0.25],
    hspace=0.05, wspace=0.05,
    figure=fig,
)

for row_idx, feat in enumerate(MATRIX_FEATURES):
    zcol = f"{feat}__z"
    for col_idx, city in enumerate(CITIES):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        cg = gdf[gdf["city"] == city]
        ax.set_facecolor("white")
        cg.plot(
            ax=ax, column=zcol, cmap=CMAP,
            edgecolor="white", linewidth=0.3,
            vmin=Z_VMIN, vmax=Z_VMAX, alpha=1.0,
        )
        cx, cy = city_info[city]["center"]
        r = city_info[city]["radius"]
        ax.set_xlim(cx - r, cx + r)
        ax.set_ylim(cy - r, cy + r)
        ax.axis("off")
        if row_idx == 0:
            ax.set_title(CITY_LABELS[city], fontsize=16, pad=8, fontweight="bold")
        if col_idx == 0:
            ax.text(
                -0.04, 0.5, FEATURE_LABELS[feat],
                transform=ax.transAxes, fontsize=12,
                va="center", ha="right", rotation=90,
            )

# Shared colorbar
cax = fig.add_subplot(gs[n_rows, :])
norm = plt.Normalize(Z_VMIN, Z_VMAX)
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_ticks([-2, -1, 0, 1, 2])
cb.set_ticklabels(["-2\u03c3", "-1\u03c3", "0", "1\u03c3", "2\u03c3"], fontsize=12)
cax.set_title("Z-score", fontsize=12, pad=4)

out1 = OUT_MATRIX / "all_cities_features_matrix_zscore.png"
plt.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {out1}")

# ---------------------------------------------------------------------------
# Figure 2: Appendix ZIP inclusion/exclusion map
# ---------------------------------------------------------------------------
print("\n--- Figure 2: Appendix ZIP inclusion/exclusion map ---")

mask_pc   = gdf["participant_count"]  >= MIN_PARTICIPANT
mask_grid = gdf["grid_coverage_mean"] >= MIN_GRID_COVERAGE

def _status(pc_ok, grid_ok):
    if pc_ok and grid_ok:         return "included"
    if not pc_ok and grid_ok:     return "excl_count"
    if pc_ok and not grid_ok:     return "excl_grid"
    return "excl_both"

gdf["_status"] = [_status(pc, gr) for pc, gr in zip(mask_pc, mask_grid)]

n = {s: int((gdf["_status"] == s).sum()) for s in
     ["included", "excl_count", "excl_grid", "excl_both"]}
print(f"  included:   {n['included']}")
print(f"  excl_count: {n['excl_count']}  (participant_count < {MIN_PARTICIPANT})")
print(f"  excl_grid:  {n['excl_grid']}  (grid_coverage_mean < {MIN_GRID_COVERAGE})")
print(f"  excl_both:  {n['excl_both']}  (both filters)")
print(f"  total:      {sum(n.values())}")

STATUS_COLORS = {
    "included":   "#2ca02c",
    "excl_count": "#d62728",
    "excl_grid":  "#ff7f0e",
    "excl_both":  "#9467bd",
}
STATUS_LABELS = {
    "included":   f"Included in model  (n={n['included']})",
    "excl_count": f"Excluded: participant count < {MIN_PARTICIPANT}  (n={n['excl_count']})",
    "excl_grid":  f"Excluded: grid coverage < {MIN_GRID_COVERAGE}  (n={n['excl_grid']})",
    "excl_both":  f"Excluded: both filters  (n={n['excl_both']})",
}

# Draw included ZIPs first (bottom layer), excluded on top so they are visible
DRAW_ORDER = ["included", "excl_count", "excl_grid", "excl_both"]

fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.5))
fig.patch.set_facecolor("white")

for col_idx, city in enumerate(CITIES):
    ax = axes[col_idx]
    cg = gdf[gdf["city"] == city]
    ax.set_facecolor("white")
    for status in DRAW_ORDER:
        sub = cg[cg["_status"] == status]
        if len(sub):
            sub.plot(ax=ax, color=STATUS_COLORS[status],
                     edgecolor="white", linewidth=0.3, alpha=0.9)
    cx, cy = city_info[city]["center"]
    r = city_info[city]["radius"]
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.axis("off")
    ax.set_title(CITY_LABELS[city], fontsize=14, pad=6, fontweight="bold")

patches = [
    mpatches.Patch(color=STATUS_COLORS[s], label=STATUS_LABELS[s])
    for s in DRAW_ORDER
]
fig.legend(
    handles=patches, loc="lower center", ncol=2,
    fontsize=10, framealpha=0.92,
    bbox_to_anchor=(0.5, -0.10),
)
n_total    = sum(n.values())
n_excluded = n_total - n["included"]
fig.suptitle(
    f"ZIP code model inclusion  "
    f"[total: {n_total}  |  included: {n['included']}  |  excluded: {n_excluded}]",
    fontsize=13, fontweight="bold", y=1.02,
)

out2 = OUT_APPENDIX / "zipcode_model_inclusion_mask.png"
plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved: {out2}")

print("\nDone.")
