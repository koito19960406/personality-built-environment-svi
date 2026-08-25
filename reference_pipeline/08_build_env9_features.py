# ---------------------------------------------------------------------------
# REFERENCE ONLY -- not runnable from this repository.
#
# This script belongs to the upstream pipeline that produced the analysis
# dataset. It requires a Google Street View API key and/or the panorama store
# held on the lab server, neither of which is redistributable. It is included
# to document how the published features were derived, not to be re-executed.
#
# Set PERSONALITY_SVI_ROOT and SVI_IMAGE_DIR if you adapt it to your own data.
# ---------------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""
Build 9 environmental composite features from zipcode-level spatial dataset.

# NOTE: The output parquet files still use 'env8' in their filenames for backward compatibility.

Inputs:
  - final_dataset_zipcode_spatial2.parquet (must contain columns like *_ratios_mean, *_counts_mean, etc.)
Outputs:
  - <out_prefix>_env9_summary.csv  (what was used, missingness, reliability stats)
  - <out_prefix>_env9_corr.csv    (correlation matrix among env9)
  - <out_prefix>_env9_corr.png (optional) (heatmap)

  python No no no"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Optional plotting (won't fail the whole run if not available)
try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:
    HAVE_PLT = False


# -----------------------------
# Utilities
# -----------------------------
def zscore(series: pd.Series) -> pd.Series:
    """Robust z-score: ignore NaNs; if std==0 return zeros."""
    s = series.astype(float)
    mu = np.nanmean(s.values)
    sd = np.nanstd(s.values)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mu) / sd


def cronbach_alpha(df_items: pd.DataFrame) -> float:
    """
    Cronbach's alpha for a set of items (columns). NaNs handled by pairwise covariance via pandas.
    Returns NaN if fewer than 2 items.
    """
    if df_items.shape[1] < 2:
        return np.nan

    # Drop rows where all items are NA (keeps partial rows for cov calculations)
    x = df_items.copy()
    # Variance of total score
    total = x.sum(axis=1, skipna=True)
    var_total = np.nanvar(total.values, ddof=1)
    if not np.isfinite(var_total) or var_total == 0:
        return np.nan

    # Sum of item variances
    var_items = np.nansum([np.nanvar(x[c].values, ddof=1) for c in x.columns])
    k = x.shape[1]
    alpha = (k / (k - 1)) * (1 - (var_items / var_total))
    return float(alpha)


def mean_offdiag_corr(df_items: pd.DataFrame) -> float:
    """Mean of off-diagonal correlations among items."""
    if df_items.shape[1] < 2:
        return np.nan
    corr = df_items.corr()
    mask = ~np.eye(corr.shape[0], dtype=bool)
    vals = corr.values[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.nanmean(vals)) if len(vals) else np.nan


def build_composite(df: pd.DataFrame, cols: list, name: str, min_items: int = 2):
    """
    Build composite = mean(zscore(item_i)) across available items.
    Returns composite series + diagnostics dict.
    """
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]

    diag = {
        "composite": name,
        "n_items_requested": len(cols),
        "n_items_present": len(present),
        "missing_columns": ";".join(missing) if missing else "",
    }

    if len(present) == 0:
        # Nothing to build
        comp = pd.Series(np.nan, index=df.index, name=name, dtype=float)
        diag.update({
            "status": "FAILED_no_columns",
            "alpha": np.nan,
            "mean_offdiag_corr": np.nan,
            "mean_row_missing_frac": np.nan,
        })
        return comp, diag

    # z-score each item
    z_items = pd.DataFrame({c: zscore(df[c]) for c in present}, index=df.index)

    # row-level missingness among present items (after zscore still NaN where original NaN)
    row_missing_frac = z_items.isna().mean(axis=1)

    # Composite: mean of available z-items
    comp = z_items.mean(axis=1, skipna=True)
    comp.name = name

    # Reliability stats (computed on z_items)
    alpha = cronbach_alpha(z_items) if len(present) >= min_items else np.nan
    mcor = mean_offdiag_corr(z_items) if len(present) >= min_items else np.nan

    diag.update({
        "status": "OK" if len(present) >= 1 else "FAILED",
        "alpha": alpha,
        "mean_offdiag_corr": mcor,
        "mean_row_missing_frac": float(np.nanmean(row_missing_frac.values)),
    })
    return comp, diag

def compute_flag_density_per_capita(
    df: pd.DataFrame,
    flag_density_col: str = "flag_density_per_km2_mean",
    pop_density_col: str = "us_census_population_density_km2",
    out_col: str = "flag_density_per_capita"
):
    """
    flags per capita = flag density (per km2) / population density (per km2)
    """
    df[out_col] = (
        df[flag_density_col] /
        df[pop_density_col].replace(0, np.nan)
    )
    return df

# -----------------------------
# Feature specification (edit here if needed)
# -----------------------------
DIRECT_FEATURES = {"env_symbolic_us_flag", "env_surveillance_cctv"}
def feature_spec():
    """
    9 composites: define each as a list of columns to be z-scored and averaged.
    Updated based on advisor feedback (Feb 2026).
    """
    return {
        # greenery on its own:
        "env_greenery": [
            "Vegetation_ratios_mean",
        ],
        
        # open space:
        "env_open_space": [
            "Sky_ratios_mean",
            "Terrain_ratios_mean",
            "Sand_ratios_mean",
        ],
        
        # building on its own:
        "env_building": [
            "Building_ratios_mean",
        ],
        
        # road simplified:
        "env_road": [
            "Road_ratios_mean",
            "Service Lane_ratios_mean",
        ],

        # Active Mobility Infrastructure
        "env_active_infra": [
            "Sidewalk_ratios_mean",
            "Bike Lane_ratios_mean",
            "Pedestrian Area_ratios_mean",
        ],

        # Active Mobility Presence
        "env_active_presence": [
            "Person_counts_mean",
            "Bicycle_counts_mean",
            "Bicyclist_counts_mean",
        ],

        # Vehicle Presence
        "env_vehicle_presence": [
            "Car_counts_mean",
            "Truck_counts_mean",
            "Bus_counts_mean",
            "Motorcycle_counts_mean",
            "Trailer_counts_mean",
            "Other Vehicle_counts_mean",
            "Caravan_counts_mean",
        ],

        # Physical Boundaries
        "env_physical_boundaries": [
            "Fence_ratios_mean",
            "Wall_ratios_mean",
            "Barrier_ratios_mean",
        ],

        # Cultural & Symbolic Elements
        "env_symbolic_us_flag": [
            "flag_count_mean",
        ],

        # 9) Surveillance (CCTV) - added as extra feature
        "env_surveillance_cctv": [
            "CCTV Camera_counts_mean"
        ],
    }


# -----------------------------
# Main
# -----------------------------
def main(in_path: str, out_prefix: str, keep_geometry: bool = True, make_plot: bool = True):
    in_path = str(in_path)
    out_prefix = str(out_prefix)

    # Read parquet (try geopandas first, fallback to pandas)
    gdf = None
    try:
        import geopandas as gpd
        gdf = gpd.read_parquet(in_path)
        df = pd.DataFrame(gdf)  # keeps geometry column
        have_geo = True
    except Exception:
        df = pd.read_parquet(in_path)
        have_geo = "geometry" in df.columns

    # Basic sanity
    print(f"[INFO] Loaded: {in_path}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Has geometry column: {have_geo}")

    # Compute any derived columns needed
    if "flag_density_per_capita" not in df.columns:
        # ---- Flag per-capita metric (NEW, density-based) ----
        required_cols = [
            "flag_density_per_km2_mean",
            "us_census_population_density_km2",
        ]

        if all(c in df.columns for c in required_cols):
            df = compute_flag_density_per_capita(df)
            print("[OK] Computed flag_density_per_capita (density-based)")

            # NEW: replace NaN with 0 by design choice
            df["flag_density_per_capita"] = df["flag_density_per_capita"].fillna(0.0)
            print("[INFO] Replaced NaN flag_density_per_capita with 0")

    # Build composites
    specs = feature_spec()
    diags = []

    for feat_name, cols in specs.items():
        # If you want a direct feature (no z-score / no averaging), take the first available column
        if feat_name in DIRECT_FEATURES:
            col = cols[0]
            if col in df.columns:
                df[feat_name] = df[col].astype(float)
                diag = {
                    "composite": feat_name,
                    "n_items_requested": len(cols),
                    "n_items_present": 1,
                    "missing_columns": "",
                    "status": "OK_direct",
                    "alpha": np.nan,
                    "mean_offdiag_corr": np.nan,
                    "mean_row_missing_frac": float(df[col].isna().mean()),
                }
                diags.append(diag)
                print(f"[OK] Built {feat_name} (direct): using {col}")
            else:
                df[feat_name] = np.nan
                diag = {
                    "composite": feat_name,
                    "n_items_requested": len(cols),
                    "n_items_present": 0,
                    "missing_columns": col,
                    "status": "FAILED_missing_direct_col",
                    "alpha": np.nan,
                    "mean_offdiag_corr": np.nan,
                    "mean_row_missing_frac": np.nan,
                }
                diags.append(diag)
                print(f"[WARN] Failed {feat_name}: missing {col}")
            continue

        # Default: composite behavior
        comp, diag = build_composite(df, cols, feat_name, min_items=2)
        df[feat_name] = comp
        diags.append(diag)
        print(f"[OK] Built {feat_name}: used {diag['n_items_present']}/{diag['n_items_requested']} items")

    diag_df = pd.DataFrame(diags)

    controls = [
        # --- exposure / sampling quality ---
        "grid_coverage_mean",
        "gsv_density_mean",
        "gsv_point_count_mean",
        "image_count",
        "zipcode_area_m2_mean",

        # --- city indicators ---
        "city_austin_texas",
        "city_dallas_texas",
        "city_houston_texas",
        "city_san_antonio_texas",

        # --- modeling controls (RAW) ---
        "visual_complexity_mean",
        "us_census_male_share",
        "us_census_age_15_29_share",
        "us_census_age_30_44_share",
        "us_census_age_45_59_share",
        "us_census_age_60_plus_share",
        "us_census_population_density_km2",

        # --- Big Five (RAW) ---
        "extraversion_mean",
        "agreeableness_mean",
        "conscientiousness_mean",
        "neuroticism_mean",
        "openness_mean",

        # --- core identifiers ---
        "now_zip",
        "zipcode",
        "city",
        "participant_count",
    ]


    present_controls = [c for c in controls if c in df.columns]
    missing_controls = [c for c in controls if c not in df.columns]
    if missing_controls:
        print(f"[WARN] Missing some suggested controls (ok): {missing_controls}")

    # Output dataframe: keep id + geometry + env9 + controls
    env9_cols = list(specs.keys())
    keep_cols = []
    for c in ["now_zip", "zipcode", "city", "participant_count", "geometry"]:
        if c in df.columns:
            keep_cols.append(c)

    keep_cols += env9_cols
    for c in present_controls:
        if c not in keep_cols:
            keep_cols.append(c)

    out_df = df[keep_cols].copy()

    # Optionally drop geometry if you want pure tabular
    if (not keep_geometry) and ("geometry" in out_df.columns):
        out_df = out_df.drop(columns=["geometry"])

    # Save parquet
    out_parquet = out_prefix + "_env8.parquet"
    try:
        # Write as geoparquet if geometry present
        if "geometry" in out_df.columns:
            import geopandas as gpd
            out_gdf = gpd.GeoDataFrame(out_df, geometry="geometry", crs=getattr(gdf, "crs", None) if gdf is not None else None)
            out_gdf.to_parquet(out_parquet, index=False)
        else:
            out_df.to_parquet(out_parquet, index=False)
    except Exception:
        # fallback
        out_df.to_parquet(out_parquet, index=False)

    print(f"[SAVE] {out_parquet}")

    # Save CSV (same content, different format)
    out_csv = out_prefix + "_env8.csv"

    if "geometry" in out_df.columns:
        # Convert geometry to WKT for CSV
        out_csv_df = out_df.copy()
        out_csv_df["geometry"] = out_csv_df["geometry"].astype(str)
    else:
        out_csv_df = out_df

    out_csv_df.to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")


    # Save diagnostics
    summary_csv = out_prefix + "_env9_summary.csv"
    diag_df.to_csv(summary_csv, index=False)
    print(f"[SAVE] {summary_csv}")

    # Correlation among composites
    corr = df[env9_cols].corr()
    corr_csv = out_prefix + "_env9_corr.csv"
    corr.to_csv(corr_csv)
    print(f"[SAVE] {corr_csv}")

    # Optional heatmap
    if make_plot and HAVE_PLT:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()
        im = ax.imshow(corr.values)
        ax.set_xticks(range(len(env9_cols)))
        ax.set_yticks(range(len(env9_cols)))
        ax.set_xticklabels(env9_cols, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(env9_cols, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        out_png = out_prefix + "_env9_corr.png"
        plt.savefig(out_png, dpi=200)
        print(f"[SAVE] {out_png}")

    # Print a quick recommendation preview
    print("\n=== QUICK DIAGNOSTICS (env9) ===")
    print(diag_df[["composite", "n_items_present", "alpha", "mean_offdiag_corr", "mean_row_missing_frac", "status"]])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input parquet path")
    ap.add_argument("--out-prefix", dest="out_prefix", required=True, help="Output prefix (no extension)")
    ap.add_argument("--keep-geometry", action="store_true", help="Keep geometry column in output")
    ap.add_argument("--no-plot", action="store_true", help="Disable heatmap plot output")
    args = ap.parse_args()

    main(
        in_path=args.in_path,
        out_prefix=args.out_prefix,
        keep_geometry=args.keep_geometry,
        make_plot=(not args.no_plot),
    )
