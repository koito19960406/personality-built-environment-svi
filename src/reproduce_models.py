"""Reproduce the OLS and spatial lag (SAR) models reported in the paper.

Fits, for each of the five Big Five traits:

  * pooled OLS with city fixed effects (HC1 robust standard errors),
  * city-specific OLS,
  * pooled spatial lag (ML_Lag) with Queen contiguity weights,
  * city-specific spatial lag.

Outputs are written to ``results/reproduced/`` and can be compared against the
published outputs in ``results/published_model_outputs/`` with
``src/check_reproduction.py``.

Usage:
    python src/reproduce_models.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
import shapely
import spreg
import statsmodels.api as sm
from esda.moran import Moran
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "analysis_dataset_zipcode.parquet"
DEFAULT_OUT = PROJECT_ROOT / "results" / "reproduced"

# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------

Y_COLS = [
    "extraversion_mean",
    "agreeableness_mean",
    "conscientiousness_mean",
    "neuroticism_mean",
    "openness_mean",
]

# Nine built environment composites plus the CCTV surveillance feature.
ENV_FEATURES = [
    "env_greenery",
    "env_open_space",
    "env_building",
    "env_road",
    "env_active_infra",
    "env_active_presence",
    "env_vehicle_presence",
    "env_physical_boundaries",
    "env_symbolic_us_flag",
    "env_surveillance_cctv",
]

CENSUS_CONTROLS = [
    "visual_complexity_mean",
    "us_census_male_share",
    "us_census_age_15_29_share",
    "us_census_age_30_44_share",
    "us_census_age_45_59_share",
    "us_census_age_60_plus_share",
    "us_census_population_density_km2",
]

# Already standardised in the dataset; the raw columns are the fallback.
ACS_PREF = ["acs_median_income_z", "acs_racial_diversity_z"]
ACS_FALLBACK = ["acs_median_income", "acs_racial_diversity_shannon"]

CITY_DUMMIES_ALL = [
    "city_austin_texas",
    "city_dallas_texas",
    "city_houston_texas",
    "city_san_antonio_texas",
]
CITY_BASELINE = "city_san_antonio_texas"

CITY_PATTERNS = {
    "austin": "city_austin_texas",
    "dallas": "city_dallas_texas",
    "houston": "city_houston_texas",
    r"san.antonio": "city_san_antonio_texas",
}

# 'san.antonio' matches both 'san antonio' and 'san_antonio'.
CITY_LEVELS = {
    "overall": None,
    "austin_texas": "austin",
    "dallas_texas": "dallas",
    "houston_texas": "houston",
    "san_antonio_texas": r"san.antonio",
}

MIN_N = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def coerce_geometry(series: pd.Series) -> pd.Series:
    """Coerce a geometry column of WKT strings or WKB bytes to shapely geometries."""
    if series.dropna().empty:
        return series
    sample = series.dropna().iloc[0]
    if isinstance(sample, str):
        return shapely.from_wkt(series.astype("string").to_numpy(dtype=object), on_invalid="ignore")
    if isinstance(sample, (bytes, bytearray, memoryview)):
        return shapely.from_wkb(series.to_numpy(dtype=object), on_invalid="ignore")
    return series


def normalize_city(s: pd.Series) -> pd.Series:
    """Lowercase, strip, collapse whitespace, remove non-alpha chars."""
    s = s.astype(str).str.strip().str.lower()
    s = s.str.replace(r"[^a-z\s]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True)


def subset_by_level(gdf_in: gpd.GeoDataFrame, level: str) -> gpd.GeoDataFrame:
    """Return the subset of the GeoDataFrame for a city level; 'overall' is a full copy."""
    gdf_in = gdf_in.loc[:, ~pd.Index(gdf_in.columns).duplicated()].copy()
    if level == "overall":
        return gdf_in
    pattern = CITY_LEVELS[level]
    city_norm = normalize_city(gdf_in["city"])
    return gdf_in[city_norm.str.contains(pattern, regex=True, na=False)].copy()


def load_data(data_path: Path) -> gpd.GeoDataFrame:
    """Load the analysis dataset and return it as a cleaned GeoDataFrame."""
    df = pd.read_parquet(data_path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()

    gdf = gpd.GeoDataFrame(
        df.copy(), geometry=coerce_geometry(df["geometry"]), crs="EPSG:4326"
    ).dropna(subset=["geometry"])
    gdf = gdf.loc[:, ~pd.Index(gdf.columns).duplicated()].copy()

    # Harmless if the geometries are already valid.
    gdf["geometry"] = gdf["geometry"].buffer(0)

    gdf["zipcode"] = pd.to_numeric(gdf["zipcode"], errors="coerce")
    gdf = gdf.dropna(subset=["zipcode"]).copy()
    gdf["zipcode"] = gdf["zipcode"].astype(int)
    gdf = gdf.drop_duplicates(subset=["zipcode"]).copy()
    return gdf


def build_design(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list, list, list]:
    """Rebuild city dummies, z-score predictors and outcomes, return column lists."""
    # Rebuilt from the 'city' column so stale dummy columns cannot leak in.
    city_norm = normalize_city(gdf["city"])
    for dummy in CITY_DUMMIES_ALL:
        gdf[dummy] = 0
    for pattern, dummy in CITY_PATTERNS.items():
        gdf.loc[city_norm.str.contains(pattern, regex=True, na=False), dummy] = 1
    for c in CITY_DUMMIES_ALL:
        gdf[c] = gdf[c].astype(int)
    city_dummies = [c for c in CITY_DUMMIES_ALL if c != CITY_BASELINE]

    acs_avail = [c for c in ACS_PREF if c in gdf.columns]
    acs_cols = acs_avail if len(acs_avail) == len(ACS_PREF) else [
        c for c in ACS_FALLBACK if c in gdf.columns
    ]

    cont_x = [c for c in (ENV_FEATURES + CENSUS_CONTROLS + acs_cols) if c in gdf.columns]
    to_zscore = [c for c in cont_x if not c.endswith("_z")]
    already_z = [c for c in cont_x if c.endswith("_z")]

    for c in to_zscore:
        gdf[f"{c}_z"] = stats.zscore(gdf[c].astype(float).to_numpy(), nan_policy="omit")

    x_base_z = [f"{c}_z" for c in to_zscore] + already_z
    x_pooled = x_base_z + city_dummies  # city fixed effects
    x_city = x_base_z  # city-specific models omit the dummies

    traits_z = []
    for y in Y_COLS:
        if y in gdf.columns:
            zname = f"{y}_z"
            gdf[zname] = stats.zscore(gdf[y].astype(float).to_numpy(), nan_policy="omit")
            traits_z.append(zname)

    return gdf, x_pooled, x_city, traits_z


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def fit_ols(sub: pd.DataFrame, y_col: str, x_cols: list):
    """OLS with HC1 robust SE; drops zero-variance predictors."""
    sub2 = sub.dropna(subset=[y_col] + x_cols).copy()
    x_raw = sub2[x_cols].astype(float)
    x_raw = x_raw.loc[:, ~x_raw.columns.duplicated()]
    keep = [c for c in x_raw.columns if x_raw[c].nunique(dropna=True) > 1]
    X = sm.add_constant(x_raw[keep], has_constant="add")
    y = sub2[y_col].astype(float)
    return sm.OLS(y, X).fit(cov_type="HC1")


def save_ols_outputs(model, trait: str, level: str, ols_dir: Path) -> None:
    """Save the OLS summary text and the coefficient CSV."""
    summary_path = ols_dir / "summary" / f"{trait}_{level}_summary.txt"
    try:
        summary_path.write_text(str(model.summary()), encoding="utf-8")
    except AssertionError:
        txt = (
            f"OLS fallback summary — {trait} | {level}\n\n"
            + model.params.to_string()
            + "\n\n"
            + pd.DataFrame(
                {
                    "coef": model.params,
                    "se": model.bse,
                    "t": model.tvalues,
                    "p": model.pvalues,
                }
            ).to_string()
        )
        summary_path.write_text(txt, encoding="utf-8")

    pd.DataFrame(
        {
            "variable": model.params.index.astype(str),
            "coefficient": model.params.values,
            "se": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
        }
    ).to_csv(ols_dir / "coefficients" / f"{trait}_{level}_coefs.csv", index=False)


def run_ols_models(gdf, traits_z, x_pooled, x_city, ols_dir: Path) -> pd.DataFrame:
    records = []
    for trait in traits_z:
        for level in CITY_LEVELS:
            sub = subset_by_level(gdf, level)
            x_use = x_pooled if level == "overall" else x_city
            n_ok = len(sub.dropna(subset=[trait] + x_use))
            if n_ok < MIN_N:
                print(f"OLS skip: {trait} | {level}  (N = {n_ok})")
                continue

            m = fit_ols(sub, trait, x_use)
            save_ols_outputs(m, trait, level, ols_dir)
            records.append(
                pd.DataFrame(
                    {
                        "trait": trait,
                        "level": level,
                        "term": m.params.index.astype(str),
                        "coef": m.params.values,
                        "se": m.bse.values,
                        "t": m.tvalues.values,
                        "p": m.pvalues.values,
                        "n": int(m.nobs),
                        "r2": float(m.rsquared),
                        "adj_r2": float(m.rsquared_adj),
                        "x_spec": "pooled+cityFE" if level == "overall" else "city_only",
                    }
                )
            )
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Spatial lag
# ---------------------------------------------------------------------------


def build_queen_weights(gdf_in: gpd.GeoDataFrame, drop_islands: bool = True):
    """Build Queen contiguity weights; optionally drop no-neighbour polygons."""
    g = gdf_in.dropna(subset=["geometry"]).reset_index(drop=True).copy()
    w = libpysal.weights.Queen.from_dataframe(g, use_index=False)
    islands = list(getattr(w, "islands", []))
    if islands and drop_islands:
        g = g.drop(index=islands).reset_index(drop=True)
        w = libpysal.weights.Queen.from_dataframe(g, use_index=False)
    w.transform = "r"
    info = {
        "n_before": int(len(gdf_in)),
        "n_after": int(len(g)),
        "islands_dropped": int(len(islands)) if (islands and drop_islands) else 0,
    }
    return g, w, info


def run_spatial_lag(y: pd.Series, X: pd.DataFrame, w):
    Xc = sm.add_constant(X, prepend=True)
    return spreg.ML_Lag(
        y.values.reshape(-1, 1),
        Xc.values,
        w=w,
        name_y=y.name,
        name_x=list(Xc.columns),
        method="full",
    )


def tidy_mllag(model, trait: str, level: str) -> pd.DataFrame:
    """Extract a tidy coefficient table from a spreg ML_Lag model."""
    zstat = list(getattr(model, "z_stat", []))
    stderr = np.asarray(getattr(model, "std_err", []), dtype=float).flatten()
    betas = np.asarray(getattr(model, "betas", []), dtype=float).flatten()
    name_x = list(getattr(model, "name_x", []))
    rho = float(getattr(model, "rho", np.nan))

    names = ["rho"] + name_x
    coefs = [rho] + betas.tolist()
    if len(zstat) == len(name_x) + 1:
        z_vals = [float(a) for a, _ in zstat]
        p_vals = [float(b) for _, b in zstat]
        se = stderr.tolist() if len(stderr) == len(names) else [np.nan] * len(names)
    else:
        se = stderr.tolist() if len(stderr) == len(coefs) else [np.nan] * len(coefs)
        z_vals = [np.nan] * len(coefs)
        p_vals = [np.nan] * len(coefs)

    return pd.DataFrame(
        {
            "trait": trait,
            "level": level,
            "variable": names,
            "coefficient": coefs,
            "se": se,
            "z": z_vals,
            "p": p_vals,
            "n": int(getattr(model, "n", np.nan)),
            "logll": float(getattr(model, "logll", np.nan)),
        }
    )


def run_sar_models(gdf, traits_z, x_pooled, x_city, sar_dir: Path) -> pd.DataFrame:
    records = []
    for trait in traits_z:
        for level in CITY_LEVELS:
            sub = subset_by_level(gdf, level)
            x_use = x_pooled if level == "overall" else x_city

            df_model = sub.dropna(subset=["geometry", trait] + x_use).reset_index(drop=True)
            if len(df_model) < MIN_N:
                print(f"SAR skip: {trait} | {level}  (N = {len(df_model)})")
                continue

            try:
                df_w, w, w_info = build_queen_weights(df_model, drop_islands=True)
            except Exception as e:  # noqa: BLE001
                print(f"SAR skip (weights): {trait} | {level}  ({e})")
                continue

            if len(df_w) < MIN_N:
                print(f"SAR skip after islands: {trait} | {level}  (N = {len(df_w)})")
                continue
            if w_info["islands_dropped"]:
                print(f"  {trait} | {level} | islands dropped: {w_info['islands_dropped']}")

            y = df_w[trait].astype(float)
            X = df_w[x_use].astype(float)
            X = X.loc[:, ~X.columns.duplicated()]
            X = X[[c for c in X.columns if X[c].nunique(dropna=True) > 1]]

            # Moran's I on the OLS residuals, as a spatial autocorrelation diagnostic.
            mi_I, mi_p = np.nan, np.nan
            try:
                ols0 = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
                mi = Moran(ols0.resid.values, w)
                mi_I, mi_p = float(mi.I), float(mi.p_sim)
                print(f"  {trait} | {level} | Moran I = {mi_I:.4f}  p_sim = {mi_p:.3g}")
            except Exception:  # noqa: BLE001
                pass

            try:
                model = run_spatial_lag(y, X, w)
            except Exception as e:  # noqa: BLE001
                print(f"SAR error: {trait} | {level}  ({e})")
                continue

            summary_txt = "\n".join(
                [
                    f"Trait: {trait}",
                    f"Level: {level}",
                    f"Weights info: {w_info}",
                    f"Moran_I_resid: {mi_I}",
                    f"Moran_p_sim:   {mi_p}",
                    "",
                    str(model.summary),
                ]
            )
            (sar_dir / "summary" / f"{trait}_{level}_summary.txt").write_text(
                summary_txt, encoding="utf-8"
            )

            coef_df = tidy_mllag(model, trait, level)
            coef_df["moran_I_resid"] = mi_I
            coef_df["moran_p_sim"] = mi_p
            coef_df["x_spec"] = "pooled+cityFE" if level == "overall" else "city_only"
            coef_df.to_csv(sar_dir / "coefficients" / f"{trait}_{level}_coefs.csv", index=False)
            records.append(coef_df)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Analysis dataset parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = ap.parse_args()

    ols_dir = args.out / "OLS"
    sar_dir = args.out / "spatial_lag"
    for d in [
        args.out,
        ols_dir / "summary",
        ols_dir / "coefficients",
        sar_dir / "summary",
        sar_dir / "coefficients",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    gdf = load_data(args.data)
    print(f"Loaded {len(gdf)} ZIP codes from {args.data.name}")
    print(gdf["city"].value_counts().to_string())

    gdf, x_pooled, x_city, traits_z = build_design(gdf)
    print(f"\nPredictors (pooled, {len(x_pooled)}): {x_pooled}")
    print(f"Outcomes ({len(traits_z)}): {traits_z}\n")

    print("=" * 60)
    print("OLS")
    print("=" * 60)
    ols_tidy = run_ols_models(gdf, traits_z, x_pooled, x_city, ols_dir)
    ols_tidy.to_csv(args.out / "ols_results_tidy_all.csv", index=False)

    r2 = ols_tidy[ols_tidy["term"] == "const"][["trait", "level", "n", "r2", "adj_r2"]]
    r2.to_csv(args.out / "ols_r2_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("SPATIAL LAG (ML_Lag, Queen contiguity, row-standardised)")
    print("=" * 60)
    sar_tidy = run_sar_models(gdf, traits_z, x_pooled, x_city, sar_dir)
    sar_tidy.to_csv(args.out / "sar_results_tidy_all.csv", index=False)

    rho = sar_tidy[sar_tidy["variable"] == "rho"].rename(
        columns={"coefficient": "rho", "p": "rho_p"}
    )
    rho.to_csv(args.out / "sar_rho_moran_summary.csv", index=False)

    print(f"\nDone. Outputs written to {args.out}")


if __name__ == "__main__":
    main()
