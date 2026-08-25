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

import os

print(os.environ.get("CENSUS_API_KEY"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from census import Census
from us import states


# =========================
# USER SETTINGS
# =========================
IN_CSV = "<PROJECT_ROOT>/data/processed/model/final_dataset_zipcode_spatial2_env8.csv"   # Change to your real path
OUT_CSV = "<PROJECT_ROOT>/data/processed/model/final_dataset_zipcode_spatial2_env8_medianincome_racialdiversity.csv"

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "").strip()
if not CENSUS_API_KEY:
    raise RuntimeError(
        "Missing Census API key. Set env var first:\n"
        "export CENSUS_API_KEY='YOUR_KEY'\n"
        "Then rerun."
    )

ACS_YEAR = 2020                 # ACS 2016–2020 5-year => use 2020 acs5
STATE_FIPS = states.TX.fips     # Texas only; if you want US-wide, see note below.


# =========================
# LOAD YOUR ZIP LIST
# =========================
df = pd.read_csv(IN_CSV)

# choose the right zip column (prefer 'zipcode' if exists)
zip_col = "zipcode" if "zipcode" in df.columns else "now_zip"
if zip_col not in df.columns:
    raise ValueError(f"Cannot find a zipcode column. Available columns: {list(df.columns)[:30]} ...")

# normalize zip to 5-digit strings
zips = (
    df[zip_col]
    .astype(str)
    .str.extract(r"(\d{5})", expand=False)
    .dropna()
    .unique()
    .tolist()
)

print(f"Loaded {len(df):,} rows. Unique ZIPs to query: {len(zips):,}")


# =========================
# DOWNLOAD ACS VIA CENSUS API
# =========================
c = Census(CENSUS_API_KEY, year=ACS_YEAR)

# Variables
VARS = [
    "NAME",
    "B19013_001E",  # median household income
    "B02001_001E",  # total
    "B02001_002E",  # white
    "B02001_003E",  # black
    "B02001_004E",  # native
    "B02001_005E",  # asian
    "B02001_006E",  # pacific
    "B02001_007E",  # other
    "B02001_008E",  # two+
]

rows = []
# Use the helper for ZIPs within one state (TX)
# This queries "zip code tabulation area" within state.
for z in tqdm(zips, desc="Querying ACS (TX ZCTAs)"):
    try:
        # `state_zipcode` returns a list of dicts
        res = c.acs5.state_zipcode(VARS, STATE_FIPS, z)
        if res:
            rows.extend(res)
    except Exception as e:
        # keep going; you can log failures if you want
        print(f"[WARN] ZIP {z} failed: {e}")

acs = pd.DataFrame(rows)
if acs.empty:
    raise RuntimeError("ACS result is empty. Check: API key, ZIPs, year, and that these ZIPs are valid ZCTAs.")

# Rename & clean
acs = acs.rename(
    columns={
        "zip code tabulation area": "zipcode",
        "B19013_001E": "acs_median_income",
        "B02001_001E": "race_total",
        "B02001_002E": "race_white",
        "B02001_003E": "race_black",
        "B02001_004E": "race_native",
        "B02001_005E": "race_asian",
        "B02001_006E": "race_pacific",
        "B02001_007E": "race_other",
        "B02001_008E": "race_two_plus",
    }
)

# Ensure zipcode is 5-digit string
acs["zipcode"] = acs["zipcode"].astype(str).str.zfill(5)

# Convert numeric columns
num_cols = [
    "acs_median_income",
    "race_total",
    "race_white",
    "race_black",
    "race_native",
    "race_asian",
    "race_pacific",
    "race_other",
    "race_two_plus",
]
for col in num_cols:
    acs[col] = pd.to_numeric(acs[col], errors="coerce")

# Deduplicate in case API returns multiple rows
acs = acs.drop_duplicates(subset=["zipcode"]).reset_index(drop=True)

print("ACS rows:", len(acs), "| example columns:", acs.columns.tolist())


# =========================
# SHANNON DIVERSITY INDEX
# =========================
race_parts = [
    "race_white",
    "race_black",
    "race_native",
    "race_asian",
    "race_pacific",
    "race_other",
    "race_two_plus",
]

def shannon_from_counts(counts: np.ndarray) -> float:
    total = np.nansum(counts)
    if not np.isfinite(total) or total <= 0:
        return np.nan
    p = counts / total
    p = p[(p > 0) & np.isfinite(p)]
    if len(p) == 0:
        return np.nan
    return float(-(p * np.log(p)).sum())

acs["acs_racial_diversity_shannon"] = acs[race_parts].apply(
    lambda r: shannon_from_counts(r.values.astype(float)),
    axis=1
)

# Optional: z-score controls (recommended for modeling)
acs["acs_median_income_z"] = (acs["acs_median_income"] - acs["acs_median_income"].mean()) / acs["acs_median_income"].std(ddof=1)
acs["acs_racial_diversity_z"] = (acs["acs_racial_diversity_shannon"] - acs["acs_racial_diversity_shannon"].mean()) / acs["acs_racial_diversity_shannon"].std(ddof=1)


# =========================
# MERGE BACK TO YOUR DATA
# =========================
df_merge = df.copy()
df_merge["zipcode"] = (
    df_merge[zip_col]
    .astype(str)
    .str.extract(r"(\d{5})", expand=False)
    .astype(str)
    .str.zfill(5)
)

keep = [
    "zipcode",
    "acs_median_income",
    "acs_racial_diversity_shannon",
    "acs_median_income_z",
    "acs_racial_diversity_z",
]
df_merge = df_merge.merge(acs[keep], on="zipcode", how="left")

# Quick check
print("After merge, missing median income:", df_merge["acs_median_income"].isna().mean())
print("After merge, missing shannon:", df_merge["acs_racial_diversity_shannon"].isna().mean())

df_merge.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
