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
"""
build_v2_post2010_dataset.py
----------------------------
Build the Version 2 modeling dataset by re-aggregating personality traits
using only survey responses collected in 2010 or later (year >= 2010).

This aligns the personality data more closely with the SVI imagery period
(mostly post-2010), while keeping all other features (ENV8, census, ACS)
unchanged from Version 1.

Inputs:
  - final_dataset_individual_with_year.parquet   (142,963 rows, 2003–2020)
  - final_dataset_zipcode_spatial2_env8_medianincome_racialdiversity.parquet  (340 ZIP rows, V1)

Output:
  - final_dataset_zipcode_spatial2_env8_medianincome_racialdiversity_post2010.parquet

The only columns that change vs Version 1:
  participant_count, extraversion_mean, agreeableness_mean,
  conscientiousness_mean, neuroticism_mean, openness_mean
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

BASE = Path(os.environ.get("PERSONALITY_SVI_ROOT", Path(__file__).resolve().parents[2]))
MODEL_DIR = BASE / 'data/processed/model'

IND_PATH = MODEL_DIR / 'final_dataset_individual_with_year.parquet'
V1_PATH  = MODEL_DIR / 'final_dataset_zipcode_spatial2_env8_medianincome_racialdiversity.parquet'
OUT_PATH = MODEL_DIR / 'final_dataset_zipcode_spatial2_env8_medianincome_racialdiversity_post2010.parquet'

YEAR_MIN = 2010
TRAITS   = ['extraversion', 'agreeableness', 'conscientiousness', 'neuroticism', 'openness']


def main():
    # ── 1. Load individual-level data ──────────────────────────────────────────
    print(f'Loading {IND_PATH.name} …')
    cols = ['now_zip', 'year'] + TRAITS
    ind = pd.read_parquet(IND_PATH, columns=cols)
    print(f'  Rows (all years):  {len(ind):,}')

    # ── 2. Filter to year >= 2010 ──────────────────────────────────────────────
    ind_filt = ind[ind['year'] >= YEAR_MIN].copy()
    print(f'  Rows (year >= {YEAR_MIN}): {len(ind_filt):,}')
    print(f'  Year range after filter: {ind_filt["year"].min()}–{ind_filt["year"].max()}')

    # ── 3. Normalize ZIP column to 5-digit string ──────────────────────────────
    ind_filt['now_zip'] = ind_filt['now_zip'].astype(str).str.zfill(5)

    # ── 4. Aggregate personality by ZIP ───────────────────────────────────────
    agg_dict = {t: 'mean' for t in TRAITS}
    agg_dict['year'] = 'count'   # use as participant_count proxy

    zip_agg = (
        ind_filt
        .groupby('now_zip', as_index=False)
        .agg(agg_dict)
        .rename(columns={t: f'{t}_mean' for t in TRAITS})
        .rename(columns={'year': 'participant_count'})
    )
    print(f'\nAggregated: {len(zip_agg)} ZIP codes')
    print(f'  participant_count stats:\n{zip_agg["participant_count"].describe().to_string()}')

    # ── 5. Load Version 1 dataset ──────────────────────────────────────────────
    print(f'\nLoading V1 dataset: {V1_PATH.name} …')
    try:
        v1 = gpd.read_parquet(V1_PATH)
        is_geo = True
    except Exception:
        v1 = pd.read_parquet(V1_PATH)
        is_geo = False
    print(f'  Shape: {v1.shape}')

    # Normalize zip column
    v1['zipcode'] = v1['zipcode'].astype(str).str.zfill(5)

    # ── 6. Drop old personality columns & participant_count ────────────────────
    drop_cols = ['participant_count'] + [f'{t}_mean' for t in TRAITS]
    drop_cols = [c for c in drop_cols if c in v1.columns]
    v2 = v1.drop(columns=drop_cols)

    # ── 7. Merge new personality aggregation ──────────────────────────────────
    zip_agg_merge = zip_agg.rename(columns={'now_zip': 'zipcode'})
    v2 = v2.merge(zip_agg_merge, on='zipcode', how='left')

    # Insert personality columns in the same position as V1
    # (after 'participant_count' was originally) — just re-add them:
    print(f'\nV2 shape: {v2.shape}')

    # Check ZIPs without post-2010 data
    missing_pers = v2[v2['extraversion_mean'].isna()]['zipcode'].tolist()
    if missing_pers:
        print(f'  WARNING: {len(missing_pers)} ZIPs have no post-2010 personality data: {missing_pers}')
    else:
        print('  All 340 ZIPs have post-2010 personality data.')

    # Compare participant counts: V1 vs V2
    v1_counts = v1.set_index('zipcode')['participant_count']
    v2_counts = v2.set_index('zipcode')['participant_count']
    both = v1_counts.align(v2_counts, join='inner')
    delta = (both[1] - both[0]).describe()
    print(f'\nParticipant count change (V2 - V1):\n{delta.to_string()}')

    # ── 8. Save ────────────────────────────────────────────────────────────────
    print(f'\nSaving to {OUT_PATH.name} …')
    if is_geo and hasattr(v2, 'geometry'):
        v2_geo = gpd.GeoDataFrame(v2, geometry='geometry', crs=v1.crs if hasattr(v1, 'crs') else None)
        v2_geo.to_parquet(OUT_PATH, index=False)
    else:
        v2.to_parquet(OUT_PATH, index=False)
    print(f'Saved: {OUT_PATH}')

    # ── 9. Summary ─────────────────────────────────────────────────────────────
    print('\n=== Version 2 Dataset Summary ===')
    print(f'  Year filter: year >= {YEAR_MIN}')
    print(f'  Individual records used: {len(ind_filt):,} (was {len(ind):,} in V1)')
    print(f'  ZIP codes: {len(v2):,}')
    trait_cols_v2 = [f'{t}_mean' for t in TRAITS]
    print(f'  Personality means (V2):')
    print(v2[trait_cols_v2].describe().round(4).to_string())


if __name__ == '__main__':
    main()
