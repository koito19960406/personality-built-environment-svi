# Data dictionary — `analysis_dataset_zipcode.parquet`

254 rows (ZIP codes) x 40 columns. One row per ZIP code, across Austin, Dallas,
Houston, and San Antonio, Texas.

| City | ZIP codes |
|---|---|
| Houston | 94 |
| San Antonio | 57 |
| Austin | 52 |
| Dallas | 51 |

Aggregated from **77,716 survey respondents**; per-ZIP-code respondent counts
range from 21 to 6,081 (median 233).

---

## Provenance

This table is the analysis-ready output of the pipeline documented in
`reference_pipeline/`, assembled from three streams:

1. **Personality** — Big Five trait scores from the Gosling–Potter Internet
   Personality Project, self-reported with a current ZIP code, averaged within
   ZIP code. Only respondents surveyed in **2010 or later** are included, so the
   survey window aligns with the street view imagery window.
2. **Built environment** — Google Street View panoramas segmented with
   Mask2Former (Mapillary Vistas), plus object detection, scene classification,
   and low-level visual features, aggregated to ZIP code and combined into nine
   composites plus a CCTV surveillance measure.
3. **Census** — US Census demographic shares and population density, and ACS
   5-year (2016–2020) median household income and racial diversity.

ZIP code polygons come from the US Census Bureau ZCTA file
`cb_2016_us_zcta510_500k`, stored as WKT in EPSG:4326.

### Already filtered

The two quality filters described in the paper were applied **before** this file
was written, taking 340 ZIP codes down to 254:

- `participant_count >= 20` — enough respondents for a stable trait mean.
- `grid_coverage_mean >= 0.03` — non-negligible street view coverage.

The unfiltered 340-ZIP-code table is not published, because reconstructing it
requires the respondent-level survey file.

### N = 252 in the spatial models

`libpysal`'s Queen contiguity graph leaves two ZIP codes with no neighbours.
`src/reproduce_models.py` drops these islands before fitting, so the pooled
spatial lag models report N = 252. The OLS models use all 254.

---

## Known issues

### `acs_median_income_z` does not behave as an income control

ZIP code 78712 (the University of Texas at Austin campus) carries the Census
Bureau's "no data" sentinel value of **-666666666** in `acs_median_income`. The
standardised column `acs_median_income_z` was computed over the raw column
*including* that sentinel, so the standard deviation is dominated by that single
observation.

The consequence: across the other 253 ZIP codes, `acs_median_income_z` spans
only 0.1213 to 0.1236 (SD 0.0004). In the fitted models it therefore functions
as an indicator for ZIP 78712 rather than as a measure of income variation.

**Both columns are left exactly as published**, so that this repository
reproduces the paper. Users refitting these models for their own purposes should
drop or repair the sentinel first:

```python
df = df[df["acs_median_income"] > 0]
df["acs_median_income_z"] = scipy.stats.zscore(df["acs_median_income"])
```

Sensitivity check, refitting the pooled OLS models after that repair: the
built-environment results are stable — the number of significant ENV predictors
is unchanged at three, and the only movements are two marginal terms crossing
p = 0.05 (`env_surveillance_cctv` 0.042 to 0.053; `env_symbolic_us_flag` 0.057
to 0.043). The income coefficient itself is *not* stable and should not be
interpreted: it changes sign for agreeableness (+0.34 to -0.27).

`acs_racial_diversity_z` is unaffected and standardises normally.

### Empty standard errors in the tidy spatial lag tables

See the note in the top-level README. The `se`, `z`, and `p` columns of
`sar_results_tidy_all.csv` are empty in the published outputs; the complete
values are in the per-model `spatial_lag/summary/*.txt` files.

---

## Columns

### Identifiers

| Column | Type | Notes |
|---|---|---|
| `now_zip` | int64 | Respondent-reported current ZIP code |
| `zipcode` | str | ZIP code, 254 unique |
| `city` | str | `austin_texas`, `dallas_texas`, `houston_texas`, `san_antonio_texas` |
| `geometry` | str (WKT) | ZIP code polygon, EPSG:4326 |

### Outcomes — Big Five trait means

Mean of respondent scores within the ZIP code, on the original response scale.
`reproduce_models.py` z-scores these before fitting.

| Column | Range (min / median / max) |
|---|---|
| `extraversion_mean` | 2.74 / 3.30 / 3.59 |
| `agreeableness_mean` | 3.55 / 3.79 / 4.04 |
| `conscientiousness_mean` | 3.22 / 3.63 / 3.98 |
| `neuroticism_mean` | 2.53 / 2.86 / 3.12 |
| `openness_mean` | 3.47 / 3.74 / 4.03 |
| `participant_count` | 21 / 233 / 6081 |

### Predictors — built environment composites

Nine composites plus a surveillance measure, built by
`reference_pipeline/08_build_env9_features.py` from segmentation ratios and
object counts. The first eight are already on a standardised-composite scale;
the last two are raw shares. All are z-scored again within
`reproduce_models.py`.

| Column | Range (min / median / max) |
|---|---|
| `env_greenery` | -1.73 / 0.225 / 3.14 |
| `env_open_space` | -1.30 / -0.132 / 1.73 |
| `env_building` | -0.905 / -0.086 / 4.07 |
| `env_road` | -1.49 / -0.029 / 1.55 |
| `env_active_infra` | -0.396 / -0.103 / 7.26 |
| `env_active_presence` | -1.00 / 0.134 / 7.71 |
| `env_vehicle_presence` | -0.774 / 0.043 / 1.30 |
| `env_physical_boundaries` | -0.782 / -0.130 / 2.13 |
| `env_symbolic_us_flag` | 0 / 0.0052 / 0.043 |
| `env_surveillance_cctv` | 0 / 0.000087 / 0.0028 |

Note: the parquet files produced upstream use `env8` in their filenames for
backward compatibility, though there are nine composites plus CCTV.

### Controls

| Column | Range (min / median / max) | Notes |
|---|---|---|
| `visual_complexity_mean` | 1.15 / 1.32 / 1.45 | Mean image visual complexity |
| `us_census_male_share` | 0.415 / 0.497 / 0.869 | |
| `us_census_age_15_29_share` | 0.063 / 0.254 / 0.979 | |
| `us_census_age_30_44_share` | 0 / 0.203 / 0.304 | |
| `us_census_age_45_59_share` | 0 / 0.136 / 0.330 | |
| `us_census_age_60_plus_share` | 0 / 0.177 / 0.389 | |
| `us_census_population_density_km2` | 45.3 / 1450 / 6439 | |
| `acs_median_income` | -666666666 / 60802 / 202526 | **See Known issues** — one sentinel value |
| `acs_racial_diversity_shannon` | 0.305 / 1.02 / 1.53 | Shannon entropy over race categories |
| `acs_median_income_z` | -8.17 / 0.122 / 0.124 | **See Known issues** |
| `acs_racial_diversity_z` | -2.92 / 0.058 / 2.16 | |

### City dummies

`city_austin_texas`, `city_dallas_texas`, `city_houston_texas`,
`city_san_antonio_texas` — 0/1. `reproduce_models.py` rebuilds these from the
`city` column rather than trusting the stored values, and omits San Antonio as
the baseline in the pooled models.

### Coverage and imagery diagnostics

Not used as predictors, except `grid_coverage_mean` which drove the quality
filter.

| Column | Range (min / median / max) |
|---|---|
| `grid_coverage_mean` | 0.030 / 0.547 / 0.937 |
| `gsv_density_mean` | 1.9e-05 / 8.7e-04 / 2.3e-03 |
| `gsv_point_count_mean` | 196 / 20890 / 92060 |
| `image_count` | 178 / 14510 / 86530 |
| `zipcode_area_m2_mean` | 4.25e+05 / 3.04e+07 / 3.67e+08 |

---

## What is not here

Respondent-level survey data is not published. That file contains IP addresses,
geolocation coordinates, browser fingerprints, and free-text responses. Requests
for the underlying survey data should be directed to the survey custodians
(Samuel D. Gosling, Jeff Potter).

Google Street View panoramas are not redistributable under the Google Maps
Platform terms. The derived, aggregated features in this table are.
