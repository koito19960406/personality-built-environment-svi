# Reference pipeline

These scripts document how `data/analysis_dataset_zipcode.parquet` was produced.

**They are not runnable from this repository.** They need a Google Street View
Static API key, GPU inference, a US Census API key, the respondent-level survey
file, and a multi-terabyte panorama store held on the lab server — none of which
is redistributable. They are included so the derivation of the published
features is inspectable, not so it can be re-executed.

Server paths have been replaced with two environment variables:

| Variable | Meaning |
|---|---|
| `PERSONALITY_SVI_ROOT` | Project root; defaults to this repository |
| `SVI_IMAGE_DIR` | Root of the street view panorama store |

A hardcoded US Census API key was removed from
`09_add_median_income_racial_diversity.py`; it now reads `CENSUS_API_KEY` from
the environment, as the script's own error message already instructed.

---

## Stages

| Script | Stage |
|---|---|
| `01_download_svi.py` | Download Google Street View panoramas on a 100 m grid for each city |
| `02_segment_svi.py` | Panoptic semantic segmentation (Mask2Former, Mapillary Vistas) via ZenSVI |
| `03_classify_svi.py` | Scene classification (place type, lighting, image quality) |
| `04_detect_svi.py` | Object detection, including US flags and CCTV cameras |
| `05_low_level_svi.py` | Low-level visual features, including visual complexity |
| `06_big_five_analysis.py` | Score Big Five traits from the survey item responses |
| `07_prepare_dataset.py` | Aggregate image features to ZIP code, join survey and Census data, compute grid coverage |
| `08_build_env9_features.py` | Build the nine built environment composites plus the CCTV measure |
| `09_add_median_income_racial_diversity.py` | Append ACS median income and racial diversity (Shannon entropy) |
| `10_build_v2_post2010_dataset.py` | Re-aggregate trait means over respondents surveyed in 2010 or later |

Stages 1–5 run per city over the panorama store and are the expensive part.
Stages 6–10 are cheap table operations.

## Note on filenames

Parquet files produced by stage 8 use `env8` in their names, retained for
backward compatibility with earlier runs. There are nine composites plus the
CCTV surveillance measure. Do not read the filename as a feature count.

## Known issue carried into the published data

Stage 9 does not screen the Census Bureau's `-666666666` "no data" sentinel
before standardising median income, which distorts `acs_median_income_z`. See
the Known issues section of [`../data/README.md`](../data/README.md) for the
consequences and a sensitivity check.
