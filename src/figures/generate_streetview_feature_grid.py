import os
import random
import warnings
import tempfile
import shutil
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageDraw, ImageFont

from zensvi.transform import ImageTransformer
from zensvi.cv import Segmenter

warnings.filterwarnings("ignore", message=".*DataFrame contains unsupported types.*")


# =========================================================
# 0) Configuration — edit this section to change behavior
# =========================================================
PROJECT_ROOT = Path(os.environ.get("PERSONALITY_SVI_ROOT", Path(__file__).resolve().parents[2]))

DATA_PATH = PROJECT_ROOT / "data/processed/checkpoints/segmentation_results.parquet"
CLASSIFICATION_PATH = PROJECT_ROOT / "data/processed/checkpoints/classification_results.parquet"
VISUAL_COMPLEXITY_PATH = PROJECT_ROOT / "data/processed/checkpoints/visual_complexity_results.parquet"


IMAGE_BASE_DIR = os.environ.get("SVI_IMAGE_DIR", "/path/to/streetview/")  # root directory containing all streetview jpg files

OUTPUT_DIR = PROJECT_ROOT / "reports/figures/feature_percentiles"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_FIND_RETRIES = 400
VISUAL_COMPLEXITY_COL = "visual_complexity"

# Representative sample selection: low/medium/high defined by percentile
PCTS = {"low": 10, "medium": 50, "high": 90}

# Threshold for filtering near-zero values when selecting medium/high samples (ratio features use 0.05)
RATIO_NONZERO_TH = 0.05
COUNT_NONZERO_TH = 0.0

# Temporary directory (cleaned up after run)
TMP_DIR = Path(tempfile.mkdtemp(prefix="feat_pct_combined_"))
print("TMP_DIR:", TMP_DIR)


# =========================================================
# 1) Feature definitions — composite features and their source columns
#    - Each feature defines which columns to aggregate
#    - Default aggregation is sum
# =========================================================
COMBINED_FEATURES = [
    ("env_building", {
        "mode": "ratios",
        "cols": ["Building_ratios"],
        "mask_classes": ["Building"],
    }),
    ("env_greenery", {
        "mode": "ratios",
        "cols": ["Vegetation_ratios"],
        "mask_classes": ["Vegetation"],
    }),
    ("env_open_space", {
        "mode": "ratios",
        "cols": ["Sky_ratios", "Terrain_ratios", "Sand_ratios"],
        "mask_classes": ["Sky", "Terrain", "Sand"],
    }),
    ("env_road", {
        "mode": "ratios",
        "cols": ["Road_ratios", "Service Lane_ratios"],
        "mask_classes": ["Road", "Service Lane"],
    }),
    ("env_physical_boundaries", {
        "mode": "ratios",
        "cols": ["Fence_ratios", "Wall_ratios", "Barrier_ratios"],
        "mask_classes": ["Fence", "Wall", "Barrier"],
    }),
    ("env_vehicle_presence", {
        "mode": "counts",
        "cols": ["Car_counts","Truck_counts","Bus_counts","Motorcycle_counts","Trailer_counts","Other Vehicle_counts","Caravan_counts"],
        "mask_classes": ["Car","Truck","Bus","Motorcycle","Trailer","Other Vehicle","Caravan"],
    }),
    ("env_active_presence", {
        "mode": "counts",
        "cols": ["Person_counts", "Bicycle_counts", "Bicyclist_counts"],
        "mask_classes": ["Person","Bicycle","Bicyclist"],
    }),
]

# =========================
# Manual rank offsets
# =========================
OFFSETS = {
    "env_building": {
        "low": 0,
        "medium": 0,
        "high": 1,
    },
    "env_greenery": {
        "low": 1,
        "medium": 0,
        "high": 1,
    },
    "env_open_space": {
        "low": 1,
        "medium": 0,
        "high": 0,
    },
    "env_road": {
        "low": 10,
        "medium": 1,
        "high": 1,
    },
    "env_physical_boundaries": {
        "low": 0,
        "medium": 4,
        "high": 0,
    },
    "env_vehicle_presence": {
        "low": 0,
        "medium": 0,
        "high": 2,
    },
    "env_active_presence": {
        "low": 100,
        "medium": 3,   # try second candidate
        "high": 1,     # try third candidate
    },
}


# Optional: Fence/Wall/Barrier combined feature (not included in main 7 by default)
FENCE_COMBINED = ("env_physical_boundaries", {
    "cols": ["Fence_ratios_mean", "Wall_ratios_mean", "Barrier_ratios_mean"],
    "kind": "ratio",
    "agg": "sum",
    "mask_classes": ["Fence", "Wall", "Barrier"],
})
# Uncomment to add Fence/Wall/Barrier as an 8th feature:
# COMBINED_FEATURES.append(FENCE_COMBINED)


# =========================================================
# 2) Mapillary/ZenSVI segmentation color map — RGB values per semantic class
#    These RGB values correspond to the palette in ZenSVI's "colored_segmented.png".
#    If a class mask appears all-black, the RGB value is likely wrong.
#    To debug: inspect a *_colored_segmented.png file and sample pixel colors.
# =========================================================
RGB = {
    # Confirmed values
    "Building": (70, 70, 70),
    "Fence": (190, 153, 153),
    "Wall": (102, 102, 156),
    "Vegetation": (107, 142, 35),
    "Car": (0, 0, 142),
    "Truck": (0, 0, 70),

    # Default placeholder values — verify against actual ZenSVI palette if masks appear all-black
    "Sky": (70, 130, 180),
    "Terrain": (152, 251, 152),
    "Sand": (244, 164, 96),

    "Road": (128, 64, 128),
    "Service Lane": (128, 64, 128),  # may be same class as Road or absent — verify
    "Sidewalk": (244, 35, 232),
    "Bike Lane": (119, 11, 32),      # may not exist as a separate class — verify
    "Pedestrian Area": (220, 220, 0),# may not exist — verify

    "Person": (220, 20, 60),
    "Bicycle": (119, 11, 32),
    "Bicyclist": (220, 20, 60),      # may not exist as a separate class — verify

    "Bus": (0, 60, 100),
    "Motorcycle": (0, 0, 230),
    "Trailer": (0, 0, 110),
    "Other Vehicle": (0, 0, 142),
    "Caravan": (0, 0, 142),

    "Barrier": (250, 170, 30),
}


# =========================================================
# 3) Utility functions
# =========================================================
def parse_filename_key(filename_key):
    try:
        parts = str(filename_key).split("/")
        file_part = parts[-1]
        panoid = file_part.split(".")[0]
        return panoid
    except Exception:
        return None


def find_image_file(base_dir, filename_key, max_retries=5):
    panoid = parse_filename_key(filename_key)
    if not panoid:
        return None
    base_path = Path(base_dir)
    pattern = f"**/{panoid}*.jpg"

    for _ in range(max_retries):
        try:
            matches = list(base_path.glob(pattern))
            if matches:
                # pick a random existing file from matches
                for f in random.sample(matches, len(matches)):
                    if f.is_file():
                        return f
        except Exception as e:
            print(f"Error searching {panoid}: {e}")
            break
    return None


def create_placeholder_image(text="Missing", size=(300, 200)):
    img = Image.new("RGB", size, color=(180, 180, 180))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
        bbox = d.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (size[0] - w) / 2
        y = (size[1] - h) / 2
        d.text((x, y), text, fill=(0, 0, 0), font=font)
    except Exception:
        pass
    return img


def safe_imread(path):
    try:
        return mpimg.imread(path)
    except Exception:
        return None


# ---- Shannon index / Visual Complexity ----
def shannon_index(ratios):
    ratios = np.asarray(ratios, dtype=np.float64)
    nonzero = ratios[ratios > 0]
    if nonzero.size == 0:
        return 0.0
    s = nonzero.sum()
    if s == 0:
        return 0.0
    p = nonzero / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def calculate_and_save_visual_complexity(df_pl, save_path):
    print("Calculating visual complexity (Shannon index)...")
    ratio_cols = [c for c in df_pl.columns if c.endswith("_ratios")]
    if not ratio_cols:
        raise ValueError("No *_ratios columns found for VC.")
    arr = df_pl.select(ratio_cols).to_numpy()
    vc = np.array([shannon_index(row) for row in arr], dtype=np.float64)
    df_out = df_pl.with_columns(pl.Series(VISUAL_COMPLEXITY_COL, vc))
    df_out.write_parquet(save_path)
    print("VC saved:", save_path)
    return df_out


def load_or_calculate_visual_complexity(df_pl, save_path):
    if os.path.exists(save_path):
        try:
            df_vc = pl.read_parquet(save_path)
            if df_vc.height == df_pl.height and VISUAL_COMPLEXITY_COL in df_vc.columns:
                print("Loaded VC:", save_path)
                return df_vc
        except Exception as e:
            print("VC load failed, recalculating:", e)
    return calculate_and_save_visual_complexity(df_pl, save_path)


# ---- Composite feature computation (pandas) ----
def ensure_columns_exist(df, cols, feature_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{feature_name}] missing columns: {missing}")


def compute_combined_features(df_pd):
    for feat_name, spec in COMBINED_FEATURES:
        cols = spec["cols"]
        ensure_columns_exist(df_pd, cols, feat_name)
        if spec.get("agg", "sum") == "sum":
            df_pd[feat_name] = df_pd[cols].sum(axis=1, skipna=True)
        elif spec["agg"] == "mean":
            df_pd[feat_name] = df_pd[cols].mean(axis=1, skipna=True)
        else:
            raise ValueError(f"Unsupported agg for {feat_name}: {spec['agg']}")
    return df_pd


# ---- Representative sample selection (pick row closest to the target percentile value) ----
def pick_row_by_percentile(df, value_col, pct, kind):
    s = df[value_col].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None

    # For medium/high, optionally filter near-zero values
    if kind == "ratio":
        nz_th = RATIO_NONZERO_TH
    else:
        nz_th = COUNT_NONZERO_TH

    if pct > 10:  # apply near-zero filter for medium/high (not for low)
        df_use = df[df[value_col] > nz_th].copy()
        if df_use.empty:
            df_use = df.copy()
    else:
        df_use = df.copy()

    target = np.nanpercentile(df_use[value_col].values, pct)
    idx = (df_use[value_col] - target).abs().idxmin()
    return df_use.loc[idx]


# =========================================================
# 4) ZenSVI processing: perspective transform + segmentation + union binary mask
# =========================================================
def find_colored_segmented_png(segment_dir: Path):
    files = list(segment_dir.glob("*_colored_segmented.png"))
    return files[0] if files else None


def create_union_binary_mask(colored_seg_path: Path, mask_classes: list[str], output_size):
    """Union pixels from multiple semantic classes to produce a binary white/black mask."""
    seg_img = Image.open(colored_seg_path).convert("RGB")
    seg_arr = np.array(seg_img)

    h, w = seg_arr.shape[:2]
    union = np.zeros((h, w), dtype=bool)

    used = 0
    for cls in mask_classes:
        if cls not in RGB:
            print(f"  [mask] RGB mapping missing for class={cls} (skip)")
            continue
        color = np.array(RGB[cls], dtype=np.uint8)
        match = np.all(seg_arr == color, axis=2)
        if match.any():
            union |= match
            used += 1

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[union] = 255

    mask_img = Image.fromarray(mask, mode="L")
    if mask_img.size != output_size:
        mask_img = mask_img.resize(output_size, Image.NEAREST)
    return mask_img.convert("RGB"), used


def process_one_image_to_persp_and_mask(img_path: Path, panoid: str, mask_classes: list[str], tmp_root: Path):
    """Process one image: raw jpg -> perspective transform -> segmentation -> union mask."""
    temp_img_dir = tmp_root / f"temp_{panoid}"
    temp_img_dir.mkdir(exist_ok=True)

    perspective_dir = temp_img_dir / "perspective"
    segmentation_dir = temp_img_dir / "segmented"
    perspective_dir.mkdir(exist_ok=True)
    segmentation_dir.mkdir(exist_ok=True)

    # copy/save original
    src = Image.open(img_path)
    src_jpg = temp_img_dir / f"{panoid}.jpg"
    src.save(src_jpg)

    # perspective
    transformer = ImageTransformer(dir_input=str(temp_img_dir), dir_output=str(perspective_dir))
    transformer.transform_images(
        style_list="perspective",
        FOV=90,
        theta=90,
        phi=0,
        aspects=(9, 16),
    )

    persp_files = list((perspective_dir / "perspective").glob("*Direction_0*.png"))
    if not persp_files:
        return None, None, "no_perspective"

    persp_path = persp_files[0]
    persp_img = Image.open(persp_path)

    # segmentation
    segmenter = Segmenter(
        task="semantic",
        dataset="mapillary",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
    )
    segmenter.segment(
        dir_input=str(persp_path),
        dir_image_output=str(segmentation_dir),
        dir_summary_output=str(segmentation_dir),
        batch_size=1,
        save_image_options="segmented_image",
        save_format="csv",
        csv_format="wide",
    )

    colored = find_colored_segmented_png(segmentation_dir)
    if colored is None:
        # segmentation failed
        return persp_img, create_placeholder_image("Segmentation failed", persp_img.size), "no_segmentation"

    mask_img, used = create_union_binary_mask(colored, mask_classes, persp_img.size)
    if used == 0:
        # No target-class pixels found: use placeholder image
        mask_img = create_placeholder_image(f"No target pixels:\n{mask_classes}", persp_img.size)

    return persp_img, mask_img, "ok"


# =========================================================
# 5) Step 1: Load data -> filter -> compute composite features -> select low/med/high -> find source image -> save output
# =========================================================
def step1_generate_feature_images():
    print("\n=== STEP 1: load data & generate images ===")

    df_pl = pl.read_parquet(DATA_PATH)
    print("Loaded:", DATA_PATH, "rows:", df_pl.height)

    # indoor filter
    if os.path.exists(CLASSIFICATION_PATH):
        try:
            df_cls = pl.read_parquet(CLASSIFICATION_PATH)
            df_pl = df_pl.join(
                df_cls.select(["filename_key", "environment_type"]),
                on="filename_key",
                how="left",
            )
            before = df_pl.height
            df_pl = df_pl.filter(pl.col("environment_type") != "indoor")
            after = df_pl.height
            print(f"Filtered indoor: {before-after} removed, remain {after}")
        except Exception as e:
            print("Warning: classification join/filter failed:", e)

    # visual complexity
    df_pl = load_or_calculate_visual_complexity(df_pl, VISUAL_COMPLEXITY_PATH)

    # Required columns: filename_key + VC + all columns needed by composite features
    needed = {"filename_key", VISUAL_COMPLEXITY_COL}
    for _, spec in COMBINED_FEATURES:
        needed.update(spec["cols"])
    needed = list(needed)

    missing = [c for c in needed if c not in df_pl.columns]
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    df_pl = df_pl.select(needed).drop_nulls()
    df_pl = df_pl.filter(pl.col(VISUAL_COMPLEXITY_COL) > 0.5)

    df_pd = df_pl.to_pandas()
    if df_pd.empty:
        raise RuntimeError("No valid rows after filtering (VC>0.5 & drop_nulls).")

    # compute combined feature columns
    df_pd = compute_combined_features(df_pd)

    # For each feature, select low/med/high samples and generate images
    results = {}  # feat -> level -> {panoid, value, paths}
    for feat_name, spec in COMBINED_FEATURES:
        print(f"\n[Feature] {feat_name}")
        results[feat_name] = {}
        kind = spec.get("kind", spec.get("mode", "ratios"))
        mask_classes = spec.get("mask_classes", [])
        print("mask_classes:", mask_classes)


        for level, pct in PCTS.items():

            # ---------- 1) Prepare usable dataframe ----------
            df_use = df_pd[[ "filename_key", feat_name ]].copy()
            df_use = df_use.replace([np.inf, -np.inf], np.nan).dropna()

            if kind == "ratio":
                nz_th = RATIO_NONZERO_TH
            else:
                nz_th = COUNT_NONZERO_TH

            if pct > 10:
                df_use = df_use[df_use[feat_name] > nz_th]
                if df_use.empty:
                    df_use = df_pd[[ "filename_key", feat_name ]].dropna()

            if df_use.empty:
                print(f"  {level}: no valid rows after filtering")
                continue

            # ---------- 2) Sort by feature value ----------
            df_sorted = df_use.sort_values(feat_name).reset_index(drop=True)

            # ---------- 3) Find the base index at the target percentile ----------
            target_val = np.percentile(df_sorted[feat_name].values, pct)
            base_idx = (df_sorted[feat_name] - target_val).abs().idxmin()

            # ---------- 4) Apply manual offset ----------
            offset = OFFSETS.get(feat_name, {}).get(level, 0)
            idx = base_idx + offset

            # Clamp index to valid range
            idx = max(0, min(idx, len(df_sorted) - 1))

            row = df_sorted.iloc[idx]

            print(
                f"  {level} p{pct}: base_idx={base_idx}, offset={offset}, "
                f"final_idx={idx}, value={row[feat_name]:.4f}"
            )


            filename_key = row["filename_key"]
            panoid = parse_filename_key(filename_key) or f"unknown_{random.randint(0,999999)}"
            val = float(row[feat_name])

            # Output filenames: {feature}_{level}_{panoid}_perspective.png / _mask.png
            out_p = OUTPUT_DIR / f"{feat_name}_{level}_{panoid}_perspective.png"
            out_m = OUTPUT_DIR / f"{feat_name}_{level}_{panoid}_mask.png"

            # Skip if output files already exist
            if out_p.exists() and out_m.exists():
                print(f"  {level} p{pct}: exists -> skip panoid={panoid} value={val:.4f}")
                results[feat_name][level] = {"panoid": panoid, "value": val, "perspective": str(out_p), "mask": str(out_m)}
                continue

            # Find source jpg image
            img_path = find_image_file(IMAGE_BASE_DIR, filename_key, max_retries=5)
            if img_path is None or (not img_path.exists()):
                print(f"  {level} p{pct}: image not found for filename_key={filename_key}")
                # Save placeholder images
                ph = create_placeholder_image(f"Not found\n{panoid}", (360, 240))
                ph.save(out_p)
                ph.save(out_m)
                results[feat_name][level] = {"panoid": panoid, "value": val, "perspective": str(out_p), "mask": str(out_m)}
                continue

            print(f"  {level} p{pct}: panoid={panoid} value={val:.4f} mask={mask_classes}")

            persp_img, mask_img, status = process_one_image_to_persp_and_mask(
                img_path=img_path,
                panoid=panoid,
                mask_classes=mask_classes,
                tmp_root=TMP_DIR,
            )

            if persp_img is None:
                # perspective failed
                ph = create_placeholder_image(f"Perspective failed\n{panoid}", (360, 240))
                ph.save(out_p)
                ph.save(out_m)
            else:
                persp_img.save(out_p)
                if mask_img is None:
                    create_placeholder_image(f"Mask failed\n{panoid}", persp_img.size).save(out_m)
                else:
                    mask_img.save(out_m)

            results[feat_name][level] = {"panoid": panoid, "value": val, "perspective": str(out_p), "mask": str(out_m), "status": status}

    print("\nSTEP 1 done. Output dir:", OUTPUT_DIR)
    return results


# =========================================================
# 6) Step 2: Assemble Step 1 outputs into a single grid figure
# =========================================================

# Readable feature labels (consistent with run_ours_v2_analysis.py FEATURE_LABELS)
GRID_FEATURE_LABELS = {
    "env_building":            "Building",
    "env_greenery":            "Greenery",
    "env_open_space":          "Open Space",
    "env_road":                "Road",
    "env_physical_boundaries": "Physical Boundaries",
    "env_vehicle_presence":    "Vehicle Presence",
    "env_active_presence":     "Active Presence",
}

# Manually pin which saved panoid to display per (feature, level).
# Keys: feature name; values: dict of level -> panoid string.
# Leave a level out to fall back to whatever step1 selected.
MANUAL_PANOID_OVERRIDES = {
    "env_active_presence": {
        "low": "u33iXqRsS5cyY_UNYf78GQ",   # clean empty road; black mask = correct for zero presence
    },
}


def _resolve_paths(feat: str, lvl: str, step1_rec: dict) -> tuple:
    """Return (perspective_path, mask_path) respecting MANUAL_PANOID_OVERRIDES."""
    override_panoid = MANUAL_PANOID_OVERRIDES.get(feat, {}).get(lvl)
    if override_panoid:
        p = OUTPUT_DIR / f"{feat}_{lvl}_{override_panoid}_perspective.png"
        m = OUTPUT_DIR / f"{feat}_{lvl}_{override_panoid}_mask.png"
        return str(p) if p.exists() else None, str(m) if m.exists() else None
    return step1_rec.get("perspective"), step1_rec.get("mask")


def step2_build_grid(results):
    print("\n=== STEP 2: build grid ===")
    features = list(results.keys())
    levels = ["low", "medium", "high"]
    col_labels = ["Low", "Medium", "High"]

    # Two rows per feature: perspective image + mask
    n_rows = len(features) * 2
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 2.1))
    plt.subplots_adjust(left=0.18, right=0.99, top=0.93, bottom=0.03, hspace=0.12, wspace=0.03)

    def plot_cell(ax, path, fallback_text):
        img = safe_imread(path) if path else None
        if img is None:
            img = np.array(create_placeholder_image(fallback_text, (360, 240)))
        ax.imshow(img)
        ax.axis("off")

    for fi, feat in enumerate(features):
        for ci, lvl in enumerate(levels):
            rec = results.get(feat, {}).get(lvl, {})
            p_path, m_path = _resolve_paths(feat, lvl, rec)

            r_p = fi * 2
            r_m = r_p + 1

            plot_cell(axes[r_p, ci], p_path, f"Missing\n{feat}\n{lvl}\nperspective")
            plot_cell(axes[r_m, ci], m_path, f"Missing\n{feat}\n{lvl}\nmask")

    # Column headers
    for ci, lab in enumerate(col_labels):
        ax = axes[0, ci]
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, 0.935, lab, ha="center", va="bottom", fontsize=12, weight="bold")

    # Left-side row labels + feature name (readable)
    for fi, feat in enumerate(features):
        ax0 = axes[fi * 2, 0]
        ax1 = axes[fi * 2 + 1, 0]
        ax0.text(-0.10, 0.5, "Original", ha="center", va="center", fontsize=10, rotation=90, transform=ax0.transAxes)
        ax1.text(-0.10, 0.5, "Mask", ha="center", va="center", fontsize=10, rotation=90, transform=ax1.transAxes)

        p0 = ax0.get_position()
        p1 = ax1.get_position()
        y_mid = (p0.y0 + p1.y1) / 2
        label = GRID_FEATURE_LABELS.get(feat, feat)
        fig.text(0.13, y_mid, label, ha="center", va="center", fontsize=12, weight="bold", rotation=90)

    out_grid = OUTPUT_DIR / "feature_percentiles_combined_grid.png"
    plt.savefig(out_grid, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("✓ Grid saved:", out_grid)


# =========================================================
# 7) Entry points
# =========================================================
def main():
    try:
        results = step1_generate_feature_images()
        step2_build_grid(results)
    finally:
        # Clean up temporary directory (comment out to keep tmp files for debugging)
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        print("Cleaned TMP_DIR:", TMP_DIR)


def run_grid_only():
    """Re-assemble grid from already-saved images in OUTPUT_DIR (skip step1).

    Scans OUTPUT_DIR for existing {feat}_{level}_{panoid}_perspective.png files,
    builds a results dict, then calls step2_build_grid. Use MANUAL_PANOID_OVERRIDES
    to pin specific images per (feature, level).
    """
    print("=== Grid-only rebuild from saved images ===")
    features_order = [f for f, _ in COMBINED_FEATURES]
    levels = ["low", "medium", "high"]

    results = {}
    for feat in features_order:
        results[feat] = {}
        for lvl in levels:
            # Check manual override first
            override_panoid = MANUAL_PANOID_OVERRIDES.get(feat, {}).get(lvl)
            if override_panoid:
                p = OUTPUT_DIR / f"{feat}_{lvl}_{override_panoid}_perspective.png"
                m = OUTPUT_DIR / f"{feat}_{lvl}_{override_panoid}_mask.png"
                if p.exists():
                    results[feat][lvl] = {"perspective": str(p), "mask": str(m) if m.exists() else None}
                    print(f"  {feat} {lvl}: override -> {override_panoid}")
                    continue

            # Fall back: pick first available file for this feat+level
            candidates = sorted(OUTPUT_DIR.glob(f"{feat}_{lvl}_*_perspective.png"))
            if candidates:
                p = candidates[0]
                panoid = p.stem.replace(f"{feat}_{lvl}_", "").replace("_perspective", "")
                m = OUTPUT_DIR / f"{feat}_{lvl}_{panoid}_mask.png"
                results[feat][lvl] = {"perspective": str(p), "mask": str(m) if m.exists() else None}
                print(f"  {feat} {lvl}: auto -> {panoid}")
            else:
                print(f"  {feat} {lvl}: no file found")

    step2_build_grid(results)


if __name__ == "__main__":
    run_grid_only()
