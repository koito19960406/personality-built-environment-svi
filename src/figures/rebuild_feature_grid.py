import os
"""
rebuild_grid.py
---------------
Re-assemble feature_percentiles_grid.png from already-saved images.
Does NOT require zensvi — runs in gisenv.

Usage:
    conda run -n gisenv python personality_svi/revision2/rebuild_grid.py
"""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("PERSONALITY_SVI_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT_DIR   = PROJECT_ROOT / "reports/figures/feature_percentiles"
OUT_GRID     = OUTPUT_DIR / "feature_percentiles_grid.png"

# ---------------------------------------------------------------------------
# Feature order (must match the order used when images were generated)
# ---------------------------------------------------------------------------
FEATURES_ORDER = [
    "env_building",
    "env_greenery",
    "env_open_space",
    "env_road",
    # "env_physical_boundaries",
    "env_vehicle_presence",
    # "env_active_presence",      # excluded: segmenter rarely detects people in TX suburban scenes
]

# Readable labels (consistent with run_ours_v2_analysis.py FEATURE_LABELS)
FEATURE_LABELS = {
    "env_building":            "Building",
    "env_greenery":            "Greenery",
    "env_open_space":          "Open Space",
    "env_road":                "Road",
    # "env_physical_boundaries": "Physical Boundaries",
    "env_vehicle_presence":    "Vehicle Presence",
    # "env_active_presence":     "Active Presence",
}

# ---------------------------------------------------------------------------
# Manual overrides: explicitly pin which panoid to use per (feature, level).
# Change these to swap images without re-running step1.
# ---------------------------------------------------------------------------
MANUAL_PANOID_OVERRIDES = {
    "env_road": {
        # q21fwp6ilJ0az: narrow residential alley — clearly small road, good road mask.
        "low": "q21fwp6ilJ0azOd5rFnxLA",
        # Tdk3_uOIc9PuSF: standard 2-lane suburban road — medium width, clear road mask.
        "medium": "Tdk3_uOIc9PuSFU2zJwyBA",
    },
    # "env_physical_boundaries": {
    #     # YOoSLwFypqlRIS: suburban road with visible wooden fence; mask has 3 real detections.
    #     "medium": "YOoSLwFypqlRIS-S8RMlIg",
    # },
    "env_vehicle_presence": {
        # jwRaJaogPhpnZeZx: strip mall parking lot; mask shows 3 clear vehicle blobs.
        "high": "jwRaJaogPhpnZeZx-qckQA",
    },
    # "env_active_presence": {
    #     # u33iXqRsS5cyY: clean rural road, no people — correct for low (black mask saved).
    #     "low": "u33iXqRsS5cyY_UNYf78GQ",
    #     # N67s_LFGmUU3gc: open suburban scene; one real mask detection.
    #     "high": "N67s_LFGmUU3gc4ngDKQqA",
    # },
}

LEVELS = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve(feat: str, lvl: str) -> tuple:
    """Return (perspective_path, mask_path) — respects MANUAL_PANOID_OVERRIDES
    then falls back to alphabetically first available file."""
    override = MANUAL_PANOID_OVERRIDES.get(feat, {}).get(lvl)
    if override:
        p = OUTPUT_DIR / f"{feat}_{lvl}_{override}_perspective.png"
        m = OUTPUT_DIR / f"{feat}_{lvl}_{override}_mask.png"
        return (p if p.exists() else None), (m if m.exists() else None)

    candidates = sorted(OUTPUT_DIR.glob(f"{feat}_{lvl}_*_perspective.png"))
    if not candidates:
        return None, None
    p = candidates[0]
    stem = p.stem  # e.g. env_building_low_bYgK8hZGLe8j2N4a7G5KeA_perspective → need panoid
    prefix = f"{feat}_{lvl}_"
    panoid = stem[len(prefix):].replace("_perspective", "")
    m = OUTPUT_DIR / f"{feat}_{lvl}_{panoid}_mask.png"
    return p, (m if m.exists() else None)


def _load(path) -> np.ndarray:
    if path and Path(path).exists():
        try:
            return mpimg.imread(str(path))
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
    # grey placeholder
    img = Image.new("RGB", (360, 240), color=(180, 180, 180))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
        txt = "Missing"
        bbox = d.textbbox((0, 0), txt, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((360 - w) / 2, (240 - h) / 2), txt, fill=(0, 0, 0), font=font)
    except Exception:
        pass
    return np.array(img)


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------
def build_grid():
    print("Building grid from:", OUTPUT_DIR)

    n_feat = len(FEATURES_ORDER)
    n_rows = n_feat * 2   # perspective + mask per feature
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.6, n_rows * 2.1))
    plt.subplots_adjust(left=0.18, right=0.99, top=0.93, bottom=0.03,
                        hspace=0.12, wspace=0.03)

    for fi, feat in enumerate(FEATURES_ORDER):
        for ci, lvl in enumerate(LEVELS):
            p_path, m_path = _resolve(feat, lvl)
            print(f"  {feat} {lvl}: p={p_path.name if p_path else 'MISSING'} "
                  f"m={m_path.name if m_path else 'MISSING'}")
            axes[fi * 2,     ci].imshow(_load(p_path))
            axes[fi * 2 + 1, ci].imshow(_load(m_path))
            axes[fi * 2,     ci].axis("off")
            axes[fi * 2 + 1, ci].axis("off")

    # Column headers
    for ci, lab in enumerate(["Low", "Medium", "High"]):
        ax = axes[0, ci]
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.935, lab,
                 ha="center", va="bottom", fontsize=12, weight="bold")

    # Row labels
    for fi, feat in enumerate(FEATURES_ORDER):
        ax0 = axes[fi * 2,     0]
        ax1 = axes[fi * 2 + 1, 0]
        ax0.text(-0.10, 0.5, "Original", ha="center", va="center",
                 fontsize=10, rotation=90, transform=ax0.transAxes)
        ax1.text(-0.10, 0.5, "Mask", ha="center", va="center",
                 fontsize=10, rotation=90, transform=ax1.transAxes)

        p0 = ax0.get_position()
        p1 = ax1.get_position()
        y_mid = (p0.y0 + p1.y1) / 2
        label = FEATURE_LABELS.get(feat, feat)
        fig.text(0.13, y_mid, label, ha="center", va="center",
                 fontsize=12, weight="bold", rotation=90)

    fig.savefig(OUT_GRID, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {OUT_GRID}")


if __name__ == "__main__":
    build_grid()
