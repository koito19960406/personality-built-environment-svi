"""Generate the UT News semantic-segmentation figure.

The figure uses the same Mask2Former architecture and Mapillary Vistas
training data described in the manuscript. Google Maps interface elements are
excluded from the semantic-class summaries and are marked in the output.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/raw/ut_news_segmentation/ut_austin_university_ave_2023-01.png"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "reports/figures/semantic_segmentation/ut_austin_university_ave_segmentation.png"
)
DEFAULT_SEGMENTATION_ONLY_OUTPUT = (
    PROJECT_ROOT
    / "reports/figures/semantic_segmentation/"
    "ut_austin_university_ave_segmentation_only.png"
)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ut_news_segmentation"
MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

# Google Maps controls and information cards in the supplied 1277 x 997 image.
# Coordinates are scaled automatically if the source image dimensions change.
REFERENCE_SIZE = (1277, 997)
INTERFACE_RECTS = (
    (14, 10, 392, 63),      # search bar
    (16, 70, 321, 213),     # location information card
    (1118, 8, 1269, 58),    # share and close buttons
    (16, 872, 243, 982),    # inset map
    (1215, 827, 1277, 983), # navigation controls
    (0, 982, 1277, 997),    # footer links and attribution strip
)

# Semantic anchor colours use a colour-vision-deficiency-conscious palette.
SEMANTIC_COLOURS = {
    "sky": "#56B4E9",
    "vegetation": "#009E73",
    "terrain": "#8C6D31",
    "building": "#D55E00",
    "road": "#6A3D9A",
    "lane marking": "#222222",
    "sidewalk": "#F0C808",
    "car": "#0072B2",
    "truck": "#CC79A7",
    "person": "#E41A1C",
    "fence": "#A6761D",
    "wall": "#666666",
}
FALLBACK_COLOURS = (
    "#4477AA",
    "#228833",
    "#EE6677",
    "#CCBB44",
    "#66CCEE",
    "#AA3377",
    "#BBBBBB",
    "#332288",
)
OTHER_COLOUR = "#D9D9D9"
INTERFACE_COLOUR = "#F7F7F7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--segmentation-only-output",
        type=Path,
        default=DEFAULT_SEGMENTATION_ONLY_OUTPUT,
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--max-classes", type=int, default=8)
    return parser.parse_args()


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_interface_mask(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale_x = width / REFERENCE_SIZE[0]
    scale_y = height / REFERENCE_SIZE[1]
    mask = np.zeros((height, width), dtype=bool)
    for left, top, right, bottom in INTERFACE_RECTS:
        x0, x1 = round(left * scale_x), round(right * scale_x)
        y0, y1 = round(top * scale_y), round(bottom * scale_y)
        mask[y0:y1, x0:x1] = True
    return mask


def class_colour(label: str, fallback_index: int) -> str:
    normalised = label.casefold()
    for keyword, colour in SEMANTIC_COLOURS.items():
        if keyword in normalised:
            return colour
    return FALLBACK_COLOURS[fallback_index % len(FALLBACK_COLOURS)]


def save_class_shares(
    output_path: Path,
    class_counts: list[tuple[int, int]],
    id2label: dict[int, str],
    analysed_pixels: int,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class_id", "class_name", "pixel_count", "pixel_share_percent"))
        for class_id, count in class_counts:
            writer.writerow(
                (
                    class_id,
                    id2label.get(class_id, f"Class {class_id}"),
                    count,
                    f"{100 * count / analysed_pixels:.4f}",
                )
            )


def add_interface_hatching(axis: plt.Axes, image_size: tuple[int, int]) -> None:
    """Mark Google Maps interface regions excluded from segmentation summaries."""
    scale_x = image_size[0] / REFERENCE_SIZE[0]
    scale_y = image_size[1] / REFERENCE_SIZE[1]
    for left, top, right, bottom in INTERFACE_RECTS:
        axis.add_patch(
            Rectangle(
                (left * scale_x, top * scale_y),
                (right - left) * scale_x,
                (bottom - top) * scale_y,
                fill=False,
                hatch="///",
                edgecolor="#B0B0B0",
                linewidth=0.35,
            )
        )


def render_figure(
    image: Image.Image,
    semantic_map: np.ndarray,
    interface_mask: np.ndarray,
    id2label: dict[int, str],
    output_path: Path,
    segmentation_only_output_path: Path,
    processed_dir: Path,
    max_classes: int,
) -> None:
    valid_pixels = semantic_map[~interface_mask]
    class_ids, counts = np.unique(valid_pixels, return_counts=True)
    ranked = sorted(
        zip(class_ids.tolist(), counts.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    analysed_pixels = int(valid_pixels.size)
    main_classes = ranked[:max_classes]
    main_ids = {class_id for class_id, _ in main_classes}

    processed_dir.mkdir(parents=True, exist_ok=True)
    save_class_shares(
        processed_dir / "ut_austin_university_ave_class_shares.csv",
        ranked,
        id2label,
        analysed_pixels,
    )

    labels_for_storage = semantic_map.astype(np.uint8)
    labels_for_storage[interface_mask] = 255
    Image.fromarray(labels_for_storage).save(
        processed_dir / "ut_austin_university_ave_class_ids.png"
    )

    rgb = np.full((*semantic_map.shape, 3), 217, dtype=np.uint8)
    class_colours: dict[int, str] = {}
    for fallback_index, (class_id, _) in enumerate(main_classes):
        label = id2label.get(class_id, f"Class {class_id}")
        colour = class_colour(label, fallback_index)
        class_colours[class_id] = colour
        rgb[semantic_map == class_id] = tuple(
            int(colour[index : index + 2], 16) for index in (1, 3, 5)
        )
    rgb[interface_mask] = (247, 247, 247)
    Image.fromarray(rgb).save(processed_dir / "ut_austin_university_ave_segmented.png")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    figure = plt.figure(figsize=(14, 5.35), facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1, 1, 0.60),
        left=0.025,
        right=0.985,
        top=0.92,
        bottom=0.11,
        wspace=0.035,
    )
    original_axis = figure.add_subplot(grid[0, 0])
    segmented_axis = figure.add_subplot(grid[0, 1])
    legend_axis = figure.add_subplot(grid[0, 2])

    original_axis.imshow(image)
    segmented_axis.imshow(rgb)
    for axis, panel, title in (
        (original_axis, "a", "Original street view"),
        (segmented_axis, "b", "Semantic segmentation"),
    ):
        axis.set_axis_off()
        axis.set_title(title, fontsize=13, fontweight="semibold", pad=8)
        axis.text(
            0.015,
            0.975,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "square,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )

    add_interface_hatching(segmented_axis, image.size)

    legend_axis.set_axis_off()
    legend_axis.text(
        0.0,
        0.98,
        "Main semantic classes",
        transform=legend_axis.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
    )
    handles: list[Patch] = []
    for class_id, count in main_classes:
        label = id2label.get(class_id, f"Class {class_id}")
        share = 100 * count / analysed_pixels
        handles.append(
            Patch(
                facecolor=class_colours[class_id],
                edgecolor="white",
                label=f"{label}  {share:.1f}%",
            )
        )
    other_count = sum(count for class_id, count in ranked if class_id not in main_ids)
    other_share = 100 * other_count / analysed_pixels
    if other_share > 0:
        handles.append(
            Patch(
                facecolor=OTHER_COLOUR,
                edgecolor="white",
                label=f"Other classes  {other_share:.1f}%",
            )
        )
    handles.append(
        Patch(
            facecolor=INTERFACE_COLOUR,
            edgecolor="#B0B0B0",
            hatch="///",
            label="Interface excluded",
        )
    )
    legend_axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, 0.90),
        frameon=False,
        handlelength=1.7,
        handleheight=1.1,
        labelspacing=0.78,
        borderaxespad=0,
        fontsize=10.5,
    )
    legend_axis.text(
        0.0,
        0.08,
        "Percentages show the share of analyzed pixels.\n"
        "Model: Mask2Former, trained on Mapillary Vistas.",
        transform=legend_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
        linespacing=1.35,
    )
    figure.text(
        0.025,
        0.025,
        "Source image: Google Street View, 2098 University Ave, Austin, Texas (January 2023).",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)

    segmentation_figure = plt.figure(figsize=(10, 5.35), facecolor="white")
    segmentation_grid = segmentation_figure.add_gridspec(
        1,
        2,
        width_ratios=(1, 0.48),
        left=0.035,
        right=0.98,
        top=0.91,
        bottom=0.12,
        wspace=0.04,
    )
    segmentation_axis = segmentation_figure.add_subplot(segmentation_grid[0, 0])
    segmentation_legend_axis = segmentation_figure.add_subplot(segmentation_grid[0, 1])
    segmentation_axis.imshow(rgb)
    segmentation_axis.set_axis_off()
    segmentation_axis.set_title(
        "Semantic segmentation",
        fontsize=13,
        fontweight="semibold",
        pad=8,
    )
    add_interface_hatching(segmentation_axis, image.size)

    segmentation_legend_axis.set_axis_off()
    segmentation_legend_axis.text(
        0.0,
        0.98,
        "Main semantic classes",
        transform=segmentation_legend_axis.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
    )
    segmentation_legend_axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(-0.02, 0.90),
        frameon=False,
        handlelength=1.7,
        handleheight=1.1,
        labelspacing=0.78,
        borderaxespad=0,
        fontsize=10.5,
    )
    segmentation_legend_axis.text(
        0.0,
        0.08,
        "Percentages show the share of analyzed pixels.\n"
        "Model: Mask2Former, trained on Mapillary Vistas.",
        transform=segmentation_legend_axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
        linespacing=1.35,
    )
    segmentation_figure.text(
        0.035,
        0.025,
        "Source image: Google Street View, 2098 University Ave, Austin, Texas "
        "(January 2023).",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    segmentation_only_output_path.parent.mkdir(parents=True, exist_ok=True)
    segmentation_figure.savefig(
        segmentation_only_output_path,
        dpi=300,
        facecolor="white",
    )
    plt.close(segmentation_figure)


def main() -> None:
    args = parse_args()
    image = Image.open(args.input).convert("RGB")
    device = select_device()
    print(f"Loading {args.model_id} on {device}...")
    processor = AutoImageProcessor.from_pretrained(args.model_id, use_fast=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_id,
        use_safetensors=True,
    ).to(device)
    model.eval()

    prepared_inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in prepared_inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    semantic_map = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )[0].cpu().numpy()

    id2label = {int(class_id): label for class_id, label in model.config.id2label.items()}
    interface_mask = create_interface_mask(image.size)
    render_figure(
        image=image,
        semantic_map=semantic_map,
        interface_mask=interface_mask,
        id2label=id2label,
        output_path=args.output,
        segmentation_only_output_path=args.segmentation_only_output,
        processed_dir=args.processed_dir,
        max_classes=args.max_classes,
    )
    print(f"Saved figure: {args.output}")
    print(f"Saved segmentation-only figure: {args.segmentation_only_output}")


if __name__ == "__main__":
    main()
