"""
analyze_confounders.py -- Data-driven confounder false-positive ranking.

Runs a trained baseline model over all non-flood categories in the training
partition, ranks them by false positive rate, and outputs a file listing which
individual images belong to categories that exceed the FP rate threshold --
ready for consumption by train_hnm.py.

Compute: CPU (no GPU required for inference on a small training partition).
Estimated runtime: 2-10 minutes depending on dataset size and hardware.

Usage:
    python scripts/analyze_confounders.py \
        --model_path results/checkpoints/efficientnet_phase2_best.keras \
        --arch efficientnet \
        --data_dir ./data/FloodingDataset2 \
        --fp_threshold 0.15 \
        --output_dir ./results
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))
from utils import PREPROCESS_FN  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE: Tuple[int, int] = (224, 224)
FP_DECISION_THRESHOLD: float = 0.5  # flood_prob > 0.5 → false positive (ranking only)

VALID_ARCHS: List[str] = list(PREPROCESS_FN.keys())

# Known category name prefixes in the non-flood training directory.
# Used only as documentation; actual grouping is inferred from filenames.
KNOWN_JUNK_PREFIXES: List[str] = [
    "Cats", "Deers", "Dogs", "Motorcycle", "Plants", "Swimmingpool",
    "building_exterior", "building_interior", "bus", "car",
    "house_exterior", "Racoon", "NoFlood", "parks_walkways",
    "River", "Lake", "Fountain",   # new external confounder categories
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rank non-flood training categories by false positive rate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="Path to trained baseline model (.keras file).",
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=VALID_ARCHS,
        help="Backbone architecture: 'efficientnet' or 'resnet50'.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data/FloodingDataset2",
        type=str,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--fp_threshold",
        default=0.15,
        type=float,
        help="FP rate threshold above which a category is flagged as a mining candidate.",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        type=str,
        help="Directory where result tables and candidate lists are written.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="Which data split to run confounder analysis on.",
    )
    parser.add_argument(
        "--mine_all",
        action="store_true",
        help=(
            "Write ALL non-flood images as mining candidates, regardless of FP rate. "
            "Use when binary FP rates are near zero (e.g. Phase 1 checkpoint) but "
            "probability-based mining is still desired. HNM's --top_pct then selects "
            "the hardest examples by flood probability score."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Partition safety guard
# ---------------------------------------------------------------------------


def confirm_partition_safety(train_non_flood_dir: str) -> None:
    """Print confirmation that only train/non_flood/ is being scanned.

    This is a human-readable safeguard.  Code never touches val or test
    directories; this message makes that guarantee visible in the run log.

    Args:
        train_non_flood_dir: The directory that will be scanned.
    """
    print(
        "\n[PARTITION SAFETY] Scanning train/non_flood/ only — val/test not touched."
    )
    print(f"  Scan target: {os.path.abspath(train_non_flood_dir)}\n")


# ---------------------------------------------------------------------------
# Image discovery and category grouping
# ---------------------------------------------------------------------------


def _category_from_filename(filename: str) -> str | None:
    """Extract category from filename prefix before first underscore.

    Args:
        filename: Base filename (without directory), e.g. ``Swimmingpool_001.jpg``.

    Returns:
        Category string if an underscore is present, else ``None``.
    """
    stem = Path(filename).stem  # strip extension
    if "_" in stem:
        return stem.split("_")[0]
    return None


def _load_split_csv_categories(
    data_dir: str,
    train_non_flood_dir: str,
) -> Dict[str, str] | None:
    """Attempt to load category labels from a train_split.csv in data_dir.

    Args:
        data_dir: Dataset root directory.
        train_non_flood_dir: Absolute path to the train/non_flood/ directory
            (used to filter relevant rows).

    Returns:
        A dict mapping absolute image path -> category string, or ``None``
        if the CSV does not exist or lacks required columns.
    """
    csv_path = os.path.join(data_dir, "train_split.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not read {csv_path}: {exc}")
        return None

    required_cols = {"path", "binary_label", "category"}
    if not required_cols.issubset(df.columns):
        print(
            f"[WARN] train_split.csv is missing columns {required_cols - set(df.columns)}; "
            "skipping CSV fallback."
        )
        return None

    non_flood_rows = df[df["binary_label"] == "non_flood"].copy()
    mapping: Dict[str, str] = {}
    for _, row in non_flood_rows.iterrows():
        abs_path = os.path.abspath(str(row["path"]))
        mapping[abs_path] = str(row["category"])

    return mapping if mapping else None


def collect_images_by_category(
    train_non_flood_dir: str,
    data_dir: str,
) -> Dict[str, List[str]]:
    """Group all images in train/non_flood/ by their category.

    With the new build_splits.py layout, images are stored in named category
    subdirectories (e.g. train/non_flood/river/, train/non_flood/building/).
    The category name is read directly from the subdirectory name.

    Falls back to filename-prefix inference and CSV lookup for legacy flat
    layouts where images sit directly inside train/non_flood/.

    Args:
        train_non_flood_dir: Absolute path to the train non-flood directory.
        data_dir: Dataset root, used to locate an optional split CSV.

    Returns:
        A dict mapping category name -> list of absolute image file paths.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    categories: Dict[str, List[str]] = {}
    flat_images: List[str] = []

    for entry in sorted(Path(train_non_flood_dir).iterdir()):
        if entry.is_dir():
            # New layout: subdirectory name IS the category.
            cat = entry.name
            imgs = sorted(
                str(p) for p in entry.iterdir()
                if p.is_file() and p.suffix.lower() in valid_extensions
            )
            if imgs:
                categories.setdefault(cat, []).extend(imgs)
        elif entry.is_file() and entry.suffix.lower() in valid_extensions:
            # Legacy flat layout: collect for filename-prefix fallback.
            flat_images.append(str(entry.resolve()))

    # Legacy fallback: filename prefix → CSV → "unknown".
    if flat_images:
        unresolved: List[str] = []
        for img_path in flat_images:
            cat = _category_from_filename(os.path.basename(img_path))
            if cat is not None:
                categories.setdefault(cat, []).append(img_path)
            else:
                unresolved.append(img_path)

        if unresolved:
            csv_mapping = _load_split_csv_categories(data_dir, train_non_flood_dir)
            if csv_mapping:
                still_unresolved: List[str] = []
                for img_path in unresolved:
                    if img_path in csv_mapping:
                        categories.setdefault(csv_mapping[img_path], []).append(img_path)
                    else:
                        still_unresolved.append(img_path)
                unresolved = still_unresolved
            if unresolved:
                categories.setdefault("unknown", []).extend(unresolved)

    if not categories:
        print("[WARN] No images found in train/non_flood/ directory.")
        return {}

    total = sum(len(v) for v in categories.values())
    print(
        f"[INFO] Discovered {total} images across {len(categories)} categories "
        f"in train/non_flood/."
    )
    return categories


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------


def load_model_safe(model_path: str) -> tf.keras.Model:
    """Load a saved Keras model from disk.

    Args:
        model_path: Filesystem path to the ``.keras`` checkpoint file.

    Returns:
        Loaded ``tf.keras.Model`` instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If loading fails for any other reason.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"[INFO] Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load model: {exc}") from exc
    print(f"[INFO] Model loaded. Output shape: {model.output_shape}")
    return model


def predict_flood_probability(
    model: tf.keras.Model,
    image_path: str,
    preprocess_fn,
) -> float:
    """Compute the flood probability for a single image.

    With flow_from_directory alphabetical class ordering (flood=0, non_flood=1),
    the model's sigmoid output is P(non_flood).  flood_prob = 1.0 - pred.

    We load raw [0, 255] pixels and apply the backbone-specific preprocessing
    function -- never a manual /255.

    Args:
        model: Loaded Keras model with a single sigmoid output.
        image_path: Absolute path to the image file.
        preprocess_fn: Preprocessing function from ``PREPROCESS_FN[arch]``.

    Returns:
        Float in [0.0, 1.0] representing P(flood).
    """
    img = load_img(image_path, target_size=IMAGE_SIZE)
    x = img_to_array(img)                     # shape (224, 224, 3), dtype float32, range [0, 255]
    x = preprocess_fn(x[np.newaxis, ...])     # backbone preprocessing; shape (1, 224, 224, 3)
    pred = float(model.predict(x, verbose=0)[0][0])  # P(non_flood) due to alphabetical class ordering
    flood_prob = 1.0 - pred
    return flood_prob


# ---------------------------------------------------------------------------
# Per-category FP rate computation
# ---------------------------------------------------------------------------


def compute_category_fp_rates(
    model: tf.keras.Model,
    categories: Dict[str, List[str]],
    preprocess_fn,
) -> pd.DataFrame:
    """Run inference on all images and compute per-category false positive rates.

    A non-flood image is counted as a false positive when flood_prob > 0.5.
    This threshold is only used for ranking; it is independent of the mining
    tau used in train_hnm.py.

    Args:
        model: Loaded Keras model.
        categories: Dict mapping category name -> list of image paths.
        preprocess_fn: Backbone preprocessing function.

    Returns:
        DataFrame with columns: category, n_images, n_fp, fp_rate.
        Sorted by fp_rate descending.
    """
    records: List[dict] = []

    total_images = sum(len(paths) for paths in categories.values())
    with tqdm(total=total_images, desc="Inference", unit="img") as pbar:
        for category, image_paths in sorted(categories.items()):
            n_fp = 0
            for img_path in image_paths:
                try:
                    flood_prob = predict_flood_probability(
                        model, img_path, preprocess_fn
                    )
                    if flood_prob > FP_DECISION_THRESHOLD:
                        n_fp += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"\n[WARN] Skipping {img_path}: {exc}")
                pbar.update(1)

            n_images = len(image_paths)
            fp_rate = n_fp / n_images if n_images > 0 else 0.0
            records.append(
                {
                    "category": category,
                    "n_images": n_images,
                    "n_fp": n_fp,
                    "fp_rate": fp_rate,
                }
            )

    df = pd.DataFrame(records).sort_values("fp_rate", ascending=False).reset_index(
        drop=True
    )
    return df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_ranked_table(df: pd.DataFrame) -> None:
    """Print the category FP rate table to stdout.

    Args:
        df: DataFrame with columns category, n_images, n_fp, fp_rate.
    """
    print("\n" + "=" * 60)
    print(f"{'Category':<25} {'N Images':>8} {'N FP':>6} {'FP Rate':>8}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(
            f"{row['category']:<25} {int(row['n_images']):>8} "
            f"{int(row['n_fp']):>6} {row['fp_rate']:>8.3f}"
        )
    print("=" * 60 + "\n")


def save_fp_rates_csv(df: pd.DataFrame, output_dir: str, arch: str) -> str:
    """Save the FP rate table to a CSV file.

    Args:
        df: DataFrame with columns category, n_images, n_fp, fp_rate.
        output_dir: Base output directory.
        arch: Architecture name used in the filename.

    Returns:
        Absolute path to the saved CSV.
    """
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    csv_path = os.path.join(tables_dir, f"confounder_fp_rates_{arch}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] FP rate table saved to: {csv_path}")
    return csv_path


def write_mining_candidates(
    df: pd.DataFrame,
    categories: Dict[str, List[str]],
    fp_threshold: float,
    output_dir: str,
    arch: str,
) -> Tuple[str, List[str]]:
    """Write the list of flagged image paths to a text file.

    Flags every category whose fp_rate exceeds fp_threshold and writes one
    absolute image path per line to ``mining_candidates_{arch}.txt``.  If no
    categories are flagged, the file is still written (empty) with a warning.

    Args:
        df: FP rate DataFrame (category, n_images, n_fp, fp_rate).
        categories: Dict mapping category name -> list of image paths.
        fp_threshold: FP rate above which a category is flagged.
        output_dir: Base output directory.
        arch: Architecture name used in the filename.

    Returns:
        Tuple of (output_file_path, list_of_flagged_category_names).
    """
    mining_dir = os.path.join(output_dir, "mining")
    os.makedirs(mining_dir, exist_ok=True)
    candidates_path = os.path.join(mining_dir, f"mining_candidates_{arch}.txt")

    flagged_cats = df[df["fp_rate"] > fp_threshold]["category"].tolist()

    flagged_image_paths: List[str] = []
    for cat in flagged_cats:
        flagged_image_paths.extend(categories.get(cat, []))

    with open(candidates_path, "w", encoding="utf-8") as fh:
        for img_path in sorted(flagged_image_paths):
            fh.write(img_path + "\n")

    if flagged_cats:
        print(
            f"\n[INFO] Flagged {len(flagged_cats)} categories with FP rate > "
            f"{fp_threshold:.0%}: {flagged_cats}"
        )
        print(
            f"[INFO] {len(flagged_image_paths)} candidate images written to: "
            f"{candidates_path}"
        )
    else:
        print(
            f"\n[WARN] No categories exceeded the FP threshold of {fp_threshold:.0%}. "
            "An empty candidates file has been written."
        )
        print(f"[INFO] Candidates file (empty): {candidates_path}")

    return candidates_path, flagged_cats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, run analysis, write outputs."""
    args = parse_args()

    # Resolve paths.
    data_dir: str = os.path.abspath(args.data_dir)
    output_dir: str = os.path.abspath(args.output_dir)
    split: str = args.split
    non_flood_dir: str = os.path.join(
        data_dir, "processed_data", "binary", split, "non_flood"
    )

    # Validate inputs.
    if not os.path.isdir(non_flood_dir):
        raise FileNotFoundError(
            f"Non-flood directory not found: {non_flood_dir}\n"
            "Run 02_stratified_splitting.ipynb first to create the split."
        )

    # Partition safety announcement.
    confirm_partition_safety(non_flood_dir)

    # Load model.
    model = load_model_safe(args.model_path)

    # Select preprocessing function.
    preprocess_fn = PREPROCESS_FN[args.arch]

    # Group images by category.
    categories = collect_images_by_category(non_flood_dir, data_dir)
    if not categories:
        print("[ERROR] No images found. Exiting.")
        sys.exit(1)

    # Run inference and compute FP rates.
    print(f"\n[INFO] Running inference with arch={args.arch} ...")
    df_fp_rates = compute_category_fp_rates(model, categories, preprocess_fn)

    # Print ranked table.
    print_ranked_table(df_fp_rates)

    # Save CSV (suffix split name so train/val outputs don't overwrite each other).
    arch_split_tag = f"{args.arch}_{split.upper()}"
    save_fp_rates_csv(df_fp_rates, output_dir, arch_split_tag)

    # Write mining candidates only for the train split.
    if split == "train":
        if args.mine_all:
            # Write every non-flood training image as a candidate.
            # HNM's --top_pct will rank by flood probability and select the hardest N%.
            # Used when binary FP rates are near zero (e.g. Phase 1 checkpoint) but
            # probability-based mining is still desired.
            all_images = sorted(p for paths in categories.values() for p in paths)
            mining_dir = os.path.join(output_dir, "mining")
            os.makedirs(mining_dir, exist_ok=True)
            candidates_path = os.path.join(mining_dir, f"mining_candidates_{args.arch}.txt")
            with open(candidates_path, "w", encoding="utf-8") as fh:
                for img_path in all_images:
                    fh.write(img_path + "\n")
            print(
                f"\n[INFO] --mine_all: wrote all {len(all_images)} non-flood training "
                f"images as candidates → {candidates_path}"
            )
            print("[INFO] HNM --top_pct will select the hardest examples by flood probability.")
        else:
            write_mining_candidates(
                df_fp_rates,
                categories,
                args.fp_threshold,
                output_dir,
                args.arch,
            )
    else:
        print("[INFO] --split=val: skipping mining candidates file (train split only).")

    print("\n[DONE] analyze_confounders.py completed successfully.")


if __name__ == "__main__":
    main()
