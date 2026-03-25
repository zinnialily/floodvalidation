"""
download_swimmingpool.py -- Supplement Swimmingpool category to 400 images.

Sources (tried in order until --max reached):
  1. places365 : MIT Places365 validation set, swimming_pool/outdoor category.
                 Downloads val_256.tar (~1.4 GB) and extracts the target category.
                 The val set has 100 images per category × 365 categories = 36,500 images.
                 swimming_pool/outdoor is category index 326 (0-based).
  2. openimages: Open Images v7 "Swimming pool" class via fiftyone zoo
                 (CSV-based fallback if fiftyone is unavailable).

Idempotent: counts existing Swimmingpool_*.jpg files and only downloads
the difference up to --max.

Output directory: {--output_dir}/Swimmingpool/
Naming:           Swimmingpool_0001.jpg, Swimmingpool_0002.jpg, …

Usage:
    python scripts/download_swimmingpool.py --dry_run --max 10
    python scripts/download_swimmingpool.py --max 400
    python scripts/download_swimmingpool.py --max 400 --sources openimages
"""

from __future__ import annotations

import csv
import io
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from tqdm import tqdm  # type: ignore

sys.path.insert(0, os.path.dirname(__file__))
from download_utils import (  # noqa: E402
    build_parser,
    check_fiftyone,
    get_next_index,
    save_image,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

CATEGORY = "Swimmingpool"
SOURCES = ["places365", "openimages"]

# ---------------------------------------------------------------------------
# Places365 constants
#
# The val_256 set has 36,500 images: 100 per class × 365 classes.
# Classes are ordered as in categories_places365.txt (0-based index).
# swimming_pool/outdoor is confirmed at index 326.
#
# Val images are stored in the tar as:
#   data_256/val/{letter}/{scene_name}/Places365_val_NNNNN.jpg
# ---------------------------------------------------------------------------

_P365_CATEGORIES_URL = (
    "https://raw.githubusercontent.com/CSAILVision/places365/master/"
    "categories_places365.txt"
)
_P365_VAL_TAR_URL = "http://data.csail.mit.edu/places/places365/val_256.tar"
_P365_IMAGES_PER_CLASS = 100  # exactly 100 val images per class in standard split


def _download_places365(output_dir: Path, needed: int, dry_run: bool) -> int:
    """Fetch swimming_pool/outdoor images from the Places365 validation tar."""
    import requests  # type: ignore

    saved = 0
    idx = get_next_index(output_dir, CATEGORY)

    # Step 1: resolve category index for swimming_pool/outdoor
    try:
        resp = requests.get(_P365_CATEGORIES_URL, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Places365 category list unavailable: %s", exc)
        return 0

    cat_index: int | None = None
    cat_path: str | None = None
    for line in resp.text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2 and "swimming_pool" in parts[0] and "outdoor" in parts[0]:
            cat_index = int(parts[1])
            cat_path = parts[0].lstrip("/")  # e.g. "s/swimming_pool/outdoor"
            break

    if cat_index is None:
        logger.warning("swimming_pool/outdoor not found in Places365 categories.")
        return 0

    logger.info(
        "Places365: swimming_pool/outdoor → index %d, path '%s'", cat_index, cat_path
    )

    if dry_run:
        print(
            f"[dry_run] Would download Places365 val_256.tar and extract "
            f"~{_P365_IMAGES_PER_CLASS} images from '{cat_path}'"
        )
        return min(needed, _P365_IMAGES_PER_CLASS)

    # Step 2: download val_256.tar to a temp directory and stream-extract
    # only the swimming_pool/outdoor subfolder to avoid storing 1.4 GB.
    logger.info(
        "Downloading Places365 val_256.tar (~1.4 GB) — "
        "only swimming_pool/outdoor images will be kept …"
    )

    with tempfile.TemporaryDirectory(prefix="places365_") as tmpdir:
        tar_path = Path(tmpdir) / "val_256.tar"

        try:
            with requests.get(_P365_VAL_TAR_URL, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(tar_path, "wb") as fh, tqdm(
                    total=total, unit="B", unit_scale=True, desc="Places365 val_256.tar"
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
                        pbar.update(len(chunk))
        except Exception as exc:
            logger.warning("Places365 val tar download failed: %s", exc)
            return 0

        # Step 3: stream-extract only the target category subfolder
        target_prefix = f"data_256/val/{cat_path}/"
        try:
            with tarfile.open(tar_path) as tf:
                members = [
                    m for m in tf.getmembers()
                    if m.name.startswith(target_prefix) and m.isfile()
                ]
                logger.info(
                    "Places365: %d images found under '%s'",
                    len(members), target_prefix
                )
                for member in tqdm(members[:needed], desc="Places365 Swimmingpool"):
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    img_bytes = fobj.read()
                    if save_image(img_bytes, CATEGORY, idx, output_dir, dry_run):
                        saved += 1
                        idx += 1
                    if saved >= needed:
                        break
        except Exception as exc:
            logger.warning("Places365 tar extraction failed: %s", exc)

    return saved


# ---------------------------------------------------------------------------
# Open Images v7 source
# ---------------------------------------------------------------------------

# v6 class-descriptions CSV (v7 shares the same format; v6 URL is reliable)
_OI_CLASS_DESC_URL = (
    "https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv"
)
_OI_VAL_LABELS_URL = (
    "https://storage.googleapis.com/openimages/v5/"
    "validation-annotations-human-imagelabels.csv"
)
_OI_VAL_IMAGES_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-boxable.csv"
)


def _download_openimages(output_dir: Path, needed: int, dry_run: bool) -> int:
    """Fetch Swimming pool images from Open Images v7.

    Tries fiftyone first; falls back to direct CSV download if unavailable.
    """
    saved = 0
    idx = get_next_index(output_dir, CATEGORY)

    if check_fiftyone():
        import fiftyone as fo  # type: ignore
        import fiftyone.zoo as foz  # type: ignore

        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="validation",
                label_types=["classifications"],
                classes=["Swimming pool"],
                max_samples=needed,
            )
            for sample in tqdm(dataset, desc="OpenImages Swimmingpool"):
                if save_image(sample.filepath, CATEGORY, idx, output_dir, dry_run):
                    saved += 1
                    idx += 1
                if saved >= needed:
                    break
            fo.delete_dataset(dataset.name)
            return saved
        except Exception as exc:
            logger.warning("fiftyone zoo download failed: %s — trying CSV fallback", exc)

    logger.info("Using Open Images CSV fallback for Swimmingpool.")
    return _openimages_csv(output_dir, idx, needed, dry_run, "Swimming pool")


def _openimages_csv(
    output_dir: Path,
    start_idx: int,
    needed: int,
    dry_run: bool,
    class_name: str,
) -> int:
    """Download images via Open Images CSV metadata (no fiftyone required)."""
    import requests  # type: ignore

    saved = 0
    idx = start_idx

    # Resolve LabelName MID for the target class
    try:
        resp = requests.get(_OI_CLASS_DESC_URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Open Images class descriptions unavailable: %s", exc)
        return 0

    label_name: str | None = None
    for row in csv.reader(io.StringIO(resp.text)):
        if len(row) >= 2 and row[1].strip().lower() == class_name.lower():
            label_name = row[0]
            break

    if not label_name:
        logger.warning("'%s' not found in Open Images class descriptions.", class_name)
        return 0

    logger.info("Open Images: '%s' → LabelName %s", class_name, label_name)

    # Collect image IDs with a positive human label
    try:
        resp = requests.get(_OI_VAL_LABELS_URL, timeout=180)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Open Images val labels unavailable: %s", exc)
        return 0

    image_ids: list[str] = []
    reader = csv.reader(io.StringIO(resp.text))
    next(reader, None)  # skip header: ImageID,Source,LabelName,Confidence
    for row in reader:
        if len(row) >= 4 and row[2] == label_name and row[3] == "1":
            image_ids.append(row[0])
        if len(image_ids) >= needed * 4:
            break

    logger.info("Open Images: %d '%s' image IDs found", len(image_ids), class_name)

    # Resolve image IDs to download URLs
    try:
        resp = requests.get(_OI_VAL_IMAGES_URL, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Open Images image list unavailable: %s", exc)
        return 0

    id_set = set(image_ids)
    id_to_url: dict[str, str] = {}
    for row in csv.reader(io.StringIO(resp.text)):
        # Format: ImageID, Subset, OriginalURL, OriginalLandingURL, ...
        if len(row) >= 3 and row[0] in id_set:
            id_to_url[row[0]] = row[2]

    for img_id in tqdm(image_ids, desc=f"OpenImages CSV {class_name}"):
        url = id_to_url.get(img_id)
        if not url:
            continue
        if save_image(url, CATEGORY, idx, output_dir, dry_run):
            saved += 1
            idx += 1
        if saved >= needed:
            break

    return saved


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser(CATEGORY, SOURCES)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / CATEGORY
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = get_next_index(output_dir, CATEGORY) - 1
    needed = max(0, args.max - existing)
    print(
        f"[{CATEGORY}] existing={existing}  target={args.max}  to_download={needed}"
    )

    if needed == 0:
        print(f"[{CATEGORY}] Already at or above target — nothing to do.")
        return

    totals: dict[str, int] = {}

    if "places365" in args.sources and needed > 0:
        n = _download_places365(output_dir, needed, args.dry_run)
        totals["places365"] = n
        needed -= n
        print(f"  places365    → {n} images saved")

    if "openimages" in args.sources and needed > 0:
        n = _download_openimages(output_dir, needed, args.dry_run)
        totals["openimages"] = n
        needed -= n
        print(f"  openimages   → {n} images saved")

    total = sum(totals.values())
    final_count = get_next_index(output_dir, CATEGORY) - 1
    print(f"\n[{CATEGORY}] {total} new images downloaded.  Directory total: {final_count}")


if __name__ == "__main__":
    main()
