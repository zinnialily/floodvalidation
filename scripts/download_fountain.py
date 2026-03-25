"""
download_fountain.py -- Download Fountain category images for the confounder HNM dataset.

Sources (tried in order until --max reached):
  1. openimages : Open Images v7 "Fountain" class via fiftyone zoo
                  (CSV-based fallback if fiftyone is unavailable).
  2. ade20k     : ADE20K dataset — filters images whose scene category or
                  annotated objects contain "fountain".

Output directory: {--output_dir}/Fountain/
Naming:           Fountain_0001.jpg, Fountain_0002.jpg, …

Usage:
    python scripts/download_fountain.py --dry_run --max 10
    python scripts/download_fountain.py --max 400
    python scripts/download_fountain.py --max 400 --sources ade20k
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
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

CATEGORY = "Fountain"
SOURCES = ["openimages", "ade20k"]

# Open Images CSV URLs (shared with swimmingpool script)
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

# ADE20K index JSON
# The public index JSON is hosted at the MIT CSAIL group page.
_ADE20K_INDEX_URL = (
    "https://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/"
    "index_ade20k.json"
)
# Fallback: GitHub mirror that some community members maintain
_ADE20K_INDEX_URL_FALLBACK = (
    "https://raw.githubusercontent.com/CSAILVision/ADE20K/main/"
    "index_ade20k.json"
)


# ---------------------------------------------------------------------------
# Open Images source
# ---------------------------------------------------------------------------


def _download_openimages(output_dir: Path, needed: int, dry_run: bool) -> int:
    """Fetch Fountain images from Open Images v7.

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
                classes=["Fountain"],
                max_samples=needed,
            )
            for sample in tqdm(dataset, desc="OpenImages Fountain"):
                if save_image(sample.filepath, CATEGORY, idx, output_dir, dry_run):
                    saved += 1
                    idx += 1
                if saved >= needed:
                    break
            fo.delete_dataset(dataset.name)
            return saved
        except Exception as exc:
            logger.warning("fiftyone zoo download failed: %s — trying CSV fallback", exc)

    logger.info("Using Open Images CSV fallback for Fountain.")
    return _openimages_csv(output_dir, idx, needed, dry_run)


def _openimages_csv(
    output_dir: Path, start_idx: int, needed: int, dry_run: bool
) -> int:
    """Download Fountain images via Open Images CSV metadata."""
    import requests  # type: ignore

    saved = 0
    idx = start_idx

    try:
        resp = requests.get(_OI_CLASS_DESC_URL, timeout=60)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Open Images class descriptions unavailable: %s", exc)
        return 0

    label_name: str | None = None
    for row in csv.reader(io.StringIO(resp.text)):
        if len(row) >= 2 and row[1].strip().lower() == "fountain":
            label_name = row[0]
            break

    if not label_name:
        logger.warning("'Fountain' not found in Open Images class descriptions.")
        return 0

    try:
        resp = requests.get(_OI_VAL_LABELS_URL, timeout=180)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Open Images val labels unavailable: %s", exc)
        return 0

    image_ids: list[str] = []
    reader = csv.reader(io.StringIO(resp.text))
    next(reader, None)
    for row in reader:
        if len(row) >= 4 and row[2] == label_name and row[3] == "1":
            image_ids.append(row[0])
        if len(image_ids) >= needed * 4:
            break

    try:
        resp = requests.get(_OI_VAL_IMAGES_URL, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Open Images image list unavailable: %s", exc)
        return 0

    id_set = set(image_ids)
    id_to_url: dict[str, str] = {}
    for row in csv.reader(io.StringIO(resp.text)):
        if len(row) >= 3 and row[0] in id_set:
            id_to_url[row[0]] = row[2]

    for img_id in tqdm(image_ids, desc="OpenImages CSV Fountain"):
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
# ADE20K source
# ---------------------------------------------------------------------------


def _fetch_ade20k_index() -> dict | None:
    """Try to download the ADE20K index JSON. Returns parsed dict or None."""
    import requests  # type: ignore

    for url in (_ADE20K_INDEX_URL, _ADE20K_INDEX_URL_FALLBACK):
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.debug("ADE20K index at %s unavailable: %s", url, exc)

    logger.warning(
        "ADE20K index JSON unavailable from all known URLs. "
        "If you have a local copy, place it at ./data/index_ade20k.json "
        "and re-run."
    )
    return None


def _download_ade20k(output_dir: Path, needed: int, dry_run: bool) -> int:
    """Fetch fountain images from the ADE20K dataset via its index JSON.

    Filters images where the scene category or any annotated object name
    contains the word 'fountain'.
    """
    import requests  # type: ignore

    saved = 0
    idx = get_next_index(output_dir, CATEGORY)

    # Check for local cached index first
    local_index = Path("data/index_ade20k.json")
    if local_index.exists():
        try:
            index_data = json.loads(local_index.read_text())
            logger.info("ADE20K: using local index at %s", local_index)
        except Exception:
            index_data = None
    else:
        index_data = _fetch_ade20k_index()

    if not index_data:
        return 0

    # The ADE20K index JSON format has a key "filename" listing all image paths
    # and "folder" for the base URL, along with per-image scene/object info.
    # Different versions of the index have slightly different schemas; handle both.

    # Extract filename list and corresponding scene/object annotations
    filenames: list[str] = index_data.get("filename", [])
    scene_categories: list[str] = index_data.get("scene", [])  # may be empty
    object_names: list[list[str]] = index_data.get("objectnames", [])  # may be empty
    folder: str = index_data.get("folder", "")  # base path prefix

    # ADE20K public image base URL
    ade20k_img_base = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26/"

    fountain_paths: list[str] = []
    for i, fname in enumerate(filenames):
        is_fountain = False

        scene = scene_categories[i] if i < len(scene_categories) else ""
        if "fountain" in scene.lower():
            is_fountain = True

        if not is_fountain and i < len(object_names):
            if any("fountain" in obj.lower() for obj in object_names[i]):
                is_fountain = True

        if not is_fountain and "fountain" in fname.lower():
            is_fountain = True

        if is_fountain:
            fountain_paths.append(fname)

    logger.info("ADE20K: %d fountain-related images found", len(fountain_paths))

    for fname in tqdm(fountain_paths[:needed], desc="ADE20K Fountain"):
        if saved >= needed:
            break

        # Try as a URL first
        url = ade20k_img_base + fname.lstrip("/")
        try:
            import requests as _req

            resp = _req.get(url, timeout=20)
            resp.raise_for_status()
            src: str | Path = url
        except Exception:
            # Try as local path (if user has ADE20K downloaded)
            local_path = Path(fname)
            if not local_path.exists():
                local_path = Path("data") / fname
            if local_path.exists():
                src = local_path
            else:
                logger.debug("ADE20K image not found locally or remotely: %s", fname)
                continue

        if save_image(src, CATEGORY, idx, output_dir, dry_run):
            saved += 1
            idx += 1

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

    if "openimages" in args.sources and needed > 0:
        n = _download_openimages(output_dir, needed, args.dry_run)
        totals["openimages"] = n
        needed -= n
        print(f"  openimages   → {n} images saved")

    if "ade20k" in args.sources and needed > 0:
        n = _download_ade20k(output_dir, needed, args.dry_run)
        totals["ade20k"] = n
        needed -= n
        print(f"  ade20k       → {n} images saved")

    total = sum(totals.values())
    final_count = get_next_index(output_dir, CATEGORY) - 1
    print(f"\n[{CATEGORY}] {total} new images downloaded.  Directory total: {final_count}")


if __name__ == "__main__":
    main()
