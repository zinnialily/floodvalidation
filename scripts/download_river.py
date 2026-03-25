"""
download_river.py -- Download River category images for the confounder HNM dataset.

Sources (tried in order until --max reached):
  1. atlantis    : ATLANTIS water segmentation dataset.
                   Downloaded as GitHub archive zip (no formal releases exist).
                   Filters COCO-style annotations for classes:
                     river, canal, rapids
                   (Exact class names from adk/dataset/cvat_labels_constructor.json)
  2. riwa        : RIWA river segmentation dataset (Kaggle).
                   Kaggle slug: franzwagner/river-water-segmentation-dataset
  3. waternet    : WaterNet / ADE20K water segmentation (Kaggle).
                   Kaggle slug: gvclsu/water-segmentation-dataset
                   Filters for river/canal annotations.
  4. lufi        : LuFI-RiverSnap river photo dataset (Kaggle).
                   Kaggle slug: arminmoghimi/lufi-riversnap

Kaggle sources are skipped gracefully when credentials are absent.

Output directory: {--output_dir}/River/
Naming:           River_0001.jpg, River_0002.jpg, …

Usage:
    python scripts/download_river.py --dry_run --max 10
    python scripts/download_river.py --max 400
    python scripts/download_river.py --max 400 --sources riwa lufi
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm  # type: ignore

sys.path.insert(0, os.path.dirname(__file__))
from download_utils import (  # noqa: E402
    build_parser,
    check_kaggle,
    get_next_index,
    save_image,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

CATEGORY = "River"
SOURCES = ["atlantis", "riwa", "waternet", "lufi"]

# Exact class names from ATLANTIS adk/dataset/cvat_labels_constructor.json
# Note: "stream" does not exist in ATLANTIS. "rapids" is the flowing-water class.
ATLANTIS_LABELS = {"river", "canal", "rapids"}

# Kaggle dataset slugs
_KAGGLE_RIWA = "franzwagner/river-water-segmentation-dataset"
_KAGGLE_WATERNET = "gvclsu/water-segmentation-dataset"
_KAGGLE_LUFI = "arminmoghimi/lufi-riversnap"

# WaterNet label keywords for river filtering
_WATERNET_RIVER_KEYWORDS = {"river", "canal"}

# ATLANTIS GitHub archive (no releases published — download repo zip)
_ATLANTIS_ZIP_URL = (
    "https://github.com/smhassanerfani/atlantis/archive/refs/heads/main.zip"
)

# ---------------------------------------------------------------------------
# ATLANTIS source
# ---------------------------------------------------------------------------


def _download_atlantis(
    output_dir: Path, needed: int, dry_run: bool, accept_labels: set[str]
) -> int:
    """Download images from ATLANTIS matching the given class labels.

    ATLANTIS has no GitHub releases. This function downloads the repository
    archive zip (code + annotations) and extracts any images it contains.

    The ATLANTIS dataset uses pixel-wise segmentation masks. Image files are
    .jpg and masks are .png. We collect all images whose annotation JSON
    lists at least one of the target labels.

    If the repo archive contains no actual images (they may require a
    separate request to the authors), a warning is logged and 0 is returned.
    In that case, rely on the Kaggle sources instead.
    """
    import requests  # type: ignore

    saved = 0
    idx = get_next_index(output_dir, CATEGORY)

    if dry_run:
        print(
            f"[dry_run] Would download ATLANTIS repo zip from GitHub and "
            f"filter for: {sorted(accept_labels)}"
        )
        return 0

    logger.info("Downloading ATLANTIS repository archive …")

    with tempfile.TemporaryDirectory(prefix="atlantis_") as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "atlantis_main.zip"

        try:
            with requests.get(_ATLANTIS_ZIP_URL, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with open(zip_path, "wb") as fh, tqdm(
                    total=total, unit="B", unit_scale=True, desc="ATLANTIS repo zip"
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
                        pbar.update(len(chunk))
        except Exception as exc:
            logger.warning("ATLANTIS download failed: %s", exc)
            return 0

        shutil.unpack_archive(str(zip_path), str(tmp))

        # Find all images in the extracted archive
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = [
            p for p in tmp.rglob("*")
            if p.suffix.lower() in image_exts and "_seg" not in p.stem
        ]

        if not all_images:
            logger.warning(
                "ATLANTIS archive contains no image files. "
                "The dataset images are not distributed via the GitHub repository. "
                "To use ATLANTIS, download images directly from the authors: "
                "https://github.com/smhassanerfani/atlantis"
            )
            return 0

        # If COCO-style JSON annotations exist, filter by target labels.
        # Otherwise fall back to using all images (RIWA/LuFI are all-river anyway).
        ann_files = list(tmp.rglob("*.json"))
        matched_image_paths: list[Path] = []

        for ann_file in ann_files:
            try:
                data = json.loads(ann_file.read_text())
            except Exception:
                continue
            if not isinstance(data, dict) or "annotations" not in data:
                continue

            cat_id_to_name: dict[int, str] = {
                c["id"]: c["name"].lower()
                for c in data.get("categories", [])
            }
            img_id_to_path: dict[int, Path] = {}
            for img_info in data.get("images", []):
                candidate = (ann_file.parent / img_info["file_name"]).resolve()
                if not candidate.exists():
                    for img_dir in tmp.rglob("images"):
                        alt = (img_dir / img_info["file_name"]).resolve()
                        if alt.exists():
                            candidate = alt
                            break
                img_id_to_path[img_info["id"]] = candidate

            matched_ids: set[int] = set()
            for ann in data.get("annotations", []):
                name = cat_id_to_name.get(ann.get("category_id", -1), "")
                if name in accept_labels:
                    matched_ids.add(ann["image_id"])

            for img_id in matched_ids:
                p = img_id_to_path.get(img_id)
                if p and p.exists():
                    matched_image_paths.append(p)

        # If no annotation-filtered results, use all images from the archive
        if not matched_image_paths:
            logger.info(
                "No annotation JSONs found or no label matches — using all %d images.",
                len(all_images),
            )
            matched_image_paths = all_images

        logger.info("ATLANTIS: %d matched images for %s", len(matched_image_paths), CATEGORY)

        for img_path in tqdm(matched_image_paths[:needed], desc=f"ATLANTIS {CATEGORY}"):
            if save_image(img_path, CATEGORY, idx, output_dir, dry_run):
                saved += 1
                idx += 1
            if saved >= needed:
                break

    return saved


# ---------------------------------------------------------------------------
# Shared Kaggle helpers
# ---------------------------------------------------------------------------


def _kaggle_download(slug: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset into *dest*. Returns True on success."""
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", slug, "--unzip", "-p", str(dest)],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "kaggle download failed for %s: %s", slug, exc.stderr.decode()[:200]
        )
        return False
    except FileNotFoundError:
        logger.warning(
            "kaggle CLI not found.  Install with:  pip install kaggle  "
            "and place credentials at ~/.kaggle/kaggle.json"
        )
        return False


def _images_from_dir(root: Path) -> list[Path]:
    """Recursively collect image files (excluding segmentation masks)."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [
        p for p in root.rglob("*")
        if p.suffix.lower() in exts and "_seg" not in p.stem
    ]


# ---------------------------------------------------------------------------
# RIWA source
# ---------------------------------------------------------------------------


def _download_riwa(output_dir: Path, needed: int, dry_run: bool) -> int:
    """Fetch river images from the RIWA dataset (Kaggle).

    RIWA is a binary river/water segmentation dataset — all images show
    rivers, so no label filtering is needed.
    """
    if not check_kaggle():
        logger.warning("Kaggle credentials absent — skipping RIWA.")
        return 0

    saved = 0
    idx = get_next_index(output_dir, CATEGORY)

    with tempfile.TemporaryDirectory(prefix="riwa_") as tmpdir:
        if not _kaggle_download(_KAGGLE_RIWA, Path(tmpdir)):
            return 0

        images = _images_from_dir(Path(tmpdir))
        logger.info("RIWA: %d images found", len(images))

        for img_path in tqdm(images[:needed], desc="RIWA River"):
            if save_image(img_path, CATEGORY, idx, output_dir, dry_run):
                saved += 1
                idx += 1
            if saved >= needed:
                break

    return saved


# ---------------------------------------------------------------------------
# WaterNet source
# ---------------------------------------------------------------------------


def _download_waternet(
    output_dir: Path, needed: int, dry_run: bool, keywords: set[str]
) -> int:
    """Fetch river/canal images from WaterNet (ADE20K water subset) via Kaggle.

    WaterNet is a water segmentation dataset with multiple subsets. We filter
    for images whose annotation categories or directory paths contain one of
    the target keywords.
    """
    if not check_kaggle():
        logger.warning("Kaggle credentials absent — skipping WaterNet.")
        return 0

    saved = 0
    idx = get_next_index(output_dir, CATEGORY)

    with tempfile.TemporaryDirectory(prefix="waternet_") as tmpdir:
        if not _kaggle_download(_KAGGLE_WATERNET, Path(tmpdir)):
            return 0

        tmp = Path(tmpdir)
        matched_paths: list[Path] = []

        # Try COCO-style JSON annotations first
        for ann_file in tmp.rglob("*.json"):
            try:
                data = json.loads(ann_file.read_text())
            except Exception:
                continue
            if not isinstance(data, dict) or "categories" not in data:
                continue

            accept_ids = {
                c["id"]
                for c in data.get("categories", [])
                if any(kw in c["name"].lower() for kw in keywords)
            }
            if not accept_ids:
                continue

            img_map = {
                img["id"]: ann_file.parent / img["file_name"]
                for img in data.get("images", [])
            }
            for ann in data.get("annotations", []):
                if ann.get("category_id") in accept_ids:
                    p = img_map.get(ann["image_id"])
                    if p and p.exists():
                        matched_paths.append(p)

        # Fallback: directory/filename pattern matching
        if not matched_paths:
            for kw in keywords:
                matched_paths.extend(
                    p for p in _images_from_dir(tmp) if kw in str(p).lower()
                )

        logger.info("WaterNet %s: %d matched images", CATEGORY, len(matched_paths))

        for img_path in tqdm(matched_paths[:needed], desc=f"WaterNet {CATEGORY}"):
            if save_image(img_path, CATEGORY, idx, output_dir, dry_run):
                saved += 1
                idx += 1
            if saved >= needed:
                break

    return saved


# ---------------------------------------------------------------------------
# LuFI-RiverSnap source
# ---------------------------------------------------------------------------


def _download_lufi(output_dir: Path, needed: int, dry_run: bool) -> int:
    """Fetch river images from LuFI-RiverSnap (Kaggle).

    LuFI-RiverSnap is a dedicated river snapshot dataset — all images show
    rivers from multiple lighting conditions and viewpoints.
    """
    if not check_kaggle():
        logger.warning("Kaggle credentials absent — skipping LuFI-RiverSnap.")
        return 0

    saved = 0
    idx = get_next_index(output_dir, CATEGORY)

    with tempfile.TemporaryDirectory(prefix="lufi_") as tmpdir:
        if not _kaggle_download(_KAGGLE_LUFI, Path(tmpdir)):
            return 0

        images = _images_from_dir(Path(tmpdir))
        logger.info("LuFI-RiverSnap: %d images found", len(images))

        for img_path in tqdm(images[:needed], desc="LuFI RiverSnap"):
            if save_image(img_path, CATEGORY, idx, output_dir, dry_run):
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

    if "atlantis" in args.sources and needed > 0:
        n = _download_atlantis(output_dir, needed, args.dry_run, ATLANTIS_LABELS)
        totals["atlantis"] = n
        needed -= n
        print(f"  atlantis     → {n} images saved")

    if "riwa" in args.sources and needed > 0:
        n = _download_riwa(output_dir, needed, args.dry_run)
        totals["riwa"] = n
        needed -= n
        print(f"  riwa         → {n} images saved")

    if "waternet" in args.sources and needed > 0:
        n = _download_waternet(
            output_dir, needed, args.dry_run, _WATERNET_RIVER_KEYWORDS
        )
        totals["waternet"] = n
        needed -= n
        print(f"  waternet     → {n} images saved")

    if "lufi" in args.sources and needed > 0:
        n = _download_lufi(output_dir, needed, args.dry_run)
        totals["lufi"] = n
        needed -= n
        print(f"  lufi         → {n} images saved")

    total = sum(totals.values())
    final_count = get_next_index(output_dir, CATEGORY) - 1
    print(f"\n[{CATEGORY}] {total} new images downloaded.  Directory total: {final_count}")


if __name__ == "__main__":
    main()
