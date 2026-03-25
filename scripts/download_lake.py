"""
download_lake.py -- Download Lake category images for the confounder HNM dataset.

Sources (tried in order until --max reached):
  1. atlantis : ATLANTIS water segmentation dataset.
                Downloaded as GitHub archive zip (no formal releases exist).
                Filters COCO-style annotations for classes:
                  lake, reservoir, wetland, marsh, river_delta
                (Exact class names from adk/dataset/cvat_labels_constructor.json.
                 Note: "pond" and "swamp" do NOT exist in ATLANTIS.)
  2. waternet : WaterNet / ADE20K water segmentation (Kaggle).
                Kaggle slug: gvclsu/water-segmentation-dataset
                Filters for lake/pond/reservoir annotations.

Kaggle sources are skipped gracefully when credentials are absent.

Output directory: {--output_dir}/Lake/
Naming:           Lake_0001.jpg, Lake_0002.jpg, …

Usage:
    python scripts/download_lake.py --dry_run --max 10
    python scripts/download_lake.py --max 400
    python scripts/download_lake.py --max 400 --sources waternet
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

CATEGORY = "Lake"
SOURCES = ["atlantis", "waternet"]

# Exact class names from ATLANTIS adk/dataset/cvat_labels_constructor.json.
# "pond" and "swamp" are NOT in ATLANTIS — the closest equivalents are
# "wetland" and "marsh". "river_delta" is also included as it depicts
# standing/slow water visually similar to flooded ground.
ATLANTIS_LABELS = {"lake", "reservoir", "wetland", "marsh", "river_delta"}

# WaterNet label keywords for lake filtering
_WATERNET_LAKE_KEYWORDS = {"lake", "pond", "reservoir"}

# Kaggle dataset slug
_KAGGLE_WATERNET = "gvclsu/water-segmentation-dataset"

# ATLANTIS GitHub archive (no formal releases published)
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

    ATLANTIS has no GitHub releases. Downloads the repository archive zip
    and extracts images. If the archive contains no images (they may need
    to be requested from authors separately), returns 0 with a warning.
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

    with tempfile.TemporaryDirectory(prefix="atlantis_lake_") as tmpdir:
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

        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = [
            p for p in tmp.rglob("*")
            if p.suffix.lower() in image_exts and "_seg" not in p.stem
        ]

        if not all_images:
            logger.warning(
                "ATLANTIS archive contains no image files. "
                "Dataset images are not in the GitHub repository. "
                "Request them from the authors: "
                "https://github.com/smhassanerfani/atlantis"
            )
            return 0

        # Filter by annotation labels where possible
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

        if not matched_image_paths:
            logger.info(
                "No annotation matches found — using all %d archive images.",
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
# Kaggle helpers
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
# WaterNet source
# ---------------------------------------------------------------------------


def _download_waternet(
    output_dir: Path, needed: int, dry_run: bool, keywords: set[str]
) -> int:
    """Fetch lake/pond images from WaterNet (ADE20K water subset) via Kaggle."""
    if not check_kaggle():
        logger.warning("Kaggle credentials absent — skipping WaterNet.")
        return 0

    saved = 0
    idx = get_next_index(output_dir, CATEGORY)

    with tempfile.TemporaryDirectory(prefix="waternet_lake_") as tmpdir:
        if not _kaggle_download(_KAGGLE_WATERNET, Path(tmpdir)):
            return 0

        tmp = Path(tmpdir)
        matched_paths: list[Path] = []

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

    if "waternet" in args.sources and needed > 0:
        n = _download_waternet(
            output_dir, needed, args.dry_run, _WATERNET_LAKE_KEYWORDS
        )
        totals["waternet"] = n
        needed -= n
        print(f"  waternet     → {n} images saved")

    total = sum(totals.values())
    final_count = get_next_index(output_dir, CATEGORY) - 1
    print(f"\n[{CATEGORY}] {total} new images downloaded.  Directory total: {final_count}")


if __name__ == "__main__":
    main()
