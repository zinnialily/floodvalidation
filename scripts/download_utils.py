"""
download_utils.py -- Shared utilities for confounder dataset download scripts.

Provides:
  - get_next_index  : Continue numbering idempotently from existing files.
  - save_image      : Open any image source, convert to RGB JPEG, save with
                      standard {Category}_{NNNN}.jpg naming.
  - check_kaggle    : Verify Kaggle credentials are present.
  - check_fiftyone  : Check fiftyone availability.
  - build_parser    : Shared argparse CLI for all download scripts.

Dependencies (all already in project environment):
  - requests
  - Pillow
  - tqdm

Optional:
  - kaggle  (for Kaggle dataset sources)
  - fiftyone (for Open Images v7 sources)
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def get_next_index(output_dir: Union[str, Path], category: str) -> int:
    """Return the next 1-based integer index for a category directory.

    Counts all files matching ``{category}_*.jpg`` in *output_dir* and
    returns ``count + 1``.  If the directory does not exist, returns 1.

    Args:
        output_dir: Directory that may contain existing category images.
        category:   Category prefix exactly as used in filenames, e.g.
                    ``"River"``.

    Returns:
        Integer ≥ 1 — the index to assign to the *next* image saved.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 1
    return len(list(output_dir.glob(f"{category}_*.jpg"))) + 1


# ---------------------------------------------------------------------------
# Image save helper
# ---------------------------------------------------------------------------


def save_image(
    src: Union[str, Path, bytes],
    category: str,
    index: int,
    output_dir: Union[str, Path],
    dry_run: bool = False,
) -> bool:
    """Open an image from any source, convert to RGB JPEG, and persist.

    Accepted *src* types:
      - Local file path (str or Path) — opened directly with Pillow.
      - Raw bytes / bytearray — wrapped in ``io.BytesIO`` for Pillow.
      - HTTP/HTTPS URL string — fetched with ``requests`` then opened.

    On success the file is written as ``{output_dir}/{category}_{index:04d}.jpg``
    (JPEG, quality 95, RGB).  Parent directories are created automatically.

    Args:
        src:        Image source (path, bytes, or URL).
        category:   Category prefix for naming, e.g. ``"River"``.
        index:      1-based integer index for the output filename.
        output_dir: Directory to write the file into.
        dry_run:    If True, print the target path without writing anything.

    Returns:
        True on success; False on any failure (a warning is logged — no
        exception is raised so callers can continue to the next image).
    """
    from PIL import Image  # type: ignore

    output_dir = Path(output_dir)
    out_path = output_dir / f"{category}_{index:04d}.jpg"

    if dry_run:
        print(f"[dry_run] Would save → {out_path}")
        return True

    try:
        if isinstance(src, (bytes, bytearray)):
            img = Image.open(io.BytesIO(src))
        elif isinstance(src, (str, Path)) and not str(src).startswith("http"):
            img = Image.open(src)
        else:
            import requests  # type: ignore

            resp = requests.get(str(src), timeout=20)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))

        img = img.convert("RGB")
        output_dir.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "JPEG", quality=95)
        return True

    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not save %s: %s", out_path.name, exc)
        return False


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------


def check_kaggle() -> bool:
    """Return True if the ``kaggle`` package is importable and credentials exist.

    Credentials are expected at ``~/.kaggle/kaggle.json`` (the default
    location used by the official Kaggle CLI).
    """
    try:
        import kaggle  # noqa: F401  # type: ignore
    except ImportError:
        return False
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def check_fiftyone() -> bool:
    """Return True if the ``fiftyone`` package is importable."""
    try:
        import fiftyone  # noqa: F401  # type: ignore

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Shared CLI builder
# ---------------------------------------------------------------------------


def build_parser(category: str, sources: list[str]) -> argparse.ArgumentParser:
    """Build the standard argument parser shared by all download scripts.

    All four per-category scripts expose an identical CLI surface so they can
    be called with the same flags in batch runs.

    Args:
        category: Human-readable category name, e.g. ``"River"``.
        sources:  All source identifiers available for this category (used as
                  ``choices`` for ``--sources``).

    Returns:
        Configured :class:`argparse.ArgumentParser` — call ``.parse_args()``
        on the result to obtain the namespace.
    """
    parser = argparse.ArgumentParser(
        description=f"Download {category} images for the confounder HNM dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        default="./data/FloodingDataset2/junk",
        help="Root junk directory; category sub-folder is created automatically.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=400,
        help="Maximum total images to have in the category directory after this run.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sources,
        default=sources,
        metavar="SOURCE",
        help=(
            f"Which sources to query (space-separated). "
            f"Available for {category}: {sources}."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be downloaded/saved without writing any files.",
    )
    return parser
