"""
download_usf.py -- Download the USF FloodingDataset2 from Google Drive.

The University of South Florida FloodingDataset2 is hosted on a public
Google Drive folder. This script downloads it using gdown.

Prerequisites:
    pip install gdown

Folder contents (as released):
    StreetFloodClasses/
        MajorFlood/       -- street-level major flood images
        MinorFlood/       -- street-level minor flood images
        ModerateFlood/    -- street-level moderate flood images
        NoFlood/          -- no-flood street scenes
        parks_walkways/   -- parks and walkways (no flood)
    junk/
        Swimmingpool.zip  -- ~15 swimming pool images (supplemented by download_swimmingpool.py)
        Cars.zip, Dogs.zip, Cats.zip, ...  -- other distractor categories

After downloading, run notebooks/02_prepare_confounder_data.ipynb to unpack
the junk archives and run the 70/15/15 stratified split.

Usage:
    python scripts/download_usf.py
    python scripts/download_usf.py --output_dir ./data/FloodingDataset2
    python scripts/download_usf.py --dry_run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Public Google Drive folder ID for FloodingDataset2 (USF)
USF_FOLDER_ID = "1PXc9VTeQgV5WeNxa2NOC9kbRpnQtg20o"
USF_FOLDER_URL = f"https://drive.google.com/drive/folders/{USF_FOLDER_ID}?usp=sharing"

# Expected top-level subfolders — used to verify a successful download
EXPECTED_SUBDIRS = [
    "StreetFloodClasses",
    "junk",
]

# Flood category subfolders
FLOOD_CLASSES = ["MajorFlood", "MinorFlood", "ModerateFlood", "NoFlood", "parks_walkways"]


def check_gdown() -> bool:
    """Return True if gdown is importable."""
    try:
        import gdown  # noqa: F401  # type: ignore
        return True
    except ImportError:
        return False


def download_folder(output_dir: Path, dry_run: bool = False) -> bool:
    """Download FloodingDataset2 from Google Drive using gdown.

    Args:
        output_dir: Destination directory. The folder contents are written
                    directly into this path.
        dry_run:    Print what would happen without downloading.

    Returns:
        True if download succeeded or was already complete, False on failure.
    """
    if not check_gdown():
        logger.error(
            "gdown is not installed.  Run:  pip install gdown>=4.7"
        )
        return False

    import gdown  # type: ignore

    # Check if already downloaded
    if all((output_dir / sub).exists() for sub in EXPECTED_SUBDIRS):
        logger.info(
            "FloodingDataset2 already present at %s — skipping download.", output_dir
        )
        return True

    if dry_run:
        print(f"[dry_run] Would download Google Drive folder {USF_FOLDER_ID} → {output_dir}")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FloodingDataset2 from Google Drive …")
    logger.info("  Source: %s", USF_FOLDER_URL)
    logger.info("  Destination: %s", output_dir)
    logger.info(
        "  Note: gdown may require multiple requests for large folders. "
        "If it stops early, re-run the script — it will resume."
    )

    try:
        # gdown.download_folder downloads recursively into output_dir.
        # remaining_ok=True lets it succeed even if some files hit Drive
        # rate limits (they can be retried on re-run).
        gdown.download_folder(
            id=USF_FOLDER_ID,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
    except Exception as exc:
        logger.error("gdown download failed: %s", exc)
        logger.error(
            "If you see a permission or quota error, try downloading manually:\n"
            "  1. Visit: %s\n"
            "  2. Right-click → Download → Download entire folder\n"
            "  3. Extract to: %s",
            USF_FOLDER_URL,
            output_dir,
        )
        return False

    return True


def unpack_junk_archives(output_dir: Path, dry_run: bool = False) -> None:
    """Unzip all .zip archives in the junk/ subdirectory in-place.

    Each archive (e.g. Swimmingpool.zip) is extracted to a same-named
    subdirectory (e.g. junk/Swimmingpool/).

    Args:
        output_dir: FloodingDataset2 root directory.
        dry_run:    Print what would be extracted without doing it.
    """
    junk_dir = output_dir / "junk"
    if not junk_dir.exists():
        logger.warning("junk/ directory not found at %s — skipping unpack.", junk_dir)
        return

    zips = list(junk_dir.glob("*.zip"))
    if not zips:
        logger.info("No zip archives found in %s.", junk_dir)
        return

    for zip_path in sorted(zips):
        dest = junk_dir / zip_path.stem
        if dest.exists():
            logger.info("  Already unpacked: %s — skipping.", zip_path.name)
            continue
        if dry_run:
            print(f"[dry_run] Would unzip {zip_path.name} → {dest}/")
            continue
        logger.info("  Unpacking %s …", zip_path.name)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
            logger.info("    → %d files extracted", len(list(dest.rglob("*"))))
        except Exception as exc:
            logger.warning("  Failed to unpack %s: %s", zip_path.name, exc)


def verify_download(output_dir: Path) -> None:
    """Print a summary of what was downloaded."""
    print("\nFloodingDataset2 download summary:")
    print(f"  Root: {output_dir}")

    flood_dir = output_dir / "StreetFloodClasses"
    if flood_dir.exists():
        for cls in FLOOD_CLASSES:
            cls_path = flood_dir / cls
            n = len(list(cls_path.glob("*.*"))) if cls_path.exists() else 0
            status = f"{n} images" if n > 0 else "MISSING"
            print(f"  StreetFloodClasses/{cls:<20} {status}")
    else:
        print("  StreetFloodClasses/  MISSING — download may have failed")

    junk_dir = output_dir / "junk"
    if junk_dir.exists():
        for item in sorted(junk_dir.iterdir()):
            if item.is_dir():
                n = len(list(item.glob("*.*")))
                print(f"  junk/{item.name:<25} {n} images")
            elif item.suffix == ".zip":
                print(f"  junk/{item.name:<25} (not yet unpacked)")
    else:
        print("  junk/  MISSING")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download USF FloodingDataset2 from Google Drive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        default="./data/FloodingDataset2",
        help="Destination directory for the dataset.",
    )
    parser.add_argument(
        "--skip_unpack",
        action="store_true",
        help="Do not unzip junk/*.zip archives after downloading.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without downloading anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    success = download_folder(output_dir, args.dry_run)

    if success and not args.skip_unpack:
        unpack_junk_archives(output_dir, args.dry_run)

    if not args.dry_run:
        verify_download(output_dir)


if __name__ == "__main__":
    main()
