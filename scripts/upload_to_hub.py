"""
upload_to_hub.py -- Upload processed_data/binary/ to HuggingFace Hub.

Uploads the ImageFolder-format tree built by build_splits.py to:
    zinnia82/flood-binary-hnm-benchmark  (public)

The binary/ tree contains train/, val/, and test/ splits, each with
flood/<category>/ and non_flood/<category>/ subdirectories plus a
metadata.csv that adds 'category' and 'source' columns.

HuggingFace auto-detects the ImageFolder format from this layout.
Run build_splits.py before this script.

Usage
-----
    python scripts/upload_to_hub.py
    python scripts/upload_to_hub.py --data_dir ./data/FloodingDataset2
    python scripts/upload_to_hub.py --dry_run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

HF_REPO_ID = "zinnia82/flood-binary-hnm-benchmark"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _load_credentials() -> None:
    """Load HF_TOKEN from api.md (dotenv format) into the environment."""
    api_file = Path(__file__).parent.parent / "api.md"
    if not api_file.exists():
        logger.warning("api.md not found — assuming HF_TOKEN is already in env.")
        return
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(api_file)
        logger.info("Credentials loaded from api.md")
    except ImportError:
        logger.warning(
            "python-dotenv not installed. Run: pip install python-dotenv\n"
            "Attempting to parse api.md manually..."
        )
        for line in api_file.read_text().splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload(binary_dir: Path, dry_run: bool = False) -> None:
    """Push processed_data/binary/ to HuggingFace Hub as an ImageFolder dataset."""
    if dry_run:
        logger.info("[dry_run] Would upload %s to %s", binary_dir, HF_REPO_ID)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error(
            "HF_TOKEN not found in environment.\n"
            "Make sure api.md contains:  HF_TOKEN=hf_..."
        )
        sys.exit(1)

    api = HfApi()
    api.create_repo(HF_REPO_ID, repo_type="dataset", exist_ok=True, token=token)

    logger.info("Uploading %s to %s ...", binary_dir, HF_REPO_ID)
    api.upload_folder(
        folder_path=str(binary_dir),
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=token,
        commit_message="Upload ImageFolder dataset with category subfolders",
    )
    logger.info("Done → https://huggingface.co/datasets/%s", HF_REPO_ID)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload processed_data/binary/ to HuggingFace Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="./data/FloodingDataset2",
        help="Local dataset root directory (must contain processed_data/binary/).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be uploaded without actually uploading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    binary_dir = Path(args.data_dir) / "processed_data" / "binary"

    _load_credentials()

    if not binary_dir.exists():
        logger.error(
            "processed_data/binary/ not found at %s.\n"
            "Run build_splits.py first.", binary_dir
        )
        sys.exit(1)

    logger.info("Uploading from %s ...", binary_dir)
    upload(binary_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
