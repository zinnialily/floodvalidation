"""
build_splits.py -- Build the stratified 70/15/15 train/val/test binary split.

Reads from three source trees:
  data/FloodingDataset2/StreetFloodClasses/
      MajorFlood/, ModerateFlood/, MinorFlood/  → flood label
      NoFlood/, parks_walkways/                 → non_flood label

  data/FloodingDataset2/junk/
      River/, Swimmingpool/, Lake/, Fountain/   → non_flood label (water confounders)
      Cats/, Dogs/, Cars/, ...                  → SKIPPED (not benchmark categories)

  data/FloodingDataset2/extracted/junk/
      building_exterior/, building_interior/    → non_flood/building/ label

Outputs:
  data/FloodingDataset2/processed_data/binary/
      train/flood/<category>/       val/flood/<category>/       test/flood/<category>/
      train/non_flood/<category>/   val/non_flood/<category>/   test/non_flood/<category>/

  data/FloodingDataset2/processed_data/binary/{train,val,test}/metadata.csv
  data/FloodingDataset2/split_manifest.csv   ← full audit trail

The split is deterministic: seeded with SPLIT_SEED=42.
Files are copied (not symlinked) so the binary/ tree is self-contained.

Usage:
    python scripts/build_splits.py
    python scripts/build_splits.py --data_dir ./data/FloodingDataset2
    python scripts/build_splits.py --dry_run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# StreetFloodClasses subdirs → binary label (kept for reference)
FLOOD_CATS = {"MajorFlood", "ModerateFlood", "MinorFlood"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# StreetFloodClasses subdirs → (category name, source, label)
STREET_CATS = {
    "MajorFlood":     ("street_major",    "usf_flooding", "flood"),
    "ModerateFlood":  ("street_moderate", "usf_flooding", "flood"),
    "MinorFlood":     ("street_minor",    "usf_flooding", "flood"),
    "NoFlood":        ("street_clear",    "usf_flooding", "non_flood"),
    "parks_walkways": ("park_walkway",    "usf_flooding", "non_flood"),
}

# junk/ subdirs → (category name, source)  — unlisted dirs are SKIPPED (dropped)
# NOTE: this intentionally drops non-water junk like Cats/, Dogs/, Cars/ that were
# previously included. They are not part of the benchmark categories.
JUNK_CATS = {
    "River":        ("river",         "riwa"),
    "Swimmingpool": ("swimming_pool", "places365"),
    "Lake":         ("lake",          "atlantis"),
    "Fountain":     ("fountain",      "openimages"),
}

# extracted/junk/ subdirs → category name (all non_flood, source=usf_junk)
EXTRACTED_JUNK_CATS = {
    "building_exterior": "building",
    "building_interior": "building",
}


def _collect_images(directory: Path) -> list[Path]:
    """Return sorted list of image files directly in directory (non-recursive)."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _split_category(
    images: list[Path],
    rng: np.random.Generator,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Shuffle and split image list into (train, val, test) sublists."""
    n = len(images)
    indices = rng.permutation(n)

    n_val = max(1, round(n * VAL_FRAC))
    n_test = max(1, round(n * TEST_FRAC))

    val_idx = indices[:n_val]
    test_idx = indices[n_val : n_val + n_test]
    train_idx = indices[n_val + n_test :]

    return (
        [images[i] for i in train_idx],
        [images[i] for i in val_idx],
        [images[i] for i in test_idx],
    )


def _dest(binary_dir: Path, split: str, label: str, category: str) -> Path:
    """Return the destination directory for a given split/label/category."""
    return binary_dir / split / label / category


def build_splits(data_dir: Path, dry_run: bool = False) -> None:
    """Run the full split pipeline."""
    rng = np.random.default_rng(seed=SPLIT_SEED)

    binary_dir = data_dir / "processed_data" / "binary"

    if not dry_run:
        # Clear the entire binary/ tree so stale category folders don't accumulate
        if binary_dir.exists():
            shutil.rmtree(binary_dir)
        binary_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    # ── 1. StreetFloodClasses ────────────────────────────────────────────────
    street_dir = data_dir / "StreetFloodClasses"
    if not street_dir.exists():
        print(f"[WARN] StreetFloodClasses/ not found at {street_dir} — skipping flood images.")
    else:
        for cat_dir in sorted(street_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_name = cat_dir.name
            if cat_name not in STREET_CATS:
                print(f"  [SKIP] Unknown StreetFloodClasses category: {cat_name}")
                continue
            category, source, label = STREET_CATS[cat_name]

            images = _collect_images(cat_dir)
            if not images:
                print(f"  [SKIP] {cat_name}: no images found")
                continue

            train_imgs, val_imgs, test_imgs = _split_category(images, rng)

            for split_name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                dest_dir = _dest(binary_dir, split_name, label, category)
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                for src in imgs:
                    dst = dest_dir / src.name
                    if dry_run:
                        print(f"[dry_run] {src} → {dst}")
                    else:
                        shutil.copy2(src, dst)
                    records.append({
                        "split": split_name, "label": label,
                        "category": category, "source": source,
                        "filename": src.name,
                    })

            print(
                f"  {cat_name:<25} label={label:<9}  n={len(images):>4}  "
                f"train={len(train_imgs):>3}  val={len(val_imgs):>3}  test={len(test_imgs):>3}"
            )

    # ── 2. junk/ (water confounders) ─────────────────────────────────────────
    # Only known water categories are included. Non-water dirs (Cats/, Dogs/,
    # Cars/, etc.) are intentionally skipped — they are not part of the benchmark.
    junk_dir = data_dir / "junk"
    if not junk_dir.exists():
        print(f"[WARN] junk/ not found at {junk_dir} — skipping confounder images.")
    else:
        for cat_dir in sorted(junk_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_name = cat_dir.name
            if cat_name not in JUNK_CATS:
                print(f"  [SKIP] junk/{cat_name}: not a benchmark category")
                continue
            category, source = JUNK_CATS[cat_name]

            images = _collect_images(cat_dir)
            if not images:
                print(f"  [SKIP] junk/{cat_name}: no images found")
                continue

            train_imgs, val_imgs, test_imgs = _split_category(images, rng)

            for split_name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                dest_dir = _dest(binary_dir, split_name, "non_flood", category)
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                for src in imgs:
                    dst = dest_dir / src.name
                    if dry_run:
                        print(f"[dry_run] {src} → {dst}")
                    else:
                        shutil.copy2(src, dst)
                    records.append({
                        "split": split_name, "label": "non_flood",
                        "category": category, "source": source,
                        "filename": src.name,
                    })

            print(
                f"  junk/{cat_name:<20} label=non_flood  n={len(images):>4}  "
                f"train={len(train_imgs):>3}  val={len(val_imgs):>3}  test={len(test_imgs):>3}"
            )

    # ── 3. extracted/junk/ (building images) ────────────────────────────────
    extracted_junk_dir = data_dir / "extracted" / "junk"
    if not extracted_junk_dir.exists():
        print(f"[WARN] extracted/junk/ not found — skipping building images.")
    else:
        for cat_dir in sorted(extracted_junk_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            subdir_name = cat_dir.name
            if subdir_name not in EXTRACTED_JUNK_CATS:
                print(f"  [SKIP] extracted/junk/{subdir_name}: unknown")
                continue
            category = EXTRACTED_JUNK_CATS[subdir_name]

            images = _collect_images(cat_dir)
            if not images:
                print(f"  [SKIP] extracted/junk/{subdir_name}: no images")
                continue

            train_imgs, val_imgs, test_imgs = _split_category(images, rng)

            for split_name, imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
                dest_dir = _dest(binary_dir, split_name, "non_flood", category)
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                for src in imgs:
                    # Prefix with subdir name to avoid collisions between
                    # building_exterior and building_interior in the same dest folder
                    dst_name = f"{subdir_name}_{src.name}"
                    dst = dest_dir / dst_name
                    if dry_run:
                        print(f"[dry_run] {src} → {dst}")
                    else:
                        shutil.copy2(src, dst)
                    records.append({
                        "split": split_name, "label": "non_flood",
                        "category": category, "source": "usf_junk",
                        "filename": dst_name,
                    })

            print(
                f"  extracted/junk/{subdir_name:<15} label=non_flood  n={len(images):>4}  "
                f"train={len(train_imgs):>3}  val={len(val_imgs):>3}  test={len(test_imgs):>3}"
            )

    if not records:
        print("[ERROR] No images were processed. Check that your data_dir is correct.")
        return

    manifest = pd.DataFrame(records)
    manifest_path = data_dir / "split_manifest.csv"

    # Write per-split metadata.csv for HuggingFace ImageFolder format.
    # file_name is relative to the split folder (e.g. non_flood/river/River_0001.jpg).
    # Always compute and print metadata paths, even in dry_run, so paths can be verified.
    for split_name in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split_name].copy()
        if split_df.empty:
            continue
        split_df["file_name"] = (
            split_df["label"] + "/" +
            split_df["category"] + "/" +
            split_df["filename"]
        )
        meta_path = binary_dir / split_name / "metadata.csv"
        if dry_run:
            print(f"[dry_run] metadata.csv → {meta_path}  ({len(split_df)} rows)")
        else:
            split_df[["file_name", "category", "source"]].to_csv(meta_path, index=False)
            print(f"metadata.csv → {meta_path}  ({len(split_df)} rows)")

    if not dry_run:
        manifest.to_csv(manifest_path, index=False)
        print(f"\nManifest saved → {manifest_path}  ({len(manifest)} rows)")

    # ── Summary table ────────────────────────────────────────────────────────
    summary = (
        manifest
        .groupby(["label", "split"])
        .size()
        .unstack(fill_value=0)
    )
    if set(["train", "val", "test"]).issubset(summary.columns):
        summary = summary[["train", "val", "test"]]
    summary["total"] = summary.sum(axis=1)

    print("\nSplit summary:")
    print(summary.to_string())
    print(f"\nTotal images: {summary['total'].sum()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stratified 70/15/15 binary split for FloodingDataset2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="./data/FloodingDataset2",
        help="Dataset root containing StreetFloodClasses/ and junk/.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be copied without writing any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    print(f"[INFO] Dataset root: {data_dir}")
    print(f"[INFO] Dry run: {args.dry_run}\n")
    build_splits(data_dir, dry_run=args.dry_run)
    print("\n[DONE] build_splits.py completed.")


if __name__ == "__main__":
    main()
