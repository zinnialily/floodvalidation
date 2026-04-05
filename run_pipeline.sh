#!/usr/bin/env bash
# run_pipeline.sh — Full flood-detection training pipeline (all 5 seeds, both archs)
#
# Prerequisites:
#   Run from the project root:
#       chmod +x run_pipeline.sh && ./run_pipeline.sh
#
# The dataset is downloaded automatically from HuggingFace Hub on first run:
#   zinnia82/flood-binary-hnm-benchmark
# Set HF_TOKEN env var (or add to api.md) if the repo requires authentication.
#
# Total estimated runtime on a T4 GPU: ~20-30 hours (all seeds + all conditions)
# To run just seed 42: change SEEDS=(42) below

set -euo pipefail

SEEDS=(42)
DATA_DIR="./data/FloodingDataset2"
MODELS_DIR="./models"
RESULTS_DIR="./results"

# ── Load HF_TOKEN from api.md if not already set ──────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -f "api.md" ]; then
  HF_TOKEN=$(grep -m1 '^HF_TOKEN=' api.md | cut -d'=' -f2-)
  export HF_TOKEN
  echo "Loaded HF_TOKEN from api.md"
fi

mkdir -p "$MODELS_DIR" "$RESULTS_DIR/predictions" "$RESULTS_DIR/tables" \
         "$RESULTS_DIR/figures" "$RESULTS_DIR/logs"

# ── Install dependencies ──────────────────────────────────────────────────────
echo "Installing dependencies..."
pip install -q \
  "tensorflow>=2.16,<2.18" numpy pandas scikit-learn \
  matplotlib seaborn Pillow scipy tf-keras-vis \
  requests tqdm gdown umap-learn huggingface_hub datasets

# ── Download dataset from HuggingFace Hub (skipped if already present) ────────
BINARY_DIR="$DATA_DIR/processed_data/binary"
if [ ! -d "$BINARY_DIR/train" ]; then
  echo "Downloading dataset from HuggingFace Hub (zinnia82/flood-binary-hnm-benchmark)..."
  python - <<'PYEOF'
import os
import pathlib
import pandas as pd
from datasets import load_dataset
from PIL import Image

HF_TOKEN    = os.environ.get("HF_TOKEN")
DATASET_ID  = "zinnia82/flood-binary-hnm-benchmark"
BINARY_DIR  = pathlib.Path("./data/FloodingDataset2/processed_data/binary")

print(f"Loading {DATASET_ID} via datasets library...")
ds = load_dataset(DATASET_ID, token=HF_TOKEN)

print(f"Splits found: {list(ds.keys())}")

for split_name, split_ds in ds.items():
    label_feature = split_ds.features["label"]
    has_category  = "category" in split_ds.features
    records = []

    for row in split_ds:
        label    = label_feature.int2str(row["label"])   # "flood" / "non_flood"
        category = (row.get("category") or "unknown") if has_category else "unknown"
        source   = (row.get("source")   or "")        if "source" in split_ds.features else ""

        dest_dir = BINARY_DIR / split_name / label / category
        dest_dir.mkdir(parents=True, exist_ok=True)

        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        # Use a zero-padded index as filename to guarantee uniqueness
        idx      = len(list(dest_dir.iterdir()))
        filename = f"{category}_{idx:06d}.jpg"
        img.save(dest_dir / filename, format="JPEG", quality=95)

        records.append({
            "file_name": f"{label}/{category}/{filename}",
            "category":  category,
            "source":    source,
        })

    # Write metadata.csv in HuggingFace ImageFolder format
    meta_path = BINARY_DIR / split_name / "metadata.csv"
    pd.DataFrame(records).to_csv(meta_path, index=False)
    print(f"  [{split_name}] {len(records)} images → {BINARY_DIR / split_name}  (metadata.csv written)")

print(f"Dataset ready at {BINARY_DIR}")
PYEOF
fi

if [ ! -d "$BINARY_DIR/train" ]; then
  echo "ERROR: Dataset not found at $BINARY_DIR/train after download attempt."
  exit 1
fi

echo ""
echo "============================================================"
echo "  Starting pipeline: ${#SEEDS[@]} seeds × 2 archs × all conditions"
echo "============================================================"

# ── Per-seed loop ─────────────────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do

  echo ""
  echo "============================================================"
  echo "  SEED $SEED"
  echo "============================================================"

  # ── STEP 1: Train baselines ─────────────────────────────────────────────────

  echo "[Seed $SEED] Training EfficientNet baseline (BCE)..."
  python scripts/train_baseline.py \
    --arch efficientnet --seed "$SEED" --loss binary_crossentropy \
    --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" --results_dir "$RESULTS_DIR"
  ENET_BCE_P1=$(ls -t "$MODELS_DIR"/efficientnet_bce_phase1_*.keras | head -1)
  ENET_BCE_P2=$(ls -t "$MODELS_DIR"/efficientnet_bce_phase2_*.keras | head -1)
  echo "  Phase 1 → $ENET_BCE_P1"
  echo "  Phase 2 → $ENET_BCE_P2"

  echo "[Seed $SEED] Training EfficientNet baseline (Focal)..."
  python scripts/train_baseline.py \
    --arch efficientnet --seed "$SEED" --loss focal \
    --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" --results_dir "$RESULTS_DIR"
  ENET_FOCAL_P1=$(ls -t "$MODELS_DIR"/efficientnet_focal_phase1_*.keras | head -1)
  ENET_FOCAL_P2=$(ls -t "$MODELS_DIR"/efficientnet_focal_phase2_*.keras | head -1)
  echo "  Phase 1 → $ENET_FOCAL_P1"
  echo "  Phase 2 → $ENET_FOCAL_P2"

  echo "[Seed $SEED] Training ResNet50 baseline (BCE)..."
  python scripts/train_baseline.py \
    --arch resnet50 --seed "$SEED" --loss binary_crossentropy \
    --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" --results_dir "$RESULTS_DIR"
  RNET_BCE_P1=$(ls -t "$MODELS_DIR"/resnet50_bce_phase1_*.keras | head -1)
  RNET_BCE_P2=$(ls -t "$MODELS_DIR"/resnet50_bce_phase2_*.keras | head -1)
  echo "  Phase 1 → $RNET_BCE_P1"
  echo "  Phase 2 → $RNET_BCE_P2"

  echo "[Seed $SEED] Training ResNet50 baseline (Focal)..."
  python scripts/train_baseline.py \
    --arch resnet50 --seed "$SEED" --loss focal \
    --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" --results_dir "$RESULTS_DIR"
  RNET_FOCAL_P1=$(ls -t "$MODELS_DIR"/resnet50_focal_phase1_*.keras | head -1)
  RNET_FOCAL_P2=$(ls -t "$MODELS_DIR"/resnet50_focal_phase2_*.keras | head -1)
  echo "  Phase 1 → $RNET_FOCAL_P1"
  echo "  Phase 2 → $RNET_FOCAL_P2"

  # ── STEP 2: Confounder analysis from Phase 1 checkpoints ───────────────────
  # Mining is intentionally run from Phase 1 (not Phase 2) checkpoints.
  # By end of Phase 2, the model has memorised the training non-flood categories
  # and FP rates collapse to ~0, leaving no candidates to mine.
  # Phase 1 (head + top layers trained, backbone mostly frozen) has higher FP
  # rates on visually confusing categories, producing richer mining candidates.
  # HNM then retrains from the fully converged Phase 2 checkpoint.

  echo "[Seed $SEED] Analyzing confounders from Phase 1 (EfficientNet)..."
  python scripts/analyze_confounders.py \
    --model_path "$ENET_BCE_P1" --arch efficientnet \
    --data_dir "$DATA_DIR" --output_dir "$RESULTS_DIR" --fp_threshold 0.05 --mine_all

  echo "[Seed $SEED] Analyzing confounders from Phase 1 (ResNet50)..."
  python scripts/analyze_confounders.py \
    --model_path "$RNET_BCE_P1" --arch resnet50 \
    --data_dir "$DATA_DIR" --output_dir "$RESULTS_DIR" --fp_threshold 0.05 --mine_all

  # ── STEP 3: Hard Negative Mining ────────────────────────────────────────────
  # Note: analyze_confounders wrote results/mining_candidates_efficientnet.txt
  # train_hnm.py reads it automatically from --results_dir (defaults to ./results)

  echo "[Seed $SEED] HNM — EfficientNet (BCE)..."
  python scripts/train_hnm.py \
    --arch efficientnet --model_path "$ENET_BCE_P2" \
    --mining_model_path "$ENET_BCE_P1" --loss binary_crossentropy \
    --seed "$SEED" --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR"
  ENET_HNM_BCE=$(ls -t "$MODELS_DIR"/efficientnet_hnm_percentile_*.keras | head -1)

  echo "[Seed $SEED] HNM — EfficientNet (Focal)..."
  python scripts/train_hnm.py \
    --arch efficientnet --model_path "$ENET_FOCAL_P2" \
    --mining_model_path "$ENET_FOCAL_P1" --loss focal \
    --seed "$SEED" --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR"
  ENET_HNM_FOCAL=$(ls -t "$MODELS_DIR"/efficientnet_hnm_percentile_*.keras | head -1)

  echo "[Seed $SEED] HNM — EfficientNet (no injection control)..."
  python scripts/train_hnm.py \
    --arch efficientnet --model_path "$ENET_BCE_P2" \
    --mining_model_path "$ENET_BCE_P1" --no_injection \
    --seed "$SEED" --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR"
  ENET_NO_INJ=$(ls -t "$MODELS_DIR"/efficientnet_extended_baseline_*.keras | head -1)

  echo "[Seed $SEED] HNM — EfficientNet (random injection control)..."
  python scripts/train_hnm.py \
    --arch efficientnet --model_path "$ENET_BCE_P2" \
    --mining_model_path "$ENET_BCE_P1" --random_injection \
    --seed "$SEED" --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR"
  ENET_RAND_INJ=$(ls -t "$MODELS_DIR"/efficientnet_random_inject_*.keras | head -1)

  echo "[Seed $SEED] HNM — ResNet50 (BCE)..."
  python scripts/train_hnm.py \
    --arch resnet50 --model_path "$RNET_BCE_P2" \
    --mining_model_path "$RNET_BCE_P1" --loss binary_crossentropy \
    --seed "$SEED" --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR"
  RNET_HNM_BCE=$(ls -t "$MODELS_DIR"/resnet50_hnm_percentile_*.keras | head -1)

  echo "[Seed $SEED] HNM — ResNet50 (Focal)..."
  python scripts/train_hnm.py \
    --arch resnet50 --model_path "$RNET_FOCAL_P2" \
    --mining_model_path "$RNET_FOCAL_P1" --loss focal \
    --seed "$SEED" --data_dir "$DATA_DIR" --output_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR"
  RNET_HNM_FOCAL=$(ls -t "$MODELS_DIR"/resnet50_hnm_percentile_*.keras | head -1)

  # ── STEP 4: Evaluate all models for this seed ───────────────────────────────

  echo "[Seed $SEED] Evaluating all models..."
  for MODEL_PATH in \
    "$ENET_BCE_P2" "$ENET_FOCAL_P2" \
    "$RNET_BCE_P2" "$RNET_FOCAL_P2" \
    "$ENET_HNM_BCE" "$ENET_HNM_FOCAL" \
    "$ENET_NO_INJ" "$ENET_RAND_INJ" \
    "$RNET_HNM_BCE" "$RNET_HNM_FOCAL"
  do
    # Determine arch from filename
    if [[ "$MODEL_PATH" == *"resnet50"* ]]; then
      ARCH="resnet50"
    else
      ARCH="efficientnet"
    fi
    echo "  Evaluating $(basename $MODEL_PATH)..."
    python scripts/evaluate.py \
      --model_path "$MODEL_PATH" --arch "$ARCH" \
      --data_dir "$DATA_DIR" --output_dir "$RESULTS_DIR"
  done

  echo "  Seed $SEED complete."
done

# ── STEP 5: GradCAM on best model (seed 42, EfficientNet HNM Focal) ───────────

echo ""
echo "Running GradCAM++ visualizations..."
BEST_ENET_HNM=$(ls -t "$MODELS_DIR"/efficientnet_hnm_percentile_*.keras | head -1)
BEST_PREDS=$(ls -t "$RESULTS_DIR"/predictions/efficientnet_hnm_percentile_*_predictions.csv | head -1)

if [ -f "$BEST_ENET_HNM" ] && [ -f "$BEST_PREDS" ]; then
  python scripts/grad_cam.py \
    --model_path "$BEST_ENET_HNM" --arch efficientnet \
    --predictions_csv "$BEST_PREDS" \
    --data_dir "$DATA_DIR" --output_dir "$RESULTS_DIR"
fi

# ── STEP 6: Aggregate seeds ───────────────────────────────────────────────────

echo ""
echo "Aggregating seeds — EfficientNet..."
python scripts/aggregate_seeds.py \
  --baseline_bce   "$RESULTS_DIR/predictions/efficientnet_bce_phase2_*_predictions.csv" \
  --baseline_focal "$RESULTS_DIR/predictions/efficientnet_focal_phase2_*_predictions.csv" \
  --hnm_bce        "$RESULTS_DIR/predictions/efficientnet_hnm_percentile_*_predictions.csv" \
  --no_injection   "$RESULTS_DIR/predictions/efficientnet_extended_baseline_*_predictions.csv" \
  --random_inject  "$RESULTS_DIR/predictions/efficientnet_random_inject_*_predictions.csv" \
  --arch           efficientnet \
  --output         "$RESULTS_DIR/tables/seed_aggregation_efficientnet.csv"

echo ""
echo "Aggregating seeds — ResNet50..."
python scripts/aggregate_seeds.py \
  --baseline_bce   "$RESULTS_DIR/predictions/resnet50_bce_phase2_*_predictions.csv" \
  --baseline_focal "$RESULTS_DIR/predictions/resnet50_focal_phase2_*_predictions.csv" \
  --hnm_bce        "$RESULTS_DIR/predictions/resnet50_hnm_percentile_*_predictions.csv" \
  --arch           resnet50 \
  --output         "$RESULTS_DIR/tables/seed_aggregation_resnet50.csv"

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Results in: $RESULTS_DIR/"
echo "    tables/   — seed_aggregation_*.csv (main results table)"
echo "    figures/  — PR curves, confusion matrices, GradCAM"
echo "    predictions/ — per-image CSVs for all models/seeds"
echo "============================================================"
