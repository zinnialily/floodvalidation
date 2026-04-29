"""
train_baseline.py -- Two-phase baseline training for EfficientNetB0 or ResNet50.

Replaces the old train_efficientnet.py and train_resnet50.py with a single
unified script that supports both architectures via --arch.

Usage:
    python scripts/train_baseline.py --arch efficientnet
    python scripts/train_baseline.py --arch resnet50 --seed 123

Compute: Colab T4 GPU (or any CUDA-capable GPU).
Estimated runtime: ~1-2 hours per architecture on T4.
Dependencies: tensorflow>=2.16, scikit-learn, numpy.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------------------------
# Path setup: allow running as a top-level script (python scripts/train_baseline.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    PREPROCESS_FN,
    best_epoch_metrics,
    build_callbacks,
    build_generators,
    build_model,
    compute_class_weights,
    freeze_for_phase2,
    get_loss,
    parse_phase_boundary,
    print_runtime_env,
    set_all_seeds,
    verify_preprocessing,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHASE1_LR: float = 1e-4
PHASE2_LR: float = 1e-5
PHASE1_MAX_EPOCHS: int = 15
PHASE2_MAX_EPOCHS: int = 20
EARLY_STOPPING_PATIENCE: int = 7
LR_PATIENCE_PHASE1: int = 3
LR_PATIENCE_PHASE2: int = 4


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a two-phase baseline flood detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=["efficientnet", "resnet50"],
        help="Backbone architecture to train.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data/FloodingDataset2",
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="./models",
        help="Directory where model checkpoints are saved.",
    )
    parser.add_argument(
        "--phase_boundary",
        default="30,50",
        help=(
            "Comma-separated pair n_trainable,n_frozen defining Phase 1 and "
            "Phase 2 freeze boundaries (e.g. '30,50')."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--loss",
        default="binary_crossentropy",
        choices=["binary_crossentropy", "focal"],
        help="Loss function: 'binary_crossentropy' (default) or 'focal'.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma for focal loss (ignored for BCE).",
    )
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="Directory for logs and evaluation outputs (separate from model checkpoints).",
    )
    parser.add_argument(
        "--resume_phase2",
        default=None,
        metavar="CKPT_PATH",
        help="Path to a Phase 1 checkpoint to load and skip directly to Phase 2.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse args, build data, train Phase 1 + Phase 2."""
    args = parse_args()
    n_trainable, n_frozen = parse_phase_boundary(args.phase_boundary)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_label = "focal" if args.loss == "focal" else "bce"

    # -- Environment --------------------------------------------------------
    print_runtime_env()

    # -- Reproducibility ----------------------------------------------------
    set_all_seeds(args.seed)
    print(f"Random seed: {args.seed}")

    # -- Directory setup ----------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # -- Data paths ---------------------------------------------------------
    binary_root = os.path.join(
        args.data_dir, "processed_data", "binary"
    )
    train_dir = os.path.join(binary_root, "train")
    val_dir = os.path.join(binary_root, "val")

    for split_dir in (train_dir, val_dir):
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected dataset split directory not found: {split_dir}\n"
                "Ensure --data_dir points to the dataset root and that "
                "Step 2 (stratified splitting) has been run."
            )

    # -- Generators ---------------------------------------------------------
    print("\nBuilding data generators ...")
    train_gen, val_gen = build_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        arch=args.arch,
        seed=args.seed,
    )
    print(
        f"Train samples: {train_gen.samples} | "
        f"Val samples: {val_gen.samples}"
    )
    print(f"Class indices: {train_gen.class_indices}")

    # -- Preprocessing verification (halts on failure) ----------------------
    print("\nVerifying preprocessing ...")
    verify_preprocessing(train_gen, args.arch)

    # -- Class weights ------------------------------------------------------
    class_weight_dict = compute_class_weights(train_gen)

    # -- Model --------------------------------------------------------------
    if args.resume_phase2:
        print(f"\nLoading Phase 1 checkpoint for Phase 2 resume: {args.resume_phase2}")
        model = tf.keras.models.load_model(args.resume_phase2)
        # Retrieve base_model by name for freeze_for_phase2
        base_model = model.get_layer(
            "efficientnetb0" if args.arch == "efficientnet" else "resnet50"
        )
        model.summary(print_fn=print)
        # Fake a phase1 history with 0 epochs so Phase 2 starts at epoch 0
        history_p1 = type("H", (), {"history": {"loss": []}})()
        phase1_ckpt = args.resume_phase2
        phase1_log  = "(resumed — skipped)"
        print("Phase 1 skipped (resuming from checkpoint).")
    else:
        print(f"\nBuilding model: arch={args.arch}, phase_boundary=({n_trainable}, {n_frozen})")
        model, base_model = build_model(args.arch, (n_trainable, n_frozen))
        model.summary(print_fn=print)

        # -- Phase 1 ----------------------------------------------------------
        phase1_ckpt = os.path.join(
            args.output_dir, f"{args.arch}_{loss_label}_phase1_{timestamp}.keras"
        )
        phase1_log = os.path.join(log_dir, f"{args.arch}_baseline_{loss_label}_phase1_{timestamp}.csv")

        print(f"\n{'='*60}")
        print(f"Phase 1: last {n_trainable} backbone layers trainable, LR={PHASE1_LR}")
        print(f"  Checkpoint : {phase1_ckpt}")
        print(f"  Log        : {phase1_log}")
        print(f"{'='*60}\n")

        model.compile(
            optimizer=Adam(learning_rate=PHASE1_LR),
            loss=get_loss(args),
            metrics=[
                "accuracy",
                Precision(name="precision"),
                Recall(name="recall"),
                AUC(name="auc"),
            ],
        )

        phase1_callbacks = build_callbacks(
            checkpoint_path=phase1_ckpt,
            patience=EARLY_STOPPING_PATIENCE,
            lr_patience=LR_PATIENCE_PHASE1,
            log_path=phase1_log,
        )

        history_p1 = model.fit(
            train_gen,
            epochs=PHASE1_MAX_EPOCHS,
            validation_data=val_gen,
            class_weight=class_weight_dict,
            callbacks=phase1_callbacks,
            verbose=1,
        )

    # -- Phase 2 ------------------------------------------------------------
    phase2_ckpt = os.path.join(
        args.output_dir, f"{args.arch}_{loss_label}_phase2_{timestamp}.keras"
    )
    phase2_log = os.path.join(log_dir, f"{args.arch}_baseline_{loss_label}_phase2_{timestamp}.csv")

    print(f"\n{'='*60}")
    print(f"Phase 2: freeze first {n_frozen} backbone layers, LR={PHASE2_LR}")
    print(f"  Checkpoint : {phase2_ckpt}")
    print(f"  Log        : {phase2_log}")
    print(f"{'='*60}\n")

    freeze_for_phase2(base_model, n_frozen=n_frozen)

    model.compile(
        optimizer=Adam(learning_rate=PHASE2_LR),
        loss=get_loss(args),
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ],
    )

    phase2_callbacks = build_callbacks(
        checkpoint_path=phase2_ckpt,
        patience=EARLY_STOPPING_PATIENCE,
        lr_patience=LR_PATIENCE_PHASE2,
        log_path=phase2_log,
    )

    actual_phase1_epochs = len(history_p1.history["loss"])

    history_p2 = model.fit(
        train_gen,
        epochs=actual_phase1_epochs + PHASE2_MAX_EPOCHS,
        initial_epoch=actual_phase1_epochs,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=phase2_callbacks,
        verbose=1,
    )

    # -- Save final model ---------------------------------------------------
    final_path = os.path.join(
        args.output_dir, f"{args.arch}_baseline_{loss_label}_{timestamp}.keras"
    )
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}")

    # -- Summary ------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training summary")
    print(f"{'='*60}")

    if not args.resume_phase2:
        p1_best = best_epoch_metrics(history_p1, monitor="val_loss")
        print("\nPhase 1 best epoch metrics (by val_loss):")
        for k, v in p1_best.items():
            print(f"  {k:<25s} {v:.6f}")

    p2_best = best_epoch_metrics(history_p2, monitor="val_loss")
    print("\nPhase 2 best epoch metrics (by val_loss):")
    for k, v in p2_best.items():
        print(f"  {k:<25s} {v:.6f}")

    print(f"\nCheckpoints:")
    print(f"  Phase 1 best : {phase1_ckpt}")
    print(f"  Phase 2 best : {phase2_ckpt}")
    print(f"  Final        : {final_path}")
    print(f"Logs:")
    print(f"  Phase 1      : {phase1_log}")
    print(f"  Phase 2      : {phase2_log}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
