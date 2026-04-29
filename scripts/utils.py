"""
utils.py -- Shared utilities for the flood detection study.

Single source of truth for:
  - Random seed initialisation
  - Preprocessing functions (Option B: explicit preprocessing_function in generator,
    include_preprocessing=False on backbone -- fixes the double-rescaling bug)
  - Model building and layer-freeze helpers
  - Training callbacks factory
"""

import argparse
import os
import random
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

PREPROCESS_FN = {
    # In TF 2.16 (Keras 3), EfficientNetB0 has built-in Rescaling+Normalization layers
    # and expects raw [0, 255] input — no external preprocessing needed.
    "efficientnet": lambda x: x,
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
}
"""
Preprocessing functions keyed by architecture name.

Both functions expect raw [0, 255] uint8 or float32 pixel input.
Use as ``preprocessing_function=PREPROCESS_FN[arch]`` in ImageDataGenerator.
Do NOT combine with ``rescale=1./255`` -- that would re-introduce the
double-rescaling bug this project is correcting.

Expected output ranges after preprocessing:
  efficientnet: maps [0, 255] -> [-1, 1]  (tf.keras.applications.efficientnet)
  resnet50:     subtracts ImageNet channel means, output roughly [-123, 151]
"""


# ---------------------------------------------------------------------------
# Seed initialisation
# ---------------------------------------------------------------------------


def set_all_seeds(seed: int) -> None:
    """Seed all randomness sources for reproducibility.

    Must be called at the top of every training script/notebook before any
    library that uses random state (NumPy, TensorFlow, Python random, hash
    randomisation).

    Args:
        seed: Integer seed value (e.g. 42, 123, 256, 512, 1024).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Preprocessing verification
# ---------------------------------------------------------------------------


def verify_preprocessing(generator, arch: str) -> None:
    """Assert that a data generator is producing correctly preprocessed batches.

    Grabs one batch from the generator and checks that pixel values fall in the
    range expected by the given backbone architecture.  Raises ``AssertionError``
    immediately if the range is wrong so training is halted before a silent
    double-rescaling bug contaminates results.

    Args:
        generator: A Keras ``ImageDataGenerator``-backed iterator or any
            generator that yields ``(batch_images, batch_labels)`` tuples.
        arch: One of ``"efficientnet"`` or ``"resnet50"``.

    Raises:
        AssertionError: If the batch statistics do not match the expected range
            for the chosen architecture.
        ValueError: If ``arch`` is not a recognised key.
    """
    if arch not in PREPROCESS_FN:
        raise ValueError(
            f"Unknown arch '{arch}'. Valid choices: {list(PREPROCESS_FN.keys())}"
        )

    batch, _ = next(iter(generator))
    batch_min = float(batch.min())
    batch_max = float(batch.max())

    if arch == "resnet50":
        # resnet50.preprocess_input subtracts ImageNet channel means
        # (103.939, 116.779, 123.68) from [0,255] input, so the minimum
        # value can reach roughly -123.  Values > 0 are expected for bright
        # pixels.  A min > -50 strongly indicates inputs were pre-scaled to
        # [0,1] before preprocessing, which is the double-rescaling bug.
        assert batch_min < -50, (
            f"[FAIL] ResNet50 preprocessing check: batch.min()={batch_min:.3f}, "
            "expected < -50.  "
            "Fix: remove 'rescale=1./255' from your ImageDataGenerator and pass "
            "raw [0, 255] pixels with "
            "preprocessing_function=PREPROCESS_FN['resnet50']."
        )

    elif arch == "efficientnet":
        # In TF 2.16 (Keras 3), EfficientNetB0 has built-in Rescaling+Normalization
        # layers and expects raw [0, 255] input.  Generator should pass pixels as-is.
        assert batch_min >= 0 and batch_max > 1.01, (
            f"[FAIL] EfficientNet preprocessing check: "
            f"batch range=[{batch_min:.3f}, {batch_max:.3f}], "
            "expected raw [0, 255] pixels (Keras 3 EfficientNet normalises internally). "
            "Fix: remove 'rescale=1./255' from your ImageDataGenerator and do not "
            "apply any external preprocessing_function for EfficientNet."
        )

    print(
        f"[OK] Preprocessing verified for {arch}: "
        f"range=[{batch_min:.3f}, {batch_max:.3f}]"
    )


# ---------------------------------------------------------------------------
# Layer-freeze helpers
# ---------------------------------------------------------------------------


def freeze_for_phase1(base_model: Model, n_trainable: int = 30) -> None:
    """Configure the backbone for Phase 1 partial fine-tuning.

    Makes the backbone trainable, then freezes all layers except the last
    ``n_trainable`` layers.  The unfrozen layers receive gradient updates
    while deeper pretrained features are preserved.

    Args:
        base_model: The backbone ``Model`` (EfficientNetB0 or ResNet50).
        n_trainable: Number of layers from the end of the backbone to leave
            trainable.  Default 30 (as per project hyperparameter table).
    """
    base_model.trainable = True
    freeze_cutoff = len(base_model.layers) - n_trainable
    for layer in base_model.layers[:freeze_cutoff]:
        layer.trainable = False


def freeze_for_phase2(base_model: Model, n_frozen: int = 50) -> None:
    """Configure the backbone for Phase 2 extended fine-tuning.

    Makes all layers trainable, then re-freezes the first ``n_frozen`` layers
    so that only the earliest (most general) pretrained features are protected.

    Args:
        base_model: The backbone ``Model`` (EfficientNetB0 or ResNet50).
        n_frozen: Number of layers from the start of the backbone to keep
            frozen.  Default 50 (as per project hyperparameter table).
    """
    base_model.trainable = True
    for layer in base_model.layers[:n_frozen]:
        layer.trainable = False


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(
    arch: str,
    phase_boundary: Tuple[int, int] = (30, 50),
) -> Tuple[Model, Model]:
    """Build the classification model for the given backbone architecture.

    Both backbones are instantiated with ``include_preprocessing=False``
    (Option B of the project's double-rescaling fix).  All preprocessing is
    delegated to the data generator via ``PREPROCESS_FN``.

    The classification head is identical for both architectures:
        GlobalAveragePooling2D
        -> Dropout(0.2)
        -> Dense(256, relu)
        -> BatchNormalization
        -> Dropout(0.3)
        -> Dense(1, sigmoid)

    Phase 1 freezing is applied by default via ``freeze_for_phase1``.

    Args:
        arch: One of ``"efficientnet"`` or ``"resnet50"``.
        phase_boundary: Tuple ``(n_trainable_phase1, n_frozen_phase2)``.
            First element is passed to ``freeze_for_phase1``; second is
            available for the caller to pass to ``freeze_for_phase2`` at
            Phase 2 transition.  Default ``(30, 50)``.

    Returns:
        A tuple ``(full_model, base_model)`` where ``full_model`` is the
        end-to-end Keras ``Model`` ready for compilation and ``base_model``
        is the backbone sub-model (useful for later calls to
        ``freeze_for_phase2``).

    Raises:
        ValueError: If ``arch`` is not a recognised key.
    """
    if arch not in PREPROCESS_FN:
        raise ValueError(
            f"Unknown arch '{arch}'. Valid choices: {list(PREPROCESS_FN.keys())}"
        )

    inputs = tf.keras.Input(shape=(224, 224, 3), name="input_image")

    if arch == "efficientnet":
        # include_preprocessing was removed in Keras 3; preprocessing is built into the model.
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )
    else:  # resnet50
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
        )

    # Apply Phase 1 freezing immediately after construction.
    freeze_for_phase1(base_model, n_trainable=phase_boundary[0])

    # Build the classification head.
    x = base_model(inputs)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = Dropout(0.2, name="dropout_1")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = BatchNormalization(name="bn")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    outputs = Dense(1, activation="sigmoid", name="output")(x)

    full_model = Model(inputs=inputs, outputs=outputs, name=f"{arch}_flood_detector")

    return full_model, base_model


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------


def build_callbacks(
    checkpoint_path: str,
    patience: int,
    lr_patience: int,
    log_path: str,
) -> List:
    """Create the standard set of training callbacks.

    Args:
        checkpoint_path: File path at which the best model weights are saved
            (e.g. ``"/content/drive/MyDrive/.../efficientnet_phase1_best.keras"``).
        patience: Number of epochs without ``val_loss`` improvement before
            ``EarlyStopping`` terminates training.
        lr_patience: Number of epochs without ``val_loss`` improvement before
            ``ReduceLROnPlateau`` halves the learning rate.
        log_path: File path for the ``CSVLogger`` output
            (e.g. ``"results/logs/efficientnet_phase1.csv"``).

    Returns:
        A list containing:
          [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger]
    """
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=lr_patience,
        min_lr=1e-7,
        verbose=1,
    )

    csv_logger = CSVLogger(filename=log_path)

    return [checkpoint, early_stopping, reduce_lr, csv_logger]


# ---------------------------------------------------------------------------
# Shared training constants
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 32
IMG_SIZE: Tuple[int, int] = (224, 224)


# ---------------------------------------------------------------------------
# Phase boundary parsing
# ---------------------------------------------------------------------------


def parse_phase_boundary(value: str) -> Tuple[int, int]:
    """Parse a 'n_trainable,n_frozen' string into a tuple of ints.

    Args:
        value: Comma-separated string with exactly two positive integers,
            e.g. ``"30,50"``.

    Returns:
        Tuple ``(n_trainable, n_frozen)``.

    Raises:
        argparse.ArgumentTypeError: If the string cannot be parsed or values
            are not positive integers.
    """
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--phase_boundary must be 'n_trainable,n_frozen', got '{value}'"
        )
    try:
        n_trainable, n_frozen = int(parts[0].strip()), int(parts[1].strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Both values in --phase_boundary must be integers, got '{value}'"
        ) from exc
    if n_trainable <= 0 or n_frozen <= 0:
        raise argparse.ArgumentTypeError(
            f"Both phase_boundary values must be > 0, got {n_trainable}, {n_frozen}"
        )
    return n_trainable, n_frozen


# ---------------------------------------------------------------------------
# Loss function factory
# ---------------------------------------------------------------------------


def get_loss(args: argparse.Namespace):
    """Return the Keras loss function specified by CLI args.

    Args:
        args: Parsed argument namespace with ``loss`` and ``focal_gamma``
            attributes.

    Returns:
        ``"binary_crossentropy"`` string or a
        ``BinaryFocalCrossentropy`` instance.
    """
    if args.loss == "focal":
        return BinaryFocalCrossentropy(gamma=args.focal_gamma, from_logits=False)
    return "binary_crossentropy"


# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------


def print_runtime_env() -> None:
    """Print GPU info, TensorFlow version, and Python version."""
    print("=" * 60)
    print("Runtime environment")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout[:600])
        else:
            print("nvidia-smi not available (CPU-only runtime or no driver).")
    except FileNotFoundError:
        print("nvidia-smi not found (macOS Metal or CPU-only runtime).")
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Python version     : {sys.version}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def build_generators(
    train_dir: str,
    val_dir: str,
    arch: str,
    seed: int,
) -> Tuple:
    """Create training and validation ImageDataGenerators.

    Training generator applies augmentation (rotation, shifts, flip, zoom,
    brightness).  Validation generator applies only the backbone preprocessing
    function.  Neither generator uses ``rescale`` — all normalisation is handled
    by ``PREPROCESS_FN[arch]`` to avoid the double-rescaling bug.

    Args:
        train_dir: Path to the train split root (contains flood/ and non_flood/).
        val_dir: Path to the val split root.
        arch: Architecture key used to select the preprocessing function.
        seed: Random seed for shuffling and augmentation.

    Returns:
        Tuple ``(train_gen, val_gen)`` of DirectoryIterators.
    """
    preprocess_fn = PREPROCESS_FN[arch]

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode="reflect",
        # NOTE: no rescale -- preprocessing_function handles normalisation.
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        # No augmentation, no rescale.
    )

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=seed,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
        seed=seed,
    )

    return train_gen, val_gen


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(train_gen) -> Dict[int, float]:
    """Compute balanced class weights from the training generator's label array.

    Args:
        train_gen: A DirectoryIterator with a populated ``classes`` attribute.

    Returns:
        Dictionary mapping class index to weight, e.g. ``{0: 1.2, 1: 0.85}``.
    """
    classes_array = train_gen.classes
    unique_classes = np.unique(classes_array)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=classes_array,
    )
    class_weight_dict = dict(zip(unique_classes.tolist(), weights.tolist()))
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def best_epoch_metrics(history, monitor: str = "val_loss") -> Dict:
    """Extract metrics at the epoch with the best monitored value.

    Args:
        history: Keras History object returned by ``model.fit()``.
        monitor: Metric name to minimise when selecting the best epoch.

    Returns:
        Dictionary of metric name -> value at the best epoch.
    """
    hist = history.history
    best_epoch = int(np.argmin(hist[monitor]))
    return {k: hist[k][best_epoch] for k in hist}


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------


def clopper_pearson_ci(
    k: int, n: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval.

    Args:
        k: Number of successes (e.g. observed false positives).
        n: Number of trials (e.g. total images in category).
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        Tuple ``(lower, upper)`` bounds on the true rate.
    """
    from scipy.stats import beta as beta_dist

    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    alpha: float = 0.05,
    n_comparisons: int = 1,
) -> Dict[str, float]:
    """McNemar's test with continuity correction and optional Bonferroni adjustment.

    Attempts to use statsmodels; falls back to a manual scipy implementation
    if statsmodels is not installed.

    Args:
        correct_a: Boolean array — True where model A was correct.
        correct_b: Boolean array — True where model B was correct.
        alpha: Nominal significance level before Bonferroni correction.
        n_comparisons: Number of pairwise comparisons for Bonferroni.

    Returns:
        Dict with keys: ``b``, ``c``, ``chi2``, ``p_value``,
        ``adjusted_alpha``, ``significant``.
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    try:
        from statsmodels.stats.contingency_tables import mcnemar as _mcnemar

        table = np.array(
            [
                [int(np.sum(correct_a & correct_b)), b],
                [c, int(np.sum(~correct_a & ~correct_b))],
            ]
        )
        result = _mcnemar(table, exact=False, correction=True)
        chi2 = float(result.statistic)
        p_value = float(result.pvalue)
    except ImportError:
        from scipy.stats import chi2 as chi2_dist

        n = b + c
        chi2 = float((abs(b - c) - 1) ** 2 / n) if n > 0 else 0.0
        p_value = float(chi2_dist.sf(chi2, df=1))

    adjusted_alpha = alpha / n_comparisons
    return {
        "b": b,
        "c": c,
        "chi2": chi2,
        "p_value": p_value,
        "adjusted_alpha": adjusted_alpha,
        "significant": p_value < adjusted_alpha,
    }
