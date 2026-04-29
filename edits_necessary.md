# Edits Necessary

All issues found by the post-experiment code review, cross-referenced against
`paper/latex/main.tex`. Ordered from most to least critical.

---

## 1. Split ratio mismatch — `scripts/build_splits.py`

**Severity: Critical**

The paper (main.tex:397–399) states:

> "Images are split 80/20 with stratified sampling by fine-grained category label
> (seed = 42), yielding 3,280 training and 819 validation images.
> No test split is held out; all reported metrics are on the validation set."

The script defaults are:

```python
VAL_FRAC  = 0.15   # line 44
TEST_FRAC = 0.15   # line 45
```

This produces a 70/15/15 train/val/test split, not 80/20, and creates a
`binary/test/` directory the paper says does not exist.

**Required change:**

```python
VAL_FRAC  = 0.20
TEST_FRAC = 0.00
```

Also update the module docstring (lines 2–31) which currently says
"Build the stratified 70/15/15 train/val/test binary split" → change to 80/20.

---

## 2. `training=True` in `build_model` is non-standard — `scripts/utils.py`

**Severity: High (correctness)**

Line 247:

```python
x = base_model(inputs, training=True)
```

Passing `training=True` hardcoded into the functional API call means
BatchNormalization layers in the backbone always run in **training mode**,
using batch statistics from the current batch during inference instead of
the accumulated running mean/variance. This is non-standard and not documented
anywhere in the paper. Standard practice is to let the `model.fit` / `model.predict`
call pass the correct training flag automatically.

**Required change** — remove the explicit `training` argument:

```python
x = base_model(inputs)
```

Keras propagates the training flag correctly through `model.fit` (training=True)
and `model.predict` / `model.evaluate` (training=False) when no override is given.

---

## 3. `evaluate.py` evaluates the test split — `scripts/evaluate.py`

**Severity: High (reproducibility)**

Line 493:

```python
test_dir = os.path.join(data_dir, "processed_data", "binary", "test")
```

The paper says all reported metrics are on the **validation** set and no test
split is held out. `generate_results.py` and `analyze_confounders.py` both
use `binary/val`, which is consistent with the paper.

`evaluate.py` as written evaluates a `binary/test/` directory that does not
exist in the canonical 80/20 split. A reader trying to reproduce results would
either get a `FileNotFoundError` or evaluate on the wrong split.

**Required change:**

Replace `"test"` with `"val"` on line 493:

```python
val_dir = os.path.join(data_dir, "processed_data", "binary", "val")
```

Update the variable name from `test_dir` → `val_dir` throughout the function,
and update the assertion on line 494 and the print statement on line 503.
Update the module docstring to say "val set" instead of "test".

---

## 4. HNM Phase-1 learning rate undocumented — `paper/latex/main.tex`

**Severity: High (reproducibility)**

`train_hnm.py` line 80:

```python
HNM_PHASE1_LR: float = 5e-5
```

This is **half** the baseline Phase-1 LR (`PHASE1_LR = 1e-4` in
`train_baseline.py`). The hyperparameters table (main.tex:988–1012) lists
Phase-1 and Phase-2 LRs for the baseline but has no entry for the HNM
retraining learning rates. A reader cannot reproduce training from the paper alone.

**Required change** — add two rows to the hyperparameters table after
"HNM retraining start" (currently line 1004):

```latex
HNM Phase-1 learning rate          & $5 \times 10^{-5}$ \\
HNM Phase-2 learning rate          & $1 \times 10^{-5}$ \\
```

(HNM Phase-2 LR equals the baseline Phase-2 LR, so noting it explicitly
avoids ambiguity.)

---

## 5. Duplicated utility functions — `scripts/train_baseline.py` and `scripts/train_hnm.py`

**Severity: Medium (maintainability)**

The following functions are copied verbatim between the two training scripts.
Any change to one (augmentation parameters, loss config, etc.) must currently
be applied in two places. All of them belong in `utils.py`, which both scripts
already import from.

| Function | train_baseline.py | train_hnm.py |
|---|---|---|
| `parse_phase_boundary()` | lines 129–157 | lines 202–229 |
| `get_loss()` | lines 160–164 | lines 232–236 |
| `print_runtime_env()` | lines 172–189 | lines 244–259 |
| `build_generators()` | lines 197–256 | lines 485–540 |
| `compute_class_weights()` | lines 264–282 | lines 543–561 |
| `_best_epoch_metrics()` | lines 290–303 | lines 569–581 |

**Required change:** Move all six functions into `utils.py`. Update the
`from utils import (...)` block in both training scripts to import them.
Delete the local definitions.

Note: `build_generators` in `train_baseline.py` references the module-level
`IMG_SIZE` and `BATCH_SIZE` constants. Move those constants to `utils.py` as
well, or pass them as function arguments. Since both scripts use the same
values (224×224, batch 32), moving to `utils.py` is cleaner.

---

## 6. Fragile backbone access via `model.layers[1]` — `scripts/train_hnm.py`

**Severity: Medium (fragility)**

Four locations in `train_hnm.py` use:

```python
base_model = model.layers[1]
```

Lines: 925, 1048, 1155, 1247.

This assumes the backbone is always at index 1 in the layer stack. It will
silently return the wrong layer if the model structure ever changes. Note that
`train_baseline.py` already does this correctly at lines 371–373:

```python
base_model = model.get_layer(
    "efficientnetb0" if args.arch == "efficientnet" else "resnet50"
)
```

**Required change** — replace all four `model.layers[1]` calls in
`train_hnm.py` with the same pattern:

```python
backbone_name = "efficientnetb0" if args.arch == "efficientnet" else "resnet50"
base_model = model.get_layer(backbone_name)
```

---

## 7. Duplicated statistical utilities — `scripts/evaluate.py` and `scripts/aggregate_seeds.py` / `scripts/plot_confounder_fp.py`

**Severity: Medium (maintainability)**

Two functions are independently reimplemented across scripts:

**`mcnemar_test()`** — implemented in:
- `evaluate.py` (lines 143–195, signature: `correct_a, correct_b, alpha, n_comparisons`)
- `aggregate_seeds.py` (lines 169–185, signature: `correct_a, correct_b`, returns p-value only)

These have different interfaces and different correction logic. One canonical
implementation should live in `utils.py`.

**`clopper_pearson_ci()`** — implemented in:
- `evaluate.py` (lines 118–135, named `clopper_pearson_ci`)
- `plot_confounder_fp.py` (lines 73–79, named `cp_ci`)

These are functionally identical. One should live in `utils.py` and both
scripts should import it.

**Required change:** Add `clopper_pearson_ci()` and `mcnemar_test()` to
`utils.py`. Update `evaluate.py`, `aggregate_seeds.py`, and
`plot_confounder_fp.py` to import from `utils` and delete their local copies.
Use `evaluate.py`'s more complete `mcnemar_test()` signature as the canonical
version (it supports `alpha` and `n_comparisons` for Bonferroni correction).

---

## 8. Stale category list in heatmap — `scripts/plot_confounder_fp.py`

**Severity: Low (stale dead code)**

`plot_all_categories()` at line 212 lists:

```python
categories = [
    "river", "plant", "swimming_pool", "park_walkway",
    "animal", "building", "street_clear", "vehicle",
]
```

`build_splits.py` intentionally skips non-water junk categories (plants,
animals, vehicles). These categories will never exist in any run of the
current pipeline and will always render as greyed-out NaN cells in the heatmap,
silently misleading any reader of the figure.

The actual benchmark non-flood categories from `build_splits.py` are:
`river`, `swimming_pool`, `lake`, `fountain`, `building`, `park_walkway`,
`street_clear`.

**Required change** — replace the `categories` list with the actual benchmark
categories:

```python
categories = [
    "river", "swimming_pool", "lake", "fountain",
    "building", "park_walkway", "street_clear",
]
```

---

## 9. Vestigial scripts not part of the research pipeline — `scripts/`

**Severity: Low (repo hygiene)**

The following scripts exist in `scripts/` but are not described in the paper,
are not called by any other script, and clutter the research repository:

| Script | Reason to remove |
|---|---|
| `md_to_pdf.py` | Documentation formatting utility; not research |
| `grad_cam.py` | No GradCAM figures appear in the paper |
| `download_utils.py` | Dataset download helper; dataset is already local |
| `upload_to_hub.py` | HuggingFace Hub upload; not part of experiments |

**Required change:** Delete these four files, or move them to a `scripts/dev/`
subfolder with a README noting they are not part of the reproducibility pipeline.

`evaluate.py` and `aggregate_seeds.py` are also not used for reported results
(evaluate.py targets the wrong split; aggregate_seeds.py is for multi-seed runs
the paper defers). After fixing evaluate.py (edit #3 above), it becomes the
canonical single-model evaluation tool and should be kept. `aggregate_seeds.py`
can be kept but should get a header note:

```python
# NOTE: Not used for reported results (paper reports single seed = 42).
# Designed for future multi-seed extension.
```

---

## Summary table

| # | File | Change | Severity |
|---|---|---|---|
| 1 | `build_splits.py` | VAL_FRAC=0.20, TEST_FRAC=0.00; update docstring | Critical |
| 2 | `utils.py` | Remove `training=True` from `build_model` | High |
| 3 | `evaluate.py` | Point at `binary/val`, not `binary/test` | High |
| 4 | `paper/latex/main.tex` | Add HNM Phase-1/2 LR rows to hyperparameters table | High |
| 5 | `utils.py`, `train_baseline.py`, `train_hnm.py` | Move 6 duplicated functions to utils.py | Medium |
| 6 | `train_hnm.py` | Replace 4× `model.layers[1]` with `model.get_layer(name)` | Medium |
| 7 | `utils.py`, `evaluate.py`, `aggregate_seeds.py`, `plot_confounder_fp.py` | Consolidate `mcnemar_test` and `clopper_pearson_ci` into utils.py | Medium |
| 8 | `plot_confounder_fp.py` | Fix category list to match actual benchmark | Low |
| 9 | `scripts/` | Delete/move 4 vestigial scripts | Low |
