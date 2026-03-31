# Pipeline Guide — Flood Detection with Hard Negative Mining

> **Core idea:** Train a flood classifier, find the images it gets wrong, inject
> those images back into training, and retrain. The model's own errors drive the
> next round of improvement.

---

## The Feedback Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Raw images  ──►  train_baseline.py  ──►  .keras checkpoint       │
│                                                  │                  │
│                                                  ▼                  │
│                             analyze_confounders.py                  │
│                          (which images fool the model?)             │
│                                                  │                  │
│                                                  ▼                  │
│                            mining_candidates.txt (hard negatives)   │
│                                                  │                  │
│                                                  ▼                  │
│                               train_hnm.py  ──►  better .keras      │
│                                                  │                  │
│                                                  ▼                  │
│                                           evaluate.py               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Each stage is connected by **files on disk** (`.keras` checkpoints, `.csv`
prediction tables, `.txt` candidate lists) — not in-memory state. This means
any stage can be re-run independently without repeating the whole pipeline.

---

## Full Pipeline — Step by Step

```
Step 1 ──► download_swimmingpool.py ──┐
           download_river.py          │
           download_lake.py           ├──► data/FloodingDataset2/junk/
           download_fountain.py       │
           download_usf.py      ──────┘

Step 2 ──► 02_prepare_confounder_data.ipynb
           (stratified 70/15/15 split → processed_data/binary/)

Step 3 ──► train_baseline.py  ──►  {arch}_baseline_{loss}.keras
           (EfficientNetB0 or ResNet50, BCE or Focal loss)

Step 4 ──► analyze_confounders.py  ──►  mining_candidates_{arch}.txt
           (score every non-flood training image, flag high P(flood))

Step 5 ──► train_hnm.py  ──►  {arch}_hnm_{loss}.keras
           (augment hard negatives 5×, inject, retrain)

Step 6 ──► evaluate.py  ──►  predictions CSV + metrics CSV + figures
           (PR-AUC, recall, per-category FP rates, bootstrap CIs)

Step 7 ──► grad_cam.py  ──►  heatmap PNGs
           (visualise what the model is looking at)

Step 8 ──► aggregate_seeds.py  ──►  seed_aggregation.csv
           (mean ± std across 5 seeds, McNemar pairwise tests)
```

**Execution order:** `02 → 03/04 → 05a → 05b/05c → 06 → aggregate_seeds`

---

## Dataset Structure

Everything lives inside one root folder. The split into flood vs. non-flood
is applied during Step 2 — it is a labelling decision, not a separate dataset.

```
data/FloodingDataset2/
│
├── StreetFloodClasses/          ← PRIMARY FLOOD DATASET (USF)
│   ├── MajorFlood/              ┐
│   ├── ModerateFlood/           ├── labelled flood → binary/*/flood/street_{major,moderate,minor}/
│   ├── MinorFlood/              ┘
│   ├── NoFlood/                 ┐
│   └── parks_walkways/          ┘  non-flood → binary/*/non_flood/{street_clear,park_walkway}/
│
├── junk/                        ← WATER CONFOUNDERS (benchmark categories)
│   ├── Swimmingpool/            ┐
│   ├── River/                   ├── visual confounders → binary/*/non_flood/{swimming_pool,river,lake,fountain}/
│   ├── Lake/                    │   populated by download scripts
│   ├── Fountain/                ┘
│   ├── Cats/  Dogs/  Cars/  …   ← SKIPPED (not benchmark categories)
│
├── extracted/junk/              ← USF BUILDING IMAGES
│   ├── building_exterior/       ┐
│   └── building_interior/       ┘  → binary/*/non_flood/building/
│
└── processed_data/
    └── binary/                  ← built by build_splits.py (seed 42, 70/15/15)
        ├── train/
        │   ├── flood/           street_major/  street_moderate/  street_minor/
        │   ├── non_flood/       river/  lake/  swimming_pool/  fountain/  building/
        │   │                    street_clear/  park_walkway/
        │   └── metadata.csv     ← HuggingFace ImageFolder format (file_name, category, source)
        ├── val/    (same structure + metadata.csv)
        └── test/   (same structure + metadata.csv)
```

> The confounder images (Swimmingpool, River, Lake, Fountain) are downloaded
> from external sources (Places365, Kaggle, Open Images, ADE20K) and written
> into `junk/`. `build_splits.py` copies them into the binary tree under named
> category subfolders — so category membership is encoded structurally, not
> inferred from filenames.

---

## Scripts

### `utils.py` — Shared foundation

Imported by every training script. Never run directly.

| | |
|---|---|
| **Inputs** | Called as a library — no CLI |
| **Outputs** | Shared functions (see below) |

Key functions:

```
set_all_seeds(seed)
    Seeds Python, NumPy, TensorFlow, and the hash randomiser.
    Called at the top of every training script for reproducibility.

build_model(arch, phase_boundary)
    Returns (full_model, base_model).
    Backbone returned separately so Phase 2 freeze can be applied mid-training.
    Architecture: backbone → GAP → Dropout(0.2) → Dense(256) → BN → Dropout(0.3) → sigmoid

verify_preprocessing(generator, arch)
    Grabs one batch and checks pixel value ranges.
    Halts immediately if the double-rescaling bug is detected.
    ⚠ Bug: combining rescale=1./255 with backbone preprocessing shrinks
      inputs to ~[0, 0.004] — pretrained features become non-functional.

freeze_for_phase1(base_model, n_trainable=30)
    Freezes all backbone layers except the last 30.

freeze_for_phase2(base_model, n_frozen=50)
    Unfreezes everything except the first 50 layers.

build_callbacks(checkpoint_path, patience, lr_patience, log_path)
    Returns [ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger].
```

---

### `train_baseline.py` — Two-phase transfer learning

Trains EfficientNetB0 or ResNet50 from ImageNet weights on the binary
flood/non-flood classification task.

| | |
|---|---|
| **Inputs** | `--arch`, `--data_dir`, `--seed`, `--loss`, `--focal_gamma`, `--phase_boundary` |
| **Reads** | `processed_data/binary/train/` and `val/` |
| **Outputs** | Three `.keras` checkpoints + two training log CSVs in `results/logs/` |

```
Phase 1                               Phase 2
───────────────────────────────       ───────────────────────────────
Trainable: last 30 backbone layers    Trainable: all except first 50
LR: 1e-4                              LR: 1e-5
Max epochs: 15                        Max epochs: 20
EarlyStopping patience: 7             EarlyStopping patience: 7
LR halved after: 3 bad epochs         LR halved after: 4 bad epochs
```

Sample invocations:
```bash
# BCE baseline, seed 42
python scripts/train_baseline.py --arch efficientnet --seed 42

# Focal loss baseline
python scripts/train_baseline.py --arch efficientnet --seed 42 --loss focal --focal_gamma 2.0

# ResNet50
python scripts/train_baseline.py --arch resnet50 --seed 42
```

Output files:
```
models/efficientnet_bce_phase1_20240101_120000.keras   ← best of phase 1
models/efficientnet_bce_phase2_20240101_120000.keras   ← best of phase 2  ← USE THIS
models/efficientnet_baseline_bce_20240101_120000.keras ← final weights
results/logs/efficientnet_baseline_bce_phase1_....csv
results/logs/efficientnet_baseline_bce_phase2_....csv
```

---

### `analyze_confounders.py` — Hard negative candidate discovery

Runs the trained baseline on every non-flood image in the training set.
Ranks categories by false positive rate and writes out the candidates for mining.

| | |
|---|---|
| **Inputs** | `--model_path` (Phase 2 `.keras`), `--arch`, `--fp_threshold` (default 0.15) |
| **Reads** | `processed_data/binary/train/non_flood/` only — val/test never touched |
| **Outputs** | `results/tables/confounder_fp_rates_{arch}.csv`, `results/mining_candidates_{arch}.txt` |

How category is inferred:
```
train/non_flood/swimming_pool/Swimmingpool_0023.jpg  →  category = "swimming_pool"
train/non_flood/river/River_0041.jpg                 →  category = "river"
train/non_flood/street_clear/NoFlood_img_003.jpg     →  category = "street_clear"
```
Category is read from the subdirectory name (new layout from `build_splits.py`).
Falls back to filename-prefix parsing and `train_split.csv` for legacy flat layouts.

Sample output table:
```
Category                  N Images   N FP   FP Rate
────────────────────────────────────────────────────
Swimmingpool                   280     91     0.325   ← flagged (> 0.15)
River                          280     54     0.193   ← flagged
Lake                           280     39     0.139
Fountain                       280     18     0.064
NoFlood                        420      8     0.019
Cats                           196      0     0.000
```

The `mining_candidates_{arch}.txt` file contains one image path per line for
every image in a flagged category. This file is consumed directly by `train_hnm.py`.

---

### `train_hnm.py` — Hard Negative Mining retraining

Loads a baseline checkpoint, scores the candidate images, mines the hardest
ones, augments them 5×, injects into training, and retrains.

| | |
|---|---|
| **Inputs** | `--model_path` (baseline Phase 2), `--arch`, `--tau_mode`, `--top_pct`, `--loss`, `--aug_factor` |
| **Reads** | Original `binary/train/` + `mining_candidates_{arch}.txt` |
| **Outputs** | New `.keras` checkpoint; augmented images in `binary_hnm/augmented_hard_negatives/` |

The mining step:
```
mining_candidates.txt
        │
        ▼
Score every candidate image with baseline model → P(flood)
        │
        ▼
Sort by P(flood) descending
        │
        ▼
Take top 10%  (--top_pct 0.10)
        │
        ▼
Augment each selected image 5× (rotation, flip, zoom, brightness shifts)
        │
        ▼
Copy original train/ to binary_hnm/train/
Add augmented images to binary_hnm/train/non_flood/
        │
        ▼
Retrain on augmented set
(Phase 1: LR=5e-5, 15 epochs  |  Phase 2: LR=1e-5, 10 epochs)
```

Three experimental modes:
```
Default                    --no_injection             --random_injection
───────────────────────    ───────────────────────    ───────────────────────
Mine top-10% by P(flood)   Same epoch budget          Same count injected
Augment 5× and inject      No injection               But selected randomly,
                           Isolates: is the           not by P(flood)
                           benefit from HNM or        Isolates: does the
                           just extra training?       ranking matter, or
                                                      just more data?
```

> A partition integrity assertion halts training if any augmented filename
> overlaps with val or test — data leakage is impossible by construction.

---

### `evaluate.py` — Test-set evaluation

Evaluates any saved model on the held-out test set. The test set is **never
modified** by any other script in the pipeline.

| | |
|---|---|
| **Inputs** | `--model_path`, `--arch`, `--n_bootstrap` (default 1000), optional `--compare_predictions_csv` |
| **Reads** | `processed_data/binary/test/` |
| **Outputs** | Predictions CSV, metrics CSV, four figure PNGs |

Output files:
```
results/predictions/{model}_predictions.csv
results/tables/{model}_metrics.csv
results/figures/{model}_confusion_matrix.png
results/figures/{model}_pr_curve.png          ← primary figure
results/figures/{model}_roc_curve.png
results/figures/{model}_severity_recall.png
```

Predictions CSV schema:
```
filename              true_label  predicted_label  flood_probability  correct  is_swimming_pool  category
MajorFlood_0012.jpg            1                1              0.943     True             False  street_major
Swimmingpool_0041.jpg          0                1              0.721    False              True  swimming_pool
```
`category` is taken from the subdirectory name in `test/flood/<category>/` or
`test/non_flood/<category>/`.

Metrics reported:
```
PRIMARY
  PR-AUC        with bootstrap 95% CI (1000 resamples)

SECONDARY
  Recall        with bootstrap CI
  Precision
  F1            with bootstrap CI
  ROC-AUC       with bootstrap CI  ⚠ can be misleading under class imbalance
  Accuracy

CONFOUNDER-SPECIFIC
  Per-category FP rate  with Clopper-Pearson exact binomial CI
  e.g.  Swimmingpool: 3/60 = 5.0%  CI: [1.0%, 13.9%]

SEVERITY-STRATIFIED
  Recall broken down by MajorFlood / ModerateFlood / MinorFlood

PAIRWISE (when --compare_predictions_csv is provided)
  McNemar's test (Bonferroni-corrected)
  Fisher's exact test on pool FP rate
```

Why PR-AUC and not ROC-AUC as the primary metric? With class imbalance,
ROC-AUC is overly optimistic because it credits the classifier for correctly
labelling the many easy negatives. PR-AUC focuses on the precision-recall
tradeoff that actually matters for flood screening.

---

### `grad_cam.py` — GradCAM++ visualizations

Generates heatmap overlays on top of images showing which pixels the model
used to make its prediction.

| | |
|---|---|
| **Inputs** | `--model_path`, `--arch`, `--predictions_csv` (from `evaluate.py`), `--n_per_set` |
| **Outputs** | `results/figures/gradcam_{false_negatives,false_positives,pool_images,random_correct}.png` |

Four image sets:
```
false_negatives   → floods the model missed (high operational cost)
false_positives   → non-flood predicted as flood
pool_images       → all swimming pool images (key confounder)
random_correct    → randomly sampled correct predictions (sanity check)
```

Layer used for gradient computation:
```
EfficientNetB0  →  top_conv          (final conv layer)
ResNet50        →  conv5_block3_out  (final conv layer)
```

---

### `aggregate_seeds.py` — Multi-seed comparison table

Collects all per-seed prediction CSVs, computes mean ± std across seeds,
and runs pairwise McNemar tests between conditions.

| | |
|---|---|
| **Inputs** | Glob patterns for each condition (one pattern per condition, quoted) |
| **Outputs** | `results/tables/seed_aggregation_{arch}.csv` |

```bash
python scripts/aggregate_seeds.py \
  --baseline_bce   "results/predictions/efficientnet_baseline_bce_seed*.csv" \
  --baseline_focal "results/predictions/efficientnet_baseline_focal_seed*.csv" \
  --hnm_bce        "results/predictions/efficientnet_hnm_bce_seed*.csv" \
  --hnm_focal      "results/predictions/efficientnet_hnm_focal_seed*.csv" \
  --no_injection   "results/predictions/efficientnet_no_injection_seed*.csv" \
  --random_inject  "results/predictions/efficientnet_random_inject_seed*.csv" \
  --arch           efficientnet \
  --output         results/tables/seed_aggregation_efficientnet.csv
```

Sample output table:
```
Condition              Arch           Seeds     PR-AUC         Recall           F1    Pool FP
─────────────────────────────────────────────────────────────────────────────────────────────
baseline_bce           efficientnet       5  0.923±0.008  0.941±0.012  0.918±0.009  0.108±0.031
baseline_focal         efficientnet       5  0.931±0.007  0.948±0.011  0.926±0.008  0.094±0.027
no_injection           efficientnet       5  0.934±0.007  0.950±0.010  0.929±0.008  0.091±0.025
random_inject          efficientnet       5  0.938±0.006  0.954±0.009  0.933±0.007  0.072±0.022
hnm_bce                efficientnet       5  0.951±0.006  0.963±0.009  0.944±0.007  0.033±0.018
hnm_focal              efficientnet       5  0.958±0.005  0.969±0.008  0.951±0.006  0.021±0.014
```

The `no_injection` and `random_inject` rows are the critical ablations:
- If `hnm_bce` beats `no_injection` → the benefit is from HNM, not extra epochs
- If `hnm_bce` beats `random_inject` → the benefit is from selecting *hard* negatives, not just more data

---

## Download Scripts

All four scripts (`download_swimmingpool.py`, `download_river.py`,
`download_lake.py`, `download_fountain.py`) share the same CLI via `download_utils.py`.

| Script | Sources |
|---|---|
| `download_swimmingpool.py` | Places365 (MIT), Open Images v7 |
| `download_river.py` | ATLANTIS, RIWA (Kaggle), WaterNet (Kaggle), LuFI-RiverSnap (Kaggle) |
| `download_lake.py` | ATLANTIS, WaterNet (Kaggle) |
| `download_fountain.py` | Open Images v7, ADE20K (MIT CSAIL) |

All scripts are idempotent — re-running after a partial failure is safe:

```
get_next_index() counts {Category}_*.jpg files already in the output dir
→ only downloads the difference to reach --max
→ filenames continue from where the last run left off
```

```bash
# Download up to 400 images per category
python scripts/download_swimmingpool.py --max 400
python scripts/download_river.py        --max 400
python scripts/download_lake.py         --max 400
python scripts/download_fountain.py     --max 400

# Dry run (print what would be downloaded, no files written)
python scripts/download_river.py --max 400 --dry_run
```

---

## Notebooks

The notebooks are thin interactive wrappers around the scripts.
All core logic lives in the scripts.

| Notebook | What it does | Output used by |
|---|---|---|
| `02_prepare_confounder_data.ipynb` | Runs download scripts + stratified 70/15/15 split | All training scripts |
| `03_baseline_efficientnetb0.ipynb` | Runs `train_baseline.py --arch efficientnet` | `05a` |
| `04_baseline_resnet50.ipynb` | Runs `train_baseline.py --arch resnet50` | `05a` |
| `05a_confounder_analysis.ipynb` | Runs `analyze_confounders.py` | `05b`, `05c` |
| `05b_hnm_efficientnetb0.ipynb` | Runs `train_hnm.py --arch efficientnet` | `06` |
| `05c_hnm_resnet50.ipynb` | Runs `train_hnm.py --arch resnet50` | `06` |
| `06_evaluation.ipynb` | Runs `evaluate.py` + `grad_cam.py` | `aggregate_seeds.py` |

Key hand-off between notebooks: after `05a`, the confounder analysis prints
the checkpoint path used for mining. Copy that path into `MODEL_PATH` at the
top of `05b` and `05c` before running them.

---

## File Connections at a Glance

```
train_baseline.py
    └── writes ──► {arch}_baseline_{loss}_{ts}.keras
                            │
                            ▼
            analyze_confounders.py
                    └── writes ──► mining_candidates_{arch}.txt
                                            │
                                            ▼
                            train_hnm.py
                                    └── writes ──► {arch}_hnm_{loss}_{ts}.keras
                                                            │
                                                            ▼
                                            evaluate.py
                                                    └── writes ──► {model}_predictions.csv
                                                                            │
                                                    ┌───────────────────────┤
                                                    ▼                       ▼
                                            grad_cam.py           aggregate_seeds.py
                                            (heatmap PNGs)        (seed_aggregation.csv)
```
