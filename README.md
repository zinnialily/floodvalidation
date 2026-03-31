# First-Pass Flood Detection from Street-Level Imagery: Hard Negative Mining for Confounder-Robust Screening

Binary flood classification using progressive transfer learning and explicit hard negative mining (HNM).
Target journal: *Computers & Geosciences* (Elsevier, ISSN 0098-3004).

> **High recall is the primary objective.** In emergency response, a missed flood is far more costly than a false alarm.

---

## Relevant Prior Work

**AlleyFloodNet (Lee et al., 2025)** applies ConvNeXt-Large to street-level flood binary classification in urban alleys, achieving 96.56% accuracy, 97.67% recall, and an F1 of 96.55% — establishing that high-accuracy binary detection is achievable on curated street-level imagery.
*Gap:* Does not study false positive rates on specific confounding categories (swimming pools, wet roads) and does not apply hard negative mining or report PR-AUC.

**CrisisMMD (Alam et al., 2018)** is an 18,082-image multimodal Twitter dataset from seven 2017 natural disasters annotated for informativeness, humanitarian category, and damage severity.
*Gap:* Not flood-specific; social media domain introduces large intra-class variance and unlabelled near-duplicates. Binary informativeness framing does not target the precision-recall tradeoff relevant to operational flood screening.

**MediaEval 2020 Flood Task (Papadimos et al., 2023)** is a binary relevance classification challenge on 7,698 Italian tweets from 2017–2019 flood events; the best multimodal GNN approach achieved F1 = 0.5379.
*Gap:* Open-domain social media with 21% positive prevalence makes the task fundamentally different from street-level camera screening. Low absolute F1 reflects classifier confusion from non-flood Twitter content, not a property of flood imagery itself.

**Khan et al. (2023)** propose a modified ResNet50 for UAV flood classification achieving 96.43% accuracy and demonstrate deployment on NVIDIA Jetson Nano (820 ms inference at 6.9 W).
*Gap:* UAV/aerial domain; does not address street-level confounders. No PR-AUC reported, no hard negative mining, and single-seed results on a small test set without confidence intervals.

**OHEM — Shrivastava et al. (2016)** introduce Online Hard Example Mining for object detection, showing that focusing gradient updates on misclassified examples outperforms random sampling and reduces training cost.
*Gap:* Designed for two-stage detectors on PASCAL VOC; not demonstrated for binary flood classification or for the swimming-pool / wet-road confounders that are specific to street-level imagery.

**Focal Loss — Lin et al. (2017)** propose a soft down-weighting of well-classified examples via a modulating factor (1 − p_t)^γ, underpinning RetinaNet.
*Gap:* Focal loss modulates gradients within each epoch's existing data distribution. Unlike explicit HNM, it does not permanently add augmented confounders to the training set. The two approaches are complementary; their relative contribution in flood detection has not been evaluated.

**FloodTrace (Dyken et al., 2024)** is a crowdsourced annotation web application for flood extents with a median annotation time less than half the state of the art.
*Gap:* Addresses labelling bottleneck but not classifier robustness to confounders.

**Esparza et al. (2022)** analyse sample, spatial, and demographic biases in crowdsourced flood reports, showing systematic under-representation of rural and economically marginalised communities.
*Gap:* Identifies bias in crowdsourced training data but does not propose a mitigation at the classifier level.

---

## Motivation

### The visual confounder problem

A largely underexplored failure mode in flood binary classification is the **false positive from visually confusing non-flood categories**. Swimming pools share three properties with flooded streets: an open reflective water surface at approximately ground level, proximity to urban infrastructure, and similar colour statistics under overcast conditions. Rivers and lakes produce the same visual signature — still or slow-moving water surfaces that closely resemble inundated ground. Fountains introduce water in motion in urban settings. Wet roads and reflective pavement produce specular reflections that can mimic standing water, particularly at night (Schumann et al., 2023). None of these confounders are systematically represented in existing flood detection datasets, and none of the reviewed papers explicitly report false positive rates on confounding categories separately from aggregate precision.

Misclassifying a swimming pool or river as flooded would trigger a false emergency dispatch. This category of error — high-confidence false positive on a semantically distinct but visually similar class — is precisely what hard negative mining is designed to correct.

### The confirmation bias problem in self-mining

A known limitation of offline HNM is **confirmation bias**: a model that mines its own hard negatives identifies examples that are hard specifically for its current representational state, potentially reinforcing its existing biases rather than resolving genuine semantic confusion (Shrivastava et al., 2016). Cross-seed mining — using one model checkpoint to mine negatives for a differently-initialised model — is a proposed mitigation that has not been evaluated in disaster detection.

### Statistically underpowered comparisons

Most flood detection papers report single-run results on test sets of 500–600 images without confidence intervals. On 564 images, a 5-point accuracy difference corresponds to approximately 28 images — a quantity sensitive to random split composition. McNemar's test and bootstrap CIs are necessary to distinguish genuine improvements from sampling variation. This project applies both.

---

## Key Contributions

1. **Explicit offline HNM for street-level flood classification.** No prior paper applies mining-and-augmentation of specific confounder categories (swimming pools, rivers, lakes, fountains) to the flood binary classification problem. This project implements offline HNM with 5× augmentation injection for both EfficientNetB0 and ResNet50, enabling a direct architectural comparison of HNM benefit.

2. **Two-phase progressive fine-tuning protocol.** Progressive unfreezing is known to mitigate catastrophic forgetting (Lyu et al., 2025; Neupane et al., 2025) but has not been evaluated on flood datasets. This project ablates phase boundaries for both backbones.

3. **Systematic confounder FP analysis with statistically powered test sets.** The original dataset contained only 15 swimming pool test images and no river, lake, or fountain images — insufficient for meaningful Clopper-Pearson CIs. Four confounder categories are expanded to 400 images each (target ~60 test images per category after 70/15/15 split), pushing the CI upper bound to ~5.9% at 0% FP rate. False positive rates on each category are reported separately with binomial CIs.

4. **Three-way ablation design.** Two controls isolate exactly what drives performance: (a) `--no_injection` matches the epoch budget but injects nothing — isolates HNM from extra training time; (b) `--random_injection` injects the same number of non-flood images selected randomly (not by flood probability) from the same candidate pool — isolates the hard-negative ranking from a simple data-augmentation effect. Together these are the two most critical ablations in the pipeline.

5. **Focal loss as a complementary comparison.** Focal loss (Lin et al., 2017) addresses the same gradient-imbalance problem as HNM via a different mechanism. A full 2×2 factorial design (`{BCE, Focal} × {no HNM, HNM}`) evaluates whether the approaches are complementary or redundant, directly answering the primary reviewer objection against HNM.

6. **PR-AUC as primary metric.** ROC-AUC is known to be misleadingly optimistic under class imbalance (Davis & Goadrich, 2006). PR-AUC directly captures the precision-recall tradeoff that matters for screening: high recall subject to a manageable false positive rate.

7. **Statistically rigorous evaluation.** Multi-seed runs (seeds 42, 123, 256, 512, 1024) with bootstrap CIs and McNemar's test for pairwise model comparisons. Fisher's exact test for per-category FP rate comparisons.

8. **Severity-stratified recall.** The dataset preserves flood severity labels (MajorFlood, ModerateFlood, MinorFlood). Recall broken down by severity identifies which flood presentations the model misses most, which has direct operational implications for emergency response triage.

---

## Methods

### Architecture

```
Input (224 × 224 × 3)
  -> backbone-native preprocessing (no double-rescaling — see below)
  -> EfficientNetB0 or ResNet50 convolutional backbone (ImageNet weights)
  -> GlobalAveragePooling2D
  -> Dropout(0.2)
  -> Dense(256, relu)
  -> BatchNormalization
  -> Dropout(0.3)
  -> Dense(1, sigmoid)        # flood probability in [0, 1]
```

### Two-Phase Progressive Fine-Tuning

| Phase | Trainable layers | Learning rate | Max epochs |
|-------|-----------------|---------------|------------|
| Phase 1 | Last 30 | 1e-4 | 15 |
| Phase 2 | All except first 50 | 1e-5 | 20 |

EarlyStopping (patience = 7) and ReduceLROnPlateau (factor = 0.5, patience = 3, min_lr = 1e-7) on validation loss. Class weights computed via `sklearn.utils.class_weight.compute_class_weight`.

### Hard Negative Mining Pipeline

1. **Confounder analysis** (`scripts/analyze_confounders.py`): predict on all `train/non_flood/` images; rank by flood probability; report per-category FP rates to identify mining targets.
2. **Mine**: flag training-set non-flood images with p(flood) > tau (default tau = 0.3, selected by validation sweep). Mining runs on `train/non_flood/` only — never on val or test.
3. **Augment**: apply 5× augmentation to each mined image (same transforms as training). Save to `processed_data/binary_hnm/augmented_hard_negatives/`.
4. **Inject**: copy original training set to `processed_data/binary_hnm/`; add augmented hard negatives to `non_flood/`. Val and test directories are never modified.
5. **Retrain**: Phase 1 (LR = 5e-5, 15 epochs), Phase 2 (LR = 1e-5, 10 epochs) on the augmented set. Validation monitored on the original unmodified val partition.
6. **Partition integrity assertion**: halts training if any filename overlap between train augmentations and val/test is detected.

### Loss Function

Both `train_baseline.py` and `train_hnm.py` accept `--loss` and `--focal_gamma`:

```bash
# Binary cross-entropy (default)
python scripts/train_baseline.py --arch efficientnet --loss binary_crossentropy

# Focal loss γ=2.0 (Lin et al., 2017)
python scripts/train_baseline.py --arch efficientnet --loss focal --focal_gamma 2.0
```

The 2×2 factorial design (`{BCE, Focal} × {no HNM, HNM}`) produces four baseline conditions per architecture.

### Ablation Controls

**Extended-training control** (`--no_injection`): same total epoch budget as HNM, no injection. Isolates HNM from extra training time.

**Random-injection control** (`--random_injection`): injects the same number of non-flood images as HNM would mine, selected randomly (not by flood probability) from the same candidate pool. Isolates the hard-negative ranking from a simple data-size effect.

```bash
# Extended-training control
python scripts/train_hnm.py --arch efficientnet --model_path models/... --no_injection

# Random-injection control
python scripts/train_hnm.py --arch efficientnet --model_path models/... --random_injection
```

### Preprocessing

Backbone-native preprocessing is used via `tf.keras.applications.efficientnet.preprocess_input` / `resnet50.preprocess_input` with `include_preprocessing=False`. **Do not combine `ImageDataGenerator(rescale=1./255)` with `include_preprocessing=True`** — this double-rescales inputs to approximately [0, 0.004], rendering pretrained features non-functional. A `verify_preprocessing()` assertion runs before every training loop.

### Threshold Selection

Thresholds are swept from 0.05 to 0.95 on the **validation set** to find the operating point maximising flood recall subject to 0% swimming pool FP on validation. The test set is evaluated at the selected threshold exactly once per model configuration.

---

## Experimental Setup

### Ablations

| Ablation | Script flag | Purpose |
|----------|-------------|---------|
| Extended training without HNM | `--no_injection` | Isolates HNM from extra training epochs |
| Random injection (same count, random selection) | `--random_injection` | Isolates hard-negative ranking from data-size effect |
| Focal loss baseline | `--loss focal` | Tests whether loss-level reweighting alone is sufficient |
| 2×2 factorial: HNM × loss function | `--loss focal` on both scripts | Tests complementarity of HNM and focal loss |
| Tau sweep (0.2, 0.3, 0.4, 0.5) | `--tau_mode sweep` | Select optimal mining threshold on validation |
| Augmentation factor (2×, 5×, 10×) | `--aug_factor` | Sensitivity to augmentation multiplier |
| Phase boundary | `--phase_boundary` | Contribution of freeze/unfreeze boundaries |

### Statistical Validation

- 5-seed runs (42, 123, 256, 512, 1024): mean ± std reported for all metrics
- Bootstrap 95% CIs (1,000 resamples) on all test-set metrics
- McNemar's test for pairwise model accuracy comparisons
- Clopper-Pearson binomial CIs for per-category confounder FP rates
- Fisher's exact test for pre/post-HNM FP comparisons per category

### Visualisations

- Precision-Recall and ROC curves per model
- Confusion matrices with cell counts
- GradCAM++ heatmaps for FN, FP, and confounder category sets (`scripts/grad_cam.py`)
- t-SNE / UMAP on Dense(256) embeddings, coloured by true label and confounder category
- Reliability diagrams (pre- and post-calibration ECE)
- Severity-stratified recall bar chart
- Per-category FP rate table with Clopper-Pearson CIs across all model variants

---

## Datasets

### Flood Classification — Primary Dataset

**FloodingDataset2** (University of South Florida) — 3,754 street-level images with severity labels.

```
FloodingDataset2/
  StreetFloodClasses/
    MajorFlood/    MinorFlood/    ModerateFlood/
    NoFlood/       parks_walkways/
  junk/
    Swimmingpool/  River/  Lake/  Fountain/   ← water confounder categories (benchmark)
    Cats/  Dogs/  Cars/  ...                  ← skipped (not benchmark categories)
  extracted/junk/
    building_exterior/  building_interior/    ← merged into building/ in processed split
  processed_data/
    binary/
      train/
        flood/     street_major/  street_minor/  street_moderate/
        non_flood/ river/  lake/  swimming_pool/  fountain/  building/
                   street_clear/  park_walkway/
        metadata.csv                         — HuggingFace ImageFolder format
      val/   (same structure)   —   ~15% images  +  metadata.csv
      test/  (same structure)   —   ~15% images  +  metadata.csv
  split_manifest.csv                         — full per-image audit trail (all splits)
```

Splits are stratified by category (seed 42, 70/15/15). Built by `build_splits.py`.
Each `metadata.csv` has columns `file_name`, `category`, `source` for HuggingFace ImageFolder auto-detection.
Run `notebooks/02_prepare_confounder_data.ipynb` to populate and re-split the confounder categories.

### Backbone Pretraining

**ImageNet** — pretrained weights for EfficientNetB0 and ResNet50 loaded via `tf.keras.applications`. Not used directly for training or evaluation.

### Confounder Category Sources

The four high-risk visual confounder categories are populated from external datasets via the download scripts in `scripts/`. Target: 400 images per category (~60 test images after split).

| Category | Visual similarity to flood | Source dataset(s) | Script |
|---|---|---|---|
| **Swimmingpool** | High — flat reflective water at ground level | Places365 (MIT), Open Images v7 (Google) | `download_swimmingpool.py` |
| **River** | High — flowing water, often muddy or turbulent | ATLANTIS, RIWA, WaterNet (ADE20K subset), LuFI-RiverSnap | `download_river.py` |
| **Lake** | High — still water surface / shoreline | ATLANTIS, WaterNet (ADE20K subset) | `download_lake.py` |
| **Fountain** | Medium — water in motion, urban setting | Open Images v7 (Google), ADE20K (MIT CSAIL) | `download_fountain.py` |

**Source details:**

| Dataset | Access | Used for |
|---|---|---|
| Places365 (MIT CSAIL) | Public HTTP | Swimmingpool outdoor category |
| Open Images v7 (Google) | fiftyone zoo / CSV fallback | Swimmingpool, Fountain |
| ATLANTIS (Erfani et al.) | GitHub releases, COCO JSON | River (river/canal/stream), Lake (lake/pond/reservoir/wetland) |
| RIWA | Kaggle `franzwagner/river-water-segmentation-dataset` | River |
| WaterNet / ADE20K subset | Kaggle `gvclsu/water-segmentation-dataset` | River, Lake |
| LuFI-RiverSnap | Kaggle `arminmoghimi/lufi-riversnap` | River |
| ADE20K (MIT CSAIL) | Public index JSON | Fountain |

All confounder images are converted to RGB JPEG, named `{Category}_{NNNN}.jpg`, and written to `data/FloodingDataset2/junk/{Category}/`. Download scripts are idempotent and can be re-run safely.

### Evaluation Metrics

| Metric | Role |
|--------|------|
| **PR-AUC** | Primary discrimination metric |
| Recall | Primary operational metric (high recall ≥ 95% target) |
| F1, Precision, Accuracy | Secondary |
| ROC-AUC | Reported for comparison only |
| Per-category FP rate + Clopper-Pearson CI | Confounder-specific FP analysis (Swimmingpool, River, Lake, Fountain) |
| ECE, reliability diagram | Probability calibration quality |
| Severity-stratified recall | Geoscience contribution |

---

## Pipeline

| Step | Notebook | Script | Compute |
|------|----------|--------|---------|
| 1. Data exploration | `01_data_exploration.ipynb` | — | CPU |
| 2. Confounder download + stratified split | `02_prepare_confounder_data.ipynb` | `download_{category}.py` | CPU |
| 3a. Baseline EfficientNetB0 (BCE) | `03_baseline_efficientnetb0.ipynb` | `train_baseline.py --arch efficientnet --loss binary_crossentropy` | T4 GPU |
| 3b. Baseline EfficientNetB0 (Focal) | — | `train_baseline.py --arch efficientnet --loss focal` | T4 GPU |
| 4a. Baseline ResNet50 (BCE) | `04_baseline_resnet50.ipynb` | `train_baseline.py --arch resnet50 --loss binary_crossentropy` | T4 GPU |
| 4b. Baseline ResNet50 (Focal) | — | `train_baseline.py --arch resnet50 --loss focal` | T4 GPU |
| 5a. Confounder analysis | `05a_confounder_analysis.ipynb` | `analyze_confounders.py` | T4 GPU |
| 5b. HNM — EfficientNetB0 (BCE) | `05b_hnm_efficientnetb0.ipynb` | `train_hnm.py --arch efficientnet` | T4 GPU |
| 5c. HNM — EfficientNetB0 (Focal) | — | `train_hnm.py --arch efficientnet --loss focal` | T4 GPU |
| 5d. HNM — ResNet50 (BCE) | `05c_hnm_resnet50.ipynb` | `train_hnm.py --arch resnet50` | T4 GPU |
| 5e. HNM — ResNet50 (Focal) | — | `train_hnm.py --arch resnet50 --loss focal` | T4 GPU |
| 5f. Extended-training control | — | `train_hnm.py --arch efficientnet --no_injection` | T4 GPU |
| 5g. Random-injection control | — | `train_hnm.py --arch efficientnet --random_injection` | T4 GPU |
| 6. Evaluation | `06_evaluation.ipynb` | `evaluate.py` | CPU/GPU |
| 7. Seed aggregation | — | `aggregate_seeds.py` | CPU |

**Execution order:** `02 → 03a/3b → 04a/4b → 05a → 05b–5g → 06 → 07`

After step 05a, copy the printed checkpoint path into `MODEL_PATH` in the HNM notebooks before running.

### Seed Aggregation

After all evaluation CSVs are produced, aggregate across seeds:

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

Outputs a unified CSV with mean ± std for PR-AUC, Recall, F1, and per-category FP rates across all conditions, plus Bonferroni-corrected McNemar p-values vs. the baseline BCE condition.

---

## Ideal Results

The best-case outcome demonstrates four things:

1. **HNM beats both controls.** `--no_injection` (epoch-matched) and `--random_injection` (data-size-matched) both fall short of HNM — establishing that the benefit comes specifically from the hard-negative nature of the injected images, not from extra training time or more non-flood data.
2. **Focal loss and HNM are complementary.** In the 2×2 factorial, HNM+Focal outperforms HNM+BCE and Focal alone — showing the two mechanisms target different aspects of the confounder problem.
3. **Cross-seed consistency.** Hard negatives identified by different random seeds substantially overlap, suggesting the mining step targets a stable region of the decision boundary rather than reflecting initialisation noise.
4. **PR-AUC ≥ 0.97 with ≥ 95% recall and ≤ 5% FP rate across all four confounder categories**, with 95% CIs that do not overlap the baseline. This would support the claim of a confounder-robust first-pass screening system.

---

## Known Limitations

1. **Dataset size and source diversity.** The flood images (3,754) come from a single institutional source. Geographic and photographer diversity is not documented. The confounder images are aggregated from seven external datasets with different domains, resolutions, and collection conditions.
2. **Random splitting vs. event-based splitting.** If images from the same flood event appear in both train and test, the model may recognise the scene rather than the flood. Without event metadata this cannot be fully mitigated.
3. **Confounder sample size pre-expansion.** The original 15 pool test images gave a CI of [0%, 21.8%] at 0% FP — scientifically indefensible. The download scripts target 400 images per category to reach a defensible ~5.9% upper bound. Results should not be reported until the expanded test set is in place.
4. **Confirmation bias in self-mining.** A model mines hard negatives that are hard for its current state. If seeds mine different sets, the benefit may not generalise. The cross-seed overlap analysis addresses this directly.
5. **Validation set double-duty in HNM pipeline.** The same 563-image validation set selects both the baseline checkpoint used for mining and the post-HNM checkpoint. There is a weak data-dependency between the mining selection and the model being evaluated on validation — test results are the only clean evaluation.
6. **Geoscience framing.** This is a first-stage filter within a larger geoscience pipeline, not a standalone hydrological analysis system. The contribution must be connected to operational flood monitoring infrastructure to meet *Computers & Geosciences* scope.

---

## Setup

### Local (CPU tasks)

```bash
pip install -r requirements.txt
```

### Dataset download (CPU, run once)

```bash
# Required: pip install requests Pillow tqdm gdown
# Optional: pip install kaggle fiftyone  (for Kaggle/Open Images sources)

# 1. Download USF FloodingDataset2 from Google Drive (primary flood dataset)
python scripts/download_usf.py --output_dir ./data/FloodingDataset2
#    This also unpacks junk/*.zip archives in place.
#    If gdown hits rate limits, re-run — it resumes automatically.

# 2. Supplement/download the four visual confounder categories
python scripts/download_swimmingpool.py --max 400  # Places365 val tar + Open Images
python scripts/download_river.py        --max 400  # ATLANTIS + RIWA + WaterNet + LuFI
python scripts/download_lake.py         --max 400  # ATLANTIS + WaterNet
python scripts/download_fountain.py     --max 400  # Open Images + ADE20K

# 3. Or run everything including the stratified split via notebook:
#    notebooks/02_prepare_confounder_data.ipynb
```

All scripts are idempotent — re-running after a partial failure continues from where it left off.

**ATLANTIS note:** The ATLANTIS dataset images are not bundled in the GitHub repository (only code and annotations are). If `download_river.py` / `download_lake.py` report 0 images from the atlantis source, the images must be requested directly from the authors at https://github.com/smhassanerfani/atlantis. The Kaggle sources (RIWA, WaterNet, LuFI-RiverSnap) are sufficient substitutes for reaching 400 images.

### Colab (GPU training)

```python
from google.colab import drive
drive.mount('/content/drive')

%pip install -q tf-keras-vis umap-learn

!python scripts/train_baseline.py \
  --arch efficientnet \
  --data_dir /content/drive/MyDrive/FloodingDataset2 \
  --output_dir /content/drive/MyDrive/models
```

### Requirements

```
tensorflow>=2.16,<2.18
numpy>=1.26
pandas>=2.2
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
Pillow>=10.0
scipy>=1.12
tf-keras-vis>=0.8
umap-learn>=0.5
requests>=2.31
tqdm>=4.66
```

---

## Repository Structure

```
imagevalidation2/
  notebooks/
    01_data_exploration.ipynb
    02_prepare_confounder_data.ipynb   ← download scripts + stratified split
    03_baseline_efficientnetb0.ipynb
    04_baseline_resnet50.ipynb
    05a_confounder_analysis.ipynb
    05b_hnm_efficientnetb0.ipynb
    05c_hnm_resnet50.ipynb
    06_evaluation.ipynb
  scripts/
    utils.py                    # seeding, model building, preprocessing, callbacks
    train_baseline.py           # two-phase fine-tuning; --loss {bce,focal}
    analyze_confounders.py      # rank train/non_flood categories by FP rate
    train_hnm.py                # HNM, --no_injection, --random_injection, --loss {bce,focal}
    evaluate.py                 # PR-AUC primary, bootstrap CI, McNemar, severity recall
    grad_cam.py                 # GradCAM++ heatmaps for FN/FP/confounder sets
    aggregate_seeds.py          # unified multi-seed comparison table + McNemar tests
    download_utils.py           # shared: get_next_index, save_image, check_kaggle/fiftyone
    download_usf.py             # USF FloodingDataset2 from Google Drive (gdown)
    download_swimmingpool.py    # Places365 val tar + Open Images v7
    download_river.py           # ATLANTIS + RIWA + WaterNet + LuFI-RiverSnap
    download_lake.py            # ATLANTIS + WaterNet
    download_fountain.py        # Open Images v7 + ADE20K
  results/
    figures/          PR curves, confusion matrices, GradCAM grids, reliability diagrams
    tables/           Metric CSVs, confounder FP rates, tau sweep results
    logs/             Training history CSVs
    predictions/      Per-image prediction CSVs
  methodology.mmd     Pipeline diagram (Mermaid)
  requirements.txt
```

---

## License

MIT

## Citation

> [To be added upon acceptance]
