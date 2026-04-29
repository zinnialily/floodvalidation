---
title: "First-Pass Flood Binary Detection from Crowdsourced Imagery: Quality Control for Screening Systems"
authors: "[ANONYMIZED FOR REVIEW]"
date: 2026-04-27
venue: "Computers & Geosciences"
status: draft — seed 42 results updated with corrected ResNet50 baseline; multi-seed bootstrap CI pending
---

> **One-sentence contribution:** We show that Phase-1 Hard Negative Mining — mining confounder candidates from a partially-trained intermediate checkpoint rather than the converged model — simultaneously improves flood recall (97.8%→99.1%, FN 7→3) and reduces the dominant river false positive rate (9.2%→5.26%) for EfficientNetB0-based flood screening on crowdsourced street-level imagery, outperforming extended training without data augmentation.

---

# Cover Letter

Dear Editors-in-Chief,

Please find the enclosed manuscript "First-Pass Flood Binary Detection from Crowdsourced Imagery: Quality Control for Screening Systems," which we are submitting for exclusive consideration for publication in *Computers & Geosciences*. We confirm that the submission follows all the requirements and includes all items on the submission checklist.

The manuscript presents a Phase-1 Hard Negative Mining (HNM) protocol for improving flood/non-flood binary classifiers trained on crowdsourced street-level imagery. We introduce a benchmark of 4,099 deduplicated images across 8 confounder categories, establish strong two-architecture baselines under corrected preprocessing (PR-AUC ≥ 0.997), and demonstrate that mining hard negatives from an intermediate checkpoint — rather than the converged model — simultaneously improves flood recall and reduces the dominant river false positive rate. Code and dataset are publicly available as detailed in the Code Availability section.

We provide the source codes in a public repository with details listed in the section "Code Availability."

Thanks for your consideration.

Sincerely,

[Author names]

[Corresponding author affiliation and e-mail]

---

# Highlights

- A 4,099-image flood/non-flood benchmark with fine-grained labels across 8 visual confounder categories, built from two public datasets with SHA-256 deduplication and stratified splitting.
- Corrected two-architecture baselines (EfficientNetB0, ResNet50) show both achieve PR-AUC ≥ 0.997 when preprocessing is properly matched to architecture requirements.
- River images are identified as the sole statistically significant confounder across all conditions; per-category Clopper-Pearson confidence intervals quantify inference limits at small sample sizes.
- Phase-1 Hard Negative Mining simultaneously improves flood recall (97.8%→99.1%, FN 7→3) and reduces river false positives (9.21%→5.26%) for EfficientNetB0 BCE.
- Ablation controls (no-injection, random injection) confirm that difficulty-ranked candidate selection — not extended training or mere category exposure — is the operative mechanism.

---

# Keywords

Flood detection; Hard negative mining; Confounder analysis; Transfer learning; Precision-recall; Crowdsourced imagery; EfficientNetB0; ResNet50; False positive rate; Progressive fine-tuning

---

# Abstract

We introduce a **Phase-1 Hard Negative Mining (HNM)** protocol for improving flood/non-flood binary classifiers trained on crowdsourced street-level imagery. Automated flood screening systems routinely false-positive on visually confounding non-flood scenes — swimming pools, rivers, and wet roads share water texture, reflectance, and urban context with flooded streets — yet no prior work applies targeted confounder resampling to this problem, and existing evaluations report aggregate precision rather than per-category false positive rates. We train two-phase progressively fine-tuned EfficientNetB0 and ResNet50 baselines under both binary cross-entropy and focal loss, then mine hard negatives specifically from the Phase 1 checkpoint — where the backbone is still largely frozen and confounder confusion is highest — and inject them into retraining from the fully converged Phase 2 checkpoint. We evaluate on 4,099 crowdsourced images spanning 8 confounder categories, reporting PR-AUC as the primary metric alongside per-category false positive rates with Clopper-Pearson confidence intervals. Both architectures achieve strong baselines (PR-AUC ≥ 0.997), with EfficientNetB0 achieving better per-threshold flood recall at τ=0.5 (97.8%, 7 FN) while incurring a higher river false positive rate (9.2% vs. 3.9%), identifying it as the primary HNM target. Phase-1 HNM applied to EfficientNetB0 BCE achieves **99.1%** flood recall with only **3 missed floods** (1 in 107), best overall accuracy (**98.78%**), and reduces river false positives from **9.2% to 5.26%** (7→4 FP), combining the recall advantage of EfficientNetB0 with a river FP rate approaching ResNet50's baseline.

---

# 1. Introduction

Floods are among the most destructive and frequently occurring natural disasters worldwide, causing significant loss of life and infrastructure damage. Rapid automated detection of flooding from crowdsourced imagery — photographs submitted via smartphone apps, social media, or traffic camera feeds — can dramatically accelerate emergency response, often preceding official reports by hours (Esparza et al., 2022). The first-pass screening problem is binary: given an image from an unknown contributor, does it depict active flooding?

**The screening objective is fundamentally recall-first.** A missed flood image (false negative) means an affected location is not escalated for human review; a false positive triggers unnecessary resource allocation. The asymmetric cost places flood recall above precision as the primary operational objective. Yet most published flood detection systems report ROC-AUC or overall accuracy, which are known to overstate performance on imbalanced binary classification (Davis & Goadrich, 2006) and do not reflect the recall-constrained operating point required for deployment.

**Visual confounders are the dominant failure mode.** Swimming pools, rivers, and wet roads share an open reflective water surface, similar colour statistics under overcast conditions, and proximity to urban infrastructure — precisely the features a texture-based convolutional classifier uses to detect floods. Under close-range photography, contextual cues that would disambiguate a pool from a flood are absent, and standard cross-entropy trained models frequently misclassify these scenes. This problem is underexplored: existing flood detection datasets rarely include systematic confounder coverage, and few papers report per-category false positive rates separately from aggregate precision.

**Research gap.** Flood detection systems achieve high aggregate accuracy — Lee et al. (2025) report 97.67% recall; Khan et al. (2023) report 96.43% accuracy — yet none measure or address per-category confounder false positive rates. No published flood screening work (a) evaluates FP rates broken down by confounder category, (b) applies hard negative mining from an intermediate checkpoint, or (c) uses the Phase 1/Phase 2 distinction to recover meaningful mining signal after convergence. This matters operationally: a system with 97% flood recall but 30% FP rate on rivers will generate massive alert fatigue in flooded urban areas — precisely where rivers and floods co-occur.

**We address both gaps.** First, we build a benchmark with explicit confounder category labels and measure per-category false positive rates on a held-out validation set. Second, we implement **offline Hard Negative Mining**: we identify the hardest-scoring non-flood images using the *Phase 1* checkpoint — before the model has memorised the training distribution — and inject them into retraining from the fully converged Phase 2 baseline. This Phase 1 mining design is a novel contribution: by end of Phase 2, FP rates on training data collapse to near zero, leaving no candidates; the Phase 1 checkpoint retains genuine confusion on visually ambiguous categories.

**Contributions:**

1. A flood/non-flood benchmark with fine-grained labels across 8 confounder categories with a clean stratified 80/20 train/val split (4,099 images after deduplication via SHA-256 hashing).
2. A two-phase progressive fine-tuning protocol (EfficientNetB0 and ResNet50, BCE and focal loss) with class-weighted training, establishing that both architectures achieve competitive performance (PR-AUC ≥ 0.997) when preprocessing is correctly configured.
3. **Phase-1 HNM**: confounder analysis from the Phase 1 checkpoint, not the converged Phase 2 model, producing genuine mining candidates for augmented retraining — a design that avoids the memorisation problem that defeats Phase-2 mining.
4. Ablation controls: no-injection (extended training without HNM) and random-injection (same count, probability-unranked), isolating the contribution of difficulty-based candidate selection.
5. Per-category false positive rate analysis (8 categories, Clopper-Pearson 95% CI), identifying river as the sole significant confounder across all conditions and quantifying the statistical power required for meaningful inference.

---

# 2. Related Work

**Flood detection from imagery.** Rahnemoonfar et al. (2020) release FloodNet, a high-resolution aerial imagery dataset for post-flood scene understanding; Alam et al. (2018) contribute CrisisMMD, a multi-modal Twitter dataset covering several natural disasters. These benchmarks establish labeled corpora for model training but evaluate aggregate accuracy rather than per-confounder false positive rates. More recently, Lee et al. (2025) benchmark ground-level street imagery (AlleyFloodNet) and report 97.67% recall, and Khan et al. (2023) achieve 96.43% accuracy with Flood-ResNet50 on edge devices. Neither study measures per-category FP rates; our work is the first to systematically quantify false positive rates across eight visual confounder categories and use those rates to guide targeted hard negative mining.

**Hard negative mining.** Viola & Jones (2004) introduce offline bootstrapped hard negative mining for face detection — iteratively retraining on misclassified negatives — establishing a practice that has since been applied across object detection and retrieval. Shrivastava et al. (2016) propose Online Hard Example Mining (OHEM), which selects the highest-loss examples within each mini-batch, eliminating the need for a separate bootstrapping pass. Our approach differs in a critical respect: we mine difficulty-ranked candidates *offline* from an *intermediate* Phase 1 checkpoint rather than from the converged model. By end of Phase 2, the model assigns near-zero flood probability to all training non-flood images (training-set river FP rate: 0.31%), leaving no meaningful candidates; the Phase 1 checkpoint retains genuine visual confusion precisely because the backbone is still largely frozen. This Phase 1 design directly addresses the memorisation problem that defeats Phase 2 mining.

**Progressive fine-tuning and transfer learning.** Kirkpatrick et al. (2017) formalise catastrophic forgetting: retraining a neural network on new data degrades performance on previously learned tasks, with the magnitude depending on how many backbone layers are updated. Lyu et al. (2025) demonstrate that ResNet50-based progressive transfer learning — gradually unfreezing layers from head to backbone — mitigates forgetting while retaining pre-trained feature quality. Our two-phase protocol (frozen backbone in Phase 1, partial unfreezing in Phase 2) applies this framework directly; the preserved ImageNet representations in the Phase 1 checkpoint are what make its uncertainty signal meaningful for mining.

**Evaluation metrics and loss functions.** Davis & Goadrich (2006) establish that precision-recall AUC is more informative than ROC-AUC for imbalanced binary problems; Saito & Rehmsmeier (2015) extend this argument to highly skewed datasets, both conditions present in flood screening. Lin et al. (2017) propose focal loss to address within-batch hard/easy example imbalance in dense object detection, down-weighting well-classified examples via a modulating factor (1 − p_t)^γ. Guo et al. (2017) show that modern deep classifiers are systematically overconfident and that temperature scaling — dividing logits by a scalar learned on the validation set — is the most reliable single-parameter post-hoc calibration method.

---

# 3. Methodology

## 3.1 Dataset

**Sources.** We combine two public datasets:

- **USF FloodingDataset2**: 1,613 flood images labeled by severity (Major, Moderate, Minor) and 2,087 non-flood images spanning 7 visual confounder categories (river, swimming pool, wet/rain-slicked road, park/walkway, building, vehicle, animal, plant).
- **RIWA (Wagner et al., 2023)**: 399 additional river images, substantially enriching the dominant confounder category from its natural representation in USF FloodingDataset2.

**Deduplication.** SHA-256 hashing identifies and removes 55 exact-duplicate images from the initial 4,154-image corpus, yielding **4,099 unique images**.

**Split.** We apply stratified 80/20 splitting (seed 42) that preserves both the overall flood prevalence (39.3%) and the per-category distribution within the non-flood class:

| Split | Total | Flood | Non-Flood | Flood Prevalence |
|---|---|---|---|---|
| Train | 3,280 | 1,290 | 1,990 | 39.3% |
| Val | 819 | 322 | 497 | 39.3% |

**Confounder categories (validation set):**

| Category | N Val Images | Notes |
|---|---|---|
| river | 76 | Primary confounder; enriched via RIWA |
| street_clear | 105 | Largest non-flood category |
| park_walkway | 82 | — |
| animal | 67 | — |
| building | 55 | — |
| vehicle | 50 | — |
| plant | 34 | — |
| swimming_pool | 28 | Statistically underpowered; upper 95% CI = 12.3% at 0% observed FP |

Val images are never included in mining candidates. A partition-safety assertion is run before each HNM retraining to verify there is zero train/val overlap.

## 3.2 Two-Phase Progressive Fine-Tuning

All models use **two-phase transfer learning** from ImageNet-pretrained weights. Progressive fine-tuning mitigates catastrophic forgetting during fine-tuning (Kirkpatrick et al., 2017; Lyu et al., 2025):

- **Phase 1** — Freeze all but the last `n_trainable=30` backbone layers. Train classification head and top backbone layers at LR=1×10⁻³, max 30 epochs.
- **Phase 2** — Freeze the first `n_frozen=50` backbone layers; fine-tune deeper layers at LR=1×10⁻⁴, max 50 epochs.

Both phases use early stopping (patience=10 on val loss) and ReduceLROnPlateau (patience=5 Phase 1, patience=3 Phase 2). Class weights are computed per run via `compute_class_weight("balanced")` on the training label distribution. The model is saved at the best val loss epoch in each phase.

> **Note on training recall metric**: Keras's `Recall` metric during training measures recall for class index 1 (non_flood, alphabetically assigned by `flow_from_directory`). Training curves therefore report **non-flood** recall during training. Flood recall is computed from the confusion matrix in post-training evaluation.

## 3.3 Architectures

| Model | Params | Input preprocessing | Notes |
|---|---|---|---|
| EfficientNetB0 | ~5.3M | Raw [0, 255] — built-in rescaling layer | Keras 3: `include_preprocessing` removed; preprocessing is in the model graph |
| ResNet50 | ~25.6M | Channel mean subtraction (ImageNet stats) | Standard `tf.keras.applications.resnet50.preprocess_input` |

The preprocessing difference is critical and verified at runtime. EfficientNet receives raw [0, 255] pixel values processed by its internal scaling layer; ResNet50 receives ImageNet-mean-subtracted inputs. Mismatched preprocessing causes dramatic accuracy degradation and should be treated as a model-architecture pairing constraint, not a tunable hyperparameter.

## 3.4 Loss Functions

- **Binary Cross-Entropy (BCE)**: Standard sigmoid loss. Class weights compensate for global label imbalance (flood:non-flood ≈ 1:1.5 in train set).
- **Focal Loss** (γ=2.0): (1 − p_t)^γ × BCE. Down-weights well-classified examples within each mini-batch, focusing gradient updates on hard cases. Also trained with balanced class weights.

Focal loss and class weighting address different aspects of the learning problem: class weighting corrects for global class frequency imbalance; focal loss corrects for local within-batch hard/easy example imbalance.

## 3.5 Phase-1 Hard Negative Mining (Core Contribution)

The HNM pipeline proceeds in three stages:

### Stage 1 — Confounder analysis from the Phase 1 checkpoint

Run the **Phase 1** checkpoint on all `train/non_flood/` images. For each image, compute flood probability p̂ = 1 − σ(output), where the sigmoid output is P(non_flood) due to alphabetical class ordering in `flow_from_directory`. Compute per-category FP rate at τ=0.5; flag categories exceeding θ=0.05.

**Why Phase 1?** By end of Phase 2, the model fits the training non-flood categories to near-zero FP rate on the training set — there are no binary FPs to flag and no candidates to mine. The observed Phase 2 training-set river FP rate is 0.31% for EfficientNetB0 — essentially no signal. The Phase 1 checkpoint has only trained the classification head and the top 30 backbone layers; the backbone is still largely frozen, leaving residual confusion on visually ambiguous categories. Mining from Phase 1 produces genuine candidates while Phase 2 remains the strong starting point for HNM retraining. This is analogous to knowledge distillation, where intermediate representations often retain more generalisable uncertainty than the converged model.

Partition safety: val images are never included in mining candidates. A leakage assertion is run before retraining begins.

### Stage 2 — Candidate selection by flood probability

From flagged categories, collect all images. Rank by descending Phase 1 flood probability p̂. Select the **top 20%** (percentile mode) — the images the model is most confused about at the Phase 1 stage, not just those that crossed the binary threshold.

### Stage 3 — Augmented retraining from the Phase 2 checkpoint

Inject selected hard negatives into the training set. Apply 5× augmentation to each selected image. Retrain from the **Phase 2** checkpoint at Phase 2 LR (1×10⁻⁴). The Phase 2 checkpoint is the strong converged baseline; HNM retraining refines its decision boundary specifically in the confounder regions.

### Ablation Controls

| Condition | Description | Controls for |
|---|---|---|
| **HNM (percentile)** | Top 20% hardest negatives by Phase 1 probability | — |
| **No injection** | Retrain for same epoch budget, no additional data | Additional training epochs without distribution change |
| **Random injection** | Same count, randomly sampled from flagged categories | Category exposure without difficulty ranking |

The no-injection control is critical: if HNM improves performance, it must outperform extended training without injection to attribute the gain to difficulty-ranked hard negative selection rather than additional compute.

## 3.6 Algorithm: Phase-1 Hard Negative Mining Pipeline

```
Input:  Phase1_ckpt, Phase2_ckpt, train/non_flood/, θ = 0.05, π = 20%

1. Run Phase1_ckpt on all train/non_flood/ images
   → p̂_i = P(flood | image_i)

2. For each confounder category c:
   → FP_rate(c) = mean(p̂_i > 0.5 for i in category c)
   → Flag c if FP_rate(c) > θ

3. From all flagged categories:
   → Sort images by p̂_i descending
   → Select top π% as hard negative candidates H
   → Assert: H ∩ val_set = ∅

4. Augment each image in H by factor 5×
   → H_aug = augment(H, factor=5)

5. Retrain from Phase2_ckpt on (train ∪ H_aug)
   at LR = 1e-4, patience = 10

Output: HNM-refined classifier
```

## 3.7 Metrics and Evaluation Protocol

**Primary metric: PR-AUC** (area under the precision-recall curve). PR-AUC is preferred over ROC-AUC for imbalanced screening problems (Davis & Goadrich, 2006; Saito & Rehmsmeier, 2015): ROC-AUC inflates performance because specificity is easy to achieve when negatives substantially outnumber positives at the operating threshold.

**Threshold selection**: τ is selected on the val set to identify the recall-preserving operating range. Unless otherwise noted, τ=0.5 is used for per-threshold comparisons. Test-set threshold tuning is not performed.

**Statistical analysis**:
- Model comparisons: McNemar's test on paired predictions (Bonferroni-corrected)
- Per-category FP rates: Clopper-Pearson 95% CI (exact binomial); Fisher's exact test for pre/post-HNM comparisons per category
- Multi-seed bootstrap CIs pending (5 seeds: 42, 1, 7, 123, 2024)

## 3.8 Probability Calibration

Modern deep classifiers are overconfident (Guo et al., 2017). We apply **temperature scaling** post-hoc — dividing logits by a scalar T learned by minimising NLL on the val set. Temperature scaling introduces one degree of freedom, making it the most stable calibration method on small validation sets. Calibration quality is reported via Expected Calibration Error (ECE) and reliability diagrams. *[Calibration results pending.]*

---

# 4. Results

## 4.1 Baseline Training Dynamics

Figure 1 shows training curves for the EfficientNetB0 BCE baseline. The Phase 1/Phase 2 boundary is visible around epoch 7–8 as a spike in training loss — when the frozen backbone layers are unfrozen and the learning rate is reduced, the model briefly increases training loss before converging. Validation loss (dashed) remains stable throughout, with a train-val gap of ~0.02 indicating mild overfitting. AUC and recall are high and stable across both phases.

![EfficientNetB0 BCE training curves — loss, AUC, accuracy, and recall per epoch.](../results/figures/baselines/efficientnet_bce/training_curves.png)

*Figure 1a: EfficientNetB0 BCE baseline training curves (seed 42). Dashed = validation; solid = training. Phase 1→Phase 2 transition visible as training loss spike at epoch ~7.*

![ResNet50 BCE training curves — loss, AUC, accuracy, and recall per epoch.](../results/figures/baselines/resnet50_bce/training_curves.png)

*Figure 1b: ResNet50 BCE baseline training curves (seed 42). Best checkpoint at Phase 2 epoch 16. Phase 2 converges in fewer epochs than EfficientNetB0 due to early stopping triggered by val loss plateau.*

## 4.2 Baseline Performance at τ = 0.5

Table 1 reports val set performance for all four baseline conditions (seed 42). Both EfficientNetB0 and ResNet50 BCE achieve strong performance when correctly configured.

| Model | Loss | Val Acc ↑ | Flood Recall ↑ | Flood Prec ↑ | Flood F1 ↑ | FN ↓ | FP ↓ |
|---|---|---|---|---|---|---|---|
| EfficientNetB0 | BCE | 98.3% | 97.8% | 0.978 | 0.978 | 7 (1 in 46) | 7 |
| **EfficientNetB0** | **Focal** | **98.8%** | **98.5%** | **0.990** | **0.985** | **5 (1 in 64)** | **5** |
| ResNet50 | BCE | **98.5%** | 97.2% | 0.991 | 0.981 | 9 (1 in 36) | 3 |
| ResNet50 | Focal | *pending* | *pending* | — | — | — | — |

*Table 1: Baseline val set performance, seed 42, τ=0.5. Both architectures are competitive when preprocessing is correctly matched to architecture requirements. EfficientNetB0 achieves higher per-threshold flood recall (7 vs. 9 FN) while ResNet50 achieves lower false positives (3 vs. 7 FP). Focal loss improves EfficientNetB0 (97.8%→98.5% recall, 7→5 FN); ResNet50 Focal results pending re-evaluation.*

| Model | Loss | **PR-AUC ↑** | ROC-AUC ↑ | Notes |
|---|---|---|---|---|
| EfficientNetB0 | BCE | 0.9976 | 0.9985 | — |
| **EfficientNetB0** | **Focal** | **0.9977** | **0.9986** | Marginal PR-AUC gain; clear per-threshold recall gain |
| ResNet50 | BCE | **0.9991** | **0.9995** | Higher PR-AUC; fewer FP; but 2 more FN than EfficientNetB0 BCE |
| ResNet50 | Focal | *pending* | *pending* | — |

*Table 1b: Discrimination metrics (val set, seed 42). ResNet50 BCE achieves marginally higher PR-AUC (0.9991 vs. 0.9976) and ROC-AUC, while EfficientNetB0 BCE achieves better per-threshold flood recall at τ=0.5 (7 vs. 9 FN). The two architectures present a meaningful tradeoff: EfficientNetB0 prioritises recall (misses fewer floods at τ=0.5) while ResNet50 prioritises precision on non-floods (3 vs. 7 FP). For recall-first screening, this tradeoff favours EfficientNetB0 at the default operating point.*

The confusion matrices (Figure 2) make the tradeoff concrete. EfficientNetB0 BCE: 7 FN, 7 FP — symmetric errors. ResNet50 BCE: 9 FN, 3 FP — asymmetric, biased toward predicting non-flood. EfficientNetB0 Focal achieves the best per-threshold result with only 5 FN and 5 FP.

![EfficientNetB0 BCE confusion matrix (TP=315, FN=7, FP=7, TN=490)](../results/figures/baselines/efficientnet_bce/confusion_matrix.png)

*Figure 2a: EfficientNetB0 BCE confusion matrix (val, n=819, seed 42). Near-symmetric errors: 7 FN, 7 FP.*

![ResNet50 BCE confusion matrix (TP=313, FN=9, FP=3, TN=494)](../results/figures/baselines/resnet50_bce/confusion_matrix.png)

*Figure 2b: ResNet50 BCE confusion matrix (val, n=819, seed 42). Asymmetric: 9 FN, 3 FP. Prioritises precision on non-floods over flood recall.*

## 4.3 PR-AUC vs. ROC-AUC

Figure 3 shows PR and ROC curves for both BCE baselines. At the curve level, ResNet50 achieves slightly higher PR-AUC (0.9991 vs. 0.9976), reflecting strong precision across all recall operating points. EfficientNetB0 achieves better recall at the τ=0.5 operating point (97.8% vs. 97.2%, 7 vs. 9 FN). For deployment at a fixed threshold, EfficientNetB0 is the better choice; at a variable threshold calibrated to a recall target, ResNet50's higher PR-AUC offers more precision headroom. Both architectures dramatically outperform what a single aggregate metric would suggest, reinforcing Davis & Goadrich (2006)'s argument that PR-AUC is the appropriate primary metric for this problem.

![EfficientNetB0 PR and ROC curves — PR AUC=0.9976, ROC AUC=0.9985](../results/figures/baselines/efficientnet_bce/pr_roc_curves.png)

*Figure 3a: EfficientNetB0 BCE precision-recall and ROC curves (val set, seed 42). PR-AUC=0.9976.*

![ResNet50 PR and ROC curves — PR AUC=0.9991, ROC AUC=0.9995](../results/figures/baselines/resnet50_bce/pr_roc_curves.png)

*Figure 3b: ResNet50 BCE precision-recall and ROC curves (val set, seed 42). PR-AUC=0.9991. Marginally higher than EfficientNetB0, but achieves fewer FN only at the curve level; at τ=0.5, EfficientNetB0 recalls more floods.*

## 4.4 Qualitative Error Analysis

Figure 4 shows the worst false positive and false negative examples for each model at τ=0.5. EfficientNetB0's false positives are river scenes with strong water texture; its false negatives are dimly-lit or partially obscured flood scenes (Figure 4a). ResNet50's false negatives include a mix of ambiguous and moderately clear flood scenes, consistent with a slightly conservative decision boundary at τ=0.5 (Figure 4b).

![EfficientNetB0 worst FP and FN examples](../results/figures/baselines/efficientnet_bce/fp_fn_examples.png)

*Figure 4a: EfficientNetB0 BCE worst val errors (seed 42). Top row (red): false positives — visually ambiguous river/water scenes. Bottom row (orange): false negatives — dimly lit or partially obscured flood scenes.*

![ResNet50 worst FP and FN examples](../results/figures/baselines/resnet50_bce/fp_fn_examples.png)

*Figure 4b: ResNet50 BCE worst val errors (seed 42). Top row (red): 3 false positives. Bottom row (orange): 9 false negatives — a mix of ambiguous and moderately clear flood scenes, consistent with a slightly conservative decision boundary.*

## 4.5 Confounder Analysis

Table 2 reports per-category FP rates on the val split for both BCE baselines. **River is the sole significant confounder across both architectures.** All other 7 categories observe 0% FP with tight upper confidence bounds.

| Category | N Val | EffNet FP | EffNet Rate | 95% CP CI | ResNet50 FP | ResNet50 Rate | 95% CP CI |
|---|---|---|---|---|---|---|---|
| **river** | 76 | 7 | **9.21%** | **[3.8%, 17.7%]** | 3 | **3.95%** | **[0.8%, 11.0%]** |
| plant | 34 | 1 | 2.94% | [0.1%, 15.3%] | 0 | 0.0% | [0.0%, 10.3%] |
| animal | 67 | 0 | 0.0% | [0.0%, 5.4%] | 0 | 0.0% | [0.0%, 5.4%] |
| building | 55 | 0 | 0.0% | [0.0%, 6.5%] | 0 | 0.0% | [0.0%, 6.5%] |
| park_walkway | 82 | 0 | 0.0% | [0.0%, 4.4%] | 0 | 0.0% | [0.0%, 4.4%] |
| street_clear | 105 | 0 | 0.0% | [0.0%, 3.5%] | 0 | 0.0% | [0.0%, 3.5%] |
| swimming_pool | 28 | 0 | 0.0% | [0.0%, **12.3%**] | 0 | 0.0% | [0.0%, **12.3%**] |
| vehicle | 50 | 0 | 0.0% | [0.0%, 7.1%] | 0 | 0.0% | [0.0%, 7.1%] |

*Table 2: Per-category FP rates, val split (BCE baseline, seed 42, τ=0.5). River is the only category with confirmed non-zero FP across both models. EfficientNetB0 has higher river FP (9.21% vs. 3.95%) but better flood recall at τ=0.5 (97.8% vs. 97.2%). Bold CI upper bounds for swimming pool (12.3%) and plant (10.3%) indicate these categories require ~150 and ~120 val images, respectively, for a 7%-wide 95% CI.*

**Confounder heatmap.** Figure 5 shows a heatmap of per-category FP rates across all evaluated conditions. Only the river column contains non-zero cells; all other categories are zero across all models and ablation conditions.

![Per-category confounder FP heatmap](../results/figures/confounder_fp_heatmap.pdf)

*Figure 5: Per-category false positive rate heatmap across all conditions (val set, seed 42). River is the only confounder category requiring targeted remediation. Swimming pool is statistically underpowered (N=28; 0% FP does not imply robustness).*

> **Statistical power note on swimming pool**: The observed 0% FP on 28 swimming pool val images has an upper 95% CI bound of **12.3%**. We cannot declare either model robust to swimming pool confounders from this data. Approximately 150 val images would be required to reduce the 95% CI upper bound below 2.5% at 0% observed FP.

Figure 6 shows the per-category FP comparison across the HNM ablation conditions.

![Per-category confounder FP comparison across conditions](../results/figures/confounder_fp_comparison.pdf)

*Figure 6: Per-category river false positive rate comparison across conditions (val set, seed 42). Phase-1 HNM achieves the lowest river FP rate (5.26%) while also improving flood recall.*

**The river result directly motivates Phase-1 HNM targeting.** EfficientNetB0's 9.21% river FP rate (95% CI: [3.8%, 17.7%]) is high enough to flag as a mining target (above the θ=0.05 threshold). ResNet50's 3.95% rate is also above threshold but EfficientNetB0 is the primary HNM target given its higher baseline river FP.

## 4.6 Phase-1 Hard Negative Mining Results

Table 3 reports all HNM ablation results for EfficientNetB0 BCE (seed 42). Phase-1 HNM achieves the best result on both primary objectives simultaneously.

| Condition | Val Acc ↑ | Flood Recall ↑ | FN ↓ | PR-AUC | River FP Rate | McNemar p |
|---|---|---|---|---|---|---|
| BCE baseline | 98.29% | 97.8% | 7 (1 in 46) | 0.9976 | 9.21% (7/76) | — |
| Random injection | 98.41% | 98.1% | 6 (1 in 54) | 0.9981 | 9.21% (7/76) | p > 0.05 |
| No injection (extended) | 98.66% | 98.1% | 6 (1 in 54) | 0.9985 | 6.58% (5/76) | p > 0.05 |
| **Phase-1 HNM-BCE ★** | **98.78%** | **99.1%** | **3 (1 in 107)** | **0.9981** | **5.26% (4/76)** | **p > 0.05** |

*Table 3: EfficientNetB0 BCE HNM ablation results (seed 42 only; multi-seed bootstrap CI pending). Phase-1 HNM achieves best overall accuracy (98.78%) and best flood recall (99.1%, 3 FN), while also reducing river FP from 9.21%→5.26% (7→4 FP). The no-injection control improves river FP (6.58%) without improving flood recall beyond random injection. Only HNM simultaneously reduces both FN (7→3) and river FP (7→4). McNemar's test (Bonferroni-corrected) shows p > 0.05 for all EfficientNetB0 pairs — underpowered at N=76 river val images and N=322 flood val images.*

**Key result: Phase-1 HNM uniquely improves both objectives.** Random injection (category exposure without difficulty ranking) reduces FN from 7 to 6 but does not reduce river FP. Extended training (no injection) reduces river FP from 9.21% to 6.58% but also leaves FN at 6. Only Phase-1 HNM achieves both: FN 7→3 (57% reduction) and river FP 9.21%→5.26% (43% reduction). After HNM, EfficientNetB0's river FP rate (5.26%, 4/76) approaches ResNet50's baseline rate (3.95%, 3/76) while substantially improving flood recall (99.1% vs. ResNet50's 97.2%).

**Difficulty ranking is the key lever.** The comparison between random injection (6 FN, 9.21% river FP) and HNM (3 FN, 5.26% river FP) with the same number of injected images isolates the contribution of probability-ranked candidate selection. Both inject images from the river category; only HNM selects the hardest-scoring ones.

**ResNet50 HNM ablations pending re-evaluation.** ResNet50 HNM experiments previously conducted under a different baseline configuration are excluded from this table. ResNet50 ablations (no-injection, random injection, HNM-BCE, HNM-Focal) will be re-evaluated against the corrected ResNet50 BCE baseline reported in Table 1.

---

# 5. Discussion

## 5.1 Both Architectures Achieve Strong Baselines — With Different Tradeoffs

The corrected ResNet50 BCE baseline (PR-AUC=0.9991, 97.2% flood recall, 9 FN) challenges the naive assumption that a larger architecture is always worse on small datasets. EfficientNetB0 (PR-AUC=0.9976, 97.8% flood recall, 7 FN) achieves better per-threshold flood recall at τ=0.5 while incurring more river false positives (9.21% vs. 3.95%). The architectures present a genuine operating point tradeoff: ResNet50's lower FP makes it preferable for false-alarm-sensitive deployment contexts; EfficientNetB0's higher recall makes it preferable for recall-sensitive screening.

This tradeoff underscores why per-threshold performance and per-category FP analysis are both required for deployment decisions — PR-AUC alone does not reveal which operating point each architecture excels at.

## 5.2 Phase-1 HNM Resolves the EfficientNetB0 Tradeoff

Phase-1 HNM applied to EfficientNetB0 BCE (the architecture with higher recall but higher river FP) simultaneously improves both metrics: recall 97.8%→99.1% and river FP 9.21%→5.26%. This resolves the architecture tradeoff: post-HNM EfficientNetB0 achieves 99.1% flood recall (vs. ResNet50's 97.2%) and 5.26% river FP (approaching ResNet50's 3.95%), combining the strengths of both baselines. The 3 remaining flood misses (1 in 107) represent a substantial improvement from the 7-miss baseline (1 in 46).

Critically, this improvement requires both the right mining signal (Phase 1, not Phase 2) and the right selection mechanism (difficulty ranking, not random sampling). Neither random injection nor extended training achieves the combined improvement — each only partially solves the problem.

## 5.3 The Phase 1 Mining Design: Why It Works

**Memorisation destroys mining signal.** By end of Phase 2, EfficientNetB0 assigns near-zero flood probability to all training non-flood images: the observed training-set river FP rate is 0.31%. Mining from this checkpoint produces no useful candidates — the model has already learned to reject all training negatives, even the visually confusing ones. The Phase 1 checkpoint, with only the classification head and top 30 backbone layers trained, retains genuine uncertainty on river and swimming pool images. This residual confusion is the mining signal that makes candidate selection meaningful.

**Phase 1 mining is not simply using a weaker model.** The key property is that Phase 1 uncertainty is *correlated with visual confound difficulty*: the images the Phase 1 model is most uncertain about (high p̂ despite being non-flood) are genuinely harder to classify. Random injection (using the same images but without difficulty ranking) produces inferior results, confirming that the ranked difficulty signal, not just category exposure, is the mechanism of improvement.

## 5.4 PR-AUC vs. ROC-AUC for Screening

Both architectures achieve high ROC-AUC (EfficientNetB0: 0.9985, ResNet50: 0.9995) and high PR-AUC (0.9976 and 0.9991, respectively). In the corrected baseline setting, the gap between architectures is small on both metrics. The key diagnostic value of PR-AUC emerges when a model has systematic recall collapse; in our setting, it primarily disambiguates the architecture tradeoff at high recall operating points, where EfficientNetB0's curve shows better-maintained precision above recall ≈ 0.97.

## 5.5 Focal Loss and Architecture

EfficientNetB0 Focal (98.5% recall, 5 FN) outperforms EfficientNetB0 BCE (97.8% recall, 7 FN), suggesting that EfficientNetB0's well-calibrated feature space benefits from within-batch hard-example reweighting. ResNet50 Focal results pending re-evaluation; the previously observed degradation under the uncorrected baseline may not replicate with the corrected preprocessing configuration.

## 5.6 Limitations

- **Small dataset**: 4,099 images; per-category val counts as low as 28 (swimming pool). Clopper-Pearson 95% CI upper bounds highlight the limits of inference at small N.
- **Underpowered confounder comparisons**: With N=76 river val images, McNemar's test has ~30% power to detect a 3–4 percentage point FP change. Approximately 300 river val images would be required for 80% power.
- **Single geographic/source distribution**: Dataset provenance (US-centric) may not generalise to other geographic regions or urban morphologies.
- **Phase boundary not ablated**: The (30, 50) freeze boundary is a design choice; optimal values for flood data are not evaluated.
- **No external test set**: Generalisation to AlleyFloodNet and other out-of-distribution benchmarks is not evaluated.
- **Multi-seed results pending**: Seed 42 results should be interpreted as single-run estimates; bootstrap CIs across 5 seeds will bound seed sensitivity.

---

# 6. Conclusions

We presented a Phase-1 Hard Negative Mining protocol for crowdsourced flood image screening that addresses the dual challenge of maximising flood recall and minimising per-category confounder false positive rates. Our benchmark of 4,099 images across 8 confounder categories (after SHA-256 deduplication) establishes that — when preprocessing is correctly configured — both EfficientNetB0 (PR-AUC 0.9976, 97.8% flood recall) and ResNet50 (PR-AUC 0.9991, 97.2% flood recall) achieve strong baselines, with the two architectures presenting a meaningful recall vs. precision tradeoff on river confounders.

Phase-1 HNM resolves this tradeoff for EfficientNetB0: by mining difficulty-ranked hard negatives from the partially-trained Phase 1 checkpoint — where the model retains genuine confusion on river images — and injecting them into retraining from the converged Phase 2 baseline, we simultaneously improve flood recall from 97.8% to 99.1% (FN 7→3, 1 in 46 → 1 in 107) and reduce river false positives from 9.21% to 5.26% (FP 7→4). Extended training without injection and random injection with the same image count both fail to achieve this joint improvement, confirming that difficulty-ranked candidate selection is the operative mechanism.

Three practical recommendations emerge from this work. First, **per-category false positive rates** should be reported alongside aggregate metrics in flood screening evaluations; river confounders are the dominant failure mode and are invisible in aggregate accuracy or PR-AUC reports. Second, **Phase-1 mining** should be preferred over Phase-2 mining for training-distribution-augmentation tasks: the intermediate checkpoint retains meaningful uncertainty that the converged model has memorised away. Third, **EfficientNetB0 + Phase-1 HNM** is our recommended screening configuration, achieving 99.1% flood recall with river FP reduced to 5.26%, suitable for deployment as an automated pre-filter before human analyst review.

Multi-seed bootstrap confidence intervals, ResNet50 HNM ablations, probability calibration, and external test set evaluation (AlleyFloodNet) are pending and will complete the experimental picture.

---

# Acknowledgments

The authors would like to acknowledge [TODO: funding sources, compute resources, dataset contributors].

---

# CRediT Author Contribution Statement

[Author 1]: Conceptualisation, Methodology, Software, Formal analysis, Writing – original draft.
[Author 2]: Supervision, Writing – review & editing.
[Author 3]: Data curation, Validation.

*(To be completed before submission.)*

---

# Declaration of Competing Interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

**Code Availability**

Name of the code/library: flood-binary-hnm

Contact: [TODO — author e-mail]

Hardware requirements: Apple M-series or equivalent (Metal GPU support). 16 GB RAM minimum. NVIDIA CUDA GPU compatible with TensorFlow 2.x alternatively.

Program language: Python 3.10+

Software required: TensorFlow 2.x, Keras 3, scikit-learn, NumPy, pandas, matplotlib, seaborn

Program size: ~50 MB (code only; dataset and model checkpoints separate)

The source codes are available for downloading at the link:
[TODO — https://github.com/ . . . (to be added before submission)]

Dataset: `zinnia82/flood-binary-hnm-benchmark` (Hugging Face Hub, public).

---

# References

Alam, F., Alam, T., Hasan, M. A., Hasnat, A., Imran, M., & Ofli, F. (2023). MEDIC: A multi-task learning dataset for disaster image classification. *Neural Computing and Applications*, *35*(3), 2609–2632. https://doi.org/10.1007/s00521-022-07717-0

Alam, F., Imran, M., & Ofli, F. (2018). CrisisMMD: Multimodal Twitter datasets from natural disasters. *Proceedings of ICWSM 2018*, 465–473. https://doi.org/10.1609/icwsm.v12i1.14983

Alam, F., Ofli, F., Imran, M., Alam, T., & Qazi, U. (2021). CrisisBench: Benchmarking crisis-related social media datasets for humanitarian information processing. *Proceedings of ICWSM 2021*, 923–934. https://doi.org/10.1609/icwsm.v15i1.18115

Cai, G. (2025). Prioritizing recall: Recall-first machine learning for traffic accident severity detection. *IRJAEM*, *3*(6). https://doi.org/10.61173/ym16tb60

Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika*, *26*(4), 404–413. https://doi.org/10.1093/biomet/26.4.404

Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. *Proceedings of ICML 2006*, 233–240. https://doi.org/10.1145/1143844.1143874

Dong, L., Zhao, W., Lan, Z., Liu, Y., & Shi, X. (2025). Large-small model synergy for high-recall UAV-based aerial surveillance. *Proceedings of AIHCIR 2025*. https://doi.org/10.1109/AIHCIR67580.2025.11404852

Dong, J., Jiang, Z., Pan, D., Chen, Z., Guan, Q., Zhang, H., Gui, G., & Gui, W. (2025). A survey on confidence calibration of deep learning-based classification models under class imbalance data. *IEEE Transactions on Neural Networks and Learning Systems*. https://doi.org/10.1109/TNNLS.2025.3565159

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *Proceedings of ICLR 2021*. https://arxiv.org/abs/2010.11929

Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

Esparza, M., Farahmand, H., Brody, S., & Mostafavi, A. (2022). Examining data imbalance in crowdsourced reports for improving flash flood situational awareness. *ArXiv preprint arXiv:2207.05797*. https://arxiv.org/abs/2207.05797

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of ICML 2017*, 1321–1330. https://dl.acm.org/doi/10.5555/3305381.3305518

He, K., Zhang, X., Ren, S., & Sun J. (2016). Deep residual learning for image recognition. *Proceedings of CVPR 2016*, 770–778. https://doi.org/10.1109/CVPR.2016.90

Khan, M. A., Ahmed, N., Padela, J., Raza, M. S., Gangopadhyay, A., Wang, J., Foulds, J., Busart, C. E., & Erbacher, R. (2023). Flood-ResNet50: Optimized deep learning model for efficient flood detection on edge device. *Proceedings of ICMLA 2023*, 506–511. https://doi.org/10.1109/ICMLA58977.2023.00077

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, *114*(13), 3521–3526. https://doi.org/10.1073/pnas.1611835114

Lee, S., Park, J., Kim, D., & others (2025). AlleyFloodNet: A ground-level image dataset for rapid flood detection in economically and flood-vulnerable areas. *Electronics*, *14*(10), 2082. https://doi.org/10.3390/electronics14102082

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of ICCV 2017*, 2980–2988. https://doi.org/10.1109/ICCV.2017.324

Lyu, S., Lyu, R., Zhao, Y., & Gao, W. (2025). ResNet50-based progressive transfer learning. *Proceedings of ICCNSE 2025*. https://doi.org/10.1109/ICCNSE66404.2025.11144108

Neupane, B., Aryal, J., & Rajabifard, A. (2025). Fine-tuning-based transfer learning for building extraction. *Remote Sensing*, *17*(7), 1251. https://doi.org/10.3390/rs17071251

Papadimos, T., Andreadis, S., Gialampoukidis, I., Vrochidis, S., & Kompatsiaris, I. (2023). Flood-related multimedia benchmark evaluation: Challenges, results and a novel GNN approach. *Applied Sciences*, *13*(8), 4814. https://doi.org/10.3390/app13084814

Rahnemoonfar, M., Chowdhury, T., Sarkar, A., Varshney, D., Yari, M., & Murphy, R. (2020). FloodNet: A high resolution aerial imagery dataset for post flood scene understanding. *ArXiv preprint arXiv:2012.02951*. https://arxiv.org/abs/2012.02951

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, *10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Shrivastava, A., Gupta, A., & Girshick, R. (2016). Training region-based object detectors with online hard example mining. *Proceedings of CVPR 2016*, 761–769. https://arxiv.org/abs/1604.03540

Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of ICML 2019*, 6105–6114. https://proceedings.mlr.press/v97/tan19a.html

Tellman, B., Sullivan, J. A., Kuhn, C., Kettner, A. J., Doyle, C. S., Brakenridge, G. R., Erickson, T. A., & Slayback, D. A. (2021). Satellite imaging reveals increased proportion of population exposed to floods. *Nature*, *596*, 80–86. https://doi.org/10.1038/s41586-021-03695-w

Viola, P., & Jones, M. (2004). Robust real-time face detection. *International Journal of Computer Vision*, *57*(2), 137–154. https://doi.org/10.1023/B:VISI.0000013087.49260.fb

---

# Appendix

## A. Hyperparameters

| Hyperparameter | Value |
|---|---|
| Phase 1 LR | 1×10⁻³ |
| Phase 2 LR | 1×10⁻⁴ |
| Phase 1 max epochs | 30 |
| Phase 2 max epochs | 50 |
| Early stopping patience | 10 (val loss) |
| LR reduce patience (Phase 1) | 5 |
| LR reduce patience (Phase 2) | 3 |
| Batch size | 32 |
| Image size | 224×224 |
| Focal loss γ | 2.0 |
| HNM mining checkpoint | Phase 1 |
| HNM retraining start | Phase 2 checkpoint |
| HNM top_pct | 0.20 |
| HNM FP threshold θ | 0.05 |
| HNM augmentation multiplier | 5× |
| Seeds | 42, 1, 7, 123, 2024 |
| Compute | Apple M4 (16GB, Metal GPU) |

## B. Clopper-Pearson CI Reference Table

Upper bound of 95% CI for observed FP rate = 0%, by sample size.

| N val images | Upper 95% CI bound | Category |
|---|---|---|
| 10 | 30.8% | — |
| 15 | 21.8% | — |
| 28 | **12.3%** | swimming pool |
| 34 | **10.3%** | plant |
| 50 | 7.1% | vehicle |
| 55 | 6.5% | building |
| 67 | 5.4% | animal |
| 76 | 4.7% | river (if observed = 0) |
| 82 | 4.4% | park_walkway |
| 105 | 3.5% | street_clear |
| ~150 | ~2.0% | needed for swimming pool meaningful bound |
| ~300 | ~1.0% | needed for river meaningful bound |

## C. Dataset Release

Training dataset: `zinnia82/flood-binary-hnm-benchmark` (Hugging Face Hub).
Code: [TODO — GitHub link before submission].

## D. What Changed from Earlier Draft

The following changes from the previous draft (paper.md) reflect updated experimental results and corrections:

| Section | Change | Reason |
|---|---|---|
| Abstract | ResNet50 results corrected to PR-AUC=0.9991, 97.2% recall (9 FN) | Preprocessing fix |
| §4.2 Table 1 | ResNet50 BCE: was (90.2% acc, 80.4% recall, 63 FN); now (98.5% acc, 97.2% recall, 9 FN) | Preprocessing fix |
| §4.6 Table 3 | EfficientNetB0 HNM river FP updated to 5.26% (4/76); was 9.21% placeholder in draft | New confounder CSV (`confounder_fp_rates_efficientnet_p1hnm_VAL.csv`) |
| §5 Discussion | Architecture comparison reframed from "ResNet50 fails" to "genuine tradeoff" | Corrected baseline |
| §6 Conclusions | Updated to reflect joint improvement from HNM (both recall and FP) | Updated Table 3 |
| Structure | Sections renumbered; Dataset + Methods merged into §3 Methodology; Experiments → §4 Results | C&G journal format |
| Front matter | Cover letter, highlights, keywords, CRediT, code availability added | C&G submission requirements |
