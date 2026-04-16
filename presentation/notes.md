# Presenter Notes — Flood Detection Talk
**"Confounder-Robust Flood Detection from Street-Level Imagery via Phase-1 Hard Negative Mining"**
Computers & Geosciences audience (~15 min + Q&A)

---

## Slide 1 — Title (~1 min)
Welcome. This talk is about a specific engineering problem: automated first-pass screening of street-level images for flood detection. When we built a screener, we ran into two problems that had to be fixed before the system was fit for purpose. The first is about evaluation — the metrics most teams report will actively mislead you on this kind of data. The second is about visual confounders — images that look like floods but aren't. This talk covers both problems and what we did about them.

**Key acronyms (defined here, reused throughout):**
- PR-AUC = Precision-Recall Area Under Curve
- ROC-AUC = Receiver Operating Characteristic AUC
- HNM = Hard Negative Mining; P1-HNM = Phase-1 HNM
- FP = False Positive; FN = False Negative
- BCE = Binary Cross-Entropy

---

## Slide 2 — Floods demand fast, reliable automated screening (~1 min)
Floods kill roughly 10,000 people per year globally and cause hundreds of billions in economic damage. After a flood event, citizen science platforms receive thousands of street-level photos from community members. A first-pass screener routes flood images to risk models and human review, and archives the rest.

**Key point:** The screener must be recall-first. A missed flood means a neighborhood never gets flagged for risk modelling or human review. A false alarm costs a reviewer one click. These failure modes are not equivalent — that asymmetry defines every design and evaluation decision in this work.

---

## Slide 3 — Visual confounders are the dominant failure mode (~1 min)
Show the false positive / false negative examples. EfficientNet's 7 false positives were all river photographs — taken from bridges or embankments where the reflective surface, ripple pattern, and cropped horizon are visually indistinguishable from a flooded street.

**Key point:** The model isn't broken. It's responding to the correct visual features (standing water, reflective texture), just in the wrong context. Residents photograph rivers, retention ponds, and swimming pools after storms — these are the images the platform receives that aren't floods but look like them.

---

## Slide 4 — Standard metrics overstate performance (~1 min)
Accuracy is the default classification metric, but it's wrong for imbalanced screening data. With ~84% non-flood images in validation, a model that labels everything non-flood achieves 84% accuracy. Accuracy is dominated by the majority class.

ROC-AUC integrates over all operating thresholds, including very low false positive rate regions irrelevant to real deployment.

**PR-AUC is the right metric:** it integrates precision (how often a flood prediction is correct) against recall (how many floods were found). Both axes are directly relevant. The 3.6-point PR-AUC gap between EfficientNet and ResNet50 corresponds to 56 missed floods in a single 819-image validation set. We use PR-AUC throughout.

---

## Slide 5 — Goal (dark slide, ~30 sec)
Three sub-questions structure the paper:
1. Which architecture and loss function best preserves recall?
2. Which evaluation metric correctly ranks models for this task?
3. Does difficulty-ranked mining from Phase-1 outperform extended training and random injection?

The ablation controls (questions 2 and 3) are what make this rigorous rather than a demo.

---

## Slide 6 — Dataset (~1 min)
- **4,099 images**: 1,657 flood (40.4%), 2,442 non-flood (59.6%)
- **Train / Val / Test split**: 2,870 / 819 / 410
- **11 fine-grained categories**: 7 flood subtypes + 2 water confounders (river/stream, swimming pool/pond) + 2 street confounders (wet road, dry street)
- Source: CRIS-HAZARD platform + open datasets

The two water-confounder categories are the primary evaluation targets for FP analysis. The dataset is more balanced (40/60) than a real deployment scenario — real deployments may have fewer flood images, which makes PR-AUC even more important.

---

## Slide 7 — Two-phase progressive fine-tuning (~1.5 min)
The training strategy: Phase 1 unfreezes the top 30 backbone layers + classification head, LR=1e-4, ≤15 epochs. Phase 2 unfreezes all layers, LR=1e-5, ≤15 epochs.

**The key insight for HNM:** By the end of Phase 2, every non-flood training image has been assigned near-zero flood probability. There are no candidates to mine from the converged model. Phase 1 is the right moment — the model has learned enough to identify difficult examples (rivers, pools that look like floods) but hasn't yet resolved its uncertainty about them.

We take the top 10% of non-flood training images by Phase-1 flood probability, augment each five times, and add them to training before retraining from the Phase-2 checkpoint.

---

## Slide 8 — EfficientNetB0 vs ResNet50 confusion matrices (~1.5 min)
Show the side-by-side confusion matrices. Both models achieve ~98% accuracy:
- **EfficientNetB0**: 98.3% accuracy, **7 FN, 7 FP**, flood recall **97.8%**
- **ResNet50**: 98.5% accuracy, **63 FN, 17 FP**, flood recall **80.4%**

EfficientNet misses 7 floods. ResNet50 misses 63. That's **9× more missed floods** at essentially the same accuracy. Accuracy not only failed to detect this — it ranked ResNet50 slightly higher (98.5% vs 98.3%).

This is the core motivation for PR-AUC. For all ablation experiments, we use EfficientNetB0 with BCE loss.

---

## Slide 9 — PR-AUC exposes 56 missed floods (~1 min)
The two panels show PR-ROC curves for EfficientNet and ResNet50 (BCE, baseline).

- **EfficientNetB0**: PR-AUC = 0.9976, ROC-AUC = 0.9985
- **ResNet50**: PR-AUC = 0.9614, ROC-AUC = 0.9728

PR-AUC gap: **3.6 points** = 56 more missed floods.
ROC-AUC gap: 2.6 points — smaller, and harder to connect intuitively to missed floods.

Both metrics identified EfficientNet as better, but PR-AUC makes the stakes concrete: 56 missed flood reports in a single 819-image validation set.

---

## Slide 10 — River is the only confounder (~1.5 min)
The heatmap shows per-category FP rates across model conditions. River scenes are the only category producing false positives at τ=0.5.

**EfficientNetB0 BCE river FP rate**: 9.2%, Clopper-Pearson 95% CI = [3.8%, 17.7%] (7 FP / 76 images)

**All other categories**: 0% FP.

But "zero" requires careful interpretation:
- Swimming pools: N=28 → 95% CI upper bound **12.3%**. We cannot conclude robustness.
- Wet roads, street scenes, animals, buildings: also small N.

**Implication**: Before deployment, expand water-confounder validation sets. Target N≥300 for river, N≥100 for pool. This is a data collection problem, not a modeling problem.

---

## Slide 11 — HNM ablation results (~2 min)
The four panels show PR-ROC curves for all conditions. Focus on the strict ordering in accuracy improvement:

| Condition | Accuracy | Δ vs Baseline |
|---|---|---|
| Baseline | 98.29% | — |
| Random injection | 98.41% | +0.12 pp |
| Extended training | 98.66% | +0.37 pp |
| **P1-HNM** | **98.78%** | **+0.49 pp** |

**What each comparison isolates:**
- Random vs Baseline: *category exposure* helps (even random samples from river/pool categories improve performance)
- Extended vs Baseline: *more compute* helps beyond more data
- P1-HNM vs Random: same count, same categories, same budget — only **difficulty ranking** differs. P1-HNM wins.

The ordering is consistent with the hypothesis that Phase-1 uncertainty identifies genuinely harder examples. Note: river FP rate changes are not statistically significant (McNemar, p>0.05 after Bonferroni) — the accuracy improvement is real but per-category FP confirmation requires more data.

---

## Slide 12 — Conclusions (dark, ~1 min)
Four takeaways:
1. **Use EfficientNetB0**: 97.8% vs 80.4% recall — 9× fewer missed floods at equal accuracy
2. **Use PR-AUC**: 3.6-pt gap = 56 missed floods. Accuracy and ROC-AUC actively mislead
3. **Phase-1 HNM works**: +0.49 pp > Extended +0.37 pp > Random +0.12 pp > Baseline
4. **River remains hard; pool data is too sparse**: River 9.2% [3.8, 17.7]; pool "zero" at N=28 → CI upper bound 12.3%

---

## Slide 13 — Future directions (~1 min)
Five directions:
1. Expand water-confounder validation (N≥300 river, N≥100 pool) — highest priority
2. Per-category CIs as deployment gates: flag FP upper CI > 5% before deployment
3. Iterative HNM across multiple checkpoints; tune mining fraction (10%) and augmentation factor (5×)
4. Geographic/temporal generalisation: test on post-Helene imagery, other coastal geographies
5. Threshold calibration: τ=0.5 default; calibrate on held-out deployment data using PR curve

---

## Slide 14 — Acknowledgments (~15 sec)
Acknowledge CRIS-HAZARD platform, USF School of Geosciences, open-source tools (PyTorch, HuggingFace, python-pptx), and reviewers. Fill in funding grant number before presenting.

---

## Slide 15 — Q&A
**Quick reference numbers for Q&A:**
- EfficientNet recall: 97.8% vs ResNet50 80.4% (both ~98.5% accuracy)
- PR-AUC gap: 3.6 pts = 56 missed floods in 819-image validation set
- River FP: 9.2% [3.8%, 17.7%] Clopper-Pearson 95% CI, N=76
- P1-HNM: +0.49 pp; Extended: +0.37 pp; Random: +0.12 pp
- Pool "zero" at N=28 → upper CI bound 12.3% — cannot confirm robust
- Phase-1 mining: top 10% by flood probability, 5× augmentation per candidate

**Likely questions:**
- *Why not focal loss?* — Focal down-weights easy negatives, which reduces the mining signal. BCE baseline showed better PR-AUC in our experiments.
- *Why 10% mining fraction?* — Tuned on validation; 5% left too few candidates, 20% diluted the signal.
- *Does this generalise beyond Pinellas County?* — Unknown; geographic generalisation is future work (slide 13).
- *Why not Phase-2 mining?* — Converged model assigns near-zero probability to all training non-floods; no candidates remain.
