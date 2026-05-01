# Paper Revision Progress

## Checklist

- [x] **1. Remove focal loss entirely (rebase to BCE only)**
  - [x] Remove EfficientNet Focal and ResNet Focal rows from results table
  - [x] Remove focal loss equation and paragraph in methodology — replaced with BCE-only section
  - [x] Remove focal loss discussion subsection (sec:discussion:loss)
  - [x] Remove related work mention of focal loss (Lin2017 "continuous gradient analogue")
  - [x] Remove recommendation bullet about focal loss
  - [x] Remove focal loss hyperparameter from appendix table (Focal γ = 2.0)
  - [x] Removed \ref{eq:focal} (equation deleted; no dangling refs remain)
  - [x] BCE section now states: "All models are trained with BCE + balanced class weights"

- [x] **2. Reframe metrics as recall-first (not "PR-AUC is definitively better")**
  - [x] Abstract: "conceal/overstate" → "neither of which surfaces category-level FP rates critical for the recall-first screening objective"
  - [x] Highlights bullet: rewritten as recall-first evaluation framing
  - [x] Introduction: removed "inflating ROC-AUC" language; reframed as "neither of which surfaces per-threshold differences"
  - [x] Contributions #1: reframed as "Recall-first evaluation with per-category FP analysis"
  - [x] Related work: "proved that PR-AUC is more informative… ROC-AUC is inflated" → "showed that PR-AUC better reflects recall-oriented performance"
  - [x] Results 4.2: removed "confirming the recommendation of Davis2006/Saito2015"
  - [x] Discussion: subsection renamed; rewritten to focus on per-category FP and recall, not metric superiority

- [x] **3. Fix Figure 1 split: 70/15/15 → 80/20** (Split node in tikz pipeline diagram)

- [x] **4. Fix Stage 2 percentage: 10% → 20%** (line ~568 text; line ~480 tikz figure label)

- [x] **5. Rewrite partition-safety sentence in plain English** (math symbols removed)

- [x] **6. Replace all ">" comparison chains with prose**
  - [x] Ablation design section (lines ~610–611)
  - [x] Ablation results section (lines ~780–782) — softened to "directionally consistent"
  - [x] Figure caption (lines ~845–846)
  - [x] Discussion phase1 subsection (line ~897) — softened to "directionally consistent"

- [x] **7. Progress file created and updated** ✓

## Status
Completed: 2026-04-30
All 7 change groups implemented. Verified: no focal loss references remain; no ">" chains remain.
Next: compile PDF to confirm clean output.
