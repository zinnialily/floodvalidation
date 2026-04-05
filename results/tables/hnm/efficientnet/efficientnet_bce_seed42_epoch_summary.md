# EFFICIENTNET Baseline (BCE, Seed 42) — Epoch Summary

**Dataset:** Clean deduplicated split (4,099 unique images, 55 dupes removed)

## Per-Epoch Metrics

| Epoch | LR | Train Loss | Val Loss | Train Acc | Val Acc | Train AUC | Val AUC | Train Prec | Val Prec | Train Recall | Val Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 0.0000 | 0.0246 | 0.0503 | 0.9902 | 0.9878 | 0.9995 | 0.9976 | 0.9976 | 0.9939 | 0.9883 | 0.9859 | ⭐
| 15 | 0.0000 | 0.0281 | 0.0537 | 0.9895 | 0.9878 | 0.9994 | 0.9975 | 0.9956 | 0.9939 | 0.9893 | 0.9859 |
| 16 | 0.0000 | 0.0254 | 0.0524 | 0.9904 | 0.9878 | 0.9992 | 0.9976 | 0.9966 | 0.9939 | 0.9896 | 0.9859 |
| 17 | 0.0000 | 0.0339 | 0.0519 | 0.9881 | 0.9878 | 0.9990 | 0.9975 | 0.9949 | 0.9939 | 0.9879 | 0.9859 |
| 18 | 0.0000 | 0.0271 | 0.0524 | 0.9890 | 0.9890 | 0.9993 | 0.9975 | 0.9966 | 0.9959 | 0.9876 | 0.9859 |
| 19 | 0.0000 | 0.0183 | 0.0534 | 0.9941 | 0.9890 | 0.9998 | 0.9975 | 0.9973 | 0.9959 | 0.9943 | 0.9859 |
| 20 | 0.0000 | 0.0277 | 0.0518 | 0.9902 | 0.9878 | 0.9992 | 0.9975 | 0.9963 | 0.9939 | 0.9896 | 0.9859 |
| 21 | 0.0000 | 0.0246 | 0.0525 | 0.9902 | 0.9878 | 0.9996 | 0.9976 | 0.9970 | 0.9959 | 0.9889 | 0.9839 |

## Val Set Performance (Best Checkpoint)

| Metric | Flood | Non-Flood | Weighted Avg |
|--------|-------|-----------|--------------|
| Precision | 0.9785 | 0.9939 | 0.9879 |
| Recall    | 0.9907    | 0.9859    | 0.9878 |
| F1-Score  | 0.9846  | 0.9899  | 0.9878 |

**Confusion Matrix:** TP=319 | FP=7 | FN=3 | TN=490

## Discrimination (Post-hoc, sklearn)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9981 |
| ROC-AUC | 0.9989 |

## Figures
- `results/figures/baselines/efficientnet_bce/training_curves.png`
- `results/figures/baselines/efficientnet_bce/confusion_matrix.png`
- `results/figures/baselines/efficientnet_bce/pr_roc_curves.png`
- `results/figures/baselines/efficientnet_bce/fp_fn_examples.png`