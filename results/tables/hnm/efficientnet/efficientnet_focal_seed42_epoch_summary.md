# EFFICIENTNET Baseline (FOCAL, Seed 42) — Epoch Summary

**Dataset:** Clean deduplicated split (4,099 unique images, 55 dupes removed)

## Per-Epoch Metrics

| Epoch | LR | Train Loss | Val Loss | Train Acc | Val Acc | Train AUC | Val AUC | Train Prec | Val Prec | Train Recall | Val Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.0000 | 0.0193 | 0.0246 | 0.9789 | 0.9866 | 0.9985 | 0.9981 | 0.9966 | 0.9939 | 0.9731 | 0.9839 | ⭐
| 11 | 0.0000 | 0.0116 | 0.0253 | 0.9848 | 0.9866 | 0.9992 | 0.9981 | 0.9952 | 0.9939 | 0.9829 | 0.9839 |
| 12 | 0.0000 | 0.0173 | 0.0256 | 0.9808 | 0.9866 | 0.9981 | 0.9980 | 0.9935 | 0.9959 | 0.9789 | 0.9819 |
| 13 | 0.0000 | 0.0152 | 0.0257 | 0.9796 | 0.9853 | 0.9990 | 0.9981 | 0.9969 | 0.9939 | 0.9738 | 0.9819 |
| 14 | 0.0000 | 0.0128 | 0.0261 | 0.9827 | 0.9853 | 0.9989 | 0.9981 | 0.9962 | 0.9939 | 0.9789 | 0.9819 |
| 15 | 0.0000 | 0.0156 | 0.0261 | 0.9799 | 0.9853 | 0.9983 | 0.9981 | 0.9918 | 0.9939 | 0.9792 | 0.9819 |
| 16 | 0.0000 | 0.0147 | 0.0254 | 0.9834 | 0.9878 | 0.9986 | 0.9981 | 0.9936 | 0.9939 | 0.9825 | 0.9859 |
| 17 | 0.0000 | 0.0178 | 0.0255 | 0.9801 | 0.9878 | 0.9982 | 0.9981 | 0.9922 | 0.9939 | 0.9792 | 0.9859 |

## Val Set Performance (Best Checkpoint)

| Metric | Flood | Non-Flood | Weighted Avg |
|--------|-------|-----------|--------------|
| Precision | 0.9755 | 0.9939 | 0.9867 |
| Recall    | 0.9907    | 0.9839    | 0.9866 |
| F1-Score  | 0.9831  | 0.9889  | 0.9866 |

**Confusion Matrix:** TP=319 | FP=8 | FN=3 | TN=489

## Discrimination (Post-hoc, sklearn)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9971 |
| ROC-AUC | 0.9984 |

## Figures
- `results/figures/baselines/efficientnet_focal/training_curves.png`
- `results/figures/baselines/efficientnet_focal/confusion_matrix.png`
- `results/figures/baselines/efficientnet_focal/pr_roc_curves.png`
- `results/figures/baselines/efficientnet_focal/fp_fn_examples.png`