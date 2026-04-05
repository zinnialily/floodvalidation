# RESNET50 Baseline (FOCAL, Seed 42) — Epoch Summary

**Dataset:** Clean deduplicated split (4,099 unique images, 55 dupes removed)

## Per-Epoch Metrics

| Epoch | LR | Train Loss | Val Loss | Train Acc | Val Acc | Train AUC | Val AUC | Train Prec | Val Prec | Train Recall | Val Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.0000 | 0.0017 | 0.0090 | 0.9981 | 0.9890 | 1.0000 | 0.9997 | 0.9987 | 0.9880 | 0.9987 | 0.9940 | ⭐
| 17 | 0.0000 | 0.0015 | 0.0152 | 0.9979 | 0.9902 | 1.0000 | 0.9996 | 0.9997 | 0.9959 | 0.9973 | 0.9879 |
| 18 | 0.0000 | 0.0028 | 0.0111 | 0.9977 | 0.9915 | 0.9999 | 0.9996 | 0.9980 | 0.9940 | 0.9987 | 0.9920 |
| 19 | 0.0000 | 0.0008 | 0.0157 | 0.9988 | 0.9915 | 1.0000 | 0.9986 | 0.9997 | 0.9960 | 0.9987 | 0.9899 |
| 20 | 0.0000 | 0.0012 | 0.0194 | 0.9986 | 0.9890 | 1.0000 | 0.9995 | 0.9997 | 0.9959 | 0.9983 | 0.9859 |
| 21 | 0.0000 | 0.0018 | 0.0148 | 0.9984 | 0.9878 | 1.0000 | 0.9996 | 0.9990 | 0.9919 | 0.9987 | 0.9879 |
| 22 | 0.0000 | 0.0016 | 0.0183 | 0.9984 | 0.9902 | 1.0000 | 0.9986 | 1.0000 | 0.9959 | 0.9977 | 0.9879 |
| 23 | 0.0000 | 0.0007 | 0.0153 | 0.9993 | 0.9890 | 1.0000 | 0.9996 | 0.9993 | 0.9919 | 0.9997 | 0.9899 |

## Val Set Performance (Best Checkpoint)

| Metric | Flood | Non-Flood | Weighted Avg |
|--------|-------|-----------|--------------|
| Precision | 0.9767 | 0.8737 | 0.9142 |
| Recall    | 0.7795    | 0.9879    | 0.9060 |
| F1-Score  | 0.8670  | 0.9273  | 0.9036 |

**Confusion Matrix:** TP=251 | FP=6 | FN=71 | TN=491

## Discrimination (Post-hoc, sklearn)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9736 |
| ROC-AUC | 0.9809 |

## Figures
- `results/figures/baselines/resnet50_focal/training_curves.png`
- `results/figures/baselines/resnet50_focal/confusion_matrix.png`
- `results/figures/baselines/resnet50_focal/pr_roc_curves.png`
- `results/figures/baselines/resnet50_focal/fp_fn_examples.png`