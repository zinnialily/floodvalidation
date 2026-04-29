# RESNET50 Baseline (BCE, Seed 42) — Epoch Summary

**Dataset:** Clean deduplicated split (4,099 unique images, 55 dupes removed)

## Per-Epoch Metrics

| Epoch | LR | Train Loss | Val Loss | Train Acc | Val Acc | Train AUC | Val AUC | Train Prec | Val Prec | Train Recall | Val Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.0000 | 0.0107 | 0.0386 | 0.9960 | 0.9853 | 0.9999 | 0.9986 | 0.9960 | 0.9821 | 0.9975 | 0.9940 | ⭐
| 17 | 0.0000 | 0.0055 | 0.0430 | 0.9979 | 0.9829 | 1.0000 | 0.9986 | 0.9985 | 0.9801 | 0.9980 | 0.9920 |
| 18 | 0.0000 | 0.0076 | 0.0439 | 0.9976 | 0.9829 | 1.0000 | 0.9985 | 0.9985 | 0.9801 | 0.9975 | 0.9920 |
| 19 | 0.0000 | 0.0068 | 0.0418 | 0.9985 | 0.9853 | 0.9996 | 0.9985 | 0.9990 | 0.9879 | 0.9985 | 0.9879 |
| 20 | 0.0000 | 0.0075 | 0.0452 | 0.9985 | 0.9841 | 0.9996 | 0.9984 | 0.9990 | 0.9840 | 0.9985 | 0.9899 |
| 21 | 0.0000 | 0.0059 | 0.0436 | 0.9976 | 0.9829 | 1.0000 | 0.9985 | 0.9975 | 0.9820 | 0.9985 | 0.9899 |
| 22 | 0.0000 | 0.0033 | 0.0411 | 0.9994 | 0.9841 | 1.0000 | 0.9985 | 0.9995 | 0.9859 | 0.9995 | 0.9879 |
| 23 | 0.0000 | 0.0025 | 0.0408 | 0.9994 | 0.9853 | 1.0000 | 0.9985 | 0.9995 | 0.9879 | 0.9995 | 0.9879 |

## Val Set Performance (Best Checkpoint)

| Metric | Flood | Non-Flood | Weighted Avg |
|--------|-------|-----------|--------------|
| Precision | 0.9905 | 0.9821 | 0.9854 |
| Recall    | 0.9720    | 0.9940    | 0.9853 |
| F1-Score  | 0.9812  | 0.9880  | 0.9853 |

**Confusion Matrix:** TP=313 | FP=3 | FN=9 | TN=494

## Discrimination (Post-hoc, sklearn)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9991 |
| ROC-AUC | 0.9995 |

## Figures
- `results/figures/baselines/resnet50_bce/training_curves.png`
- `results/figures/baselines/resnet50_bce/confusion_matrix.png`
- `results/figures/baselines/resnet50_bce/pr_roc_curves.png`
- `results/figures/baselines/resnet50_bce/fp_fn_examples.png`