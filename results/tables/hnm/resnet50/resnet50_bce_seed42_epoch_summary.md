# RESNET50 Baseline (BCE, Seed 42) — Epoch Summary

**Dataset:** Clean deduplicated split (4,099 unique images, 55 dupes removed)

## Per-Epoch Metrics

| Epoch | LR | Train Loss | Val Loss | Train Acc | Val Acc | Train AUC | Val AUC | Train Prec | Val Prec | Train Recall | Val Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.0000 | 0.0024 | 0.0616 | 0.9995 | 0.9805 | 1.0000 | 0.9955 | 0.9993 | 0.9744 | 1.0000 | 0.9940 |
| 17 | 0.0000 | 0.0025 | 0.0596 | 0.9991 | 0.9866 | 1.0000 | 0.9984 | 0.9997 | 0.9860 | 0.9990 | 0.9920 | ⭐
| 18 | 0.0000 | 0.0021 | 0.0630 | 0.9995 | 0.9866 | 1.0000 | 0.9984 | 1.0000 | 0.9860 | 0.9993 | 0.9920 |
| 19 | 0.0000 | 0.0012 | 0.0693 | 0.9998 | 0.9853 | 1.0000 | 0.9975 | 0.9997 | 0.9860 | 1.0000 | 0.9899 |
| 20 | 0.0000 | 0.0022 | 0.0847 | 0.9995 | 0.9853 | 1.0000 | 0.9964 | 1.0000 | 0.9860 | 0.9993 | 0.9899 |
| 21 | 0.0000 | 0.0016 | 0.0712 | 0.9995 | 0.9841 | 1.0000 | 0.9975 | 1.0000 | 0.9859 | 0.9993 | 0.9879 |
| 22 | 0.0000 | 0.0021 | 0.0729 | 0.9995 | 0.9853 | 1.0000 | 0.9975 | 1.0000 | 0.9860 | 0.9993 | 0.9899 |
| 23 | 0.0000 | 0.0011 | 0.0693 | 1.0000 | 0.9866 | 1.0000 | 0.9974 | 1.0000 | 0.9860 | 1.0000 | 0.9920 |
| 24 | 0.0000 | 0.0009 | 0.0690 | 1.0000 | 0.9866 | 1.0000 | 0.9974 | 1.0000 | 0.9860 | 1.0000 | 0.9920 |

## Val Set Performance (Best Checkpoint)

| Metric | Flood | Non-Flood | Weighted Avg |
|--------|-------|-----------|--------------|
| Precision | 0.9478 | 0.8766 | 0.9046 |
| Recall    | 0.7888    | 0.9718    | 0.8999 |
| F1-Score  | 0.8610  | 0.9218  | 0.8979 |

**Confusion Matrix:** TP=254 | FP=14 | FN=68 | TN=483

## Discrimination (Post-hoc, sklearn)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9715 |
| ROC-AUC | 0.9777 |

## Figures
- `results/figures/baselines/resnet50_bce/training_curves.png`
- `results/figures/baselines/resnet50_bce/confusion_matrix.png`
- `results/figures/baselines/resnet50_bce/pr_roc_curves.png`
- `results/figures/baselines/resnet50_bce/fp_fn_examples.png`