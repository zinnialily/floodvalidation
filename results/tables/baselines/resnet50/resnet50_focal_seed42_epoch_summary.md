# RESNET50 Baseline (FOCAL, Seed 42) — Epoch Summary

**Dataset:** Clean deduplicated split (4,099 unique images, 55 dupes removed)

## Per-Epoch Metrics

| Epoch | LR | Train Loss | Val Loss | Train Acc | Val Acc | Train AUC | Val AUC | Train Prec | Val Prec | Train Recall | Val Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 0.0000 | 0.0076 | 0.0218 | 0.9905 | 0.9841 | 0.9995 | 0.9984 | 0.9930 | 0.9802 | 0.9915 | 0.9940 |
| 16 | 0.0000 | 0.0082 | 0.0195 | 0.9915 | 0.9902 | 0.9994 | 0.9981 | 0.9940 | 0.9939 | 0.9920 | 0.9899 | ⭐
| 17 | 0.0000 | 0.0034 | 0.0210 | 0.9954 | 0.9890 | 0.9999 | 0.9980 | 0.9970 | 0.9880 | 0.9955 | 0.9940 |
| 18 | 0.0000 | 0.0067 | 0.0202 | 0.9921 | 0.9866 | 0.9997 | 0.9988 | 0.9930 | 0.9880 | 0.9940 | 0.9899 |
| 19 | 0.0000 | 0.0036 | 0.0198 | 0.9945 | 0.9878 | 0.9999 | 0.9988 | 0.9955 | 0.9899 | 0.9955 | 0.9899 |
| 20 | 0.0000 | 0.0043 | 0.0196 | 0.9951 | 0.9841 | 0.9999 | 0.9990 | 0.9960 | 0.9840 | 0.9960 | 0.9899 |
| 21 | 0.0000 | 0.0038 | 0.0204 | 0.9951 | 0.9853 | 0.9998 | 0.9990 | 0.9970 | 0.9860 | 0.9950 | 0.9899 |
| 22 | 0.0000 | 0.0026 | 0.0201 | 0.9963 | 0.9829 | 0.9999 | 0.9990 | 0.9980 | 0.9840 | 0.9960 | 0.9879 |
| 23 | 0.0000 | 0.0022 | 0.0197 | 0.9966 | 0.9841 | 1.0000 | 0.9991 | 0.9975 | 0.9840 | 0.9970 | 0.9899 |

## Val Set Performance (Best Checkpoint)

| Metric | Flood | Non-Flood | Weighted Avg |
|--------|-------|-----------|--------------|
| Precision | 0.9582 | 0.8397 | 0.8862 |
| Recall    | 0.7112    | 0.9799    | 0.8742 |
| F1-Score  | 0.8164  | 0.9044  | 0.8698 |

**Confusion Matrix:** TP=229 | FP=10 | FN=93 | TN=487

## Discrimination (Post-hoc, sklearn)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9600 |
| ROC-AUC | 0.9721 |

## Figures
- `results/figures/baselines/resnet50_focal/training_curves.png`
- `results/figures/baselines/resnet50_focal/confusion_matrix.png`
- `results/figures/baselines/resnet50_focal/pr_roc_curves.png`
- `results/figures/baselines/resnet50_focal/fp_fn_examples.png`