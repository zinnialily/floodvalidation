# EFFICIENTNET Baseline (BCE, Seed 42) — Epoch Summary

**Dataset:** Clean deduplicated split (4,099 unique images, 55 dupes removed)

## Per-Epoch Metrics

| Epoch | LR | Train Loss | Val Loss | Train Acc | Val Acc | Train AUC | Val AUC | Train Prec | Val Prec | Train Recall | Val Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.0000 | 0.0211 | 0.0387 | 0.9930 | 0.9866 | 0.9997 | 0.9984 | 0.9955 | 0.9880 | 0.9930 | 0.9899 | ⭐
| 17 | 0.0000 | 0.0194 | 0.0397 | 0.9930 | 0.9866 | 0.9995 | 0.9975 | 0.9965 | 0.9880 | 0.9920 | 0.9899 |
| 18 | 0.0000 | 0.0225 | 0.0404 | 0.9921 | 0.9866 | 0.9995 | 0.9975 | 0.9970 | 0.9880 | 0.9899 | 0.9899 |
| 19 | 0.0000 | 0.0210 | 0.0417 | 0.9918 | 0.9866 | 0.9991 | 0.9983 | 0.9940 | 0.9880 | 0.9925 | 0.9899 |
| 20 | 0.0000 | 0.0186 | 0.0429 | 0.9918 | 0.9866 | 0.9998 | 0.9974 | 0.9960 | 0.9880 | 0.9904 | 0.9899 |
| 21 | 0.0000 | 0.0142 | 0.0418 | 0.9939 | 0.9866 | 0.9997 | 0.9983 | 0.9970 | 0.9880 | 0.9930 | 0.9899 |
| 22 | 0.0000 | 0.0189 | 0.0409 | 0.9939 | 0.9866 | 0.9995 | 0.9983 | 0.9975 | 0.9880 | 0.9925 | 0.9899 |
| 23 | 0.0000 | 0.0160 | 0.0415 | 0.9942 | 0.9866 | 0.9998 | 0.9983 | 0.9950 | 0.9880 | 0.9955 | 0.9899 |

## Val Set Performance (Best Checkpoint)

| Metric | Flood | Non-Flood | Weighted Avg |
|--------|-------|-----------|--------------|
| Precision | 0.9844 | 0.9880 | 0.9866 |
| Recall    | 0.9814    | 0.9899    | 0.9866 |
| F1-Score  | 0.9829  | 0.9889  | 0.9866 |

**Confusion Matrix:** TP=316 | FP=5 | FN=6 | TN=492

## Discrimination (Post-hoc, sklearn)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.9985 |
| ROC-AUC | 0.9991 |

## Figures
- `results/figures/baselines/efficientnet_bce/training_curves.png`
- `results/figures/baselines/efficientnet_bce/confusion_matrix.png`
- `results/figures/baselines/efficientnet_bce/pr_roc_curves.png`
- `results/figures/baselines/efficientnet_bce/fp_fn_examples.png`