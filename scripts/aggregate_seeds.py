"""
aggregate_seeds.py — Aggregate multi-seed evaluation results into a unified table.

Loads per-seed predictions CSVs produced by evaluate.py, computes mean ± std
across seeds for all metrics, runs pairwise McNemar tests between conditions,
and outputs a single unified comparison table suitable for inclusion in the paper.

Usage:
    python scripts/aggregate_seeds.py \\
        --baseline_bce   "results/predictions/efficientnet_baseline_bce_seed*.csv" \\
        --baseline_focal "results/predictions/efficientnet_baseline_focal_seed*.csv" \\
        --hnm_bce        "results/predictions/efficientnet_hnm_bce_seed*.csv" \\
        --hnm_focal      "results/predictions/efficientnet_hnm_focal_seed*.csv" \\
        --no_injection   "results/predictions/efficientnet_no_injection_seed*.csv" \\
        --random_inject  "results/predictions/efficientnet_random_inject_seed*.csv" \\
        --arch           efficientnet \\
        --output         results/tables/seed_aggregation_efficientnet.csv

Notes:
    - All glob patterns must be quoted to prevent shell expansion.
    - At least --baseline_bce must be provided; all other conditions are optional.
    - McNemar p-values are Bonferroni-corrected for the number of conditions - 1.
    - PR-AUC, Recall, F1, and pool FP rate are reported as mean ± std across seeds.
"""

# NOTE: Not used for reported results (paper reports single seed = 42).
# Designed for future multi-seed extension.

from __future__ import annotations

import argparse
import glob
import os
import sys
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import mcnemar_test as _mcnemar_canonical  # noqa: E402

# ---------------------------------------------------------------------------
# Metrics computed per seed CSV
# ---------------------------------------------------------------------------

METRIC_COLS = ["pr_auc", "recall", "precision", "f1", "roc_auc", "pool_fp_rate"]


def compute_metrics_from_csv(path: str) -> Dict:
    """Compute per-seed metrics from a predictions CSV produced by evaluate.py.

    Args:
        path: Path to predictions CSV with columns:
              filename, true_label, predicted_label, flood_probability,
              correct, is_swimming_pool, category.

    Returns:
        Dict of metric_name -> float.
    """
    df = pd.read_csv(path)

    required = {"true_label", "predicted_label", "flood_probability", "correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing columns: {missing}")

    y_true = df["true_label"].values.astype(int)
    y_pred = df["predicted_label"].values.astype(int)
    y_prob = df["flood_probability"].values.astype(float)

    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pr_auc = average_precision_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    pool_fp_rate = float("nan")
    if "is_swimming_pool" in df.columns:
        pool_df = df[df["is_swimming_pool"].astype(bool)]
        if len(pool_df) > 0:
            pool_fp_rate = float((pool_df["predicted_label"] == 1).sum()) / len(pool_df)

    return {
        "pr_auc": pr_auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc,
        "pool_fp_rate": pool_fp_rate,
        "n_test": len(df),
    }


def load_condition(
    glob_pattern: str, condition_name: str
) -> Tuple[List[Dict], List[pd.DataFrame]]:
    """Load all seed CSVs matching a glob pattern.

    Returns:
        Tuple of (list of metric dicts, list of DataFrames).
    """
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        print(f"  [WARN] No files matched pattern for '{condition_name}': {glob_pattern}")
        return [], []

    metrics_list = []
    df_list = []
    for p in paths:
        try:
            m = compute_metrics_from_csv(p)
            metrics_list.append(m)
            df_list.append(pd.read_csv(p))
            print(f"  Loaded [{condition_name}] seed file: {os.path.basename(p)}")
        except Exception as e:
            print(f"  [WARN] Could not load {p}: {e}")

    return metrics_list, df_list


def aggregate_metrics(
    metrics_list: List[Dict], condition: str, arch: str
) -> Dict:
    """Compute mean ± std across seeds for all metrics.

    Args:
        metrics_list: List of per-seed metric dicts.
        condition: Condition label (e.g. 'baseline_bce').
        arch: Architecture name.

    Returns:
        Dict with mean and std for each metric.
    """
    if not metrics_list:
        return {}

    result = {"condition": condition, "arch": arch, "n_seeds": len(metrics_list)}
    for metric in METRIC_COLS:
        vals = [
            m[metric]
            for m in metrics_list
            if m.get(metric) is not None and not np.isnan(m[metric])
        ]
        if vals:
            result[f"{metric}_mean"] = float(np.mean(vals))
            result[f"{metric}_std"] = float(
                np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            )
        else:
            result[f"{metric}_mean"] = float("nan")
            result[f"{metric}_std"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """McNemar's test with continuity correction.

    Thin wrapper around utils.mcnemar_test that returns just the p-value
    for use in the pairwise sweep loop below.

    Args:
        correct_a: Boolean array of per-image correctness for model A.
        correct_b: Boolean array of per-image correctness for model B.

    Returns:
        p-value (two-sided).
    """
    if int(np.sum(correct_a & ~correct_b)) + int(np.sum(~correct_a & correct_b)) == 0:
        return 1.0
    return _mcnemar_canonical(correct_a, correct_b)["p_value"]


def run_mcnemar_pairs(
    condition_dfs: Dict[str, List[pd.DataFrame]],
    n_comparisons: int,
) -> Dict[Tuple[str, str], float]:
    """Run McNemar tests between all condition pairs (pooled across seeds).

    Pools per-image correct/incorrect across all seeds for each condition,
    then runs McNemar on the pooled vectors with Bonferroni correction.

    Args:
        condition_dfs: Dict mapping condition name to list of DataFrames.
        n_comparisons: Number of comparisons for Bonferroni correction.

    Returns:
        Dict mapping (condition_a, condition_b) -> corrected p-value.
    """
    pooled: Dict[str, np.ndarray] = {}
    for cond, dfs in condition_dfs.items():
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        if "correct" not in combined.columns:
            continue
        pooled[cond] = combined["correct"].values.astype(bool)

    results = {}
    pairs = list(combinations(pooled.keys(), 2))
    for a, b in pairs:
        vec_a = pooled[a]
        vec_b = pooled[b]
        min_len = min(len(vec_a), len(vec_b))
        p_raw = mcnemar_test(vec_a[:min_len], vec_b[:min_len])
        p_corrected = min(1.0, p_raw * n_comparisons)
        results[(a, b)] = p_corrected

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def print_summary_table(rows: List[Dict], mcnemar_results: Dict) -> None:
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 95)
    print(
        f"{'Condition':<22} {'Arch':<14} {'Seeds':>5} "
        f"{'PR-AUC':>14} {'Recall':>14} {'F1':>14} {'Pool FP':>10}"
    )
    print("-" * 95)
    for row in rows:
        if not row:
            continue
        cond = row.get("condition", "")
        arch = row.get("arch", "")
        n = row.get("n_seeds", 0)
        pr = f"{row.get('pr_auc_mean', float('nan')):.3f}±{row.get('pr_auc_std', float('nan')):.3f}"
        rec = f"{row.get('recall_mean', float('nan')):.3f}±{row.get('recall_std', float('nan')):.3f}"
        f1 = f"{row.get('f1_mean', float('nan')):.3f}±{row.get('f1_std', float('nan')):.3f}"
        pfp = f"{row.get('pool_fp_rate_mean', float('nan')):.3f}±{row.get('pool_fp_rate_std', float('nan')):.3f}"
        print(f"{cond:<22} {arch:<14} {n:>5} {pr:>14} {rec:>14} {f1:>14} {pfp:>10}")

    if mcnemar_results:
        print("\nMcNemar pairwise tests (Bonferroni-corrected p-values):")
        print("-" * 60)
        for (a, b), p in sorted(mcnemar_results.items()):
            sig = "  *" if p < 0.05 else ""
            print(f"  {a} vs {b}: p = {p:.4f}{sig}")
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed evaluation results into a unified comparison table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline_bce",
        type=str,
        default=None,
        help="Glob pattern for baseline BCE seed prediction CSVs (quote to prevent shell expansion).",
    )
    parser.add_argument(
        "--baseline_focal",
        type=str,
        default=None,
        help="Glob pattern for baseline focal loss seed prediction CSVs.",
    )
    parser.add_argument(
        "--hnm_bce",
        type=str,
        default=None,
        help="Glob pattern for HNM BCE seed prediction CSVs.",
    )
    parser.add_argument(
        "--hnm_focal",
        type=str,
        default=None,
        help="Glob pattern for HNM focal loss seed prediction CSVs.",
    )
    parser.add_argument(
        "--no_injection",
        type=str,
        default=None,
        help="Glob pattern for extended-training control (no-injection) seed CSVs.",
    )
    parser.add_argument(
        "--random_inject",
        type=str,
        default=None,
        help="Glob pattern for random-injection control seed CSVs.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="efficientnet",
        help="Architecture label for the output table.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables/seed_aggregation.csv",
        help="Output CSV path for the unified comparison table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    condition_patterns = {
        "baseline_bce":   args.baseline_bce,
        "baseline_focal": args.baseline_focal,
        "hnm_bce":        args.hnm_bce,
        "hnm_focal":      args.hnm_focal,
        "no_injection":   args.no_injection,
        "random_inject":  args.random_inject,
    }

    # Drop conditions with no pattern provided.
    condition_patterns = {k: v for k, v in condition_patterns.items() if v is not None}

    if not condition_patterns:
        print("[ERROR] No condition patterns provided. Pass at least --baseline_bce.")
        sys.exit(1)

    print("\nLoading seed CSVs...")
    condition_metrics: Dict[str, List[Dict]] = {}
    condition_dfs: Dict[str, List[pd.DataFrame]] = {}

    for cond, pattern in condition_patterns.items():
        metrics_list, df_list = load_condition(pattern, cond)
        condition_metrics[cond] = metrics_list
        condition_dfs[cond] = df_list

    print("\nAggregating metrics across seeds...")
    rows = []
    for cond, metrics_list in condition_metrics.items():
        if not metrics_list:
            print(f"  [WARN] No data for condition '{cond}', skipping.")
            continue
        row = aggregate_metrics(metrics_list, cond, args.arch)
        rows.append(row)
        print(f"  {cond}: {len(metrics_list)} seeds loaded.")

    # McNemar tests — Bonferroni over number of active conditions minus 1.
    n_comparisons = max(1, len(condition_metrics) - 1)
    mcnemar_results = run_mcnemar_pairs(condition_dfs, n_comparisons)

    # Attach McNemar p-value vs baseline_bce to each row.
    for row in rows:
        cond = row.get("condition", "")
        if cond == "baseline_bce":
            row["mcnemar_vs_baseline_bce_p"] = float("nan")
            continue
        p = mcnemar_results.get(
            ("baseline_bce", cond),
            mcnemar_results.get((cond, "baseline_bce"), float("nan")),
        )
        row["mcnemar_vs_baseline_bce_p"] = p

    print_summary_table(rows, mcnemar_results)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nUnified comparison table saved to: {args.output}")


if __name__ == "__main__":
    main()
