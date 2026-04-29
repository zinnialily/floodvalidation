"""
plot_confounder_fp.py — Generate Figure 5: per-category FP bar chart with
Clopper-Pearson 95% confidence intervals across baseline and HNM conditions.

Usage:
    python scripts/plot_confounder_fp.py
    python scripts/plot_confounder_fp.py --results_dir ./results --threshold 0.05

Output:
    <results_dir>/figures/confounder_fp_comparison.pdf
    <results_dir>/figures/confounder_fp_comparison.png
    <results_dir>/figures/confounder_fp_heatmap.pdf
    <results_dir>/figures/confounder_fp_heatmap.png
"""

import argparse
import os
import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import clopper_pearson_ci  # noqa: E402


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot per-category confounder FP rates with Clopper-Pearson CIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results_dir",
        default="./results",
        help="Root results directory. CSV files are read from "
             "<results_dir>/tables/confounders/ and "
             "<results_dir>/confounder_analysis/*/tables/.",
    )
    p.add_argument(
        "--threshold",
        default=0.05,
        type=float,
        help="HNM flagging threshold θ shown as a dashed line on the bar chart.",
    )
    return p.parse_args()


# ── Build conditions dict from results_dir ────────────────────────────────────

def build_conditions(results_dir: str) -> dict:
    """Return an ordered dict mapping label -> CSV path for each condition.

    All paths are relative to results_dir so the script works regardless of
    where the repository is cloned.
    """
    rd = pathlib.Path(results_dir)
    conf = rd / "tables" / "confounders"
    ablation = rd / "confounder_analysis"
    return {
        "EffNet BCE":           str(conf / "confounder_fp_rates_efficientnet_VAL.csv"),
        "ResNet BCE":           str(conf / "confounder_fp_rates_resnet50_VAL.csv"),
        "EffNet Ext. Training": str(ablation / "extended_baseline" / "tables" /
                                    "confounder_fp_rates_efficientnet_VAL.csv"),
        "EffNet Rand. Inject":  str(ablation / "random_inject" / "tables" /
                                    "confounder_fp_rates_efficientnet_VAL.csv"),
        "EffNet P1-HNM \u2605": str(conf / "confounder_fp_rates_efficientnet_p1hnm_VAL.csv"),
    }


# ── Load all conditions ───────────────────────────────────────────────────────

def load_all(conditions: dict) -> dict:
    frames = {}
    for label, path in conditions.items():
        p = pathlib.Path(path)
        if not p.exists():
            print(f"  [WARN] missing: {p}")
            continue
        df = pd.read_csv(p)
        df = df.set_index("category")
        # Ensure n_fp column exists (old files use different schema)
        if "n_fp" not in df.columns and "fp_rate" in df.columns:
            df["n_fp"] = (df["fp_rate"] * df["n_images"]).round().astype(int)
        frames[label] = df
    return frames


# ── Build river-only summary ──────────────────────────────────────────────────

def build_river_summary(frames: dict) -> pd.DataFrame:
    rows = []
    for label, df in frames.items():
        if "river" not in df.index:
            continue
        row = df.loc["river"]
        n   = int(row["n_images"])
        k   = int(row["n_fp"])
        lo, hi = clopper_pearson_ci(k, n)
        rows.append({
            "condition": label,
            "n":         n,
            "k":         k,
            "fp_rate":   k / n,
            "ci_lo":     lo,
            "ci_hi":     hi,
        })
    return pd.DataFrame(rows)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_river_fp(summary: pd.DataFrame, output_dir: pathlib.Path, hnm_threshold: float) -> None:
    n_cond = len(summary)
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [
        "#3498db",   # EffNet BCE
        "#95a5a6",   # ResNet BCE
        "#f39c12",   # EffNet Ext. Training
        "#9b59b6",   # EffNet Rand. Inject
        "#e74c3c",   # EffNet P1-HNM
    ][:n_cond]

    x = np.arange(n_cond)
    bars = ax.bar(
        x,
        summary["fp_rate"] * 100,
        color=colors,
        width=0.6,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    # Error bars (Clopper-Pearson)
    yerr_lo = (summary["fp_rate"] - summary["ci_lo"]) * 100
    yerr_hi = (summary["ci_hi"] - summary["fp_rate"]) * 100
    ax.errorbar(
        x,
        summary["fp_rate"] * 100,
        yerr=[yerr_lo.values, yerr_hi.values],
        fmt="none",
        color="black",
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        zorder=4,
    )

    # HNM flagging threshold line
    ax.axhline(
        y=hnm_threshold * 100,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=r"HNM flagging threshold $\theta = " + f"{hnm_threshold:.0%}" + r"$",
        zorder=2,
    )

    # Annotate counts above bars
    for bar, row in zip(bars, summary.itertuples()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + row.ci_hi * 100 - row.fp_rate * 100 + 0.4,
            f"{row.k}/{row.n}",
            ha="center", va="bottom",
            fontsize=9, color="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["condition"], rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("River FP Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "River False Positive Rate by Training Condition\n"
        "with Clopper–Pearson 95% Binomial Confidence Intervals "
        r"($N_{\mathrm{val}}$ = 76)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(summary["ci_hi"].max() * 100 + 5, hnm_threshold * 100 + 8))
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = output_dir / f"confounder_fp_comparison.{ext}"
        dpi = 300 if ext == "png" else None
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close()


# ── All-category heatmap (supplementary) ─────────────────────────────────────

def plot_all_categories(frames: dict, output_dir: pathlib.Path) -> None:
    """Generate a heatmap of FP rates for all categories × all conditions."""
    categories = [
        "river", "swimming_pool", "lake", "fountain",
        "building", "park_walkway", "street_clear",
    ]
    labels = list(frames.keys())
    matrix = np.full((len(categories), len(labels)), np.nan)

    for j, (label, df) in enumerate(frames.items()):
        for i, cat in enumerate(categories):
            if cat in df.index:
                matrix[i, j] = df.loc[cat, "fp_rate"] * 100

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    cmap = plt.get_cmap("Reds")
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(
        np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=20, aspect="auto"
    )
    plt.colorbar(im, ax=ax, label="FP rate (%)")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=10)

    # Annotate cells
    for i in range(len(categories)):
        for j in range(len(labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if val > 10 else "black")

    ax.set_title(
        "Per-Category False Positive Rate (%) by Training Condition\n"
        "(Val set, N=497, seed 42, decision threshold τ = 0.5)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    for ext in ("pdf", "png"):
        out = output_dir / f"confounder_fp_heatmap.{ext}"
        dpi = 300 if ext == "png" else None
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")

    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = pathlib.Path(args.results_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = build_conditions(args.results_dir)

    print("Loading confounder FP rate CSVs...")
    frames = load_all(conditions)
    print(f"Loaded {len(frames)} conditions: {list(frames.keys())}")

    summary = build_river_summary(frames)
    print("\nRiver FP rate summary:")
    print(summary.to_string(index=False))

    print("\nGenerating Figure 5: river FP bar chart...")
    plot_river_fp(summary, output_dir, args.threshold)

    print("\nGenerating supplementary all-category heatmap...")
    plot_all_categories(frames, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
