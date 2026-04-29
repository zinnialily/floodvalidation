"""
generate_results.py — Generate figures and summary table for a trained baseline model.

Usage:
    python scripts/generate_results.py --arch efficientnet
    python scripts/generate_results.py --arch resnet50
"""

import argparse
import os
import pathlib
import sys
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve,
    roc_curve, auc, classification_report,
)
from PIL import Image
from utils import PREPROCESS_FN

IMG_SIZE  = (224, 224)
BATCH     = 32


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--arch", required=True, choices=["efficientnet", "resnet50"])
    p.add_argument("--ckpt",       default=None, help="Override checkpoint path")
    p.add_argument("--log_csv",    default=None, help="Override log CSV path")
    p.add_argument("--results_dir",default="./results")
    p.add_argument("--loss_label", default="bce", choices=["bce", "focal"])
    p.add_argument("--seed",       default=42, type=int)
    p.add_argument("--val_dir",
                   default="data/FloodingDataset2/processed_data/binary/val",
                   help="Path to the validation split directory (binary/val/).")
    p.add_argument("--subdir",     default="baselines",
                   help="Subfolder under results/figures/ and results/tables/. "
                        "Use 'hnm' for HNM models.")
    p.add_argument("--condition",  default="",
                   help="Optional suffix to distinguish runs with same arch+loss "
                        "(e.g. 'no_injection', 'random_injection').")
    return p.parse_args()


def resolve_paths(args):
    models_dir = pathlib.Path("models")
    logs_dir   = pathlib.Path(args.results_dir) / "logs"
    tag = f"{args.arch}_{args.loss_label}"

    ckpt = pathlib.Path(args.ckpt) if args.ckpt else sorted(
        models_dir.glob(f"{args.arch}_{args.loss_label}_phase2_*.keras") or
        models_dir.glob(f"{args.arch}_{args.loss_label}_phase1_*.keras")
    )[-1]

    log_csv = pathlib.Path(args.log_csv) if args.log_csv else sorted(
        list(logs_dir.glob(f"{args.arch}_baseline_{args.loss_label}_phase2_*.csv")) or
        list(logs_dir.glob(f"{args.arch}_baseline_{args.loss_label}_phase1_*.csv"))
    )[-1]

    fig_dir   = pathlib.Path(args.results_dir) / "figures" / args.subdir / tag
    table_dir = pathlib.Path(args.results_dir) / "tables"  / args.subdir / args.arch
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    return ckpt, log_csv, fig_dir, table_dir, tag


def run_inference(model, val_dir: str, arch: str):
    gen = ImageDataGenerator(
        preprocessing_function=PREPROCESS_FN[arch],
    ).flow_from_directory(
        val_dir, target_size=IMG_SIZE, batch_size=BATCH,
        class_mode='binary', shuffle=False,
    )
    print(f"Val samples: {gen.samples}  |  class_indices: {gen.class_indices}")
    y_prob = model.predict(gen, verbose=0).flatten()
    return y_prob, gen.classes, gen.filenames, gen.class_indices


def plot_training_curves(df, fig_dir, arch, tag):
    epochs = df['epoch'] + 1
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f'{arch.upper()} Baseline — Per-Epoch Training Curves\n'
        f'(Clean deduplicated split, {tag.split("_")[1].upper()} loss, seed 42)',
        fontsize=13, fontweight='bold',
    )
    metrics = [
        ('loss',     'val_loss',     'Loss',     '#e74c3c', '#c0392b'),
        ('auc',      'val_auc',      'AUC',      '#3498db', '#2980b9'),
        ('accuracy', 'val_accuracy', 'Accuracy', '#2ecc71', '#27ae60'),
        ('recall',   'val_recall',   'Recall',   '#9b59b6', '#8e44ad'),
    ]
    for ax, (tr_col, vl_col, title, tc, vc) in zip(axes.flat, metrics):
        ax.plot(epochs, df[tr_col],  'o-',  color=tc, label='Train',      lw=2, ms=5)
        ax.plot(epochs, df[vl_col],  's--', color=vc, label='Validation', lw=2, ms=5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, len(epochs) + 0.5)
        if title not in ('Loss',):
            ax.set_ylim(max(0, df[[tr_col, vl_col]].min().min() - 0.05), 1.02)
    plt.tight_layout()
    out = fig_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def plot_confusion_matrix(cm, n_val, n_epochs, fig_dir, arch):
    TP, FN, FP, TN = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = LinearSegmentedColormap.from_list('flood', ['#ffffff', '#2980b9'])
    im = ax.imshow(cm, cmap=cmap)
    labels = ['Flood (Positive)', 'Non-Flood (Negative)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label',      fontsize=12, fontweight='bold')
    ax.set_title(
        f'{arch.upper()} — Confusion Matrix (Val, n={n_val})\nEpoch {n_epochs}',
        fontsize=12, fontweight='bold',
    )
    cell_labels = [[f'TP\n{TP}', f'FN\n{FN}'], [f'FP\n{FP}', f'TN\n{TN}']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else 'black'
            ax.text(j, i, cell_labels[i][j], ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out = fig_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return TP, FP, FN, TN


def plot_pr_roc(y_true_flood, flood_prob, TP, FP, FN, TN, fig_dir, arch):
    prec, rec, _ = precision_recall_curve(y_true_flood, flood_prob)
    pr_auc = auc(rec, prec)
    fpr, tpr, _ = roc_curve(y_true_flood, flood_prob)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{arch.upper()} Baseline — PR & ROC Curves (Val Set)',
                 fontsize=13, fontweight='bold')

    ax1.plot(rec, prec, lw=2.5, color='#e74c3c', label=f'PR AUC={pr_auc:.4f}')
    ax1.axhline(y=np.mean(y_true_flood), color='grey', ls='--', lw=1.5, label='Random')
    ax1.fill_between(rec, prec, alpha=0.15, color='#e74c3c')
    ax1.set_xlabel('Recall (Flood)', fontsize=12)
    ax1.set_ylabel('Precision (Flood)', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontweight='bold')
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
    ax1.annotate(f'TP={TP}  FP={FP}\nFN={FN}  TN={TN}',
                 xy=(0.05, 0.08), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle='round', fc='lightyellow', ec='orange'))

    ax2.plot(fpr, tpr, lw=2.5, color='#3498db', label=f'ROC AUC={roc_auc:.4f}')
    ax2.plot([0, 1], [0, 1], color='grey', ls='--', lw=1.5, label='Random')
    ax2.fill_between(fpr, tpr, alpha=0.15, color='#3498db')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate',  fontsize=12)
    ax2.set_title('ROC Curve', fontweight='bold')
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "pr_roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return pr_auc, roc_auc


def plot_fp_fn_examples(y_true, y_pred, y_prob, filenames, fig_dir, arch, val_dir: str, n=8):
    def load_img(path):
        return np.array(Image.open(path).convert('RGB').resize((112, 112)))

    fp_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    fn_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fp_sorted = fp_idx[np.argsort(y_prob[fp_idx])][:n]
    fn_sorted = fn_idx[np.argsort(-y_prob[fn_idx])][:n]

    TP = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = len(fp_idx); FN = len(fn_idx)
    TN = int(np.sum((y_true == 1) & (y_pred == 1)))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0

    val_root = pathlib.Path(val_dir)
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f'{arch.upper()} Baseline — Val Set Worst Errors\n'
        f'TP={TP}  TN={TN}  FP={FP}  FN={FN}  '
        f'(Precision={precision:.3f}, Recall={recall:.3f})',
        fontsize=13, fontweight='bold',
    )
    gs = gridspec.GridSpec(2, n, hspace=0.5, wspace=0.15)

    for row, (indices, title, color, prob_dir) in enumerate([
        (fp_sorted, 'FP: Non-Flood → Flood',   '#e74c3c', 1),
        (fn_sorted, 'FN: Flood → Non-Flood',    '#f39c12', -1),
    ]):
        for col, idx in enumerate(indices):
            ax = fig.add_subplot(gs[row, col])
            try:
                ax.imshow(load_img(val_root / filenames[idx]))
            except Exception:
                ax.text(0.5, 0.5, '?', ha='center', va='center',
                        transform=ax.transAxes)
            p = y_prob[idx]
            cat = pathlib.Path(filenames[idx]).parts[0]
            ax.set_title(f'p={p:.2f}\n{cat}', fontsize=7, pad=2)
            for spine in ax.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=9, fontweight='bold', color=color)

    out = fig_dir / "fp_fn_examples.png"
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


def write_summary_table(df, report, cm, table_dir, arch, tag, seed, pr_auc=None, roc_auc=None):
    TP, FN, FP, TN = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    summary = df.copy()
    summary['epoch'] = summary['epoch'] + 1
    summary = summary.rename(columns={
        'epoch':'Epoch','loss':'Train Loss','val_loss':'Val Loss',
        'accuracy':'Train Acc','val_accuracy':'Val Acc',
        'auc':'Train AUC','val_auc':'Val AUC',
        'precision':'Train Prec','val_precision':'Val Prec',
        'recall':'Train Recall','val_recall':'Val Recall',
        'learning_rate':'LR',
    })
    cols = ['Epoch','LR','Train Loss','Val Loss','Train Acc','Val Acc',
            'Train AUC','Val AUC','Train Prec','Val Prec','Train Recall','Val Recall']
    summary = summary[cols]
    for c in cols[1:]:
        summary[c] = summary[c].map(lambda x: f'{x:.4f}')
    best_row = df['val_loss'].idxmin()
    summary.insert(0, '', ['⭐' if i == best_row else '' for i in summary.index])

    csv_path = table_dir / f"{arch}_{tag.split('_')[1]}_seed{seed}_epoch_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    md_lines = [
        f'# {arch.upper()} Baseline ({tag.split("_")[1].upper()}, Seed {seed}) — Epoch Summary',
        '',
        '**Dataset:** Clean deduplicated split (4,099 unique images, 55 dupes removed)',
        '',
        '## Per-Epoch Metrics',
        '',
        '| ' + ' | '.join(cols) + ' |',
        '|' + '|'.join(['---'] * len(cols)) + '|',
    ]
    for _, row in summary.iterrows():
        star = ' ⭐' if row[''] == '⭐' else ''
        md_lines.append('| ' + ' | '.join(str(row[c]) for c in cols) + f' |{star}')

    md_lines += [
        '',
        '## Val Set Performance (Best Checkpoint)',
        '',
        '| Metric | Flood | Non-Flood | Weighted Avg |',
        '|--------|-------|-----------|--------------|',
        f'| Precision | {report["Flood"]["precision"]:.4f} | {report["Non-Flood"]["precision"]:.4f} | {report["weighted avg"]["precision"]:.4f} |',
        f'| Recall    | {report["Flood"]["recall"]:.4f}    | {report["Non-Flood"]["recall"]:.4f}    | {report["weighted avg"]["recall"]:.4f} |',
        f'| F1-Score  | {report["Flood"]["f1-score"]:.4f}  | {report["Non-Flood"]["f1-score"]:.4f}  | {report["weighted avg"]["f1-score"]:.4f} |',
        '',
        f'**Confusion Matrix:** TP={TP} | FP={FP} | FN={FN} | TN={TN}',
        '',
        '## Discrimination (Post-hoc, sklearn)',
        '',
        f'| Metric | Value |',
        f'|--------|-------|',
        f'| PR-AUC | {pr_auc:.4f} |' if pr_auc is not None else '| PR-AUC | — |',
        f'| ROC-AUC | {roc_auc:.4f} |' if roc_auc is not None else '| ROC-AUC | — |',
        '',
        '## Figures',
        f'- `results/figures/baselines/{tag}/training_curves.png`',
        f'- `results/figures/baselines/{tag}/confusion_matrix.png`',
        f'- `results/figures/baselines/{tag}/pr_roc_curves.png`',
        f'- `results/figures/baselines/{tag}/fp_fn_examples.png`',
    ]
    md_path = table_dir / f"{arch}_{tag.split('_')[1]}_seed{seed}_epoch_summary.md"
    md_path.write_text('\n'.join(md_lines))
    print(f"Saved: {md_path}")


def main():
    args = parse_args()
    ckpt, log_csv, fig_dir, table_dir, tag = resolve_paths(args)
    print(f"Arch       : {args.arch}")
    print(f"Checkpoint : {ckpt}")
    print(f"Log CSV    : {log_csv}")

    print("Loading model...")
    model = tf.keras.models.load_model(str(ckpt))

    val_dir = os.path.abspath(args.val_dir)
    y_prob, y_true, filenames, class_indices = run_inference(model, val_dir, args.arch)
    y_pred = (y_prob >= 0.5).astype(int)

    df = pd.read_csv(log_csv)
    cm = confusion_matrix(y_true, y_pred)
    flood_prob = 1.0 - y_prob
    report = classification_report(y_true, y_pred,
                                   target_names=['Flood', 'Non-Flood'],
                                   output_dict=True)

    plot_training_curves(df, fig_dir, args.arch, tag)
    TP, FP, FN, TN = plot_confusion_matrix(cm, len(y_true), len(df), fig_dir, args.arch)
    pr_auc, roc_auc = plot_pr_roc(y_true == 0, flood_prob, TP, FP, FN, TN, fig_dir, args.arch)
    plot_fp_fn_examples(y_true, y_pred, y_prob, filenames, fig_dir, args.arch, val_dir)
    write_summary_table(df, report, cm, table_dir, args.arch, tag, args.seed,
                        pr_auc=pr_auc, roc_auc=roc_auc)
    print(f"\nPR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")

    print(f"\nAll results for {args.arch} saved to results/figures/{args.arch}/ and results/tables/{args.arch}/")


if __name__ == "__main__":
    main()
