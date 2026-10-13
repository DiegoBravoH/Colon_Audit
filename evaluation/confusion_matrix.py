"""Confusion matrices for the Multi-frame / Multi-frame+HMM model (window = 588).

Reads the released frame-level test predictions (frame_predictions_soft_vote_val_mcc_hmm.json,
one per fold, downloadable from docs/models.md) and writes one figure per protocol with two
row-normalized confusion matrices side by side: without and with HMM smoothing. Test frames
are pooled over all folds of the protocol, the same aggregation used for the per-class F1 in
Table 1. Each cell shows the percentage of that ground-truth class and, below it, the frame
count over the class total.

Expected layout under --pred_root (same as the download links in docs/models.md):

    <pred_root>/
    ├── 5fold/fold1/588/frame_predictions_soft_vote_val_mcc_hmm.json
    ├── ...
    ├── 5fold/fold5/588/frame_predictions_soft_vote_val_mcc_hmm.json
    ├── 4fold/fold1/588/frame_predictions_soft_vote_val_mcc_hmm.json
    ├── ...
    └── 4fold/fold4/588/frame_predictions_soft_vote_val_mcc_hmm.json

Run:
    python evaluation/confusion_matrix.py --pred_root <pred_root> --output_dir figures

Output: <output_dir>/confusion_matrix_5fold.png and <output_dir>/confusion_matrix_4fold.png
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 1. Argument Parsing
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot row-normalized confusion matrices from the released frame-level predictions."
    )
    parser.add_argument("--pred_root",  type=str, required=True,
                        help="Folder holding <protocol>/fold<N>/<window>/frame_predictions_soft_vote_val_mcc_hmm.json")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--window",     type=int, default=588)
    return parser.parse_args()


# --------------------------------------------------
# 2. Constants
# --------------------------------------------------

CLASS_NAMES = ["Outside", "Insertion", "Cecum", "Ileum", "Ascending",
               "Transverse", "Descending", "Sigmoid", "Rectum"]
PROTOCOLS = {
    "5fold": (5, "5-fold protocol"),
    "4fold": (4, "Cross-center 4-fold protocol"),
}
PRED_FILE = "frame_predictions_soft_vote_val_mcc_hmm.json"
# (column in PRED_FILE, panel title)
PANELS = [("PredictedClass", "Multi-frame"),
          ("PredictedClass_HMM", "Multi-frame + HMM")]

# Single-hue sequential scale (white -> dark blue): 0% recedes into the background.
CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, MUTED = "#1f1f1f", "#5f5f5f"


def fmt_count(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


# --------------------------------------------------
# 3. Load + Plot
# --------------------------------------------------

def load_protocol(pred_root, protocol, n_folds, window):
    """All test frames of every fold of `protocol`, pooled into one DataFrame."""
    dfs = []
    for fold in range(1, n_folds + 1):
        path = os.path.join(pred_root, protocol, f"fold{fold}", str(window), PRED_FILE)
        dfs.append(pd.read_json(path)[["patient", "frame", "anatomy"] + [c for c, _ in PANELS]])
    return pd.concat(dfs, ignore_index=True)


def plot_matrix(ax, y_true, y_pred, title):
    labels = range(len(CLASS_NAMES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1)
    cm_pct = cm / np.maximum(row_totals[:, None], 1) * 100

    im = ax.imshow(cm_pct, cmap=CMAP, vmin=0, vmax=100)
    for i in labels:
        for j in labels:
            pct = cm_pct[i, j]
            if pct < 0.5:  # leave near-empty cells blank instead of printing "0%"
                continue
            color = "white" if pct >= 50 else INK
            ax.text(j, i - 0.13, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"{fmt_count(cm[i, j])}/{fmt_count(row_totals[i])}",
                    ha="center", va="center", fontsize=6.5, color=color)

    ax.set_xticks(labels, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(labels, CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    # 1.5px white gap between cells
    ax.set_xticks(np.arange(-.5, len(CLASS_NAMES)), minor=True)
    ax.set_yticks(np.arange(-.5, len(CLASS_NAMES)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

    for protocol, (n_folds, protocol_title) in PROTOCOLS.items():
        df = load_protocol(args.pred_root, protocol, n_folds, args.window)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        for ax, (col, title) in zip(axes, PANELS):
            im = plot_matrix(ax, df["anatomy"], df[col], title)

        cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01)
        cb.set_label("% of ground-truth frames (row-normalized)", color=MUTED)
        cb.outline.set_visible(False)
        fig.suptitle(f"Colon-Audit confusion matrices — {protocol_title} "
                     f"(T = {args.window}, 5 fps, test frames pooled over {n_folds} folds)", fontsize=12)

        out_path = os.path.join(args.output_dir, f"confusion_matrix_{protocol}.png")
        fig.savefig(out_path, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        print(f"{protocol}: {len(df)} frames -> {out_path}")


if __name__ == "__main__":
    main()
