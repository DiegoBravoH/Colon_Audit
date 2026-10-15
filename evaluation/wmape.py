"""Withdrawal-time error (WMAPE) for the Multi-frame / Multi-frame+HMM model (window = 588).

Reads the released frame-level test predictions (frame_predictions_soft_vote_val_mcc_hmm.json,
one per fold, downloadable from docs/models.md) and prints the WMAPE reported in the README:

    WMAPE = 1/N * sum_i |A_i - P_i| / A_i * 100

where A_i and P_i are the actual and predicted number of withdrawal frames (cecum, ileum,
ascending, transverse, descending, sigmoid and rectum) in video i, and N is the number of
videos. Test frames are pooled over all folds of the protocol first, so every one of the 60
videos contributes exactly once per protocol.

Expected layout under --pred_root (same as confusion_matrix.py and the download links in
docs/models.md):

    <pred_root>/<protocol>/fold<N>/588/frame_predictions_soft_vote_val_mcc_hmm.json

Run:
    python evaluation/wmape.py --pred_root <pred_root>

Optional: --per_video_csv <path> also writes the per-video actual/predicted counts and errors.
"""
import os
import argparse

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Argument Parsing
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute withdrawal-time WMAPE from the released frame-level predictions."
    )
    parser.add_argument("--pred_root",     type=str, required=True,
                        help="Folder holding <protocol>/fold<N>/<window>/frame_predictions_soft_vote_val_mcc_hmm.json")
    parser.add_argument("--window",        type=int, default=588)
    parser.add_argument("--per_video_csv", type=str, default=None,
                        help="Optional path to save per-video counts and errors")
    return parser.parse_args()


# --------------------------------------------------
# 2. Constants
# --------------------------------------------------

# Label indices: 0 outside, 1 insertion, 2 cecum, 3 ileum, 4 ascending,
# 5 transverse, 6 descending, 7 sigmoid, 8 rectum.
WITHDRAWAL_LABELS = list(range(2, 9))
PROTOCOLS = {"4fold": 4, "5fold": 5}
PRED_FILE = "frame_predictions_soft_vote_val_mcc_hmm.json"
# (column in PRED_FILE, model name)
MODELS = [("PredictedClass", "Multi-frame"),
          ("PredictedClass_HMM", "Multi-frame+HMM")]


# --------------------------------------------------
# 3. Load + WMAPE
# --------------------------------------------------

def load_protocol(pred_root, protocol, n_folds, window):
    """All test frames of every fold of `protocol`, pooled into one DataFrame."""
    dfs = []
    for fold in range(1, n_folds + 1):
        path = os.path.join(pred_root, protocol, f"fold{fold}", str(window), PRED_FILE)
        dfs.append(pd.read_json(path)[["patient", "frame", "anatomy"] + [c for c, _ in MODELS]])
    return pd.concat(dfs, ignore_index=True)


def per_video_errors(df, pred_col):
    """Actual/predicted withdrawal-frame counts and absolute percentage error per video."""
    rows = []
    for patient, g in df.groupby("patient"):
        actual = int(g["anatomy"].isin(WITHDRAWAL_LABELS).sum())
        predicted = int(g[pred_col].isin(WITHDRAWAL_LABELS).sum())
        if actual == 0:  # undefined denominator -- does not occur in REAL-Colon
            print(f"  warning: {patient} has no ground-truth withdrawal frames, skipped")
            continue
        rows.append({"patient": patient, "actual": actual, "predicted": predicted,
                     "abs_pct_error": abs(actual - predicted) / actual * 100})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    summary, per_video = [], []

    for protocol, n_folds in PROTOCOLS.items():
        df = load_protocol(args.pred_root, protocol, n_folds, args.window)
        for pred_col, model in MODELS:
            errors = per_video_errors(df, pred_col)
            summary.append({"protocol": protocol, "model": model, "videos": len(errors),
                            "WMAPE": errors["abs_pct_error"].mean(),
                            "median": errors["abs_pct_error"].median()})
            per_video.append(errors.assign(protocol=protocol, model=model))

    table = pd.DataFrame(summary).pivot(index="model", columns="protocol", values="WMAPE")
    print("WMAPE (%), mean over videos:")
    print(table[list(PROTOCOLS)].round(1).to_string())
    print("\nDetail:")
    print(pd.DataFrame(summary).round(2).to_string(index=False))

    if args.per_video_csv:
        pd.concat(per_video, ignore_index=True).to_csv(args.per_video_csv, index=False)
        print(f"\nPer-video errors -> {args.per_video_csv}")


if __name__ == "__main__":
    main()
