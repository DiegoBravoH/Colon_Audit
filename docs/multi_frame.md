# Multi-Frame Classification (Main Method)

Code: `classification/multi_frame/` — **embargoed until the paper is accepted**, this document describes what it does and how it's used ahead of the code itself being published. Pretrained checkpoints are already available, see [`models.md`](models.md).

This is the **"Multi-frame" / "Multi-frame+HMM" rows of Tables 1 and 2** in the paper — the temporal ViT-Small model (sequence-to-sequence over sliding windows of frame embeddings), the framework's main contribution. Compare against `classification/single_frame/` (see [`classification.md`](classification.md)), which has no temporal context at all.

**Verified working** (tested end-to-end — training with real warm-up + fine-tuning + early stopping, inference, frame-level soft-voting, and HMM decoding — on a small real subset of frames/patients/one window size; test output was discarded, not committed).

## Requirements

- ViT-Small embeddings already extracted (see [`features.md`](features.md)). Only needed for **training** from scratch or re-running **inference** — not needed just to look at the pretrained checkpoints.
- Precomputed sequence windows (train/valid/test JSON, ~2.3 GB, Google Drive) — download details live in `classification/multi_frame/experimental_setup/PRECOMPUTED_DATA.md`, embargoed with the rest of the code until the paper is accepted. Only needed for training/inference, same as above.
- The precomputed HMM transition matrices (`classification/multi_frame/hmm/transition_matrices_5fps/`, ~192 KB) — bundled alongside the code, so embargoed with it until the paper is accepted too (only needed for the optional HMM step below).
- The pretrained checkpoints — see [`models.md`](models.md) for the download link, if you want to skip training entirely.

## Pipeline (4 stages, in order)

### 1. Training — `multi_frame/training/12_train_5fps_offset-aug.py`

Trains the temporal ViT-Small model for every `(protocol, fold, temporal_window)` combination: a 10-epoch warm-up (projection + classification heads only) followed by partial fine-tuning (last 3 transformer blocks + norm + head, up to 100 epochs, early stopping on validation MCC, patience 5).

```bash
conda activate colon-audit
cd classification/multi_frame/training
python 12_train_5fps_offset-aug.py --loss weighted_ce
```

**`--loss weighted_ce` is required** to reproduce the paper — the script's own default (`focal`) was used during loss-function comparisons in development, not what's reported in Tables 1/2. Passing it changes the checkpoint output path (`.../weighted_ce/...` vs `.../focal_g2.0/...`), so getting it wrong silently trains a different, unreported configuration rather than erroring.

Output: `<output_dir>/best-model-{val_mcc,val_f1_macro,val_wf1_paper}.ckpt` per `(protocol, fold, window)`. `output_dir` (top of the script) defaults to a path outside the repo (too large to commit — 5.9 GB just for `val_mcc`, see [`models.md`](models.md) for the pretrained download instead); edit it to your own output location.

### 2. Inference — `multi_frame/testing/predict.py`

Loads each `(protocol, fold, window)` checkpoint and runs causal sliding-window inference on that fold's test split. `MODEL_ROOT` (top of the script) defaults to a path outside the repo — point it at wherever you downloaded/placed the pretrained checkpoints (see [`models.md`](models.md)) or at your own training output.

```bash
cd ../testing
python predict.py
```

Output: `window_level_predictions_raw_{checkpoint_tag}.json` — one row per window (not yet frame-level).

### 3. Frame-level soft-voting — `multi_frame/testing/flatten_and_metrics.py`

Expands window-level predictions to frame-level and averages the class probabilities of every window that covers a given frame (soft voting) — this is what makes the causal, sliding-window model produce one stable prediction per frame. Also computes classification metrics directly.

```bash
python flatten_and_metrics.py
# or scope a single run, e.g.: --protocols 5fold --folds 1 --windows 16
```

Output: `frame_predictions_soft_vote_{checkpoint_tag}.json` plus a printed metrics table.

### 4. HMM smoothing (optional) — `multi_frame/testing/apply_hmm.py`

Decodes each patient's frame-by-frame soft-voted predictions with Viterbi, using the precomputed transition matrices bundled in `../hmm/`. This produces the "+HMM" variant.

```bash
python apply_hmm.py
```

Output: `frame_predictions_soft_vote_{checkpoint_tag}_hmm.json`, adds a `PredictedClass_HMM` column.

## Notes

- Only the **12_5fps_offset-aug** setup (the winning one) has training code and precomputed windows prepared — two other setups were explored and compared against it in the paper's ablation (a plain 5FPS grid without offset augmentation, and a native-fps grid), but only the winning one will be published (once the code embargo lifts, see the note at the top of this document).
- `multi_frame/testing/compare_to_baseline.py` (not yet verified against this cleaned-up code) computes the final Table 1/2 comparison against the ColonTCN baseline — same role `classification/single_frame/testing/metrics.ipynb` plays for the single-frame baseline.
