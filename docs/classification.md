# Single-Frame Classification (Baseline)

Code: `classification/single_frame/` — **embargoed until the paper is accepted**, this document describes what it does and how it's used ahead of the code itself being published. Pretrained checkpoints are already available, see [`models.md`](models.md).

This is the **"Single-frame" row of Table 1** in the paper — a plain linear probe (`nn.Linear(384, 9)`) trained directly on the frozen ViT-Small embeddings, with no temporal context at all. It's the reference point that shows what the frame-level embedding alone gives you before any temporal modeling (`classification/multi_frame/`, reported as "Multi-frame" in the same table).

**Verified working** (tested end-to-end — training, inference, and HMM decoding — on a small real subset of frames/patients; test output was discarded, not committed).

## Requirements

- ViT-Small embeddings already extracted (see [`features.md`](features.md)) — this baseline reads the *same* `features.npy`/`frame_numbers.npy` files `classification/multi_frame/` does. Only needed for **training** your own probe or re-running **inference** from scratch — not needed just to look at the pretrained checkpoints.
- The `set_type_{5,4}fold_5fps.csv` fold-assignment files (same canonical 5fps grid used across this project) — see [`data.md`](data.md).
- The precomputed HMM transition matrices (`classification/multi_frame/hmm/transition_matrices_5fps/`) — bundled alongside the code, so embargoed with it until the paper is accepted (only needed for the optional HMM step below).
- The trained checkpoints themselves — see [`models.md`](models.md) for the download link.

## Pipeline (3 stages, in order)

### 1. Training — `single_frame/training/train_linear_probe.py`

Reads every labeled 5fps frame directly (no windowing — a single frame is already a full training example) and fits one linear probe per `(protocol, fold)`, with early stopping on validation MCC.

```bash
conda activate colon-audit
cd classification/single_frame/training
python train_linear_probe.py
```

Output: `<OUT_ROOT>/VITS_GastroNet-5M_DINOv1/<protocol>/fold<N>/linear_probe.pt`. `OUT_ROOT` (top of the script) defaults to a path outside the repo — same convention as `classification/multi_frame/`; edit it to your own output location. Only needed if you want to retrain from scratch — see [`models.md`](models.md) for the already-trained checkpoints.

### 2. Inference — `single_frame/testing/predict.py`

Loads each fold's checkpoint (`../checkpoints/` by default; edit `MODEL_ROOT` to point at wherever you downloaded the pretrained ones from [`models.md`](models.md), or at your own training output) and runs it on that fold's **test** split.

```bash
cd ../testing
python predict.py
```

Output: `test_predictions.json` next to the checkpoint — columns `patient`, `frame`, `anatomy` (ground truth), `PredictedClass`, `ClassProbabilities`. Already frame-level (no soft-voting/flattening needed, unlike the windowed `classification/multi_frame/`).

### 3. HMM smoothing (optional) — `single_frame/testing/apply_hmm.py`

Decodes each patient's frame-by-frame predictions with Viterbi, using the same precomputed transition matrices `classification/multi_frame/` uses (fit purely from ground-truth label sequences, independent of which model produced the predictions being decoded).

```bash
python apply_hmm.py
```

Output: `test_predictions_hmm.json`, same as step 2 plus a `PredictedClass_HMM` column. This is the "Single-frame+HMM" variant.

## Notes

- Only the **ViT-Small** embedding is included — the paper's Table 1 "Single-frame" row is ViT-Small only. An earlier ResNet50 comparison existed during development but isn't part of the published result, so its code path was removed for clarity.
- `single_frame/testing/metrics.ipynb` computes the actual Table 1 metrics from this pipeline's output — verified: re-run end-to-end against the real predictions, reproduces the paper's exact wF1 (52.9% 5-fold / 48.6% 4-fold for "Single-frame").
