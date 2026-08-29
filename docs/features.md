# Frame-Level Feature Extraction

Encodes every video frame independently through a frozen, pretrained ViT-Small backbone (DINOv1, GastroNet-5M weights — see paper Section 3.1), producing the 384-dimensional embeddings that everything downstream (`../classification/multi_frame/`, `../classification/single_frame/`) is built on. Code: [`../features/features_ViT_Small.py`](../features/features_ViT_Small.py).

**Verified working** (tested end-to-end on a 10-frame sample; test output was discarded, not committed).

## Requirements

- The REAL-Colon dataset downloaded and extracted (see [`data.md`](data.md)) — this script reads raw frame images, not precomputed features.
- The GastroNet-5M ViT-Small (DINOv1) pretrained checkpoint. These are third-party weights (Boers et al. 2024; Jong et al. 2025 — see main [README](../README.md#citation)) — obtain them from the GastroNet-5M authors; they aren't redistributed here.

## Input format

A single manifest CSV (`--dataframe`) listing every frame to extract, with columns:

| Column | Example | Meaning |
|---|---|---|
| `patient` | `004-006` | Video/patient ID |
| `filename` | `004-006_0.jpg` | Frame image filename, expected at `<data_path>/<patient>_frames/<filename>` |
| `pt_name` | `000000.pt` | Legacy naming: `f"{frame_number:06d}.pt"` — only its numeric stem is used (to recover `frame_number`); no `.pt` file is actually read. |
| `label` | `outside` | Not read by this script — carried over from the annotation source, harmless if present. |

## Usage

```bash
conda activate colon-audit  # see libraries.md
cd features/

python features_ViT_Small.py \
    --weights_path /path/to/VITS_GastroNet-5M_DINOv1.pth \
    --output_dir   ./GastroNet-5M/ViT-Small/original \
    --dataframe    /path/to/labels.csv \
    --data_path    /path/to/REAL-Colon \
    --batch_size   64 \
    --num_workers  8 \
    --device       cuda:0 \
    --augmentation original
```

`--output_dir`'s last path component must match `--augmentation` (a safety guard against accidentally overwriting a different variant's output — see the script's docstring). `original` is the only variant `../classification/multi_frame/` and `../classification/single_frame/` consume; the other choices (`flip-h`, `flip-v`, `rotation-90/180/270`) were a separate, unrelated spatial-augmentation ablation.

## Output

One folder per patient under `--output_dir`:

```
<output_dir>/<patient>/
├── features.npy        # [N_frames, 384] float32
└── frame_numbers.npy   # [N_frames] int64 — the real frame_number of each row (not necessarily contiguous)
```

`../classification/multi_frame/training/12_train_5fps_offset-aug.py` and `../classification/multi_frame/testing/predict.py` expect this output at `./GastroNet-5M/ViT-Small/original/` relative to this folder (i.e. `features/GastroNet-5M/ViT-Small/original/`) by default.
