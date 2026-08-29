# Pretrained Models

Trained checkpoints for both classification pipelines, released so results can be inspected/reproduced without retraining. Both cover the official REAL-Colon **5-fold** and cross-center **4-fold** protocols.

**Note:** the checkpoints below are shared now. The training/testing *code* that produced them (`classification/`) is embargoed until the paper is accepted — see [`classification.md`](classification.md) and [`multi_frame.md`](multi_frame.md) for what that code does and its exact usage, ahead of the code itself being published.

## Single-frame

The "Single-frame" baseline (Table 1): 9 linear-probe checkpoints (5-fold + 4-fold), tiny (~16 KB each, ~190 KB total).

🔗 **Google Drive**: *(link here — publicly shared folder)*

| Protocol | Folds | Checkpoint |
|---|---|---|
| 5-fold | 1, 2, 3, 4, 5 | `linear_probe.pt` |
| 4-fold | 1, 2, 3, 4 | `linear_probe.pt` |

One `ViT-Small` backbone, one `nn.Linear(384, 9)` each.

## Multi-frame

The "Multi-frame" / "Multi-frame+HMM" main method (Tables 1 and 2): temporal ViT-Small checkpoints, `val_mcc`-selected only (the only checkpoint-selection metric the pipeline uses — see [`multi_frame.md`](multi_frame.md)).

### Window = 588 (Table 1's headline result)

All 9 folds (5-fold + 4-fold) at the temporal window that produces Table 1's reported numbers — this is the released subset (see note below on why the full ablation isn't distributed this way).

**Setup** — precomputed sequence windows (train/valid/test JSON) for this `(protocol, fold, 588)` combination, needed if you want to run training or inference yourself:

| Protocol | Fold | Tokens | Windows |
|---|---|---|---|
| 4-fold | 1 | 588 | [train](https://drive.google.com/uc?id=1JLK4Sgn9vZB_tmLjwJgikEwEpJlTd2qL) · [valid](https://drive.google.com/uc?id=1h072Oce0FFdvVvEssLWtGUpfFrNLzWKG) · [test](https://drive.google.com/uc?id=1jaVe2b-c4eHq7NwBJOVejr0Ne6svVnEQ) |
| 4-fold | 2 | 588 | [train](https://drive.google.com/uc?id=1nK6s1CsSDAQkAgTeYcjn536T2qI053FM) · [valid](https://drive.google.com/uc?id=1G983l7IFqqo6B36ReMWiTI9OgjD8kOzL) · [test](https://drive.google.com/uc?id=1Y5z4tp9RgJ6M9ol0qPkY7XoAC0WqYpJV) |
| 4-fold | 3 | 588 | [train](https://drive.google.com/uc?id=14wZFfAD2lFXT_Jh2pD-n2sq8ApJO3u9m) · [valid](https://drive.google.com/uc?id=1L1c7gSYGVNFbUfCLHHwUgwNB1p1d24Zy) · [test](https://drive.google.com/uc?id=1-XCL7lkPlxqORC7mX19yjlLJSfJaRGrs) |
| 4-fold | 4 | 588 | [train](https://drive.google.com/uc?id=1q9OOS5njKcemuFFmVKzs_2q6NLIeczCc) · [valid](https://drive.google.com/uc?id=17mw_WmDiyhKk5nwgqnTCnVDlLQKhbpdx) · [test](https://drive.google.com/uc?id=1FuU07ZCGCoNoB9pXWpn4BahkP5yuEG3m) |
| 5-fold | 1 | 588 | [train](https://drive.google.com/uc?id=1-bLaTmdzwor-ETn7eNoALKTIncoaWv3L) · [valid](https://drive.google.com/uc?id=1wAXpE-9D6vQdn1SXFTd4UIYPW9EDV6Fc) · [test](https://drive.google.com/uc?id=1H3JMgUsXDItRv62LQ03UALexFeqPi8KC) |
| 5-fold | 2 | 588 | [train](https://drive.google.com/uc?id=10IQH6ApVKwhvS35lCP4CXT6W1JFHaXUF) · [valid](https://drive.google.com/uc?id=17lZdANOV5SwzZi9fEjt18IVmu65v0scw) · [test](https://drive.google.com/uc?id=1Tmt19jx3TWCvoMJai2VrD30sHvuSjOhe) |
| 5-fold | 3 | 588 | [train](https://drive.google.com/uc?id=1JDdzwad40073T2igMI9V11jf5JVkCd0z) · [valid](https://drive.google.com/uc?id=1Ccy_JY5rnqLgICTnubPZ0KdgJVlImK5F) · [test](https://drive.google.com/uc?id=1rPLDsKe__ImYeB7JUqSu3pLY7EMF5dNJ) |
| 5-fold | 4 | 588 | [train](https://drive.google.com/uc?id=102tm3Ag6ETEsF-S0TQaJMqYbUvMKQ0Zy) · [valid](https://drive.google.com/uc?id=1xyk76tF6CaI_m_FkTQQYwE6n4h0dJN3w) · [test](https://drive.google.com/uc?id=17ACtdvkjON2d9lhUEm32ipBoZTOoyQOe) |
| 5-fold | 5 | 588 | [train](https://drive.google.com/uc?id=1DCeuNE7gLofZnHxZBcNAYP9NhM6qMzTu) · [valid](https://drive.google.com/uc?id=1G7u6wx4PLN2jZJHVJeAjuFn69CiXNat7) · [test](https://drive.google.com/uc?id=1kubwsjcsOU3ZEHsTjGF2nHnEt6D6olqG) |

**Model** — the trained checkpoint (`best-model-val_mcc.ckpt`) and its HMM-decoded predictions on the test split (`frame_predictions_soft_vote_val_mcc_hmm.json`) for the same `(protocol, fold, 588)`:

| Protocol | Fold | Tokens | Downloads |
|---|---|---|---|
| 4-fold | 1 | 588 | [model](https://drive.google.com/uc?id=1DgP7F1XW3LrtJh8KVtw5Efr7adi5h8kr) · [prediction](https://drive.google.com/uc?id=1bZ9raAC6LQdB8RuFRI_piwFI-yYVSleL) |
| 4-fold | 2 | 588 | [model](https://drive.google.com/uc?id=1j7SI6P5JlzrAR8xlm2nxVYfszjNNR8HR) · [prediction](https://drive.google.com/uc?id=1d1AjVVJOBC3ZQ1kwCSfXz1b-bjyc6gP9) |
| 4-fold | 3 | 588 | [model](https://drive.google.com/uc?id=1p56Wo3h5XNuz6CFAvhg2GtaWyqLua39p) · [prediction](https://drive.google.com/uc?id=1zS2v-l09rlKNFBiVhgkHRgSEpObZYy8S) |
| 4-fold | 4 | 588 | [model](https://drive.google.com/uc?id=1fyg-R9fn_rp4B6CCNwY6qf3V3wOmWOXN) · [prediction](https://drive.google.com/uc?id=1FsQS66PMVGB5qTX1X4AiiK6hjUdBK6cm) |
| 5-fold | 1 | 588 | [model](https://drive.google.com/uc?id=1VlF3FNrBpZgbv6PQx8OHlvMV7f26-zkV) · [prediction](https://drive.google.com/uc?id=1DhAiTr71lOwkt4qZrgb2hvxuNoRwBbrX) |
| 5-fold | 2 | 588 | [model](https://drive.google.com/uc?id=1ef1N5jRFcbqt2wnh5TS7ADqIix44dbVN) · [prediction](https://drive.google.com/uc?id=132G-UWQ1Z60kRMipNkBma-ozl7e0d0SB) |
| 5-fold | 3 | 588 | [model](https://drive.google.com/uc?id=1ZIzWpX22jGYLft-orTgSfcsCCm_WVI6p) · [prediction](https://drive.google.com/uc?id=1KuqWfBKcAL8mBvC6h7hdwCcJ_rdeNsFv) |
| 5-fold | 4 | 588 | [model](https://drive.google.com/uc?id=1xqIDKUj3df_khiTK5W3BBp1JkACLQUL1) · [prediction](https://drive.google.com/uc?id=1DFkQhLKMa2qymWBmbpNEibnoyRvd89gj) |
| 5-fold | 5 | 588 | [model](https://drive.google.com/uc?id=1QHUGFT31R9fxEa2tnYq5xI8LLAxjxB8e) · [prediction](https://drive.google.com/uc?id=1m8bylk15dwb093c_HWT2X4s982VFMXYG) |

Only window 588 is released — the full 8-window ablation (72 checkpoints, 5.9 GB) exists but isn't being distributed this way; it's a lot of large files for what the other 7 windows would add beyond what's already in the paper's Table 2 ablation numbers.

## Using these checkpoints

Loading and running inference with them requires the `classification/` code (model architecture, inference loop, HMM decoding) — not yet public, see the note above. Once released, [`classification.md`](classification.md) and [`multi_frame.md`](multi_frame.md) will document exactly where each download goes and how to point the scripts at it.
