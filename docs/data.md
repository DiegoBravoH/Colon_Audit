# 📂 Data Preparation — REAL-Colon Dataset

## 1️⃣ Download the REAL-Colon Dataset

The proposed framework was evaluated on the **REAL-Colon dataset** (Biffi et al., 2024), comprising 60 fully annotated colonoscopy procedures (∼2.7 million frames) from six medical centers across three continents.

🔗 Available on Figshare:
https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866

## 2️⃣ Annotations

Each frame is labeled as one of nine anatomical phases:

| Label | Phase |
|---|---|
| 0 | outside |
| 1 | insertion |
| 2 | cecum |
| 3 | ileum |
| 4 | ascending |
| 5 | transverse |
| 6 | descending |
| 7 | sigmoid |
| 8 | rectum |

## 3️⃣ Evaluation Protocols

The official evaluation protocols are followed without modification, using the provided annotations:

- **5-fold** — standard cross-validation, mixed centers per fold.
- **4-fold (cross-center)** — mutually exclusive medical centers per fold; better reflects real-world cross-center generalization.

## 4️⃣ Reference

```bibtex
@article{biffi2024realcolon,
  title   = {REAL-Colon: A dataset for developing real-world AI applications in colonoscopy},
  author  = {Biffi, Carlo and Antonelli, Giulio and Bernhofer, Sebastian and Hassan, Cesare and Hirata, Daizen and Iwatate, Mineo and Maieron, Andreas and Salvagnini, Pietro and Cherubini, Andrea},
  journal = {Scientific Data},
  volume  = {11},
  number  = {1},
  pages   = {539},
  year    = {2024}
}
```
