# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/DiegoBravoH/Colon_Audit.git
cd Colon_Audit
```

## 2. Environment

**Python:** 3.9 or newer.
**CUDA:** ≥ 11.8 (for GPU training/inference; PyTorch was tested against this version).

```bash
conda create -n colon-audit python=3.9
conda activate colon-audit
```

## 3. Requirements

Install the exact versions used to produce the results in the paper:

```bash
pip install -r requirements.txt
```

This installs, among others:

| Package | Version |
|---|---|
| torch | 2.5.1 |
| torchvision | 0.23.0 |
| pytorch-lightning | 2.4.0 |
| timm | 1.0.22 |
| torchmetrics | 1.6.0 |
| scikit-learn | 1.5.2 |
| matplotlib | 3.9.2 |
| pandas | 2.2.2 |
| numpy | 1.26.4 |
| tqdm | 4.67.1 |
| ijson | 3.5.1 |

To verify the install:

```python
import torch, pytorch_lightning as pl, timm
print(torch.__version__, pl.__version__, timm.__version__)
# Expected: 2.5.1 2.4.0 1.0.22
```
