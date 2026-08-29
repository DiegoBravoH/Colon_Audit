import os
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm


# --------------------------------------------------
# 1. Argument Parsing
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract CNN features using a pretrained ViT-Small (DINOv1) backbone."
    )
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--dataframe",    type=str, required=True)
    parser.add_argument("--data_path",    type=str, required=True)
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=8)
    parser.add_argument(
        "--device", type=str, default="cuda:1",
        help="PyTorch device string (e.g. cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--augmentation", type=str, default="original",
        choices=["original", "flip-h", "flip-v", "rotation-90", "rotation-180", "rotation-270"],
        help="Whole-video-consistent geometric augmentation (2026-07-27): the SAME transform is "
             "applied to every frame of every patient for a given value -- builds a full augmented "
             "COPY of the feature set (same idea as the temporal 'offset' augmentation, but spatial). "
             "'original' = no-op, identical to this script's pre-existing behavior."
    )
    return parser.parse_args()


# --------------------------------------------------
# 2. Robust Weight Loading (ViT-Small / DINOv1 -- unchanged from the
#    GastroHUN version of this script; this is the part worth keeping as-is)
# --------------------------------------------------

def _interpolate_pos_embed(state_dict, model, key="pos_embed", num_extra_tokens=1):
    """
    Reescala pos_embed del checkpoint al tamaño esperado por el modelo actual.
    Detecta automáticamente si el modelo sigue la convención:
      - "con cls en pos_embed"  -> N_model = num_patches + num_extra_tokens
      - "no_embed_class=True"   -> N_model = num_patches (p.ej. DINOv2 reg4 en timm)
    Para ViT-Small DINOv1 a 224 el checkpoint y el modelo coinciden (197 tokens),
    por lo que la función retorna sin modificar el state_dict.
    """
    if key not in state_dict or not hasattr(model, key):
        return state_dict

    pos_ckpt = state_dict[key]
    pos_model = getattr(model, key)
    if pos_ckpt.shape == pos_model.shape:
        return state_dict

    D = pos_ckpt.shape[-1]
    N_ckpt_total = pos_ckpt.shape[1]
    N_model_total = pos_model.shape[1]

    gs_model_if_no_extra = int(round(math.sqrt(N_model_total)))
    model_has_extra_in_pos = (gs_model_if_no_extra ** 2 != N_model_total)

    if model_has_extra_in_pos:
        N_model = N_model_total - num_extra_tokens
        model_extra = num_extra_tokens
    else:
        N_model = N_model_total
        model_extra = 0

    N_ckpt = N_ckpt_total - num_extra_tokens
    gs_ckpt = int(round(math.sqrt(N_ckpt)))
    gs_model = int(round(math.sqrt(N_model)))
    assert gs_ckpt * gs_ckpt == N_ckpt, f"pos_embed del checkpoint no es cuadrado ({N_ckpt})"
    assert gs_model * gs_model == N_model, f"pos_embed del modelo no es cuadrado ({N_model})"

    extra = pos_ckpt[:, :num_extra_tokens]
    grid = pos_ckpt[:, num_extra_tokens:]

    grid = grid.reshape(1, gs_ckpt, gs_ckpt, D).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(gs_model, gs_model),
                         mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, gs_model * gs_model, D)

    if model_extra == 0:
        state_dict[key] = grid
        print(f"  pos_embed interpolated: {gs_ckpt}x{gs_ckpt} -> {gs_model}x{gs_model} "
              f"(dropped {num_extra_tokens} prefix token(s) from ckpt to match model convention)")
    else:
        state_dict[key] = torch.cat([extra, grid], dim=1)
        print(f"  pos_embed interpolated: {gs_ckpt}x{gs_ckpt} -> {gs_model}x{gs_model}")

    return state_dict


def load_backbone_weights(model, path, arch="ViT_Small"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights file not found: {path}")

    print(f"Loading weights from: {os.path.basename(path)}")
    checkpoint = torch.load(path, map_location="cpu")

    # 1) Detectar sub-estructura del checkpoint
    if "teacher" in checkpoint:
        state_dict, found_key = checkpoint["teacher"], "teacher"
    elif "student" in checkpoint:
        state_dict, found_key = checkpoint["student"], "student"
    elif "state_dict" in checkpoint:
        state_dict, found_key = checkpoint["state_dict"], "state_dict"
    elif "model" in checkpoint:
        state_dict, found_key = checkpoint["model"], "model"
    else:
        state_dict, found_key = checkpoint, "flat structure (direct)"
    print(f"Detected structure: weights loaded from key '{found_key}'")

    # 2) Limpiar prefijos DDP / DINO y renombrar claves específicas de convención
    PRETRAIN_ONLY_PREFIXES = ("dino_head.", "ibot_head.")
    PRETRAIN_ONLY_KEYS = {"mask_token"}
    # Mapeos de nombres: DINOv2 oficial -> timm
    RENAME = {
        "register_tokens": "reg_token",   # timm usa singular
    }

    clean_state_dict = {}
    dropped = []
    for k, v in state_dict.items():
        k = k.replace("module.", "").replace("backbone.", "")
        k = k.replace("teacher.", "").replace("student.", "")

        if k in PRETRAIN_ONLY_KEYS or any(k.startswith(p) for p in PRETRAIN_ONLY_PREFIXES):
            dropped.append(k)
            continue

        if k in RENAME:
            k = RENAME[k]

        clean_state_dict[k] = v

    if dropped:
        print(f"  Dropped {len(dropped)} pretraining-only keys "
              f"(e.g. {dropped[:3]}{'...' if len(dropped) > 3 else ''})")

    # 3) Si es ViT, interpolar pos_embed a la resolución actual del modelo
    if arch in ("ViT_Small", "ViT_Base"):
        clean_state_dict = _interpolate_pos_embed(
            clean_state_dict, model, key="pos_embed", num_extra_tokens=1
        )

    # 4) Cargar
    msg = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Load report: {len(msg.missing_keys)} missing keys, "
          f"{len(msg.unexpected_keys)} unexpected keys.")
    if msg.unexpected_keys:
        print(f"  unexpected (primeras 5): {msg.unexpected_keys[:5]}")
    if msg.missing_keys:
        print(f"  missing (primeras 5):    {msg.missing_keys[:5]}")

    # 5) Chequeo crítico según arquitectura
    critical_key = {
        "ResNet50":  "conv1.weight",
        "ViT_Small": "patch_embed.proj.weight",
        "ViT_Base":  "patch_embed.proj.weight",
    }[arch]
    if critical_key in msg.missing_keys:
        raise RuntimeError(
            f"⚠️ CRITICAL ERROR: '{critical_key}' was not loaded. "
            f"Check checkpoint prefixes or architecture mismatch."
        )

    print("✅ Weights loaded successfully.")


@torch.no_grad()
def extract_vit_features(model, x):
    """
    Extract the cls_token feature from a timm ViT.
    forward_features(x) -> [B, N_tokens, D]   (N_tokens = 1 cls + num_patches)
    Returns the cls_token: [B, D] (D=384 for ViT-Small).
    """
    tokens = model.forward_features(x)   # [B, N_tokens, D]
    cls_feat = tokens[:, 0]              # [B, D] -> cls_token
    return cls_feat


# --------------------------------------------------
# 3. Dataset (idéntico a features_ResNet50.py -- lee imágenes ya extraídas
#    por frame, no video crudo; ese era el manejo de datos de la version
#    GastroHUN de este script, ya no aplica aquí)
# --------------------------------------------------

def apply_geometric_augmentation(img, augmentation):
    """Applies ONE deterministic, exact geometric transform to a PIL image.
    "original" is a no-op -- kept as a valid choice so this same code path
    can also (re)produce the baseline set.

    flip-h/flip-v use PIL's own mirror constants (exact reflection, no
    interpolation). rotation-* use .transpose(), NOT .rotate(angle):
    .transpose() is an EXACT pixel permutation for multiples of 90 degrees
    (no resampling, no black-corner padding), unlike .rotate() which
    resamples the image and would introduce interpolation artifacts even
    at 90-degree multiples. Arbitrary-angle rotation (e.g. 30 degrees) is
    deliberately NOT offered here -- it requires resampling/padding and
    pushes inputs further from what the frozen DINOv1 backbone saw during
    its own pretraining (see 2026-07-27 discussion); safer to validate the
    cheap, artifact-free transforms first.
    """
    if augmentation == "original":
        return img
    elif augmentation == "flip-h":
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif augmentation == "flip-v":
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    elif augmentation == "rotation-90":
        return img.transpose(Image.ROTATE_90)
    elif augmentation == "rotation-180":
        return img.transpose(Image.ROTATE_180)
    elif augmentation == "rotation-270":
        return img.transpose(Image.ROTATE_270)
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")


class ImageFeatureDataset(Dataset):
    def __init__(self, df, transform=None, augmentation="original"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, "image_name"]).convert("RGB")
        img = apply_geometric_augmentation(img, self.augmentation)
        if self.transform:
            img = self.transform(img)
        return img, self.df.loc[idx, "pt_name"]


# --------------------------------------------------
# 4. Main
# --------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        # device.index is relative to CUDA_VISIBLE_DEVICES, not physical GPU id
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print(f"  GPU: {torch.cuda.get_device_name(idx)}")

    transform = T.Compose([
        T.Resize((224, 224), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Safety guard (2026-07-27): --output_dir's last folder must match
    # --augmentation, so a typo/forgotten flag can't silently overwrite a
    # DIFFERENT augmentation's (or the original's) already-extracted
    # features -- each variant takes ~2.5h for all 60 patients, expensive
    # to redo by accident.
    actual_suffix = os.path.basename(os.path.normpath(args.output_dir))
    if actual_suffix != args.augmentation:
        raise ValueError(
            f"--output_dir's last folder ('{actual_suffix}') must match --augmentation "
            f"('{args.augmentation}') -- e.g. --augmentation flip-h --output_dir '.../ViT-Small/flip-h'."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.dataframe, index_col=0).reset_index(drop=True)

    # Instantiate ViT-Small (DINOv1: patch_size=16, embed_dim=384)
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=False,
        num_classes=0,    # quita la cabeza; equivalente a nn.Identity()
    )
    load_backbone_weights(model, args.weights_path, arch="ViT_Small")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    patients = df["patient"].unique()[::-1]
    print(f"Starting feature extraction for {len(patients)} patients... (augmentation={args.augmentation})")

    for name_patient in tqdm(patients, desc="Patients"):
        df_patient = df[df["patient"] == name_patient].copy().reset_index(drop=True)

        out_patient = os.path.join(args.output_dir, name_patient)

        # Sin idempotencia a proposito (2026-07-27, decision explicita del
        # usuario): SIEMPRE re-extrae y sobreescribe, incluso si
        # features.npy ya existe para este paciente. np.save() mas abajo
        # sobreescribe el archivo existente sin necesidad de borrarlo antes.
        if os.path.exists(os.path.join(out_patient, "features.npy")):
            print(f"  ♻️  Ya existe, sobreescribiendo: {out_patient}")

        os.makedirs(out_patient, exist_ok=True)

        df_patient["image_name"] = df_patient["filename"].apply(
            lambda f: os.path.join(args.data_path, f"{name_patient}_frames", f)
        )

        dataset = ImageFeatureDataset(df_patient, transform=transform, augmentation=args.augmentation)
        loader  = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            shuffle=False,   # order must be preserved -- it becomes the row order of the saved array
        )

        # Consolidar por paciente en vez de un .pt por frame -- mismo cambio
        # y mismo motivo que en features_ResNet50.py (cuello de botella de
        # I/O medido en entrenamiento: millones de archivos ~8KB, costo
        # dominado por overhead de apertura/deserializacion por archivo).
        # No hace falta .clone() aca: torch.cat() siempre aloja memoria
        # nueva, a diferencia de indexar una fila de un tensor de batch
        # (que SI comparte storage con el batch completo y por eso el path
        # antiguo, por frame, necesitaba features_cpu[i].clone() antes de
        # guardar cada fila sola).
        all_feats = []
        all_frame_ids = []
        for imgs, pt_paths in tqdm(loader, desc=name_patient, leave=False):
            imgs = imgs.to(device)
            feats = extract_vit_features(model, imgs).cpu()
            all_feats.append(feats)
            all_frame_ids.extend(int(os.path.splitext(p)[0]) for p in pt_paths)

        features = torch.cat(all_feats, dim=0).numpy()          # [N_frames, 384]
        frame_numbers = np.array(all_frame_ids, dtype=np.int64)  # [N_frames]

        np.save(os.path.join(out_patient, "features.npy"), features)
        np.save(os.path.join(out_patient, "frame_numbers.npy"), frame_numbers)

    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
