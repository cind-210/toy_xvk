import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_unique_default_out_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        cand = f"{path}({idx})"
        if not os.path.exists(cand):
            return cand
        idx += 1


def parse_pred_specs(spec: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for raw in [x.strip().lower() for x in spec.split(",") if x.strip()]:
        if raw in ("x", "e", "v"):
            out.append({"name": raw, "pred_type": raw})
            continue
        raise ValueError(f"Unsupported pred spec: {raw!r}. Use x/e/v.")
    return out


def parse_noise_scale_modes(spec: str) -> List[Dict[str, Optional[float]]]:
    parts = [p.strip().lower() for p in str(spec).split(",") if p.strip() != ""]
    if len(parts) == 0:
        return [{"mode": "fixed", "value": 1.0}]

    modes: List[Dict[str, Optional[float]]] = []
    for tok in parts:
        if tok == "auto":
            modes.append({"mode": "auto", "value": None})
            continue
        if tok == "e":
            modes.append({"mode": "e", "value": None})
            continue
        if tok == "var":
            modes.append({"mode": "var", "value": None})
            continue
        if tok == "ep":
            modes.append({"mode": "ep", "value": None})
            continue
        val = float(tok)
        if val < 0.0:
            raise ValueError("--noise_scale numeric entries must be >= 0.")
        modes.append({"mode": "fixed", "value": val})
    return modes


def parse_high_dims(spec: str) -> List[int]:
    parts = [p.strip() for p in str(spec).split(",") if p.strip() != ""]
    if len(parts) == 0:
        return [64]
    out: List[int] = []
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError("--high_dim entries must be positive integers.")
        out.append(v)
    return out


def _sanitize_alnum(text: str) -> str:
    out = "".join(ch for ch in text if ch.isalnum())
    return out if out else "x"


def build_default_out_dir(
    pred_types_spec: str,
    high_dims: List[int],
    shape: str,
    noise_modes: List[Dict[str, Optional[float]]],
    width: int,
) -> str:
    pred_tag = _sanitize_alnum(pred_types_spec.lower())
    has_numeric_noise = any(str(m["mode"]) == "fixed" for m in noise_modes)
    letter_modes = {"e", "var", "ep"}
    ordered_letters: List[str] = []
    for m in noise_modes:
        mode = str(m["mode"])
        if mode in letter_modes and mode not in ordered_letters:
            ordered_letters.append(mode)
    noise_suffix = ""
    if (not has_numeric_noise) and len(ordered_letters) > 0:
        noise_suffix = "_" + "".join(ordered_letters)
    hd_tag = str(high_dims[0]) if len(high_dims) == 1 else "".join(str(x) for x in high_dims)
    line_suffix = "_line" if shape == "line" else ""
    width_suffix = f"_{width}" if width != 256 else ""
    return f"./out/{pred_tag}_{hd_tag}{noise_suffix}{line_suffix}{width_suffix}"
