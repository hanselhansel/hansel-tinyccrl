import struct
from pathlib import Path

import numpy as np
import torch

from tinyccrl.model import NnueModel


def export_weights(checkpoint: Path, ft_hidden: int, hidden1_size: int, out_path: Path):
    model = NnueModel(ft_hidden, hidden1_size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    tensors = [
        model.ft.weight.detach().cpu().numpy(),
        model.ft.bias.detach().cpu().numpy(),
        model.hidden1.weight.detach().cpu().numpy(),
        model.hidden1.bias.detach().cpu().numpy(),
        model.head.weight.detach().cpu().numpy(),
        model.head.bias.detach().cpu().numpy(),
    ]
    max_abs = max(np.abs(t).max() for t in tensors)
    max_abs = max(max_abs, 1e-8)
    scale = max_abs / 32767.0

    quantized = [np.clip(np.round(t / scale), -32768, 32767).astype(np.int16) for t in tensors]
    ft_w_q, ft_b_q, h1_w_q, h1_b_q, h2_w_q, h2_b_q = quantized

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", ft_hidden, hidden1_size))
        for t in quantized:
            t.tofile(f)

    print(f"exported {out_path}: {out_path.stat().st_size} bytes")
