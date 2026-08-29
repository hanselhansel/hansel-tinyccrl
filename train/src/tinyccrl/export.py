import struct
from pathlib import Path

import numpy as np
import torch

from tinyccrl.model import NnueModel


def export_weights(checkpoint: Path, hidden_size: int, out_path: Path):
    model = NnueModel(hidden_size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    ft_w = model.ft.weight.detach().cpu().numpy()
    ft_b = model.ft.bias.detach().cpu().numpy()
    hid_w = model.head.weight.detach().cpu().numpy()
    hid_b = model.head.bias.detach().cpu().numpy()

    # Scale all weights into i16 range using a single shared scale.
    max_abs = max(
        np.abs(ft_w).max(),
        np.abs(ft_b).max(),
        np.abs(hid_w).max(),
        np.abs(hid_b).max(),
        1e-8,
    )
    scale = max_abs / 32767.0

    ft_w_q = np.clip(np.round(ft_w / scale), -32768, 32767).astype(np.int16)
    ft_b_q = np.clip(np.round(ft_b / scale), -32768, 32767).astype(np.int16)
    hid_w_q = np.clip(np.round(hid_w / scale), -32768, 32767).astype(np.int16)
    hid_b_q = np.clip(np.round(hid_b / scale), -32768, 32767).astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", hidden_size))
        ft_w_q.tofile(f)
        ft_b_q.tofile(f)
        hid_w_q.tofile(f)
        hid_b_q.tofile(f)

    print(f"exported {out_path}: {out_path.stat().st_size} bytes")
