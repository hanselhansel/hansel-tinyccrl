import struct
from pathlib import Path

import numpy as np
import torch

from tinyccrl.model import NnueModel


def _transpose_2d(t: np.ndarray, expected_rows: int, expected_cols: int) -> np.ndarray:
    """Return C-contiguous, little-endian f32 array in the shape the engine expects."""
    assert t.shape == (expected_rows, expected_cols), t.shape
    return np.ascontiguousarray(t.T.astype("<f4"))


def export_weights(checkpoint: Path, ft_hidden: int, hidden1_size: int, out_path: Path):
    model = NnueModel(ft_hidden, hidden1_size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    ft_w = _transpose_2d(model.ft.weight.detach().cpu().numpy(), ft_hidden, 768)
    ft_b = np.ascontiguousarray(model.ft.bias.detach().cpu().numpy().astype("<f4"))
    h1_w = _transpose_2d(model.hidden1.weight.detach().cpu().numpy(), hidden1_size, ft_hidden)
    h1_b = np.ascontiguousarray(model.hidden1.bias.detach().cpu().numpy().astype("<f4"))
    h2_w = np.ascontiguousarray(model.head.weight.detach().cpu().numpy().reshape(-1).astype("<f4"))
    h2_b = np.ascontiguousarray(model.head.bias.detach().cpu().numpy().reshape(-1).astype("<f4"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", ft_hidden, hidden1_size))
        for t in (ft_w, ft_b, h1_w, h1_b, h2_w, h2_b):
            t.tofile(f)

    print(f"exported {out_path}: {out_path.stat().st_size} bytes")
