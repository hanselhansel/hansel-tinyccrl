import json
import math
import random
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset

from tinyccrl.features import fen_to_indices


def score_to_wdl(score: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-score / 400.0))


def wdl_to_cp(wdl: float) -> float:
    """Convert a white win-probability to centipawns."""
    w = max(1e-6, min(1.0 - 1e-6, wdl))
    return -400.0 * math.log10((1.0 - w) / w)


def sample_fens(pgn_path: Path, max_games: int = 1000, positions_per_game: int = 4) -> Iterator[str]:
    with open(pgn_path) as f:
        for _ in range(max_games):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            plies = list(game.mainline_moves())
            start = 8
            if start >= len(plies):
                continue
            end = min(len(plies), start + positions_per_game)
            chosen = random.sample(range(start, end), min(positions_per_game, end - start))
            for i in sorted(chosen):
                board.reset()
                for mv in plies[: i + 1]:
                    board.push(mv)
                yield board.fen()


def indices_to_dense(indices: list[list[int]], dim: int = 768) -> torch.Tensor:
    B = len(indices)
    out = torch.zeros(B, dim, dtype=torch.float32)
    for b, idxs in enumerate(indices):
        out[b, idxs] = 1.0
    return out


class FenDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in open(path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        white_idx, black_idx, stm = fen_to_indices(row["fen"])

        if "score" in row:
            cp = float(row["score"])
        elif "expected" in row:
            cp = wdl_to_cp(float(row["expected"]))
        else:
            cp = 0.0

        # The model sees the position from the side-to-move's perspective,
        # so the target must also be from the side-to-move's perspective.
        # Stockfish labels are from White's perspective, flip for Black.
        target = (cp / 1000.0) if stm == 0 else -(cp / 1000.0)
        return white_idx, black_idx, stm, target

    def collate(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        white_idx, black_idx, stm, expected = zip(*batch)
        white = indices_to_dense(white_idx)
        black = indices_to_dense(black_idx)
        stm_t = torch.tensor(stm, dtype=torch.long)
        targets = torch.tensor(expected, dtype=torch.float32)
        return white, black, stm_t, targets
