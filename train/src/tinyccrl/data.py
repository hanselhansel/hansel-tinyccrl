import json
import random
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset

from tinyccrl.features import fen_to_indices


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
        return white_idx, black_idx, stm, float(row["score"])

    def collate(self, batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        white_idx, black_idx, stm, scores = zip(*batch)
        white = indices_to_dense(white_idx)
        black = indices_to_dense(black_idx)
        stm_t = torch.tensor(stm, dtype=torch.long)
        targets = torch.tensor(scores, dtype=torch.float32) / 1000.0
        return white, black, stm_t, targets
