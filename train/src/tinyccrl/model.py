import torch
import torch.nn as nn


class NnueModel(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.ft = nn.Linear(768, hidden_size, bias=True)
        self.head = nn.Linear(hidden_size, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.ft.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.ft.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.head.bias)

    def forward(self, white: torch.Tensor, black: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        white_acc = self.ft(white).clamp(min=0)
        black_acc = self.ft(black).clamp(min=0)
        acc = torch.where(stm.view(-1, 1).bool(), black_acc, white_acc)
        return self.head(acc).squeeze(1)
