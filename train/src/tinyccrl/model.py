import torch
import torch.nn as nn


class NnueModel(nn.Module):
    def __init__(self, ft_hidden: int = 256, hidden1_size: int = 32):
        super().__init__()
        self.ft_hidden = ft_hidden
        self.hidden1_size = hidden1_size
        self.ft = nn.Linear(768, ft_hidden, bias=True)
        self.hidden1 = nn.Linear(ft_hidden, hidden1_size, bias=True)
        self.head = nn.Linear(hidden1_size, 1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in [self.ft, self.hidden1, self.head]:
            nn.init.normal_(m.weight, mean=0.0, std=0.001)
            nn.init.zeros_(m.bias)

    def forward(self, white: torch.Tensor, black: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        white_acc = self.ft(white).clamp(min=0)
        black_acc = self.ft(black).clamp(min=0)
        acc = torch.where(stm.view(-1, 1).bool(), black_acc, white_acc)
        h = self.hidden1(acc).clamp(min=0)
        return self.head(h).squeeze(1)
