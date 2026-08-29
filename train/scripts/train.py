import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tinyccrl.data import FenDataset
from tinyccrl.model import NnueModel


def train(data_path: Path, hidden_size: int = 256, epochs: int = 10, batch_size: int = 1024, lr: float = 1e-3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"training on {device}")

    model = NnueModel(hidden_size).to(device)
    ds = FenDataset(data_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for white, black, stm, targets in loader:
            white = white.to(device)
            black = black.to(device)
            stm = stm.to(device)
            targets = targets.to(device)
            pred = model(white, black, stm)
            loss = F.mse_loss(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss {total_loss / len(loader):.4f}")

    out = Path("checkpoints")
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / "tinyccrl.pt")
    print(f"saved {out / 'tinyccrl.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/labeled.jsonl"))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data, args.hidden, args.epochs, args.batch, args.lr)


if __name__ == "__main__":
    main()
