import argparse
from pathlib import Path

from tinyccrl.export import export_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tinyccrl.pt"))
    parser.add_argument("--ft-hidden", type=int, default=512)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("../../engine/assets/tinyccrl.nnue"))
    args = parser.parse_args()
    export_weights(args.checkpoint, args.ft_hidden, args.hidden1, args.out)


if __name__ == "__main__":
    main()
