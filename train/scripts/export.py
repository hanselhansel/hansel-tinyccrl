import argparse
from pathlib import Path

from tinyccrl.export import export_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tinyccrl.pt"))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("../../engine/assets/tinyccrl.nnue"))
    args = parser.parse_args()
    export_weights(args.checkpoint, args.hidden, args.out)


if __name__ == "__main__":
    main()
