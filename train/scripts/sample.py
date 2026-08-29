import argparse
import json
from pathlib import Path

from tinyccrl.data import sample_fens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/fens.jsonl"))
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--per-game", type=int, default=4)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for fen in sample_fens(args.pgn, args.games, args.per_game):
            f.write(json.dumps({"fen": fen}) + "\n")
    print(f"sampled FENs to {args.out}")


if __name__ == "__main__":
    main()
