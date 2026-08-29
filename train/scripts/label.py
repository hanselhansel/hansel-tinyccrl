import argparse
import json
from pathlib import Path

import chess.engine


def label(data_path: Path, stockfish: Path, depth: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(str(stockfish))
    try:
        with open(data_path) as infile, open(out_path, "w") as outfile:
            for line in infile:
                row = json.loads(line)
                board = chess.Board(row["fen"])
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info["score"].white().score(mate_score=100_000)
                outfile.write(json.dumps({"fen": row["fen"], "score": score}) + "\n")
    finally:
        engine.quit()
    print(f"labeled {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/fens.jsonl"))
    parser.add_argument("--stockfish", type=Path, default=Path("/opt/homebrew/bin/stockfish"))
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("data/labeled.jsonl"))
    args = parser.parse_args()
    label(args.data, args.stockfish, args.depth, args.out)


if __name__ == "__main__":
    main()
