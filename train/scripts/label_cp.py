import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import chess
import chess.engine


def label_chunk(stockfish: str, rows: list[dict]) -> list[dict]:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    try:
        out = []
        for row in rows:
            board = chess.Board(row["fen"])
            info = engine.analyse(board, chess.engine.Limit(depth=1))
            score = info["score"].white().score(mate_score=100_000)
            score = max(-10_000, min(10_000, score))
            out.append({"fen": row["fen"], "score": score})
        return out
    finally:
        engine.quit()


def label(data_path: str, stockfish: str, out_path: str, workers: int = 4):
    with open(data_path) as f:
        rows = [json.loads(line) for line in f]

    chunk_size = max(1, len(rows) // workers)
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(label_chunk, stockfish, chunk) for chunk in chunks]
        results = [future.result() for future in futures]

    with open(out_path, "w") as f:
        for chunk in results:
            for row in chunk:
                f.write(json.dumps(row) + "\n")
    print(f"labeled {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/fens_all.jsonl")
    parser.add_argument("--stockfish", type=str, default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--out", type=str, default="data/labeled_cp.jsonl")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    label(args.data, args.stockfish, args.out, args.workers)


if __name__ == "__main__":
    main()
