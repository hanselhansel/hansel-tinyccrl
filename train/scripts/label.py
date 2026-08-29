import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import chess.engine


def label_chunk(stockfish: str, depth: int, rows: list[dict]) -> list[dict]:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish)
    try:
        out = []
        for row in rows:
            board = chess.Board(row["fen"])
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white().score(mate_score=100_000)
            out.append({"fen": row["fen"], "score": score})
        return out
    finally:
        engine.quit()


def label(data_path: Path, stockfish: Path, depth: int, out_path: Path, workers: int = 4):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in open(data_path)]
    chunk_size = (len(rows) + workers - 1) // workers
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(label_chunk, str(stockfish), depth, chunk): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            i = futures[future]
            results.append((i, future.result()))

    results.sort(key=lambda x: x[0])
    with open(out_path, "w") as f:
        for _, chunk in results:
            for row in chunk:
                f.write(json.dumps(row) + "\n")
    print(f"labeled {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/fens.jsonl"))
    parser.add_argument("--stockfish", type=Path, default=Path("/opt/homebrew/bin/stockfish"))
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("data/labeled.jsonl"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    label(args.data, args.stockfish, args.depth, args.out, args.workers)


if __name__ == "__main__":
    main()
