import argparse
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def play_game(engine_path: Path, opponent_path: Path, fen: str | None, nodes: int, engine_white: bool) -> dict:
    env = {"NNUE_WEIGHTS": str(Path("engine/assets/tinyccrl.nnue").resolve())} if Path("engine/assets/tinyccrl.nnue").exists() else None
    engine = chess.engine.SimpleEngine.popen_uci(str(engine_path), env=env)
    opponent = chess.engine.SimpleEngine.popen_uci(str(opponent_path))
    try:
        board = chess.Board(fen) if fen else chess.Board()
        players = [engine, opponent] if engine_white else [opponent, engine]
        while not board.is_game_over():
            current = players[0 if board.turn == chess.WHITE else 1]
            limit = chess.engine.Limit(nodes=nodes)
            result = current.play(board, limit)
            board.push(result.move)
        return {
            "fen": fen or "startpos",
            "engine_white": engine_white,
            "result": board.result(claim_draw=True),
            "pgn": str(chess.pgn.Game.from_board(board)),
        }
    finally:
        engine.quit()
        opponent.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, default=Path("/opt/homebrew/bin/stockfish"))
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=800)
    parser.add_argument("--out", type=Path, default=Path("gauntlet.jsonl"))
    args = parser.parse_args()

    results = []
    for i in range(args.rounds):
        engine_white = i % 2 == 0
        r = play_game(args.engine, args.opponent, None, args.nodes, engine_white)
        results.append(r)
        print(r["result"], "as White" if engine_white else "as Black")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
