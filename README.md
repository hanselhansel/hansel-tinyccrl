# Hansel TinyCCRL

A sub-2M parameter chess engine distilled from Stockfish, trained on a MacBook,
and compiled to WASM for browser play.

- [Design spec](docs/superpowers/specs/2026-08-29-hansel-tinyccrl-design.md)
- [Implementation plan](docs/superpowers/plans/2026-08-29-hansel-tinyccrl-plan.md)

## Build

```bash
# Rust engine
cargo build --release -p tinyccrl-engine

# Python training environment
cd train
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Web app
cd ../web
npm install
npm run build
```

## Run the engine

```bash
echo -e "position startpos\ngo nodes 8000\nquit" | ./target/release/tinyccrl-engine
```

## Train a net

```bash
cd train
source .venv/bin/activate
python scripts/sample.py --pgn data/lichess_db_standard_rated_2013-01.pgn --out data/fens.jsonl --games 50000 --per-game 4
python scripts/label.py --data data/fens.jsonl --stockfish /opt/homebrew/bin/stockfish --depth 8 --out data/labeled.jsonl
python scripts/train.py --data data/labeled.jsonl --hidden 512 --epochs 10 --batch 1024 --lr 1e-3
python scripts/export.py --checkpoint checkpoints/tinyccrl.pt --hidden 512 --out ../engine/assets/tinyccrl.nnue
```

## Gauntlet

```bash
cargo build --release -p tinyccrl-engine
python scripts/gauntlet.py --engine target/release/tinyccrl-engine --rounds 4 --nodes 800
```

## Web app

```bash
cd web
npm run dev
```
