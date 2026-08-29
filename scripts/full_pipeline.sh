#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PATH="/opt/homebrew/opt/rustup/bin:$PATH:/Users/hansel/.cargo/bin:$PATH"

DATA_DIR=train/data
PGN_DIR=$DATA_DIR
mkdir -p "$PGN_DIR"

echo "=== Downloading 2013 Lichess PGNs ==="
for m in $(seq -w 1 12); do
    url="https://database.lichess.org/standard/lichess_db_standard_rated_2013-${m}.pgn.zst"
    file="$PGN_DIR/lichess_db_standard_rated_2013-${m}.pgn.zst"
    if [ ! -f "$file" ]; then
        curl -L -o "$file" "$url"
    fi
done

echo "=== Extracting PGNs ==="
for zst in "$PGN_DIR"/*.pgn.zst; do
    pgn="${zst%.zst}"
    if [ ! -f "$pgn" ]; then
        zstd -d "$zst"
    fi
done

echo "=== Sampling FENs ==="
source train/.venv/bin/activate
python train/scripts/sample.py --pgn "$PGN_DIR/lichess_db_standard_rated_2013-01.pgn" --out "$DATA_DIR/fens_01.jsonl" --games 50000 --per-game 4
for pgn in "$PGN_DIR"/lichess_db_standard_rated_2013-*.pgn; do
    if [ "$pgn" != "$PGN_DIR/lichess_db_standard_rated_2013-01.pgn" ]; then
        month=$(basename "$pgn" .pgn | sed 's/.*_//')
        python train/scripts/sample.py --pgn "$pgn" --out "$DATA_DIR/fens_${month}.jsonl" --games 20000 --per-game 4
    fi
done

cat "$DATA_DIR"/fens_*.jsonl > "$DATA_DIR/fens_all.jsonl"

echo "=== Labeling with Stockfish WDL ==="
python train/scripts/label.py \
    --data "$DATA_DIR/fens_all.jsonl" \
    --stockfish /opt/homebrew/bin/stockfish \
    --depth 6 \
    --out "$DATA_DIR/labeled_all.jsonl" \
    --workers 4

echo "=== Training NNUE ==="
python train/scripts/train.py \
    --data "$DATA_DIR/labeled_all.jsonl" \
    --ft-hidden 1024 \
    --hidden1 128 \
    --epochs 50 \
    --batch 1024 \
    --lr 1e-3

echo "=== Exporting weights ==="
python train/scripts/export.py \
    --checkpoint train/checkpoints/tinyccrl.pt \
    --ft-hidden 1024 \
    --hidden1 128 \
    --out engine/assets/tinyccrl.nnue

echo "=== Building engine ==="
cargo build --release -p tinyccrl-engine

echo "=== Running gauntlet ==="
python scripts/gauntlet.py \
    --engine target/release/tinyccrl-engine \
    --opponent-elo 1320 \
    --rounds 20 \
    --nodes 8000 \
    --out gauntlet_result.jsonl

echo "=== Done ==="
