# Hansel TinyCCRL — Sub-2M Parameter Chess Engine Distilled from Stockfish

Date: 2026-08-29
Status: approved
Repo: `hanselhansel/hansel-tinyccrl`

## Claim

> A sub-2M-parameter chess engine distilled from Stockfish, trained entirely on a MacBook, compiled to WASM for in-browser play, and measured on CCRL.

Headline benchmark: **≥2,500 CCRL-equivalent Elo** with **<2M parameters** at the published search budget.

This is a deliberate departure from `hansel-chess-ai`: that repo pursues from-scratch AlphaZero self-play with a policy-value net and browser MCTS. This repo asks what is the strongest small chess engine you can build on a laptop when time is the binding constraint, and answers it with an NNUE-style evaluation inside a classic alpha-beta search engine.

## Success criteria

1. Reach **≥2,500 CCRL-equivalent Elo** with **<2M parameters**.
2. Complete training on an Apple M4 MacBook (24 GB, MPS) in **under two weeks wall-clock**.
3. Ship a live website where anyone can play the engine in the browser via WASM.
4. Make every step — data generation, training, engine build, evaluation, WASM compile — reproducible from the README.

## Scope

**In scope**
- Sub-2M parameter NNUE-style evaluation network.
- Alpha-beta search engine in Rust with UCI support.
- Stockfish-distilled training data on local PC.
- CCRL-style gauntlet for rating.
- WASM build for browser play.
- React + Vite website with board, engine output, and efficiency card.
- Public weights, training logs, and gauntlet results.

**Out of scope for v1**
- From-scratch training.
- Policy-value architecture with MCTS as the rated search.
- Lichess BOT rating as the primary claim.
- Cloud training or distributed self-play.
- Server-side play; the engine runs in the browser via WASM.

## Architecture

### Engine

- **Language**: Rust.
- **Search**: principal-variation alpha-beta with iterative deepening, transposition table, quiescence search, and basic move ordering.
- **Evaluation**: NNUE-style network. Start with a feature transformer `(768 -> N)x2 -> 1` where N is tuned for the parameter budget. A `(768 -> 256)x2 -> 1` baseline is roughly **400–500K parameters** and is the architecture class used by Tcheran to reach ~2,500 CCRL 40/15 on self-play data.
- **UCI**: standard UCI protocol for gauntlets.
- **WASM**: same engine compiled to WebAssembly via `wasm-pack` / `wasm-bindgen` for browser play.

### Network

- **Input**: 768-element sparse feature vector (piece × square × side to move). Also known as "HalfKP" or a simple piece-square encoding.
- **Feature transformer**: two perspectives, output clipped and folded with `crelu` / `screlu`.
- **Hidden layers**: one or two small fully-connected layers.
- **Output**: single scalar win probability (WDL) or centipawn-equivalent value.
- **Parameter target**: <2M. The baseline is <1M, leaving room for output buckets or a slightly wider transformer if experiments show it helps.

### Training signal

Distill from Stockfish 18. For each sampled FEN:
- **Value target**: Stockfish WDL (win/draw/loss probabilities) or centipawn score converted to a win probability.
- Optionally **policy target** (best move) is not required for NNUE alpha-beta, but can be generated if we later add a policy head for MCTS variants.

Loss: value cross-entropy or mean squared error against the Stockfish target.

### Data generation

- Sample FENs from Lichess PGNs (2013–2014 is a clean, small archive; scale up if needed).
- Add synthetic positions for openings and endgames.
- Label each FEN with Stockfish at fixed depth or nodes.
- Deduplicate and split by game to avoid leakage.

### Training

- PyTorch training script running on MPS (Apple GPU).
- Export trained weights to a compact binary format consumed by the Rust engine.
- Embed weights into the Rust binary at build time, including the WASM build.

## Data flow

```
Lichess PGNs / synthetic FENs
        |
        v
   stockfish-label  -->  train/data/labeled.binpack
        |
        v
   train-nnue (MPS)  -->  train/checkpoints/tinyccrl.nnue
        |
        v
   Rust engine build (weights embedded)
        |
        +-- UCI engine  -->  CCRL-style gauntlet
        |
        +-- wasm-pack  -->  browser demo
```

1. **Generate**: sample FENs and label with Stockfish WDL/score.
2. **Train**: train NNUE on MPS with value loss.
3. **Export**: write quantized weights to `engine/assets/tinyccrl.nnue`.
4. **Build**: compile Rust engine for native UCI and WASM browser targets.
5. **Evaluate**: run CCRL-style gauntlet with the native UCI engine.
6. **Serve**: website loads WASM engine and lets visitors play.

## Benchmarking

### CCRL-style gauntlet

Build a **UCI engine** from the Rust code. Run fixed matches with `cutechess-cli` or `c-chess-cli` against a small, reproducible opponent set:
- Stockfish 18 `UCI_Elo` 1320 / 1500 / 1800 / 2000.
- Stockfish 18 full strength (ceiling measurement).
- Optionally one or two other open-source engines for diversity.

Use a **fixed node count per move** as the published search budget so the result is reproducible on any hardware. Because this is not identical to the official CCRL time-control protocol, the headline Elo is described as "CCRL-style" or "estimated CCRL" until an official CCRL submission is made.

Compute Elo with `BayesElo` or `Ordo`. Report confidence intervals, not a single rounded number.

### Actual CCRL submission

Optional. The UCI engine and reproducible build are designed to support it, but the launch does not depend on an official CCRL entry.

### Browser demo claim

The website efficiency card shows:
- Parameter count.
- Search budget (fixed nodes / fixed depth).
- Estimated CCRL Elo at that budget.
- Training time on a MacBook.
- Link to weights, logs, and gauntlet PGNs.

## Website

Single-page React + Vite app.

Features:
- Play as White or Black against the WASM engine.
- Show engine output: evaluation, depth, principal variation, nodes per second.
- Toggle search strength (fixed nodes / fixed depth) or a friendly Elo slider that maps to depth/nodes.
- Efficiency card with params, search budget, estimated CCRL Elo, and training time.
- Download PGN of the current game and links to GitHub / logs.

## Error handling and invariants

- **Never claim "trained from scratch."** README and website state "distilled from Stockfish" clearly.
- **Elo always paired with search budget.** No bare Elo number without nodes or depth.
- **No guessed Elo.** If Stockfish or gauntlet opponents are missing, the rating card says "not rated."
- **Atomic promotion.** Candidate weights and engine builds are evaluated before replacing public artifacts.
- **Reproducibility.** Pin Stockfish version, random seed, dataset version, and architecture hash.

## Testing

- Unit tests for board representation, move generation, alpha-beta, NNUE inference, and UCI protocol.
- Sanity test: engine produces legal moves from the start position.
- Overfit test: NNUE memorizes a tiny labeled dataset.
- Gauntlet smoke test: run a 4-game match at reduced strength with no crashes.
- WASM smoke test: browser build loads and plays a move.

## Repo structure

```
hansel-tinyccrl/
├── engine/                 Rust UCI + WASM engine
│   ├── src/
│   ├── wasm/
│   └── assets/             Embedded NNUE weights
├── train/                  Python: data generation, NNUE training, export
│   ├── src/tinyccrl/
│   ├── scripts/
│   └── checkpoints/
├── web/                    TypeScript + Vite website
│   ├── src/
│   └── public/
├── scripts/                Build, eval, deploy helpers
├── docs/superpowers/specs/ Design docs
└── README.md
```

## Milestones

1. **Rust engine skeleton + NNUE eval + UCI** (week 1)
   - Legal move generator, alpha-beta, UCI.
   - Load a randomly-initialized NNUE for inference.

2. **Stockfish label generator + trainer** (week 1)
   - Generate ~1M labeled FENs.
   - Train first NNUE on MPS.
   - Export weights to engine format.

3. **First trained net + CCRL-style gauntlet** (week 1–2)
   - Run fixed-opponent matches.
   - Tune data size, depth, and search budget.

4. **WASM build + browser integration** (week 2)
   - Compile engine to WASM.
   - Build the single-page website.

5. **Tune to target + publish** (week 2–3)
   - Reach ≥2,500 CCRL-equivalent Elo with <2M params.
   - Publish website, weights, logs, and gauntlet PGNs.

## First experiment

Generate **100,000 FENs** labeled by Stockfish 18 at depth 12. Train a `(768 -> 256)x2 -> 1` NNUE for 10 epochs on MPS. Run a 4-game smoke gauntlet at fixed nodes. Success: the UCI engine plays legal moves, finishes games, and does not crash. Elo is not the gate for the first experiment.

## Non-goals

- From-scratch AlphaZero training.
- Policy-value MCTS as the rated search.
- Lichess BOT rating as the primary metric.
- Models at or above 2M parameters for the headline claim.
- Cloud compute or distributed training.
- Forking `hansel-chess-ai` or `hansel-chesslite`.

## Never

- Claim the model was trained from scratch.
- Report an Elo without the search budget.
- Guess an Elo when the gauntlet is missing.
- Promote public weights before the gauntlet passes.
