# hansel-tinyccrl

A sub-2M parameter NNUE chess engine distilled from Stockfish, trained locally and
compiled to WASM for browser play.

- Design spec: `docs/superpowers/specs/2026-08-29-hansel-tinyccrl-design.md`
- Implementation plan: `docs/superpowers/plans/2026-08-29-hansel-tinyccrl-plan.md`

Read both before judging structure. The intended design lives there, and a change
that diverges from it needs a stated reason.

## Layout

| Path | Owns |
|---|---|
| `engine/src/search.rs` | Search. No protocol parsing, no NNUE internals. |
| `engine/src/nnue.rs` | Evaluation, network shape, quantisation scales. |
| `engine/src/uci.rs` | UCI protocol only. |
| `engine/src/lib.rs` | The shared core. |
| `engine/src/main.rs` | Native binary. Thin shell over the core. |
| `engine/src/wasm.rs` | WASM binding. Thin shell over the same core. |
| `train/src/tinyccrl/` | Feature extraction, model, exporter. |
| `train/scripts/` | Sample, label, train, export entrypoints. |
| `web/src/` | Vite and TypeScript app that loads the WASM engine. |

## Commands

```bash
cargo build --release -p tinyccrl-engine
cargo test -p tinyccrl-engine
cargo clippy -p tinyccrl-engine -- -D warnings
make all                  # engine plus web
echo -e "position startpos\ngo nodes 8000\nquit" | ./target/release/tinyccrl-engine
```

Python training lives under `train/` behind its own venv. The web app builds with
`npm install && npm run build` inside `web/`.

## Boundaries

1. Keep search, evaluation, and protocol separable. Search must not reach into
   NNUE internals, and evaluation logic must not appear in the UCI layer.
2. `main.rs` and `wasm.rs` stay thin. Logic duplicated between them, or behavior
   that diverges by target, is a bug.
3. Search and eval tunables (depths, margins, quantisation scales, network shape)
   live in one declared place, never hardcoded at call sites.
4. The NNUE file format is a contract between `engine/src/nnue.rs`,
   `train/src/tinyccrl/export.py`, and the web app. Change one side and you change
   all of them in the same PR.

## Build inputs

`Cargo.lock` and the generated `tinyccrl.nnue` are gitignored. The engine therefore
must not `include_bytes!` the net unconditionally. Gate loading it behind a
`build.rs` cfg with a zeroed fallback so a fresh clone builds without training
artifacts. Before claiming a build works, clone into a clean directory and build
there.

## Tests

Pin behavior with perft counts, golden UCI transcripts, and NNUE characterization
tests. Check any new expected value against an outside reference, not against what
the code prints today. The start position has 20 legal moves.

Never disable, skip, or weaken a test or a CI rule to make a check pass.
