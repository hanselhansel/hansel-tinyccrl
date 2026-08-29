mod nnue;
mod search;
mod uci;
use nnue::NnueWeights;
use uci::Uci;

fn main() {
    let weights = std::env::var("NNUE_WEIGHTS")
        .ok()
        .and_then(|path| {
            if std::path::Path::new(&path).exists() {
                Some(NnueWeights::from_file(&path))
            } else {
                None
            }
        })
        .unwrap_or_else(|| NnueWeights::zero(256));
    Uci::new(weights).run();
}
