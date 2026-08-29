use tinyccrl_engine::uci::Uci;
use tinyccrl_engine::NnueWeights;

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
        .unwrap_or_else(|| tinyccrl_engine::default_weights(512, 64));
    Uci::new(weights).run();
}
