pub mod nnue;
pub mod search;
pub mod uci;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use nnue::{Nnue, NnueWeights};
pub use search::Searcher;
pub use uci::Uci;

const NNUE_BYTES: &[u8] = include_bytes!("../assets/tinyccrl.nnue");

pub fn default_weights(ft_hidden: usize, hidden1_size: usize) -> NnueWeights {
    NnueWeights::from_bytes(NNUE_BYTES)
}
