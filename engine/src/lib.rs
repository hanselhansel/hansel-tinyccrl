pub mod nnue;
pub mod search;
pub mod uci;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use nnue::{Nnue, NnueAccumulator, NnueWeights};
pub use search::Searcher;
pub use uci::Uci;

#[cfg(nnue_asset)]
const NNUE_BYTES: &[u8] = include_bytes!("../assets/tinyccrl.nnue");

/// Trained weights embedded at build time when `engine/assets/tinyccrl.nnue`
/// exists, otherwise zeroed weights of the requested shape.
pub fn default_weights(ft_hidden: usize, hidden1_size: usize) -> NnueWeights {
    #[cfg(nnue_asset)]
    {
        let _ = (ft_hidden, hidden1_size);
        NnueWeights::from_bytes(NNUE_BYTES)
    }
    #[cfg(not(nnue_asset))]
    {
        NnueWeights::zero(ft_hidden, hidden1_size)
    }
}
