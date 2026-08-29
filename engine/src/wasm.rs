use std::str::FromStr;

use wasm_bindgen::prelude::*;

use crate::default_weights;
use crate::nnue::Nnue;
use crate::search::Searcher;
use chess::Board;

#[wasm_bindgen]
pub struct WasmEngine {
    nnue: Nnue,
}

#[wasm_bindgen]
impl WasmEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
        Self {
            nnue: Nnue::new(default_weights()),
        }
    }

    pub fn best_move(&self, fen: &str, depth: u8) -> String {
        let board = Board::from_str(fen).unwrap_or_else(|_| Board::default());
        let mut searcher = Searcher::new(&self.nnue);
        searcher
            .best_move(&board, depth)
            .map(|m| m.to_string())
            .unwrap_or_else(|| "e2e4".to_string())
    }
}
