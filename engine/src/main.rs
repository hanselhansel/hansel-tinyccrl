mod nnue;
mod search;
mod uci;
use uci::Uci;

fn main() {
    Uci::new().run();
}
