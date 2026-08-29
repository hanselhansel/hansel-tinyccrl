use crate::nnue::Nnue;
use chess::{Board, BoardStatus, ChessMove, MoveGen};

const MATE_SCORE: i32 = 100_000;

pub struct Searcher<'a> {
    nnue: &'a Nnue,
    nodes: u64,
}

impl<'a> Searcher<'a> {
    pub fn new(nnue: &'a Nnue) -> Self {
        Self { nnue, nodes: 0 }
    }

    pub fn best_move(&mut self, board: &Board, max_depth: u8) -> Option<ChessMove> {
        let legal: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal.is_empty() {
            return None;
        }
        let mut best = legal[0];
        for depth in 1..=max_depth {
            let mut current_best = best;
            let mut alpha = i32::MIN + 1;
            let beta = i32::MAX;
            for &mv in &legal {
                let next = board.make_move_new(mv);
                let score = -self.negamax(&next, depth - 1, -beta, -alpha);
                if score > alpha {
                    alpha = score;
                    current_best = mv;
                }
            }
            best = current_best;
        }
        Some(best)
    }

    fn negamax(&mut self, board: &Board, depth: u8, alpha: i32, beta: i32) -> i32 {
        self.nodes += 1;
        if depth == 0 {
            return self.nnue.evaluate(board);
        }
        let legal: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal.is_empty() {
            return match board.status() {
                BoardStatus::Checkmate => -(MATE_SCORE - depth as i32),
                BoardStatus::Stalemate => 0,
                _ => 0,
            };
        }
        let mut alpha = alpha;
        for &mv in &legal {
            let next = board.make_move_new(mv);
            let score = -self.negamax(&next, depth - 1, -beta, -alpha);
            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }
        alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nnue::NnueWeights;
    use chess::Board;

    #[test]
    fn search_returns_legal_move() {
        let nnue = Nnue::new(NnueWeights::zero(256));
        let mut searcher = Searcher::new(&nnue);
        let board = Board::default();
        let mv = searcher.best_move(&board, 3).unwrap();
        let legal: Vec<_> = MoveGen::new_legal(&board).collect();
        assert!(legal.contains(&mv));
    }
}
