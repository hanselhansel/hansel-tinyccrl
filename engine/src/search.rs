use crate::nnue::Nnue;
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece};

const MATE_SCORE: i32 = 100_000;

pub struct Searcher<'a> {
    nnue: &'a Nnue,
    nodes: u64,
    max_nodes: Option<u64>,
}

impl<'a> Searcher<'a> {
    pub fn new(nnue: &'a Nnue) -> Self {
        Self {
            nnue,
            nodes: 0,
            max_nodes: None,
        }
    }

    pub fn with_max_nodes(mut self, max_nodes: u64) -> Self {
        self.max_nodes = Some(max_nodes);
        self
    }

    fn should_stop(&self) -> bool {
        self.max_nodes.map_or(false, |max| self.nodes >= max)
    }

    pub fn best_move(&mut self, board: &Board, max_depth: u8) -> Option<ChessMove> {
        let mut legal: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal.is_empty() {
            return None;
        }
        legal.sort_by_key(|mv| -move_score(board, *mv));
        let mut best = legal[0];
        for depth in 1..=max_depth {
            if self.should_stop() {
                break;
            }
            let mut current_best = best;
            let mut alpha = i32::MIN + 1;
            let beta = i32::MAX;
            for &mv in &legal {
                if self.should_stop() {
                    break;
                }
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
        if self.should_stop() {
            return self.nnue.evaluate(board);
        }
        self.nodes += 1;
        if depth == 0 {
            return self.nnue.evaluate(board);
        }
        let mut legal: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal.is_empty() {
            return match board.status() {
                BoardStatus::Checkmate => -(MATE_SCORE - depth as i32),
                BoardStatus::Stalemate => 0,
                _ => 0,
            };
        }
        legal.sort_by_key(|mv| -move_score(board, *mv));
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

fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}

fn move_score(board: &Board, mv: ChessMove) -> i32 {
    let mut score = 0;
    if let Some(captured) = board.piece_on(mv.get_dest()) {
        let moved = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
        // MVV-LVA: value of captured minus value of attacker
        score = piece_value(captured) - piece_value(moved) + 10000;
    }
    if mv.get_promotion().is_some() {
        score += 8000;
    }
    score
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
