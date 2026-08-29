use crate::nnue::Nnue;
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece};

const MATE_SCORE: i32 = 100_000;
const TT_SIZE: usize = 1 << 20; // ~1M entries

#[derive(Clone, Copy, PartialEq, Eq)]
enum TtFlag {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    depth: u8,
    flag: TtFlag,
    score: i32,
    best_move: Option<ChessMove>,
}

pub struct Searcher<'a> {
    nnue: &'a Nnue,
    nodes: u64,
    max_nodes: Option<u64>,
    tt: Vec<Option<TtEntry>>,
}

impl<'a> Searcher<'a> {
    pub fn new(nnue: &'a Nnue) -> Self {
        Self {
            nnue,
            nodes: 0,
            max_nodes: None,
            tt: vec![None; TT_SIZE],
        }
    }

    pub fn with_max_nodes(mut self, max_nodes: u64) -> Self {
        self.max_nodes = Some(max_nodes);
        self
    }

    fn should_stop(&self) -> bool {
        self.max_nodes.map_or(false, |max| self.nodes >= max)
    }

    fn tt_index(&self, key: u64) -> usize {
        key as usize % TT_SIZE
    }

    fn probe_tt(&self, board: &Board, depth: u8, alpha: i32, beta: i32) -> Option<i32> {
        let entry = self.tt[self.tt_index(board.get_hash())]?;
        if entry.key != board.get_hash() || entry.depth < depth {
            return None;
        }
        match entry.flag {
            TtFlag::Exact => Some(entry.score),
            TtFlag::Lower if entry.score >= beta => Some(entry.score),
            TtFlag::Upper if entry.score <= alpha => Some(entry.score),
            _ => None,
        }
    }

    fn store_tt(&mut self, board: &Board, depth: u8, flag: TtFlag, score: i32, best_move: Option<ChessMove>) {
        let idx = self.tt_index(board.get_hash());
        self.tt[idx] = Some(TtEntry {
            key: board.get_hash(),
            depth,
            flag,
            score,
            best_move,
        });
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
                let score = -self.negamax(&next, depth - 1, -beta, -alpha, true);
                if score > alpha {
                    alpha = score;
                    current_best = mv;
                }
            }
            best = current_best;
        }
        Some(best)
    }

    fn negamax(&mut self, board: &Board, depth: u8, alpha: i32, beta: i32, allow_null: bool) -> i32 {
        if self.should_stop() {
            return self.nnue.evaluate(board);
        }
        self.nodes += 1;

        if let Some(score) = self.probe_tt(board, depth, alpha, beta) {
            return score;
        }

        let mut legal: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal.is_empty() {
            return match board.status() {
                BoardStatus::Checkmate => -(MATE_SCORE - depth as i32),
                BoardStatus::Stalemate => 0,
                _ => 0,
            };
        }

        if depth == 0 {
            return self.quiescence(board, alpha, beta);
        }

        legal.sort_by_key(|mv| -move_score(board, *mv));
        let mut alpha = alpha;
        let mut best_move = None;
        let mut flag = TtFlag::Upper;

        for &mv in &legal {
            let next = board.make_move_new(mv);
            let score = -self.negamax(&next, depth - 1, -beta, -alpha, true);
            if score >= beta {
                self.store_tt(board, depth, TtFlag::Lower, beta, Some(mv));
                return beta;
            }
            if score > alpha {
                alpha = score;
                best_move = Some(mv);
                flag = TtFlag::Exact;
            }
        }
        self.store_tt(board, depth, flag, alpha, best_move);
        alpha
    }

    fn quiescence(&mut self, board: &Board, mut alpha: i32, beta: i32) -> i32 {
        if self.should_stop() {
            return self.nnue.evaluate(board);
        }
        self.nodes += 1;

        let eval = self.nnue.evaluate(board);
        if eval >= beta {
            return beta;
        }
        if eval > alpha {
            alpha = eval;
        }

        let mut captures: Vec<ChessMove> = MoveGen::new_legal(board)
            .filter(|mv| board.piece_on(mv.get_dest()).is_some())
            .collect();
        captures.sort_by_key(|mv| -move_score(board, *mv));

        for &mv in &captures {
            let next = board.make_move_new(mv);
            let score = -self.quiescence(&next, -beta, -alpha);
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
        let nnue = Nnue::new(NnueWeights::zero(256, 32));
        let mut searcher = Searcher::new(&nnue);
        let board = Board::default();
        let mv = searcher.best_move(&board, 3).unwrap();
        let legal: Vec<_> = MoveGen::new_legal(&board).collect();
        assert!(legal.contains(&mv));
    }
}
