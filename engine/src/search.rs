use crate::nnue::Nnue;
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece};
use std::collections::HashMap;

const MATE_SCORE: i32 = 100_000;
const TT_SIZE: usize = 1 << 20;
const NULL_REDUCTION: u8 = 2;
const LMR_DEPTH: u8 = 3;
const LMR_MOVES: usize = 4;

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
    history: [[i32; 64]; 12],
    killers: [[Option<ChessMove>; 2]; 64],
    ply: usize,
}

impl<'a> Searcher<'a> {
    pub fn new(nnue: &'a Nnue) -> Self {
        Self {
            nnue,
            nodes: 0,
            max_nodes: None,
            tt: vec![None; TT_SIZE],
            history: [[0; 64]; 12],
            killers: [[None; 2]; 64],
            ply: 0,
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

    fn piece_index(piece: Piece, color: chess::Color) -> usize {
        color.to_index() * 6 + piece.to_index()
    }

    fn move_order_score(&self, board: &Board, mv: ChessMove, tt_move: Option<ChessMove>) -> i32 {
        if Some(mv) == tt_move {
            return 10_000_000;
        }
        if let Some(captured) = board.piece_on(mv.get_dest()) {
            let moved = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
            return piece_value(captured) - piece_value(moved) + 1_000_000;
        }
        if mv.get_promotion().is_some() {
            return 900_000;
        }
        let p = Self::piece_index(board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn), board.side_to_move());
        let s = mv.get_dest().to_index();
        let hist = self.history[p][s];
        if hist != 0 {
            return 100_000 + hist.min(99_999);
        }
        if self.killers[self.ply].contains(&Some(mv)) {
            return 50_000;
        }
        0
    }

    fn sort_moves(&self, board: &Board, moves: &mut Vec<ChessMove>, tt_move: Option<ChessMove>) {
        moves.sort_by_key(|mv| -self.move_order_score(board, *mv, tt_move));
    }

    pub fn best_move(&mut self, board: &Board, max_depth: u8) -> Option<(ChessMove, i32)> {
        let mut legal: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal.is_empty() {
            return None;
        }

        let mut scores: HashMap<ChessMove, i32> = HashMap::new();
        let mut alpha = i32::MIN + 1;
        let beta = i32::MAX;

        for depth in 1..=max_depth {
            if self.should_stop() {
                break;
            }
            legal.sort_by_key(|mv| -scores.get(mv).copied().unwrap_or(-10_000_000));

            let mut current_alpha = alpha;
            let mut current_best = legal[0];
            for &mv in &legal {
                if self.should_stop() {
                    break;
                }
                let next = board.make_move_new(mv);
                let score = -self.negamax(&next, depth - 1, -beta, -current_alpha, true, true);
                scores.insert(mv, score);
                if score > current_alpha {
                    current_alpha = score;
                    current_best = mv;
                }
            }
            alpha = current_alpha;
            legal[0] = current_best;
            eprintln!("info depth {} score {} nodes {}", depth, alpha, self.nodes);
        }

        Some((legal[0], alpha))
    }

    fn negamax(&mut self, board: &Board, depth: u8, alpha: i32, beta: i32, allow_null: bool, use_tt: bool) -> i32 {
        if self.should_stop() {
            return self.nnue.evaluate(board);
        }
        self.nodes += 1;

        let is_check = board.checkers().popcnt() > 0;

        if let Some(score) = self.probe_tt(board, depth, alpha, beta) {
            if use_tt {
                return score;
            }
        }

        let mut legal: Vec<ChessMove> = MoveGen::new_legal(board).collect();
        if legal.is_empty() {
            return if is_check { -(MATE_SCORE - self.ply as i32) } else { 0 };
        }

        if depth == 0 {
            return self.quiescence(board, alpha, beta);
        }

        let tt_move = if use_tt {
            self.tt[self.tt_index(board.get_hash())].and_then(|e| if e.key == board.get_hash() { e.best_move } else { None })
        } else {
            None
        };

        // Null move pruning
        if allow_null && !is_check && self.ply > 0 && depth > NULL_REDUCTION {
            let next = board.null_move();
            if let Some(n) = next {
                self.ply += 1;
                let score = -self.negamax(&n, depth.saturating_sub(1 + NULL_REDUCTION), -beta, -beta + 1, false, use_tt);
                self.ply -= 1;
                if score >= beta {
                    return beta;
                }
            }
        }

        self.sort_moves(board, &mut legal, tt_move);
        let mut alpha = alpha;
        let mut best_move = None;
        let mut flag = TtFlag::Upper;

        for (i, &mv) in legal.iter().enumerate() {
            self.ply += 1;
            let next = board.make_move_new(mv);
            let new_depth = if is_check { depth } else { depth.saturating_sub(1) };
            let mut score;
            if i == 0 {
                score = -self.negamax(&next, new_depth, -beta, -alpha, true, use_tt);
            } else {
                let reduction = if depth >= LMR_DEPTH && i >= LMR_MOVES && !mv.get_promotion().is_some() && board.piece_on(mv.get_dest()).is_none() {
                    1
                } else {
                    0
                };
                score = -self.negamax(&next, new_depth.saturating_sub(reduction), -alpha - 1, -alpha, true, use_tt);
                if score > alpha && (reduction > 0 || score < beta) {
                    score = -self.negamax(&next, new_depth, -beta, -alpha, true, use_tt);
                }
            }
            self.ply -= 1;

            if score >= beta {
                if board.piece_on(mv.get_dest()).is_none() {
                    let p = Self::piece_index(board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn), board.side_to_move());
                    let s = mv.get_dest().to_index();
                    self.history[p][s] += depth as i32 * depth as i32;
                    if self.killers[self.ply][0] != Some(mv) {
                        self.killers[self.ply][1] = self.killers[self.ply][0];
                        self.killers[self.ply][0] = Some(mv);
                    }
                }
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
        captures.sort_by_key(|mv| -capture_value(board, *mv));

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

fn capture_value(board: &Board, mv: ChessMove) -> i32 {
    let captured = board.piece_on(mv.get_dest()).unwrap_or(Piece::Pawn);
    let moved = board.piece_on(mv.get_source()).unwrap_or(Piece::Pawn);
    piece_value(captured) - piece_value(moved)
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
        let (mv, _) = searcher.best_move(&board, 3).unwrap();
        let legal: Vec<_> = MoveGen::new_legal(&board).collect();
        assert!(legal.contains(&mv));
    }
}
