use chess::{Board, ChessMove, Color, Piece};
use std::fs;

pub struct NnueWeights {
    pub feature_weights: Vec<i16>, // [12 * 64 * ft_hidden]
    pub feature_biases: Vec<i16>,  // [ft_hidden]
    pub hidden1_weights: Vec<i16>, // [ft_hidden * hidden1_size]
    pub hidden1_biases: Vec<i16>,  // [hidden1_size]
    pub hidden2_weights: Vec<i16>, // [hidden1_size]
    pub hidden2_bias: i16,
    pub ft_hidden: usize,
    pub hidden1_size: usize,
}

impl NnueWeights {
    pub fn zero(ft_hidden: usize, hidden1_size: usize) -> Self {
        Self {
            feature_weights: vec![0; 12 * 64 * ft_hidden],
            feature_biases: vec![0; ft_hidden],
            hidden1_weights: vec![0; ft_hidden * hidden1_size],
            hidden1_biases: vec![0; hidden1_size],
            hidden2_weights: vec![0; hidden1_size],
            hidden2_bias: 0,
            ft_hidden,
            hidden1_size,
        }
    }

    pub fn from_file(path: &str) -> Self {
        let bytes = fs::read(path).expect("nnue file");
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let ft_hidden = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let hidden1_size = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let mut offset = 8;

        let ft_w_count = 12 * 64 * ft_hidden;
        let mut feature_weights = vec![0i16; ft_w_count];
        for weight in &mut feature_weights {
            *weight = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut feature_biases = vec![0i16; ft_hidden];
        for bias in &mut feature_biases {
            *bias = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut hidden1_weights = vec![0i16; ft_hidden * hidden1_size];
        for weight in &mut hidden1_weights {
            *weight = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut hidden1_biases = vec![0i16; hidden1_size];
        for bias in &mut hidden1_biases {
            *bias = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut hidden2_weights = vec![0i16; hidden1_size];
        for weight in &mut hidden2_weights {
            *weight = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let hidden2_bias = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        Self {
            feature_weights,
            feature_biases,
            hidden1_weights,
            hidden1_biases,
            hidden2_weights,
            hidden2_bias,
            ft_hidden,
            hidden1_size,
        }
    }
}

pub struct Nnue {
    weights: NnueWeights,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NnueAccumulator {
    white: Vec<i32>,
    black: Vec<i32>,
}

impl NnueAccumulator {
    fn new(weights: &NnueWeights) -> Self {
        Self {
            white: weights.feature_biases.iter().map(|&b| b as i32).collect(),
            black: weights.feature_biases.iter().map(|&b| b as i32).collect(),
        }
    }

    fn update_piece(
        &mut self,
        nnue: &Nnue,
        piece: Piece,
        square: chess::Square,
        color: Color,
        add: i32,
    ) {
        let white_index = Nnue::feature_index(piece, square, color, Color::White);
        let black_index = Nnue::feature_index(piece, square, color, Color::Black);
        for i in 0..nnue.weights.ft_hidden {
            self.white[i] +=
                add * nnue.weights.feature_weights[white_index * nnue.weights.ft_hidden + i] as i32;
            self.black[i] +=
                add * nnue.weights.feature_weights[black_index * nnue.weights.ft_hidden + i] as i32;
        }
    }

    pub fn apply_move(&mut self, nnue: &Nnue, board: &Board, mv: ChessMove) -> Board {
        let next = board.make_move_new(mv);
        for square in chess::ALL_SQUARES {
            let before = board.piece_on(square).zip(board.color_on(square));
            let after = next.piece_on(square).zip(next.color_on(square));
            if before != after {
                if let Some((piece, color)) = before {
                    self.update_piece(nnue, piece, square, color, -1);
                }
                if let Some((piece, color)) = after {
                    self.update_piece(nnue, piece, square, color, 1);
                }
            }
        }
        next
    }
}

impl Nnue {
    pub fn new(weights: NnueWeights) -> Self {
        Self { weights }
    }

    pub fn accumulator(&self, board: &Board) -> NnueAccumulator {
        let mut accumulator = NnueAccumulator::new(&self.weights);
        for square in chess::ALL_SQUARES {
            if let Some((piece, color)) = board.piece_on(square).zip(board.color_on(square)) {
                accumulator.update_piece(self, piece, square, color, 1);
            }
        }
        accumulator
    }

    fn piece_index(piece: chess::Piece, color: chess::Color) -> usize {
        let p = piece.to_index();
        let c = color.to_index();
        c * 6 + p
    }

    fn square_index(sq: chess::Square, perspective: chess::Color) -> usize {
        let file = sq.get_file().to_index();
        let rank = sq.get_rank().to_index();
        if perspective == chess::Color::White {
            rank * 8 + file
        } else {
            (7 - rank) * 8 + file
        }
    }

    fn feature_index(
        piece: chess::Piece,
        square: chess::Square,
        piece_color: chess::Color,
        perspective: chess::Color,
    ) -> usize {
        let relative_color = if piece_color == perspective {
            chess::Color::White
        } else {
            chess::Color::Black
        };
        let pidx = Self::piece_index(piece, relative_color);
        let sqidx = Self::square_index(square, perspective);
        pidx * 64 + sqidx
    }

    pub fn evaluate_accumulator(&self, board: &Board, accumulator: &NnueAccumulator) -> i32 {
        let hidden1_size = self.weights.hidden1_size;
        let acc = if board.side_to_move() == Color::White {
            &accumulator.white
        } else {
            &accumulator.black
        };

        let mut hidden1 = vec![0i32; hidden1_size];
        for (j, hidden) in hidden1.iter_mut().enumerate() {
            let mut sum = self.weights.hidden1_biases[j] as i32;
            for (i, &value) in acc.iter().enumerate() {
                let v = value.max(0);
                sum += v * self.weights.hidden1_weights[i * hidden1_size + j] as i32;
            }
            *hidden = sum;
        }

        let mut out = self.weights.hidden2_bias as i32;
        for (j, &value) in hidden1.iter().enumerate() {
            let v = value.max(0);
            out += v * self.weights.hidden2_weights[j] as i32;
        }
        out
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        let accumulator = self.accumulator(board);
        self.evaluate_accumulator(board, &accumulator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::{Board, MoveGen};

    #[test]
    fn zero_weights_eval_is_zero() {
        let nnue = Nnue::new(NnueWeights::zero(256, 32));
        let board = Board::default();
        assert_eq!(nnue.evaluate(&board), 0);
    }

    #[test]
    fn incremental_accumulator_matches_full_recompute() {
        let nnue = Nnue::new(NnueWeights::from_bytes(include_bytes!(
            "../assets/fixtures/tinyccrl-test.nnue"
        )));
        let boards = [
            Board::default(),
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                .parse()
                .unwrap(),
            "8/P6k/8/8/8/8/6K1/8 w - - 0 1".parse().unwrap(),
            "8/8/8/3pP3/8/8/8/4K2k w - d6 0 2".parse().unwrap(),
        ];
        for board in boards {
            let accumulator = nnue.accumulator(&board);
            assert_eq!(
                nnue.evaluate_accumulator(&board, &accumulator),
                nnue.evaluate(&board)
            );

            for mv in MoveGen::new_legal(&board) {
                let mut incremental = accumulator.clone();
                let next = incremental.apply_move(&nnue, &board, mv);
                let full = nnue.accumulator(&next);
                assert_eq!(incremental, full, "incremental update for {mv}");
                assert_eq!(
                    nnue.evaluate_accumulator(&next, &incremental),
                    nnue.evaluate(&next),
                    "incremental evaluation for {mv}"
                );
            }
        }
    }

    #[test]
    fn fixture_evaluations_are_stable() {
        let nnue = Nnue::new(NnueWeights::from_bytes(include_bytes!(
            "../assets/fixtures/tinyccrl-test.nnue"
        )));
        let cases = [
            ("startpos", Board::default()),
            (
                "tactical",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
                    .parse()
                    .unwrap(),
            ),
            ("endgame", "8/8/3k4/8/3K4/8/8/8 b - - 0 1".parse().unwrap()),
        ];
        let expected = [78, 422, 100];
        for ((name, board), expected) in cases.into_iter().zip(expected) {
            assert_eq!(nnue.evaluate(&board), expected, "{name}");
        }
    }
}
