use chess::{Board, Color, Piece, Square};
use std::fs;

pub struct NnueWeights {
    pub feature_weights: Vec<i16>, // [12 * 64 * hidden_size]
    pub feature_biases: Vec<i16>,  // [hidden_size]
    pub hidden_weights: Vec<i16>,  // [hidden_size]
    pub hidden_bias: i16,
    pub hidden_size: usize,
}

impl NnueWeights {
    pub fn zero(hidden_size: usize) -> Self {
        Self {
            feature_weights: vec![0; 12 * 64 * hidden_size],
            feature_biases: vec![0; hidden_size],
            hidden_weights: vec![0; hidden_size],
            hidden_bias: 0,
            hidden_size,
        }
    }

    pub fn from_file(path: &str) -> Self {
        let bytes = fs::read(path).expect("nnue file");
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        let hidden_size = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let mut offset = 4;
        let ft_w_count = 12 * 64 * hidden_size;
        let mut feature_weights = vec![0i16; ft_w_count];
        for i in 0..ft_w_count {
            feature_weights[i] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut feature_biases = vec![0i16; hidden_size];
        for i in 0..hidden_size {
            feature_biases[i] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut hidden_weights = vec![0i16; hidden_size];
        for i in 0..hidden_size {
            hidden_weights[i] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let hidden_bias = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
        Self {
            feature_weights,
            feature_biases,
            hidden_weights,
            hidden_bias,
            hidden_size,
        }
    }
}

pub struct Nnue {
    weights: NnueWeights,
}

impl Nnue {
    pub fn new(weights: NnueWeights) -> Self {
        Self { weights }
    }

    fn piece_index(piece: Piece, color: Color) -> usize {
        let p = piece.to_index();
        let c = color.to_index();
        c * 6 + p
    }

    fn square_index(sq: Square, perspective: Color) -> usize {
        let file = sq.get_file().to_index();
        let rank = sq.get_rank().to_index();
        if perspective == Color::White {
            rank * 8 + file
        } else {
            (7 - rank) * 8 + file
        }
    }

    fn feature_index(piece: Piece, square: Square, piece_color: Color, perspective: Color) -> usize {
        let relative_color = if piece_color == perspective { Color::White } else { Color::Black };
        let pidx = Self::piece_index(piece, relative_color);
        let sqidx = Self::square_index(square, perspective);
        pidx * 64 + sqidx
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        let hidden_size = self.weights.hidden_size;
        let mut white_acc: Vec<i32> = self.weights.feature_biases.iter().map(|&b| b as i32).collect();
        let mut black_acc: Vec<i32> = self.weights.feature_biases.iter().map(|&b| b as i32).collect();

        for sq in *board.combined() {
            let piece = board.piece_on(sq).expect("occupied square");
            let color = board.color_on(sq).expect("occupied square");
            let fw = Self::feature_index(piece, sq, color, Color::White);
            let fb = Self::feature_index(piece, sq, color, Color::Black);
            for i in 0..hidden_size {
                white_acc[i] += self.weights.feature_weights[fw * hidden_size + i] as i32;
                black_acc[i] += self.weights.feature_weights[fb * hidden_size + i] as i32;
            }
        }

        let acc = if board.side_to_move() == Color::White {
            &white_acc
        } else {
            &black_acc
        };

        let mut sum = self.weights.hidden_bias as i32;
        for i in 0..hidden_size {
            let v = acc[i].max(0);
            sum += v * self.weights.hidden_weights[i] as i32;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_weights_eval_is_zero() {
        let nnue = Nnue::new(NnueWeights::zero(256));
        let board = Board::default();
        assert_eq!(nnue.evaluate(&board), 0);
    }
}
