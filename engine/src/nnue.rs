use chess::{Board, Color};
use std::fs;

pub struct NnueWeights {
    pub feature_weights: Vec<i16>, // [12 * 64 * ft_hidden]
    pub feature_biases: Vec<i16>,    // [ft_hidden]
    pub hidden1_weights: Vec<i16>,   // [ft_hidden * hidden1_size]
    pub hidden1_biases: Vec<i16>,    // [hidden1_size]
    pub hidden2_weights: Vec<i16>,  // [hidden1_size]
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
        for i in 0..ft_w_count {
            feature_weights[i] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut feature_biases = vec![0i16; ft_hidden];
        for i in 0..ft_hidden {
            feature_biases[i] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut hidden1_weights = vec![0i16; ft_hidden * hidden1_size];
        for i in 0..ft_hidden * hidden1_size {
            hidden1_weights[i] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut hidden1_biases = vec![0i16; hidden1_size];
        for i in 0..hidden1_size {
            hidden1_biases[i] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            offset += 2;
        }
        let mut hidden2_weights = vec![0i16; hidden1_size];
        for i in 0..hidden1_size {
            hidden2_weights[i] = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
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

impl Nnue {
    pub fn new(weights: NnueWeights) -> Self {
        Self { weights }
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

    fn feature_index(piece: chess::Piece, square: chess::Square, piece_color: chess::Color, perspective: chess::Color) -> usize {
        let relative_color = if piece_color == perspective { chess::Color::White } else { chess::Color::Black };
        let pidx = Self::piece_index(piece, relative_color);
        let sqidx = Self::square_index(square, perspective);
        pidx * 64 + sqidx
    }

    pub fn evaluate(&self, board: &Board) -> i32 {
        let ft_hidden = self.weights.ft_hidden;
        let hidden1_size = self.weights.hidden1_size;
        let mut white_acc: Vec<i32> = self.weights.feature_biases.iter().map(|&b| b as i32).collect();
        let mut black_acc: Vec<i32> = self.weights.feature_biases.iter().map(|&b| b as i32).collect();

        for sq in *board.combined() {
            let piece = board.piece_on(sq).expect("occupied square");
            let color = board.color_on(sq).expect("occupied square");
            let fw = Self::feature_index(piece, sq, color, Color::White);
            let fb = Self::feature_index(piece, sq, color, Color::Black);
            for i in 0..ft_hidden {
                white_acc[i] += self.weights.feature_weights[fw * ft_hidden + i] as i32;
                black_acc[i] += self.weights.feature_weights[fb * ft_hidden + i] as i32;
            }
        }

        let acc = if board.side_to_move() == Color::White {
            &white_acc
        } else {
            &black_acc
        };

        let mut hidden1 = vec![0i32; hidden1_size];
        for j in 0..hidden1_size {
            let mut sum = self.weights.hidden1_biases[j] as i32;
            for i in 0..ft_hidden {
                let v = acc[i].max(0);
                sum += v * self.weights.hidden1_weights[i * hidden1_size + j] as i32;
            }
            hidden1[j] = sum;
        }

        let mut out = self.weights.hidden2_bias as i32;
        for j in 0..hidden1_size {
            let v = hidden1[j].max(0);
            out += v * self.weights.hidden2_weights[j] as i32;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_weights_eval_is_zero() {
        let nnue = Nnue::new(NnueWeights::zero(256, 32));
        let board = Board::default();
        assert_eq!(nnue.evaluate(&board), 0);
    }
}
