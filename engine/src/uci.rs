use std::io::{self, BufRead, Write};
use std::str::FromStr;

use chess::{Board, ChessMove};

pub struct Uci {
    name: &'static str,
    author: &'static str,
    board: Option<Board>,
}

impl Uci {
    pub fn new() -> Self {
        Self {
            name: "TinyCCRL",
            author: "Hansel",
            board: None,
        }
    }

    pub fn run(&mut self) {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut lock = stdin.lock();
        let mut line = String::new();
        loop {
            line.clear();
            if lock.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            let cmd = parts.first().copied().unwrap_or("");
            match cmd {
                "uci" => {
                    let mut out = stdout.lock();
                    writeln!(out, "id name {}", self.name).unwrap();
                    writeln!(out, "id author {}", self.author).unwrap();
                    writeln!(out, "uciok").unwrap();
                }
                "isready" => println!("readyok"),
                "position" => self.handle_position(&parts),
                "go" => self.handle_go(&parts),
                "quit" => break,
                _ => {}
            }
        }
    }

    fn handle_position(&mut self, parts: &[&str]) {
        let mut board = if parts.get(1) == Some(&"startpos") {
            Board::default()
        } else if parts.get(1) == Some(&"fen") {
            let rest = &parts[2..];
            let moves_idx = rest.iter().position(|&p| p == "moves").unwrap_or(rest.len());
            let fen = rest[..moves_idx].join(" ");
            Board::from_str(&fen).ok().unwrap_or_else(Board::default)
        } else {
            Board::default()
        };

        if let Some(moves_idx) = parts.iter().position(|&p| p == "moves") {
            for mv_str in &parts[moves_idx + 1..] {
                if let Ok(mv) = ChessMove::from_str(mv_str) {
                    board = board.make_move_new(mv);
                }
            }
        }

        self.board = Some(board);
    }

    fn handle_go(&self, _parts: &[&str]) {
        if let Some(board) = self.board {
            let legal: Vec<ChessMove> = chess::MoveGen::new_legal(&board).collect();
            let mv = legal.into_iter().next().unwrap_or_else(|| {
                ChessMove::new(
                    chess::Square::make_square(chess::Rank::Second, chess::File::E),
                    chess::Square::make_square(chess::Rank::Fourth, chess::File::E),
                    None,
                )
            });
            println!("bestmove {}", mv);
        } else {
            println!("bestmove e2e4");
        }
    }
}
