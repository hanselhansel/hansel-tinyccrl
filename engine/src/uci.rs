use std::io::{self, BufRead, Write};

pub struct Uci {
    name: &'static str,
    author: &'static str,
}

impl Uci {
    pub fn new() -> Self {
        Self {
            name: "TinyCCRL",
            author: "Hansel",
        }
    }

    pub fn run(&self) {
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
                "position" => {}
                "go" => println!("bestmove e2e4"),
                "quit" => break,
                _ => {}
            }
        }
    }
}
