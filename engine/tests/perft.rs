use std::str::FromStr;

use chess::{Board, MoveGen};

fn perft(board: &Board, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }
    MoveGen::new_legal(board)
        .map(|mv| perft(&board.make_move_new(mv), depth - 1))
        .sum()
}

fn assert_counts(fen: &str, expected: &[u64]) {
    let board = Board::from_str(fen).expect("valid FEN");
    for (depth, &count) in expected.iter().enumerate() {
        assert_eq!(
            perft(&board, depth as u8 + 1),
            count,
            "{fen} depth {}",
            depth + 1
        );
    }
}

fn assert_count_at_depth(fen: &str, depth: u8, expected: u64) {
    let board = Board::from_str(fen).expect("valid FEN");
    assert_eq!(perft(&board, depth), expected, "{fen} depth {depth}");
}

#[test]
fn startpos_perft_depths_1_to_5() {
    assert_counts(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[20, 400, 8902, 197281, 4865609],
    );
}

#[test]
fn kiwipete_perft_depths_1_to_4() {
    assert_counts(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        &[48, 2039, 97862, 4085603],
    );
}

#[test]
fn position_3_perft_depths_1_to_5() {
    assert_counts(
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        &[14, 191, 2812, 43238, 674624],
    );
}

#[test]
fn position_4_perft_depths_1_to_4() {
    assert_counts(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[6, 264, 9467, 422333],
    );
}

#[test]
fn position_5_perft_depths_1_to_4() {
    assert_counts(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        &[44, 1486, 62379, 2103487],
    );
}

#[test]
#[ignore = "reference depth 5 is intentionally slow for CI"]
fn kiwipete_perft_depth_5() {
    assert_count_at_depth(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        5,
        193690690,
    );
}

#[test]
#[ignore = "reference depth 5 is intentionally slow for CI"]
fn position_4_perft_depth_5() {
    assert_count_at_depth(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        5,
        15833292,
    );
}

#[test]
#[ignore = "reference depth 5 is intentionally slow for CI"]
fn position_5_perft_depth_5() {
    assert_count_at_depth(
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        5,
        89941194,
    );
}
