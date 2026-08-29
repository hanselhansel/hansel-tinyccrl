use chess::{Board, MoveGen};

#[test]
fn startpos_has_all_legal_moves() {
    let board = Board::default();
    assert_eq!(MoveGen::new_legal(&board).len(), 20);
}
