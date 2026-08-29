import chess

from tinyccrl.features import fen_to_indices


def test_start_position_feature_indices_are_deterministic():
    white, black, stm = fen_to_indices(chess.STARTING_FEN)

    assert stm == 0
    assert len(white) == 32
    assert len(black) == 32
    assert white[:2] == [3 * 64, 1 * 64 + 1]
    assert black[:2] == [9 * 64 + 56, 7 * 64 + 57]


def test_side_to_move_is_encoded_from_fen():
    _, _, stm = fen_to_indices("8/8/8/8/8/8/8/K6k b - - 0 1")

    assert stm == 1
