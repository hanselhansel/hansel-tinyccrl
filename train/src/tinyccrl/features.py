import chess


def _piece_index(piece: chess.Piece, perspective: chess.Color) -> int:
    p = piece.piece_type - 1
    c = 0 if piece.color == perspective else 1
    return c * 6 + p


def _square_index(sq: chess.Square, perspective: chess.Color) -> int:
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    if perspective == chess.WHITE:
        return rank * 8 + file
    return (7 - rank) * 8 + file


def fen_to_indices(fen: str) -> tuple[list[int], list[int], int]:
    board = chess.Board(fen)
    stm = 0 if board.turn == chess.WHITE else 1
    white_idx = []
    black_idx = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            white_idx.append(_piece_index(piece, chess.WHITE) * 64 + _square_index(sq, chess.WHITE))
            black_idx.append(_piece_index(piece, chess.BLACK) * 64 + _square_index(sq, chess.BLACK))
    return white_idx, black_idx, stm
