"""Repetition-hash uniqueness tests (audit fix-pack v1).

Before the fix, ``board_hash`` used the first letter of each piece-kind name,
so KING / KNIGHT both became 'K' and CHARIOT / CANNON both became 'C'. Two
otherwise-different positions could collide and falsely trigger threefold
repetition.

These tests assert that:
  * KING ≠ KNIGHT for the same square (Chess side)
  * CHARIOT ≠ CANNON for the same square (Xiangqi side)
  * The new XQ_QUEEN piece collides with neither QUEEN nor any Xiangqi piece
  * Python ``board_hash`` and C++ ``Board.board_hash`` produce the same string
    for the same position+side-to-move (cross-language consistency).
"""

from __future__ import annotations

from hybrid.core.board import Board
from hybrid.core.rules import board_hash
from hybrid.core.types import Piece, PieceKind, Side

from hybrid.cpp_engine import hybrid_cpp_engine as cpp


def _royals(b: Board) -> None:
    """Both sides must have a royal for the C++ board API to behave."""
    b.set(4, 0, Piece(PieceKind.KING, Side.CHESS))
    b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))


def _build_cpp_board(py_board: Board):
    cpp_board = cpp.Board.empty()
    for x, y, p in py_board.iter_pieces():
        cpp_kind = getattr(cpp.PieceKind, p.kind.name)
        cpp_side = cpp.Side.CHESS if p.side == Side.CHESS else cpp.Side.XIANGQI
        cpp_board.set(x, y, cpp.Piece(cpp_kind, cpp_side))
    return cpp_board


# Collision tests (Python)

def test_king_and_knight_hash_differently_chess():
    a = Board.empty(); _royals(a)
    a.set(2, 2, Piece(PieceKind.KING, Side.CHESS))  # additional Chess king-shape
    # Note: two Chess kings is not a legal game state but board_hash is purely
    # spatial; that's exactly the coverage we want.

    b = Board.empty(); _royals(b)
    b.set(2, 2, Piece(PieceKind.KNIGHT, Side.CHESS))

    assert board_hash(a, Side.CHESS) != board_hash(b, Side.CHESS), (
        "KING and KNIGHT must produce different repetition-hash tokens"
    )


def test_chariot_and_cannon_hash_differently_xiangqi():
    a = Board.empty(); _royals(a)
    a.set(2, 5, Piece(PieceKind.CHARIOT, Side.XIANGQI))

    b = Board.empty(); _royals(b)
    b.set(2, 5, Piece(PieceKind.CANNON, Side.XIANGQI))

    assert board_hash(a, Side.CHESS) != board_hash(b, Side.CHESS), (
        "CHARIOT and CANNON must produce different repetition-hash tokens"
    )


def test_xq_queen_does_not_collide_with_chess_queen():
    a = Board.empty(); _royals(a)
    a.set(0, 5, Piece(PieceKind.QUEEN, Side.CHESS))

    b = Board.empty(); _royals(b)
    b.set(0, 5, Piece(PieceKind.XQ_QUEEN, Side.XIANGQI))

    assert board_hash(a, Side.CHESS) != board_hash(b, Side.CHESS)


def test_side_to_move_changes_hash():
    a = Board.empty(); _royals(a)
    assert board_hash(a, Side.CHESS) != board_hash(a, Side.XIANGQI)


# Cross-language Python ↔ C++ consistency

def _check_py_cpp_match(py_board: Board, side: Side) -> None:
    cpp_board = _build_cpp_board(py_board)
    cpp_side = cpp.Side.CHESS if side == Side.CHESS else cpp.Side.XIANGQI
    py_h = board_hash(py_board, side)
    cpp_h = cpp_board.board_hash(cpp_side)
    assert py_h == cpp_h, (
        f"Python and C++ board_hash differ for side={side}: {py_h} vs {cpp_h}"
    )


def test_py_cpp_hash_match_initial_default():
    from hybrid.core.board import initial_board
    b = initial_board()
    _check_py_cpp_match(b, Side.CHESS)
    _check_py_cpp_match(b, Side.XIANGQI)


def test_py_cpp_hash_match_xq_queen_variant():
    from hybrid.core.board import initial_board
    from hybrid.core.config import VariantConfig
    b = initial_board(VariantConfig(xq_queen=True))
    _check_py_cpp_match(b, Side.CHESS)
    _check_py_cpp_match(b, Side.XIANGQI)


def test_py_cpp_hash_match_distinguishes_king_vs_knight():
    """C++ side must also distinguish KING from KNIGHT (the actual fix)."""
    a = Board.empty(); _royals(a)
    a.set(2, 2, Piece(PieceKind.KING, Side.CHESS))
    b = Board.empty(); _royals(b)
    b.set(2, 2, Piece(PieceKind.KNIGHT, Side.CHESS))

    a_cpp = _build_cpp_board(a).board_hash(cpp.Side.CHESS)
    b_cpp = _build_cpp_board(b).board_hash(cpp.Side.CHESS)
    assert a_cpp != b_cpp


def test_py_cpp_hash_match_distinguishes_chariot_vs_cannon():
    a = Board.empty(); _royals(a)
    a.set(2, 5, Piece(PieceKind.CHARIOT, Side.XIANGQI))
    b = Board.empty(); _royals(b)
    b.set(2, 5, Piece(PieceKind.CANNON, Side.XIANGQI))

    a_cpp = _build_cpp_board(a).board_hash(cpp.Side.CHESS)
    b_cpp = _build_cpp_board(b).board_hash(cpp.Side.CHESS)
    assert a_cpp != b_cpp
