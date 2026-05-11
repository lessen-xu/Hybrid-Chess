"""XQ_QUEEN semantics tests (audit fix-pack v1).

Verifies that the new ``PieceKind.XQ_QUEEN`` is wired correctly through:
  * initial board layout under ``xq_queen=True``
  * Python and C++ move generation (queen-like = orthogonal + diagonal sliding)
  * Python and C++ attack detection (Chess King in check from XQ_QUEEN)
  * C++ fast vs. slow attack detector (must agree)
  * State encoding (XQ_QUEEN gets a *distinct* channel — channel 13 — that is
    never co-activated with the Chess QUEEN channel)
"""

from __future__ import annotations

import pytest

from hybrid.core.board import Board, initial_board
from hybrid.core.config import MAX_PLIES, VariantConfig
from hybrid.core.env import HybridChessEnv
from hybrid.core.rules import (
    generate_legal_moves,
    is_in_check,
    _piece_moves,
)
from hybrid.core.types import Move, Piece, PieceKind, Side
from hybrid.rl.az_encoding import (
    NUM_PIECE_CHANNELS,
    PIECE_CHANNELS,
    encode_state,
)
from hybrid.core.env import GameState

from hybrid.cpp_engine import hybrid_cpp_engine as cpp


# Initial-setup tests

def test_initial_setup_places_xq_queen():
    """xq_queen=True must replace the LEFT advisor with a XQ_QUEEN piece."""
    env = HybridChessEnv(variant=VariantConfig(xq_queen=True))
    state = env.reset()
    piece = state.board.get(3, 9)  # left advisor square
    assert piece is not None
    assert piece.side == Side.XIANGQI
    assert piece.kind == PieceKind.XQ_QUEEN, (
        f"Expected XQ_QUEEN at (3,9), got {piece.kind}"
    )
    # Right advisor is untouched.
    right = state.board.get(5, 9)
    assert right is not None and right.kind == PieceKind.ADVISOR


def test_default_variant_has_no_xq_queen():
    """Sanity: without xq_queen flag, the board has no XQ_QUEEN piece."""
    env = HybridChessEnv()
    state = env.reset()
    for _, _, p in state.board.iter_pieces():
        assert p.kind != PieceKind.XQ_QUEEN


# State encoding

def test_xq_queen_uses_distinct_channel():
    """Chess QUEEN and XQ_QUEEN must activate different encoding channels."""
    chess_ch = PIECE_CHANNELS[PieceKind.QUEEN]
    xq_ch = PIECE_CHANNELS[PieceKind.XQ_QUEEN]
    assert chess_ch != xq_ch, "XQ_QUEEN must have its own channel"
    assert 0 <= xq_ch < NUM_PIECE_CHANNELS


def test_encoding_distinguishes_chess_queen_from_xq_queen():
    """Place both queens on a near-empty board and check channel ownership."""
    b = Board.empty()
    b.set(4, 0, Piece(PieceKind.KING, Side.CHESS))
    b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
    b.set(0, 0, Piece(PieceKind.QUEEN, Side.CHESS))     # Chess queen at (0,0)
    b.set(0, 9, Piece(PieceKind.XQ_QUEEN, Side.XIANGQI))  # XQ queen at (0,9)

    state = GameState(board=b, side_to_move=Side.CHESS)
    t = encode_state(state)

    chess_ch = PIECE_CHANNELS[PieceKind.QUEEN]
    xq_ch = PIECE_CHANNELS[PieceKind.XQ_QUEEN]

    assert t[chess_ch, 0, 0].item() == 1.0, "Chess QUEEN channel should fire at (0,0)"
    assert t[xq_ch, 0, 0].item() == 0.0, "XQ_QUEEN channel should NOT fire where Chess Queen is"

    assert t[xq_ch, 9, 0].item() == 1.0, "XQ_QUEEN channel should fire at (0,9)"
    assert t[chess_ch, 9, 0].item() == 0.0, "Chess QUEEN channel should NOT fire where XQ Queen is"


# Move generation (Python)

def test_python_xq_queen_moves_in_open_space():
    """XQ_QUEEN on an empty board reaches like a Chess Queen (8 rays)."""
    b = Board.empty()
    # Place royals far away so legality filtering is trivial.
    b.set(4, 0, Piece(PieceKind.KING, Side.CHESS))
    b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
    b.set(4, 4, Piece(PieceKind.XQ_QUEEN, Side.XIANGQI))

    p = b.get(4, 4)
    moves = _piece_moves(b, 4, 4, p)
    targets = {(m.tx, m.ty) for m in moves}

    # Eight directions through (4,4) on a 9x10 board, blocked only by the two
    # royals on file 4 at (4,0) and (4,9).
    expected = set()
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        cx, cy = 4 + dx, 4 + dy
        while 0 <= cx < 9 and 0 <= cy < 10:
            blocker = b.get(cx, cy)
            if blocker is not None:
                # Blocker is friendly only if XIANGQI; the kings are CHESS/XIANGQI.
                if blocker.side != Side.XIANGQI:
                    expected.add((cx, cy))  # capture chess king square
                break
            expected.add((cx, cy))
            cx += dx
            cy += dy
    assert targets == expected


# Python ↔ C++ legal-move parity under xq_queen

def _build_cpp_board(py_board: Board):
    cpp_board = cpp.Board.empty()
    for x, y, p in py_board.iter_pieces():
        cpp_kind = getattr(cpp.PieceKind, p.kind.name)
        cpp_side = cpp.Side.CHESS if p.side == Side.CHESS else cpp.Side.XIANGQI
        cpp_board.set(x, y, cpp.Piece(cpp_kind, cpp_side))
    return cpp_board


def _move_set(moves):
    """Normalize Move into (fx, fy, tx, ty, promo_name_or_NONE) tuples.
    Both Python (promotion=None) and C++ (promotion=PieceKind.NONE) collapse
    to the literal string ``"NONE"``."""
    out = set()
    for m in moves:
        if hasattr(m, "promotion") and m.promotion is not None:
            name = getattr(m.promotion, "name", "NONE")
        else:
            name = "NONE"
        out.add((m.fx, m.fy, m.tx, m.ty, name))
    return out


def test_python_cpp_legal_moves_match_xq_queen_initial():
    """Initial xq_queen board: Py and C++ must agree on the legal-move set."""
    env = HybridChessEnv(variant=VariantConfig(xq_queen=True), use_cpp=False)
    state = env.reset()
    py_moves = generate_legal_moves(state.board, Side.CHESS)

    cpp_board = _build_cpp_board(state.board)
    cpp_moves = cpp.generate_legal_moves(cpp_board, cpp.Side.CHESS)

    assert _move_set(py_moves) == _move_set(cpp_moves)


# C++ fast vs. slow attack detector

def test_cpp_fast_attack_detects_xq_queen_orthogonal():
    """Chess King on (4,4); XQ_QUEEN on (4,9) on an open file → in check."""
    b = Board.empty()
    b.set(4, 4, Piece(PieceKind.KING, Side.CHESS))
    b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))  # XQ royal must exist
    b.set(4, 7, Piece(PieceKind.XQ_QUEEN, Side.XIANGQI))

    # Python is_in_check must agree.
    assert is_in_check(b, Side.CHESS) is True

    cpp_board = _build_cpp_board(b)
    fast = cpp.is_square_attacked_fast(cpp_board, 4, 4, cpp.Side.XIANGQI)
    slow = cpp.is_square_attacked_slow(cpp_board, 4, 4, cpp.Side.XIANGQI)
    assert fast == slow == True  # noqa: E712


def test_cpp_fast_attack_detects_xq_queen_diagonal():
    """Chess King on (4,4); XQ_QUEEN on (7,7) along an open diagonal → in check."""
    b = Board.empty()
    b.set(4, 4, Piece(PieceKind.KING, Side.CHESS))
    b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
    b.set(7, 7, Piece(PieceKind.XQ_QUEEN, Side.XIANGQI))

    assert is_in_check(b, Side.CHESS) is True

    cpp_board = _build_cpp_board(b)
    fast = cpp.is_square_attacked_fast(cpp_board, 4, 4, cpp.Side.XIANGQI)
    slow = cpp.is_square_attacked_slow(cpp_board, 4, 4, cpp.Side.XIANGQI)
    assert fast == slow == True  # noqa: E712


def test_cpp_fast_attack_blocked_by_intervening_piece():
    """A blocker on the diagonal should prevent the XQ_QUEEN from attacking.

    The General is placed off-file from the Chess King so the flying-general
    rule cannot independently attack (4,4) — only the XQ_QUEEN-via-diagonal
    attack is being tested.
    """
    b = Board.empty()
    b.set(4, 4, Piece(PieceKind.KING, Side.CHESS))
    b.set(3, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))  # different file from king
    b.set(7, 7, Piece(PieceKind.XQ_QUEEN, Side.XIANGQI))
    b.set(5, 5, Piece(PieceKind.PAWN, Side.CHESS))  # blocks the diagonal

    cpp_board = _build_cpp_board(b)
    fast = cpp.is_square_attacked_fast(cpp_board, 4, 4, cpp.Side.XIANGQI)
    slow = cpp.is_square_attacked_slow(cpp_board, 4, 4, cpp.Side.XIANGQI)
    assert fast == slow == False  # noqa: E712


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_cpp_fast_vs_slow_consistency_random_xq_queen_games(seed):
    """Play short random games under xq_queen and assert fast == slow attack
    detector results at every position."""
    import random

    rng = random.Random(seed)
    env = HybridChessEnv(variant=VariantConfig(xq_queen=True), use_cpp=True)
    state = env.reset()

    for _ in range(40):
        moves = env.legal_moves()
        if not moves:
            break
        mv = rng.choice(moves)
        state, _, done, _ = env.step(mv)

        cpp_board = _build_cpp_board(state.board)
        for side in (cpp.Side.CHESS, cpp.Side.XIANGQI):
            # Sample a few squares for speed; royals + a couple of empty squares.
            for sx, sy in [(4, 0), (4, 9), (0, 0), (8, 9), (4, 4)]:
                fast = cpp.is_square_attacked_fast(cpp_board, sx, sy, side)
                slow = cpp.is_square_attacked_slow(cpp_board, sx, sy, side)
                assert fast == slow, (
                    f"fast/slow mismatch seed={seed} square=({sx},{sy}) side={side}"
                )

        if done:
            break
