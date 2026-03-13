"""C++ engine tests — runs the same 40 oracle cases from test_rules.py
against the C++ engine via pybind11.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple

import pytest

from hybrid.cpp_engine import (
    Side, PieceKind, Piece, Move, Board,
    generate_legal_moves, apply_move,
    is_in_check, is_square_attacked,
    terminal_info, GameInfo,
    BOARD_W, BOARD_H, MAX_PLIES,
)


# ═══════════════════════════════════════════════════════════════
# Helper utilities (same as test_rules.py but using C++ types)
# ═══════════════════════════════════════════════════════════════

def make_board(pieces: Dict[Tuple[int, int], Piece]) -> Board:
    b = Board.empty()
    for (x, y), piece in pieces.items():
        assert b.in_bounds(x, y), f"({x},{y}) out of bounds"
        b.set(x, y, piece)
    return b


def destinations_from(moves, from_sq: Tuple[int, int]) -> Set[Tuple[int, int]]:
    return {(m.tx, m.ty) for m in moves if (m.fx, m.fy) == from_sq}


def moves_from(moves, from_sq: Tuple[int, int]):
    return [m for m in moves if (m.fx, m.fy) == from_sq]


def C(kind: PieceKind) -> Piece:
    return Piece(kind, Side.CHESS)


def X(kind: PieceKind) -> Piece:
    return Piece(kind, Side.XIANGQI)


# ═══════════════════════════════════════════════════════════════
# Group 1: Cannon (8 cases)
# ═══════════════════════════════════════════════════════════════

class TestCannon:
    def test_cannon_slide_no_obstacles(self):
        board = make_board({
            (4, 5): X(PieceKind.CANNON),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 5)
        )
        for x in range(BOARD_W):
            if x != 4:
                assert (x, 5) in dests
        for y in range(BOARD_H):
            if y in (5, 9, 0):
                continue
            assert (4, y) in dests

    def test_cannon_jump_capture(self):
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (3, 5): X(PieceKind.SOLDIER),
            (7, 5): C(PieceKind.BISHOP),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (7, 5) in dests

    def test_cannon_no_screen_cannot_capture(self):
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (7, 5): C(PieceKind.BISHOP),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (6, 5) in dests
        assert (7, 5) not in dests

    def test_cannon_double_screen_cannot_capture(self):
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (2, 5): X(PieceKind.SOLDIER),
            (5, 5): C(PieceKind.PAWN),
            (7, 5): C(PieceKind.ROOK),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (7, 5) not in dests
        assert (5, 5) in dests

    def test_cannon_cannot_jump_to_empty(self):
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (3, 5): C(PieceKind.PAWN),
            (7, 5): C(PieceKind.ROOK),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (4, 5) not in dests
        assert (5, 5) not in dests
        assert (6, 5) not in dests
        assert (7, 5) in dests

    def test_cannon_vertical_jump(self):
        board = make_board({
            (2, 0): X(PieceKind.CANNON),
            (2, 3): C(PieceKind.PAWN),
            (2, 7): C(PieceKind.BISHOP),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (2, 0)
        )
        assert (2, 7) in dests
        assert (2, 4) not in dests
        assert (2, 5) not in dests
        assert (2, 6) not in dests

    def test_cannon_friendly_screen(self):
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (3, 5): X(PieceKind.SOLDIER),
            (6, 5): C(PieceKind.ROOK),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (6, 5) in dests

    def test_cannon_cannot_capture_friendly(self):
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (3, 5): C(PieceKind.PAWN),
            (6, 5): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (6, 5) not in dests


# ═══════════════════════════════════════════════════════════════
# Group 2: Flying General (4 cases)
# ═══════════════════════════════════════════════════════════════

class TestFlyingGeneral:
    def test_flying_general_open_file(self):
        board = make_board({
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 9)
        )
        assert (4, 0) in dests

    def test_flying_general_blocked(self):
        board = make_board({
            (4, 9): X(PieceKind.GENERAL),
            (4, 5): X(PieceKind.SOLDIER),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 9)
        )
        assert (4, 0) not in dests

    def test_moving_blocker_exposes_flying_general_legal(self):
        """Flying general is ONE-DIRECTIONAL (General→King only)."""
        board = make_board({
            (4, 9): X(PieceKind.GENERAL),
            (4, 4): X(PieceKind.SOLDIER),
            (4, 0): C(PieceKind.KING),
        })
        legal = generate_legal_moves(board, Side.XIANGQI)
        soldier_dests = destinations_from(legal, (4, 4))
        assert (4, 3) in soldier_dests
        assert (3, 4) in soldier_dests
        assert (5, 4) in soldier_dests

    def test_no_flying_general_different_column(self):
        board = make_board({
            (3, 9): X(PieceKind.GENERAL),
            (5, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (3, 9)
        )
        assert (5, 0) not in dests
        assert (4, 9) in dests
        assert (3, 8) in dests


# ═══════════════════════════════════════════════════════════════
# Group 3: Horse vs Knight (5 cases)
# ═══════════════════════════════════════════════════════════════

class TestHorseKnight:
    def test_horse_unblocked_all_8(self):
        board = make_board({
            (4, 5): X(PieceKind.HORSE),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 5)
        )
        expected = {(5,7),(3,7),(5,3),(3,3),(6,6),(6,4),(2,6),(2,4)}
        assert dests == expected

    def test_horse_one_leg_blocked(self):
        board = make_board({
            (4, 5): X(PieceKind.HORSE),
            (4, 6): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 5)
        )
        assert (5, 7) not in dests
        assert (3, 7) not in dests
        assert (5, 3) in dests
        assert (3, 3) in dests

    def test_horse_all_legs_blocked(self):
        board = make_board({
            (4, 5): X(PieceKind.HORSE),
            (4, 6): X(PieceKind.SOLDIER),
            (4, 4): X(PieceKind.SOLDIER),
            (5, 5): X(PieceKind.SOLDIER),
            (3, 5): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 5)
        )
        assert len(dests) == 0

    def test_knight_ignores_blockers(self):
        board = make_board({
            (4, 5): C(PieceKind.KNIGHT),
            (4, 6): C(PieceKind.PAWN),
            (4, 4): C(PieceKind.PAWN),
            (5, 5): C(PieceKind.PAWN),
            (3, 5): C(PieceKind.PAWN),
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.CHESS), (4, 5)
        )
        expected = {(5,7),(3,7),(5,3),(3,3),(6,6),(6,4),(2,6),(2,4)}
        assert dests == expected

    def test_horse_edge_no_oob(self):
        board = make_board({
            (0, 0): X(PieceKind.HORSE),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 0)
        )
        possible_in_bounds = {(2, 1), (1, 2)}
        assert dests <= possible_in_bounds


# ═══════════════════════════════════════════════════════════════
# Group 4: Elephant (4 cases)
# ═══════════════════════════════════════════════════════════════

class TestElephant:
    def test_elephant_unblocked_all_4(self):
        board = make_board({
            (4, 7): X(PieceKind.ELEPHANT),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 7)
        )
        expected = {(6, 9), (2, 9), (6, 5), (2, 5)}
        assert dests == expected

    def test_elephant_eye_blocked(self):
        board = make_board({
            (4, 7): X(PieceKind.ELEPHANT),
            (5, 8): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (5, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 7)
        )
        assert (6, 9) not in dests
        assert (2, 9) in dests
        assert (6, 5) in dests
        assert (2, 5) in dests

    def test_elephant_cannot_cross_river(self):
        board = make_board({
            (2, 5): X(PieceKind.ELEPHANT),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (2, 5)
        )
        assert (0, 3) not in dests
        assert (4, 3) not in dests
        assert (0, 7) in dests
        assert (4, 7) in dests

    def test_elephant_river_edge_partial(self):
        board = make_board({
            (6, 6): X(PieceKind.ELEPHANT),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (6, 6)
        )
        assert (8, 8) in dests
        assert (4, 8) in dests
        assert (8, 4) not in dests
        assert (4, 4) not in dests


# ═══════════════════════════════════════════════════════════════
# Group 5: Advisor + General basics (3 cases)
# ═══════════════════════════════════════════════════════════════

class TestAdvisorGeneral:
    def test_advisor_center_palace(self):
        board = make_board({
            (4, 8): X(PieceKind.ADVISOR),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 8)
        )
        expected = {(3, 7), (5, 7), (3, 9), (5, 9)}
        assert dests == expected

    def test_advisor_corner_palace(self):
        board = make_board({
            (3, 7): X(PieceKind.ADVISOR),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (3, 7)
        )
        assert (4, 8) in dests
        assert (2, 6) not in dests
        assert (2, 8) not in dests
        assert (4, 6) not in dests

    def test_general_center_vs_edge(self):
        board_center = make_board({
            (4, 8): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests_center = destinations_from(
            generate_legal_moves(board_center, Side.XIANGQI), (4, 8)
        )
        assert (4, 9) in dests_center
        assert (4, 7) in dests_center
        assert (3, 8) in dests_center
        assert (5, 8) in dests_center

        board_corner = make_board({
            (3, 9): X(PieceKind.GENERAL),
            (5, 0): C(PieceKind.KING),
        })
        dests_corner = destinations_from(
            generate_legal_moves(board_corner, Side.XIANGQI), (3, 9)
        )
        expected_corner = {(4, 9), (3, 8)}
        assert dests_corner == expected_corner


# ═══════════════════════════════════════════════════════════════
# Group 6: Soldier (3 cases)
# ═══════════════════════════════════════════════════════════════

class TestSoldier:
    def test_soldier_before_river(self):
        board = make_board({
            (4, 6): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 6)
        )
        assert dests == {(4, 5)}

    def test_soldier_after_river(self):
        board = make_board({
            (4, 4): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 4)
        )
        expected = {(4, 3), (3, 4), (5, 4)}
        assert dests == expected

    def test_soldier_edge_after_river(self):
        board = make_board({
            (0, 3): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 3)
        )
        expected = {(0, 2), (1, 3)}
        assert dests == expected


# ═══════════════════════════════════════════════════════════════
# Group 7: Chess Pawn (4 cases)
# ═══════════════════════════════════════════════════════════════

class TestChessPawn:
    def test_pawn_initial_double_step(self):
        board = make_board({
            (4, 1): C(PieceKind.PAWN),
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.CHESS), (4, 1)
        )
        assert (4, 2) in dests
        assert (4, 3) in dests

    def test_pawn_double_step_blocked(self):
        board1 = make_board({
            (4, 1): C(PieceKind.PAWN),
            (4, 2): X(PieceKind.SOLDIER),
            (4, 0): C(PieceKind.KING),
            (3, 9): X(PieceKind.GENERAL),
        })
        dests1 = destinations_from(
            generate_legal_moves(board1, Side.CHESS), (4, 1)
        )
        assert (4, 2) not in dests1
        assert (4, 3) not in dests1

        board2 = make_board({
            (4, 1): C(PieceKind.PAWN),
            (4, 3): X(PieceKind.SOLDIER),
            (4, 0): C(PieceKind.KING),
            (3, 9): X(PieceKind.GENERAL),
        })
        dests2 = destinations_from(
            generate_legal_moves(board2, Side.CHESS), (4, 1)
        )
        assert (4, 2) in dests2
        assert (4, 3) not in dests2

    def test_pawn_diagonal_capture(self):
        board = make_board({
            (4, 4): C(PieceKind.PAWN),
            (3, 5): X(PieceKind.SOLDIER),
            (5, 5): X(PieceKind.CANNON),
            (0, 0): C(PieceKind.KING),
            (3, 9): X(PieceKind.GENERAL),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.CHESS), (4, 4)
        )
        assert (3, 5) in dests
        assert (5, 5) in dests
        assert (4, 5) in dests

    def test_pawn_promotion(self):
        board = make_board({
            (4, 8): C(PieceKind.PAWN),
            (4, 0): C(PieceKind.KING),
            (3, 9): X(PieceKind.GENERAL),
        })
        legal = generate_legal_moves(board, Side.CHESS)
        pawn_moves = moves_from(legal, (4, 8))
        promo_moves = [m for m in pawn_moves if m.ty == 9]
        assert len(promo_moves) >= 1
        promo_kinds = {m.promotion for m in promo_moves}
        assert PieceKind.QUEEN in promo_kinds
        assert PieceKind.ROOK in promo_kinds
        assert PieceKind.BISHOP in promo_kinds
        assert PieceKind.KNIGHT in promo_kinds
        non_promo_to_9 = [m for m in pawn_moves if m.ty == 9 and m.promotion == PieceKind.NONE]
        assert len(non_promo_to_9) == 0


# ═══════════════════════════════════════════════════════════════
# Group 8: Self-check filtering (4 cases)
# ═══════════════════════════════════════════════════════════════

class TestSelfCheckFiltering:
    def test_absolute_pin(self):
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 3): C(PieceKind.ROOK),
            (4, 8): X(PieceKind.CHARIOT),
            (3, 9): X(PieceKind.GENERAL),
        })
        legal = generate_legal_moves(board, Side.CHESS)
        rook_dests = destinations_from(legal, (4, 3))
        assert (4, 1) in rook_dests
        assert (4, 8) in rook_dests
        assert (0, 3) not in rook_dests
        assert (8, 3) not in rook_dests

    def test_must_resolve_check(self):
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 5): X(PieceKind.CHARIOT),
            (0, 1): C(PieceKind.ROOK),
            (3, 9): X(PieceKind.GENERAL),
        })
        assert is_in_check(board, Side.CHESS)
        legal = generate_legal_moves(board, Side.CHESS)
        for mv in legal:
            nb = apply_move(board, mv)
            assert not is_in_check(nb, Side.CHESS)

    def test_king_cannot_walk_into_attack(self):
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (3, 2): X(PieceKind.CHARIOT),
            (3, 9): X(PieceKind.GENERAL),
        })
        legal = generate_legal_moves(board, Side.CHESS)
        king_dests = destinations_from(legal, (4, 0))
        assert (3, 0) not in king_dests
        assert (3, 1) not in king_dests

    def test_cannon_pin(self):
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 3): C(PieceKind.BISHOP),
            (4, 5): X(PieceKind.SOLDIER),
            (4, 8): X(PieceKind.CANNON),
            (3, 9): X(PieceKind.GENERAL),
        })
        legal = generate_legal_moves(board, Side.CHESS)
        bishop_dests = destinations_from(legal, (4, 3))
        assert len(bishop_dests) == 0


# ═══════════════════════════════════════════════════════════════
# Group 9: Terminal state detection (5 cases)
# ═══════════════════════════════════════════════════════════════

class TestTerminal:
    def test_checkmate(self):
        board = make_board({
            (0, 0): C(PieceKind.KING),
            (0, 5): X(PieceKind.CHARIOT),
            (1, 5): X(PieceKind.CHARIOT),
            (4, 9): X(PieceKind.GENERAL),
        })
        assert is_in_check(board, Side.CHESS)
        legal = generate_legal_moves(board, Side.CHESS)
        assert len(legal) == 0
        info = terminal_info(board, Side.CHESS, {}, 0, MAX_PLIES)
        assert info.status == "xiangqi_win"
        assert info.winner == 2
        assert "Checkmate" in info.reason

    def test_stalemate(self):
        board = make_board({
            (0, 0): C(PieceKind.KING),
            (2, 1): X(PieceKind.CHARIOT),
            (1, 2): X(PieceKind.CHARIOT),
            (4, 9): X(PieceKind.GENERAL),
        })
        assert not is_in_check(board, Side.CHESS)
        legal = generate_legal_moves(board, Side.CHESS)
        assert len(legal) == 0
        info = terminal_info(board, Side.CHESS, {}, 0, MAX_PLIES)
        assert info.status == "draw"
        assert "Stalemate" in info.reason

    def test_threefold_repetition(self):
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
            (0, 5): C(PieceKind.ROOK),
        })
        key = board.board_hash(Side.CHESS)
        rep_table = {key: 3}
        info = terminal_info(board, Side.CHESS, rep_table, 0, MAX_PLIES)
        assert info.status == "draw"
        assert "repetition" in info.reason.lower()

    def test_max_ply_draw(self):
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
        })
        info = terminal_info(board, Side.CHESS, {}, MAX_PLIES, MAX_PLIES)
        assert info.status == "draw"
        assert "plies" in info.reason.lower() or "Max" in info.reason

    def test_ongoing_normal(self):
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
            (0, 0): C(PieceKind.ROOK),
        })
        info = terminal_info(board, Side.CHESS, {}, 10, MAX_PLIES)
        assert info.status == "ongoing"
        assert info.winner == 0
