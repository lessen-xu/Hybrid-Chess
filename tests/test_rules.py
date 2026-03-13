"""Comprehensive tests for rules.py — piece move generation, check detection,
and terminal state logic.  These serve as the Python oracle for the upcoming
C++ rewrite.
"""

from __future__ import annotations
from typing import Dict, Set, Tuple, Optional

import pytest

from hybrid.core.types import Side, PieceKind, Piece, Move
from hybrid.core.board import Board
from hybrid.core.config import BOARD_W, BOARD_H, MAX_PLIES
from hybrid.core.rules import (
    generate_legal_moves,
    generate_pseudo_legal_moves,
    apply_move,
    is_in_check,
    is_square_attacked,
    terminal_info,
    board_hash,
    TerminalStatus,
    GameInfo,
)


# ═══════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════

def make_board(pieces: Dict[Tuple[int, int], Piece]) -> Board:
    """Create a board from a dict of {(x, y): Piece}.

    Starts with a completely empty 9×10 board and places only the
    specified pieces.
    """
    b = Board.empty()
    for (x, y), piece in pieces.items():
        assert b.in_bounds(x, y), f"({x},{y}) out of bounds"
        b.set(x, y, piece)
    return b


def destinations_from(moves, from_sq: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """Return the set of destination (tx, ty) for all moves originating from
    *from_sq*.
    """
    return {(m.tx, m.ty) for m in moves if (m.fx, m.fy) == from_sq}


def moves_from(moves, from_sq: Tuple[int, int]):
    """Return a list of Move objects originating from *from_sq*."""
    return [m for m in moves if (m.fx, m.fy) == from_sq]


# Shorthand constructors to keep test bodies concise.
def C(kind: PieceKind) -> Piece:
    """Chess-side piece."""
    return Piece(kind, Side.CHESS)


def X(kind: PieceKind) -> Piece:
    """Xiangqi-side piece."""
    return Piece(kind, Side.XIANGQI)


# ═══════════════════════════════════════════════════════════════
# Group 1: Cannon (8 cases)
# ═══════════════════════════════════════════════════════════════

class TestCannon:
    """Xiangqi Cannon — slide non-capture + jump-capture."""

    def test_cannon_slide_no_obstacles(self):
        """Cannon on an open board slides in all orthogonal directions."""
        board = make_board({
            (4, 5): X(PieceKind.CANNON),
            (4, 9): X(PieceKind.GENERAL),   # need royals for legal-move gen
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 5)
        )
        # Should reach every empty square on row 5 and column 4
        # (excluding own squares).  No jump destinations.
        for x in range(BOARD_W):
            if x != 4:
                assert (x, 5) in dests, f"should slide to ({x},5)"
        for y in range(BOARD_H):
            if y == 5:
                continue
            if y == 9:
                continue  # General is there (friendly, can't land)
            if y == 0:
                continue  # King is there — but cannon can't capture w/o screen
            assert (4, y) in dests, f"should slide to (4,{y})"

    def test_cannon_jump_capture(self):
        """Cannon — screen — enemy target → capture is legal."""
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (3, 5): X(PieceKind.SOLDIER),    # screen (friendly)
            (7, 5): C(PieceKind.BISHOP),      # target (enemy)
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (7, 5) in dests

    def test_cannon_no_screen_cannot_capture(self):
        """Without a screen piece the cannon cannot capture."""
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (7, 5): C(PieceKind.BISHOP),      # enemy, but no screen
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        # Can slide up to (6,5) but NOT capture (7,5)
        assert (6, 5) in dests
        assert (7, 5) not in dests

    def test_cannon_double_screen_cannot_capture(self):
        """Two pieces between cannon and target → no capture."""
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (2, 5): X(PieceKind.SOLDIER),     # screen 1
            (5, 5): C(PieceKind.PAWN),         # screen 2
            (7, 5): C(PieceKind.ROOK),         # target behind two screens
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (7, 5) not in dests
        # With double screen, first screen is at (2,5); after jumping,
        # the cannon finds next piece = (5,5) which is enemy → capture that
        assert (5, 5) in dests

    def test_cannon_cannot_jump_to_empty(self):
        """After jumping over a screen, cannot land on an empty square."""
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (3, 5): C(PieceKind.PAWN),         # screen
            (7, 5): C(PieceKind.ROOK),         # target further away
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        # Empty squares (4,5), (5,5), (6,5) between screen and target → cannot land
        assert (4, 5) not in dests
        assert (5, 5) not in dests
        assert (6, 5) not in dests
        # CAN capture the target
        assert (7, 5) in dests

    def test_cannon_vertical_jump(self):
        """Vertical cannon jump capture works identically to horizontal."""
        board = make_board({
            (2, 0): X(PieceKind.CANNON),
            (2, 3): C(PieceKind.PAWN),         # screen
            (2, 7): C(PieceKind.BISHOP),       # target
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (2, 0)
        )
        assert (2, 7) in dests
        # Cannot land on empty squares between screen and target
        assert (2, 4) not in dests
        assert (2, 5) not in dests
        assert (2, 6) not in dests

    def test_cannon_friendly_screen(self):
        """Screen piece CAN be a friendly piece — jump capture still works."""
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (3, 5): X(PieceKind.SOLDIER),     # friendly screen
            (6, 5): C(PieceKind.ROOK),         # enemy target
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 5)
        )
        assert (6, 5) in dests

    def test_cannon_cannot_capture_friendly(self):
        """Cannot jump-capture a friendly piece."""
        board = make_board({
            (0, 5): X(PieceKind.CANNON),
            (3, 5): C(PieceKind.PAWN),         # enemy screen
            (6, 5): X(PieceKind.SOLDIER),      # friendly target
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
    """Flying-general rule: General may capture King on an open file."""

    def test_flying_general_open_file(self):
        """General and King on same column, no pieces between → can capture."""
        board = make_board({
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 9)
        )
        assert (4, 0) in dests

    def test_flying_general_blocked(self):
        """A piece between General and King blocks flying general."""
        board = make_board({
            (4, 9): X(PieceKind.GENERAL),
            (4, 5): X(PieceKind.SOLDIER),     # blocker
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 9)
        )
        assert (4, 0) not in dests

    def test_moving_blocker_exposes_flying_general_legal(self):
        """DISCOVERY: Flying general is ONE-DIRECTIONAL (General→King only).
        The King does NOT attack the General via flying general. Therefore,
        moving a Xiangqi piece off the General-King column is legal — it does
        not expose the General to 'check' from the King."""
        board = make_board({
            (4, 9): X(PieceKind.GENERAL),
            (4, 4): X(PieceKind.SOLDIER),     # blocker, has crossed river
            (4, 0): C(PieceKind.KING),
        })
        legal = generate_legal_moves(board, Side.XIANGQI)
        soldier_dests = destinations_from(legal, (4, 4))
        # Soldier at y<=4 can move forward (4,3) and sideways (3,4), (5,4)
        assert (4, 3) in soldier_dests  # forward
        # Sideways moves ARE legal because flying general is one-directional:
        # the King cannot 'capture' the General via flying general rule.
        assert (3, 4) in soldier_dests
        assert (5, 4) in soldier_dests

    def test_no_flying_general_different_column(self):
        """General and King on different columns → no special move."""
        board = make_board({
            (3, 9): X(PieceKind.GENERAL),
            (5, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (3, 9)
        )
        # Only normal palace moves, no (5,0)
        assert (5, 0) not in dests
        # Normal moves from (3,9) inside palace: (4,9), (3,8)
        assert (4, 9) in dests
        assert (3, 8) in dests


# ═══════════════════════════════════════════════════════════════
# Group 3: Horse vs Knight (5 cases)
# ═══════════════════════════════════════════════════════════════

class TestHorseKnight:
    """Xiangqi Horse (leg block) vs Chess Knight (no block)."""

    def test_horse_unblocked_all_8(self):
        """Horse in center with no adjacent pieces → all 8 L-shape targets."""
        board = make_board({
            (4, 5): X(PieceKind.HORSE),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 5)
        )
        expected = {
            (5, 7), (3, 7),  # leg (4,6)
            (5, 3), (3, 3),  # leg (4,4)
            (6, 6), (6, 4),  # leg (5,5)
            (2, 6), (2, 4),  # leg (3,5)
        }
        assert dests == expected

    def test_horse_one_leg_blocked(self):
        """Blocking one leg removes two targets; others unaffected."""
        board = make_board({
            (4, 5): X(PieceKind.HORSE),
            (4, 6): X(PieceKind.SOLDIER),     # block leg going up
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 5)
        )
        assert (5, 7) not in dests
        assert (3, 7) not in dests
        # Other directions still open
        assert (5, 3) in dests
        assert (3, 3) in dests

    def test_horse_all_legs_blocked(self):
        """All 4 legs blocked → Horse has no moves."""
        board = make_board({
            (4, 5): X(PieceKind.HORSE),
            (4, 6): X(PieceKind.SOLDIER),     # up
            (4, 4): X(PieceKind.SOLDIER),     # down
            (5, 5): X(PieceKind.SOLDIER),     # right
            (3, 5): X(PieceKind.SOLDIER),     # left
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 5)
        )
        assert len(dests) == 0

    def test_knight_ignores_blockers(self):
        """Chess Knight on the same square with same blockers can reach all 8."""
        board = make_board({
            (4, 5): C(PieceKind.KNIGHT),
            (4, 6): C(PieceKind.PAWN),        # would block Horse leg
            (4, 4): C(PieceKind.PAWN),
            (5, 5): C(PieceKind.PAWN),
            (3, 5): C(PieceKind.PAWN),
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.CHESS), (4, 5)
        )
        expected = {
            (5, 7), (3, 7), (5, 3), (3, 3),
            (6, 6), (6, 4), (2, 6), (2, 4),
        }
        assert dests == expected

    def test_horse_edge_no_oob(self):
        """Horse on the board edge doesn't cause index errors."""
        board = make_board({
            (0, 0): X(PieceKind.HORSE),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 0)
        )
        # From (0,0), valid L-destinations that are in bounds:
        # leg (1,0) → (2,1), (2,-1 oob)
        # leg (0,1) → (1,2), (-1,2 oob)
        possible_in_bounds = {(2, 1), (1, 2)}
        assert dests <= possible_in_bounds  # subset (leg blocking may reduce)
        # Should not raise


# ═══════════════════════════════════════════════════════════════
# Group 4: Elephant (4 cases)
# ═══════════════════════════════════════════════════════════════

class TestElephant:
    """Xiangqi Elephant — diagonal 2-step, eye block, river restriction."""

    def test_elephant_unblocked_all_4(self):
        """Elephant in center of own half, eyes clear → 4 destinations."""
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
        """Blocking the eye square removes that diagonal."""
        # NOTE: Use a Xiangqi Soldier as eye-blocker (not Chess Pawn, which
        # would attack the General diagonally at (4,9) from (5,8)).
        # King placed off column 4 to avoid flying general interference.
        board = make_board({
            (4, 7): X(PieceKind.ELEPHANT),
            (5, 8): X(PieceKind.SOLDIER),      # blocks (6,9) eye, friendly
            (4, 9): X(PieceKind.GENERAL),
            (5, 0): C(PieceKind.KING),         # off column 4!
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 7)
        )
        assert (6, 9) not in dests
        # Other 3 should still be available
        assert (2, 9) in dests
        assert (6, 5) in dests
        assert (2, 5) in dests

    def test_elephant_cannot_cross_river(self):
        """Elephant at y=5 cannot move to y=3 (crosses river)."""
        board = make_board({
            (2, 5): X(PieceKind.ELEPHANT),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (2, 5)
        )
        # (0,3) and (4,3) are across the river → forbidden
        assert (0, 3) not in dests
        assert (4, 3) not in dests
        # (0,7) and (4,7) stay on Xiangqi side → allowed
        assert (0, 7) in dests
        assert (4, 7) in dests

    def test_elephant_river_edge_partial(self):
        """Elephant at y=6, some diagonals go to y=4 (too far)."""
        # (6,6) → diag targets: (8,8), (4,8), (8,4 oob width OK?, 4,4)
        # y=4 < 5 → river crossed → illegal
        board = make_board({
            (6, 6): X(PieceKind.ELEPHANT),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (6, 6)
        )
        # (8,8): in bounds (x<9), y>=5 ✓
        assert (8, 8) in dests
        # (4,8): in bounds, y>=5 ✓
        assert (4, 8) in dests
        # (8,4): y<5 ✗
        assert (8, 4) not in dests
        # (4,4): y<5 ✗
        assert (4, 4) not in dests


# ═══════════════════════════════════════════════════════════════
# Group 5: Advisor + General basics (3 cases)
# ═══════════════════════════════════════════════════════════════

class TestAdvisorGeneral:
    """Advisor and General basic palace-restricted movement."""

    def test_advisor_center_palace(self):
        """Advisor at (4,8) — center of palace → 4 diagonal moves."""
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
        """Advisor at (3,7) — corner of palace → only 1 diagonal in palace."""
        board = make_board({
            (3, 7): X(PieceKind.ADVISOR),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (3, 7)
        )
        assert (4, 8) in dests
        # (2,6) is outside palace → not allowed
        assert (2, 6) not in dests
        assert (2, 8) not in dests
        assert (4, 6) not in dests

    def test_general_center_vs_edge(self):
        """General at center of palace has 4 orthogonal moves; at edge, fewer."""
        # Center: (4,8)
        board_center = make_board({
            (4, 8): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests_center = destinations_from(
            generate_legal_moves(board_center, Side.XIANGQI), (4, 8)
        )
        # (4,9), (4,7), (3,8), (5,8) — all in palace
        # But (4,0) could be flying general. Column 4, no blockers.
        assert (4, 9) in dests_center
        assert (4, 7) in dests_center
        assert (3, 8) in dests_center
        assert (5, 8) in dests_center

        # Corner: (3,9)
        board_corner = make_board({
            (3, 9): X(PieceKind.GENERAL),
            (5, 0): C(PieceKind.KING),
        })
        dests_corner = destinations_from(
            generate_legal_moves(board_corner, Side.XIANGQI), (3, 9)
        )
        # Possible moves from (3,9): (4,9), (3,8)
        # (2,9) is outside palace, (3,10) out of bounds
        expected_corner = {(4, 9), (3, 8)}
        assert dests_corner == expected_corner


# ═══════════════════════════════════════════════════════════════
# Group 6: Xiangqi Soldier (3 cases)
# ═══════════════════════════════════════════════════════════════

class TestSoldier:
    """Xiangqi Soldier — forward before river, +sideways after."""

    def test_soldier_before_river(self):
        """Soldier at y=6 (own side) can only move forward (0,-1)."""
        board = make_board({
            (4, 6): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (4, 6)
        )
        # Only forward to (4,5)
        # Note: forward for Xiangqi is y-1 (towards Chess side)
        assert dests == {(4, 5)}

    def test_soldier_after_river(self):
        """Soldier at y=4 (crossed river) can go forward + sideways."""
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
        """Soldier at (0,3) — edge + crossed river: limited sideways."""
        board = make_board({
            (0, 3): X(PieceKind.SOLDIER),
            (4, 9): X(PieceKind.GENERAL),
            (4, 0): C(PieceKind.KING),
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.XIANGQI), (0, 3)
        )
        # Forward: (0,2), Sideways right: (1,3), left: (-1,3) → oob
        expected = {(0, 2), (1, 3)}
        assert dests == expected


# ═══════════════════════════════════════════════════════════════
# Group 7: Chess Pawn (4 cases)
# ═══════════════════════════════════════════════════════════════

class TestChessPawn:
    """Chess Pawn — double step, diagonal capture, promotion."""

    def test_pawn_initial_double_step(self):
        """Pawn at y=1 can move 1 or 2 squares forward."""
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
        """Pawn double step blocked by piece in the way."""
        # Piece at y=2 blocks single AND double step
        board1 = make_board({
            (4, 1): C(PieceKind.PAWN),
            (4, 2): X(PieceKind.SOLDIER),     # blocks single step
            (4, 0): C(PieceKind.KING),
            (3, 9): X(PieceKind.GENERAL),
        })
        dests1 = destinations_from(
            generate_legal_moves(board1, Side.CHESS), (4, 1)
        )
        assert (4, 2) not in dests1
        assert (4, 3) not in dests1

        # Piece at y=3 blocks only double step
        board2 = make_board({
            (4, 1): C(PieceKind.PAWN),
            (4, 3): X(PieceKind.SOLDIER),     # blocks double step only
            (4, 0): C(PieceKind.KING),
            (3, 9): X(PieceKind.GENERAL),
        })
        dests2 = destinations_from(
            generate_legal_moves(board2, Side.CHESS), (4, 1)
        )
        assert (4, 2) in dests2
        assert (4, 3) not in dests2

    def test_pawn_diagonal_capture(self):
        """Pawn captures diagonally; cannot move diagonally to empty."""
        # NOTE: keep King off column 4 to avoid pawn being pinned on
        # the flying general line.
        board = make_board({
            (4, 4): C(PieceKind.PAWN),
            (3, 5): X(PieceKind.SOLDIER),     # enemy on diagonal → capture
            (5, 5):  X(PieceKind.CANNON),     # enemy on other diagonal
            (0, 0): C(PieceKind.KING),         # off column 4!
            (3, 9): X(PieceKind.GENERAL),      # off column 4!
        })
        dests = destinations_from(
            generate_legal_moves(board, Side.CHESS), (4, 4)
        )
        assert (3, 5) in dests  # capture
        assert (5, 5) in dests  # capture
        assert (4, 5) in dests  # forward non-capture

    def test_pawn_promotion(self):
        """Pawn reaching y=9 produces promotion moves Q/R/B/N."""
        board = make_board({
            (4, 8): C(PieceKind.PAWN),
            (4, 0): C(PieceKind.KING),
            (3, 9): X(PieceKind.GENERAL),
        })
        legal = generate_legal_moves(board, Side.CHESS)
        pawn_moves = moves_from(legal, (4, 8))
        # All moves that go to y=9 should have a promotion
        promo_moves = [m for m in pawn_moves if m.ty == 9]
        assert len(promo_moves) >= 1
        promo_kinds = {m.promotion for m in promo_moves}
        # By default ABLATION_NO_QUEEN_PROMOTION is False → Q included
        assert PieceKind.QUEEN in promo_kinds
        assert PieceKind.ROOK in promo_kinds
        assert PieceKind.BISHOP in promo_kinds
        assert PieceKind.KNIGHT in promo_kinds
        # No non-promotion move to y=9
        non_promo_to_9 = [m for m in pawn_moves if m.ty == 9 and m.promotion is None]
        assert len(non_promo_to_9) == 0


# ═══════════════════════════════════════════════════════════════
# Group 8: Self-check filtering (4 cases)
# ═══════════════════════════════════════════════════════════════

class TestSelfCheckFiltering:
    """generate_legal_moves must filter out moves leaving own royal in check."""

    def test_absolute_pin(self):
        """A piece pinned to its King cannot leave the pin line."""
        # Chess King at (4,0), Chess Rook at (4,3) pinned by Xiangqi Chariot at (4,8)
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 3): C(PieceKind.ROOK),
            (4, 8): X(PieceKind.CHARIOT),     # pins the Rook on column 4
            (3, 9): X(PieceKind.GENERAL),
        })
        legal = generate_legal_moves(board, Side.CHESS)
        rook_dests = destinations_from(legal, (4, 3))
        # Rook CAN move along column 4 (stays on pin line)
        assert (4, 1) in rook_dests
        assert (4, 8) in rook_dests  # capture the pinning piece
        # Rook CANNOT move off column 4
        assert (0, 3) not in rook_dests
        assert (8, 3) not in rook_dests

    def test_must_resolve_check(self):
        """When in check, all legal moves must resolve the check."""
        # Xiangqi Chariot gives check to Chess King
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 5): X(PieceKind.CHARIOT),     # checking along column 4
            (0, 1): C(PieceKind.ROOK),         # could block or King moves
            (3, 9): X(PieceKind.GENERAL),
        })
        assert is_in_check(board, Side.CHESS)
        legal = generate_legal_moves(board, Side.CHESS)
        # Every legal move must result in no longer being in check
        for mv in legal:
            nb = apply_move(board, mv)
            assert not is_in_check(nb, Side.CHESS), \
                f"Move {mv} doesn't resolve check"

    def test_king_cannot_walk_into_attack(self):
        """King/General cannot move to a square attacked by the opponent."""
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (3, 2): X(PieceKind.CHARIOT),     # attacks (3,0), (3,1) etc.
            (3, 9): X(PieceKind.GENERAL),
        })
        legal = generate_legal_moves(board, Side.CHESS)
        king_dests = destinations_from(legal, (4, 0))
        # (3,0) is attacked by the Chariot
        assert (3, 0) not in king_dests
        # (3,1) also attacked
        assert (3, 1) not in king_dests

    def test_cannon_pin(self):
        """A piece acting as cannon screen: moving it allows cannon to capture
        the King via flying capture → illegal."""
        # Cannon at (0,0), screen = Chess Pawn at (0,3), King at (0,6)
        # If pawn moves off column 0, cannon can jump-capture through... wait,
        # cannon needs a screen. Let's set up:
        # Xiangqi Cannon at (4,8), friendly soldier at (4,5) is screen,
        # Chess King at (4,0). A Chess Bishop at (4,3) sits between.
        # If Bishop leaves col 4, cannon → screen(soldier) → capture King.
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 3): C(PieceKind.BISHOP),       # if this moves, King is exposed
            (4, 5): X(PieceKind.SOLDIER),      # cannon screen
            (4, 8): X(PieceKind.CANNON),       # aims at King
            (3, 9): X(PieceKind.GENERAL),
        })
        legal = generate_legal_moves(board, Side.CHESS)
        bishop_dests = destinations_from(legal, (4, 3))
        # Bishop moves diagonally, so ALL bishop moves leave column 4 →
        # all should be illegal if the cannon can capture King through screen.
        # But wait — bishop at (4,3) is between screen(4,5) and King(4,0).
        # Cannon → screen(4,5) → skip → finds (4,3) bishop first.
        # If bishop moves away, cannon → screen(4,5) → skip → finds (4,0) King.
        # So bishop IS pinned.
        # Actually the bishop at (4,3) itself is the target between screen and
        # king. But cannon captures the FIRST piece after the screen.
        # Currently: cannon at (4,8), screen at (4,5), bishop at (4,3), king at (4,0).
        # Cannon capture: after screen(4,5), next piece = bishop(4,3) → enemy,
        # so cannon captures bishop, not the king.
        # If bishop moves away: cannon after screen → next piece = King → capture King.
        # So YES, moving the bishop away exposes King to cannon capture → illegal.
        # All bishop moves leave column 4 (diagonal) → all should be illegal.
        assert len(bishop_dests) == 0


# ═══════════════════════════════════════════════════════════════
# Group 9: Terminal state detection (5 cases)
# ═══════════════════════════════════════════════════════════════

class TestTerminal:
    """terminal_info — checkmate, stalemate, repetition, max ply."""

    def test_checkmate(self):
        """Checkmate: in check + no legal moves → loser's opponent wins."""
        # Two Chariots on columns 0 and 1 at y=5.
        # Chariot(0,5) gives check on column 0; Chariot(1,5) guards column 1.
        # King at (0,0): (0,1) blocked by col-0 chariot, (1,0) blocked by
        # col-1 chariot, (1,1) also col-1 chariot. All other squares OOB.
        board = make_board({
            (0, 0): C(PieceKind.KING),
            (0, 5): X(PieceKind.CHARIOT),     # check on column 0
            (1, 5): X(PieceKind.CHARIOT),     # guards column 1
            (4, 9): X(PieceKind.GENERAL),
        })
        assert is_in_check(board, Side.CHESS)
        legal = generate_legal_moves(board, Side.CHESS)
        assert len(legal) == 0, f"Expected checkmate but got: {legal}"
        info = terminal_info(board, Side.CHESS, {}, 0, MAX_PLIES)
        assert info.status == TerminalStatus.XIANGQI_WIN
        assert info.winner == Side.XIANGQI
        assert "Checkmate" in info.reason

    def test_stalemate(self):
        """Stalemate: not in check but no legal moves → draw."""
        # Chess King at (0,0), surrounded by attacked squares, not in check.
        # Xiangqi Chariot at (2,2) controls row 2 and col 2.
        # Another Chariot at (1,5) controls col 1.
        # King at (0,0): can go to (1,0), (0,1), (1,1)
        # We need all those squares attacked.
        board = make_board({
            (0, 0): C(PieceKind.KING),
            (1, 8): X(PieceKind.CHARIOT),     # controls column 1
            (8, 1): X(PieceKind.CHARIOT),     # controls row 1 (y=1)
            (4, 9): X(PieceKind.GENERAL),
        })
        # King at (0,0) moves: (1,0) attacked by chariot at (1,8)?
        # Chariot at (1,8) controls column 1 → (1,0) attacked ✓
        # (0,1) attacked by chariot at (8,1)? chariot at (8,1) controls row 1 → (0,1) attacked ✓
        # (1,1) attacked by chariot at (1,8) col 1 ✓ and chariot at (8,1) row 1 ✓
        # King at (0,0) is NOT in check (no piece attacks (0,0) directly)
        if is_in_check(board, Side.CHESS):
            # If it is in check, adjust setup
            pass
        legal = generate_legal_moves(board, Side.CHESS)
        if len(legal) == 0 and not is_in_check(board, Side.CHESS):
            info = terminal_info(board, Side.CHESS, {}, 0, MAX_PLIES)
            assert info.status == TerminalStatus.DRAW
            assert "Stalemate" in info.reason
        else:
            # Fallback: simpler stalemate setup
            # King at corner, both rows/cols adjacent blocked
            board2 = make_board({
                (0, 0): C(PieceKind.KING),
                (2, 1): X(PieceKind.CHARIOT),     # row 1 controlled from x=2
                (1, 2): X(PieceKind.CHARIOT),     # col 1 controlled from y=2
                (4, 9): X(PieceKind.GENERAL),
            })
            # (0,0) → not attacked directly?
            # chariot at (2,1): controls row y=1 and col x=2
            # chariot at (1,2): controls row y=2 and col x=1
            # (1,0) attacked by chariot (1,2) col 1? Yes.
            # (0,1) attacked by chariot (2,1) row 1? Yes.
            # (1,1) attacked by both
            # (0,0) not attacked (not on row 1, row 2, col 1, col 2) ✓
            assert not is_in_check(board2, Side.CHESS)
            legal2 = generate_legal_moves(board2, Side.CHESS)
            assert len(legal2) == 0
            info2 = terminal_info(board2, Side.CHESS, {}, 0, MAX_PLIES)
            assert info2.status == TerminalStatus.DRAW
            assert "Stalemate" in info2.reason

    def test_threefold_repetition(self):
        """Threefold repetition in the repetition table → draw."""
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
            (0, 5): C(PieceKind.ROOK),
        })
        key = board_hash(board, Side.CHESS)
        rep_table = {key: 3}  # already seen 3 times
        info = terminal_info(board, Side.CHESS, rep_table, 0, MAX_PLIES)
        assert info.status == TerminalStatus.DRAW
        assert "repetition" in info.reason.lower()

    def test_max_ply_draw(self):
        """Reaching max plies → draw."""
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
        })
        info = terminal_info(board, Side.CHESS, {}, MAX_PLIES, MAX_PLIES)
        assert info.status == TerminalStatus.DRAW
        assert "plies" in info.reason.lower() or "Max" in info.reason

    def test_ongoing_normal(self):
        """Normal position with legal moves and no repetition → ongoing."""
        board = make_board({
            (4, 0): C(PieceKind.KING),
            (4, 9): X(PieceKind.GENERAL),
            (0, 0): C(PieceKind.ROOK),
        })
        info = terminal_info(board, Side.CHESS, {}, 10, MAX_PLIES)
        assert info.status == TerminalStatus.ONGOING
        assert info.winner is None
