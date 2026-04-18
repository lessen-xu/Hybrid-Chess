"""Rule engine: move generation, legality checks, check/checkmate/draw detection.

Asymmetric rules: Chess Knight has no leg-block; Xiangqi Horse does.
Cannon captures require a screen piece. General/Advisor are palace-restricted.
Elephant cannot cross the river. Chess Pawn has double-step and promotion.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Iterable, Dict
import hashlib

from .config import BOARD_W, BOARD_H, ENABLE_FLYING_GENERAL_CAPTURE, DEFAULT_VARIANT

# Module-level active variant, set by HybridChessEnv to avoid threading issues.
# When None, functions fall back to legacy global flags.
_active_variant = None
from .types import Side, PieceKind, Piece, Move
from .board import Board

ORTH_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

KNIGHT_DELTAS = [
    (1, 2), (2, 1), (-1, 2), (-2, 1),
    (1, -2), (2, -1), (-1, -2), (-2, -1),
]


def _slide_moves(board: Board, x: int, y: int, side: Side, dirs: Iterable[Tuple[int, int]]) -> List[Move]:
    """Generate sliding moves for rook/bishop/queen/chariot."""
    out: List[Move] = []
    for dx, dy in dirs:
        cx, cy = x + dx, y + dy
        while board.in_bounds(cx, cy):
            p = board.get(cx, cy)
            if p is None:
                out.append(Move(x, y, cx, cy))
            else:
                if p.side != side:
                    out.append(Move(x, y, cx, cy))
                break
            cx += dx
            cy += dy
    return out


def _find_royal(board: Board, side: Side) -> Optional[Tuple[int, int]]:
    """Find the royal piece (Chess KING / Xiangqi GENERAL)."""
    target = PieceKind.KING if side == Side.CHESS else PieceKind.GENERAL
    for x, y, p in board.iter_pieces():
        if p.side == side and p.kind == target:
            return (x, y)
    return None


def _palace_contains(side: Side, x: int, y: int) -> bool:
    """Check if (x,y) is inside the side's palace."""
    if side == Side.XIANGQI:
        return (3 <= x <= 5) and (7 <= y <= 9)
    # Chess palace: only when chess_palace flag is active
    if side == Side.CHESS:
        if _active_variant is not None and _active_variant.chess_palace:
            return (3 <= x <= 5) and (0 <= y <= 2)
    return True  # no restriction


def _xiangqi_elephant_can_go(x: int, y: int, nx: int, ny: int) -> bool:
    """Elephant cannot cross the river (must stay at y >= 5)."""
    return ny >= 5

def generate_pseudo_legal_moves(board: Board, side: Side) -> List[Move]:
    """Generate pseudo-legal moves (ignoring self-check)."""
    moves: List[Move] = []
    for x, y, p in board.iter_pieces():
        if p.side != side:
            continue
        moves.extend(_piece_moves(board, x, y, p))
    return moves


def _piece_moves(board: Board, x: int, y: int, p: Piece) -> List[Move]:
    """Dispatch move generation by piece kind."""
    k = p.kind
    s = p.side

    # --- Chess pieces ---
    if k == PieceKind.ROOK:
        return _slide_moves(board, x, y, s, ORTH_DIRS)
    if k == PieceKind.BISHOP:
        return _slide_moves(board, x, y, s, DIAG_DIRS)
    if k == PieceKind.QUEEN:
        return _slide_moves(board, x, y, s, ORTH_DIRS + DIAG_DIRS)
    if k == PieceKind.KNIGHT:
        # Check if knight_block is active
        use_block = (_active_variant is not None and _active_variant.knight_block)
        if use_block:
            return _xiangqi_horse_moves(board, x, y, s)
        out = []
        for dx, dy in KNIGHT_DELTAS:
            nx, ny = x + dx, y + dy
            if not board.in_bounds(nx, ny):
                continue
            t = board.get(nx, ny)
            if t is None or t.side != s:
                out.append(Move(x, y, nx, ny))
        return out
    if k == PieceKind.KING:
        out = []
        for dx, dy in ORTH_DIRS + DIAG_DIRS:
            nx, ny = x + dx, y + dy
            if not board.in_bounds(nx, ny):
                continue
            if not _palace_contains(s, nx, ny):
                continue
            t = board.get(nx, ny)
            if t is None or t.side != s:
                out.append(Move(x, y, nx, ny))
        return out
    if k == PieceKind.PAWN:
        return _chess_pawn_moves(board, x, y, s)

    # --- Xiangqi pieces ---
    if k == PieceKind.CHARIOT:
        return _slide_moves(board, x, y, s, ORTH_DIRS)
    if k == PieceKind.CANNON:
        return _xiangqi_cannon_moves(board, x, y, s)
    if k == PieceKind.HORSE:
        return _xiangqi_horse_moves(board, x, y, s)
    if k == PieceKind.ELEPHANT:
        return _xiangqi_elephant_moves(board, x, y, s)
    if k == PieceKind.ADVISOR:
        return _xiangqi_advisor_moves(board, x, y, s)
    if k == PieceKind.GENERAL:
        return _xiangqi_general_moves(board, x, y, s)
    if k == PieceKind.SOLDIER:
        return _xiangqi_soldier_moves(board, x, y, s)

    raise ValueError(f"unknown piece kind: {k}")


def _chess_pawn_moves(board: Board, x: int, y: int, side: Side) -> List[Move]:
    """Chess Pawn: +1 forward, double-step from y=1, diagonal capture, promote at y=9.
    En passant is not implemented.
    """
    assert side == Side.CHESS
    out: List[Move] = []
    fy = y

    # Forward 1
    nx, ny = x, y + 1
    if board.in_bounds(nx, ny) and board.get(nx, ny) is None:
        out.extend(_maybe_promotions(x, fy, nx, ny))
        # Double step from starting rank
        if y == 1:
            nx2, ny2 = x, y + 2
            if board.in_bounds(nx2, ny2) and board.get(nx2, ny2) is None:
                out.append(Move(x, fy, nx2, ny2))

    # Diagonal capture
    for dx in (-1, 1):
        cx, cy = x + dx, y + 1
        if not board.in_bounds(cx, cy):
            continue
        t = board.get(cx, cy)
        if t is not None and t.side != side:
            out.extend(_maybe_promotions(x, fy, cx, cy))
    return out


def _maybe_promotions(fx: int, fy: int, tx: int, ty: int) -> List[Move]:
    """If Pawn reaches y=9, generate promotion moves (Q/R/B/N) or plain if no_promotion."""
    if ty != 9:
        return [Move(fx, fy, tx, ty)]

    # Check no_promotion first
    if _active_variant is not None and _active_variant.no_promotion:
        return [Move(fx, fy, tx, ty)]  # plain move, no promotion

    # Check variant config for no_queen_promotion, fallback to legacy global
    if _active_variant is not None:
        no_queen_promo = _active_variant.no_queen_promotion
    else:
        from .config import ABLATION_NO_QUEEN_PROMOTION
        no_queen_promo = ABLATION_NO_QUEEN_PROMOTION
    promos = [PieceKind.ROOK, PieceKind.BISHOP, PieceKind.KNIGHT]
    if not no_queen_promo:
        promos.insert(0, PieceKind.QUEEN)
    return [Move(fx, fy, tx, ty, promotion=k) for k in promos]


def _xiangqi_cannon_moves(board: Board, x: int, y: int, side: Side) -> List[Move]:
    """Cannon: slides like a rook (non-capture); captures by jumping over exactly one screen piece."""
    out: List[Move] = []

    # Non-capture: slide until blocked
    for dx, dy in ORTH_DIRS:
        cx, cy = x + dx, y + dy
        while board.in_bounds(cx, cy) and board.get(cx, cy) is None:
            out.append(Move(x, y, cx, cy))
            cx += dx
            cy += dy

    # Capture: find screen piece, then find enemy target behind it
    for dx, dy in ORTH_DIRS:
        cx, cy = x + dx, y + dy
        # Find screen
        while board.in_bounds(cx, cy) and board.get(cx, cy) is None:
            cx += dx
            cy += dy
        if not board.in_bounds(cx, cy):
            continue
        # Skip screen, find target
        cx += dx
        cy += dy
        while board.in_bounds(cx, cy) and board.get(cx, cy) is None:
            cx += dx
            cy += dy
        if not board.in_bounds(cx, cy):
            continue
        target = board.get(cx, cy)
        if target is not None and target.side != side:
            out.append(Move(x, y, cx, cy))

    return out


def _xiangqi_horse_moves(board: Board, x: int, y: int, side: Side) -> List[Move]:
    """Horse: one orthogonal step + one diagonal step. Blocked if the leg square is occupied."""
    out: List[Move] = []
    candidates = [
        # (leg_dx, leg_dy, dst_dx, dst_dy)
        (1, 0, 2, 1), (1, 0, 2, -1),
        (-1, 0, -2, 1), (-1, 0, -2, -1),
        (0, 1, 1, 2), (0, 1, -1, 2),
        (0, -1, 1, -2), (0, -1, -1, -2),
    ]
    for leg_dx, leg_dy, dst_dx, dst_dy in candidates:
        lx, ly = x + leg_dx, y + leg_dy
        if not board.in_bounds(lx, ly):
            continue
        if board.get(lx, ly) is not None:
            continue  # leg blocked
        nx, ny = x + dst_dx, y + dst_dy
        if not board.in_bounds(nx, ny):
            continue
        t = board.get(nx, ny)
        if t is None or t.side != side:
            out.append(Move(x, y, nx, ny))
    return out


def _xiangqi_elephant_moves(board: Board, x: int, y: int, side: Side) -> List[Move]:
    """Elephant: diagonal 2-step, blocked if eye square is occupied, cannot cross river."""
    out: List[Move] = []
    assert side == Side.XIANGQI
    for dx, dy in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
        nx, ny = x + dx, y + dy
        if not board.in_bounds(nx, ny):
            continue
        if not _xiangqi_elephant_can_go(x, y, nx, ny):
            continue
        ex, ey = x + dx // 2, y + dy // 2  # eye square
        if board.get(ex, ey) is not None:
            continue
        t = board.get(nx, ny)
        if t is None or t.side != side:
            out.append(Move(x, y, nx, ny))
    return out


def _xiangqi_advisor_moves(board: Board, x: int, y: int, side: Side) -> List[Move]:
    """Advisor: one diagonal step within the palace."""
    out: List[Move] = []
    for dx, dy in DIAG_DIRS:
        nx, ny = x + dx, y + dy
        if not board.in_bounds(nx, ny):
            continue
        if not _palace_contains(side, nx, ny):
            continue
        t = board.get(nx, ny)
        if t is None or t.side != side:
            out.append(Move(x, y, nx, ny))
    return out


def _xiangqi_general_moves(board: Board, x: int, y: int, side: Side) -> List[Move]:
    """General: one orthogonal step within the palace. Optional flying-general capture."""
    out: List[Move] = []

    # Normal moves
    for dx, dy in ORTH_DIRS:
        nx, ny = x + dx, y + dy
        if not board.in_bounds(nx, ny):
            continue
        if not _palace_contains(side, nx, ny):
            continue
        t = board.get(nx, ny)
        if t is None or t.side != side:
            out.append(Move(x, y, nx, ny))

    # Flying-general capture: if King is on the same file with no pieces in between
    if _active_variant is not None:
        fg_enabled = _active_variant.flying_general
    else:
        from .config import ABLATION_NO_FLYING_GENERAL
        fg_enabled = ENABLE_FLYING_GENERAL_CAPTURE and not ABLATION_NO_FLYING_GENERAL
    if fg_enabled:
        king_pos = _find_royal(board, Side.CHESS)
        if king_pos is not None:
            kx, ky = king_pos
            if kx == x:
                step = 1 if ky > y else -1
                cy = y + step
                blocked = False
                while cy != ky:
                    if board.get(x, cy) is not None:
                        blocked = True
                        break
                    cy += step
                if not blocked:
                    out.append(Move(x, y, kx, ky))

    return out


def _xiangqi_soldier_moves(board: Board, x: int, y: int, side: Side) -> List[Move]:
    """Soldier: forward only before crossing river; forward/left/right after crossing (y <= 4)."""
    out: List[Move] = []
    assert side == Side.XIANGQI

    forward = (0, -1)
    candidates = [forward]

    # After crossing the river (y <= 4), can also move sideways
    if y <= 4:
        candidates += [(1, 0), (-1, 0)]

    for dx, dy in candidates:
        nx, ny = x + dx, y + dy
        if not board.in_bounds(nx, ny):
            continue
        t = board.get(nx, ny)
        if t is None or t.side != side:
            out.append(Move(x, y, nx, ny))
    return out

def is_square_attacked(board: Board, x: int, y: int, by_side: Side) -> bool:
    """Check if square (x,y) is attacked by any piece of by_side."""
    for px, py, p in board.iter_pieces():
        if p.side != by_side:
            continue

        # Chess pawn attacks diagonally (not same as its movement)
        if p.kind == PieceKind.PAWN and p.side == Side.CHESS:
            for dx in (-1, 1):
                ax, ay = px + dx, py + 1
                if (ax, ay) == (x, y):
                    if board.in_bounds(ax, ay):
                        return True
            continue

        # Other pieces: any pseudo-legal move landing on (x,y) counts as an attack
        for mv in _piece_moves(board, px, py, p):
            if (mv.tx, mv.ty) == (x, y):
                return True
    return False


def is_in_check(board: Board, side: Side) -> bool:
    """Check if side's royal piece is in check."""
    royal = _find_royal(board, side)
    if royal is None:
        # Royal captured => treated as "in check" for terminal detection
        return True
    rx, ry = royal
    return is_square_attacked(board, rx, ry, side.opponent())

def apply_move(board: Board, mv: Move) -> Board:
    """Apply a move on a cloned board and return the new board (copy-on-write)."""
    nb = board.clone()
    piece = nb.get(mv.fx, mv.fy)
    assert piece is not None, "no piece to move"

    nb.move_piece(mv.fx, mv.fy, mv.tx, mv.ty)

    # Handle pawn promotion
    if piece.kind == PieceKind.PAWN and piece.side == Side.CHESS and mv.promotion is not None:
        nb.set(mv.tx, mv.ty, Piece(mv.promotion, Side.CHESS))

    return nb


def generate_legal_moves(board: Board, side: Side) -> List[Move]:
    """Generate legal moves (filters out moves that leave own royal in check)."""
    out: List[Move] = []
    for mv in generate_pseudo_legal_moves(board, side):
        nb = apply_move(board, mv)
        if not is_in_check(nb, side):
            out.append(mv)
    return out

class TerminalStatus:
    ONGOING = "ongoing"
    CHESS_WIN = "chess_win"
    XIANGQI_WIN = "xiangqi_win"
    DRAW = "draw"


def board_hash(board: Board, side_to_move: Side) -> str:
    """Stable hash of the position for threefold repetition detection."""
    rows = []
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            p = board.get(x, y)
            if p is None:
                rows.append(".")
            else:
                rows.append(f"{p.side.name[0]}{p.kind.name[0]}")
    rows.append(f"T{side_to_move.name[0]}")
    s = "|".join(rows).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


@dataclass
class GameInfo:
    status: str
    winner: Optional[Side] = None
    reason: str = ""


def terminal_info(board: Board, side_to_move: Side, repetition_table: Dict[str, int], ply: int, max_plies: int) -> GameInfo:
    """Determine if the game is over. Check order:
    1) Royal captured => loss
    2) Max plies => draw
    3) Threefold repetition => draw
    4) No legal moves => checkmate (loss) or stalemate (draw)
    """
    # 1) Royal existence
    chess_royal = _find_royal(board, Side.CHESS)
    xiangqi_royal = _find_royal(board, Side.XIANGQI)
    if chess_royal is None:
        return GameInfo(TerminalStatus.XIANGQI_WIN, winner=Side.XIANGQI, reason="Chess king captured")
    if xiangqi_royal is None:
        return GameInfo(TerminalStatus.CHESS_WIN, winner=Side.CHESS, reason="Xiangqi general captured")

    # 2) Move limit
    if ply >= max_plies:
        return GameInfo(TerminalStatus.DRAW, winner=None, reason="Max plies reached")

    # 3) Threefold repetition
    key = board_hash(board, side_to_move)
    if repetition_table.get(key, 0) >= 3:
        return GameInfo(TerminalStatus.DRAW, winner=None, reason="Threefold repetition")

    # 4) Legal moves
    legal = generate_legal_moves(board, side_to_move)
    if len(legal) > 0:
        return GameInfo(TerminalStatus.ONGOING, winner=None, reason="")

    # No legal moves
    if is_in_check(board, side_to_move):
        winner = side_to_move.opponent()
        status = TerminalStatus.CHESS_WIN if winner == Side.CHESS else TerminalStatus.XIANGQI_WIN
        return GameInfo(status, winner=winner, reason="Checkmate")
    else:
        # Stalemate = loss for the side with no moves (Xiangqi convention)
        winner = side_to_move.opponent()
        status = TerminalStatus.CHESS_WIN if winner == Side.CHESS else TerminalStatus.XIANGQI_WIN
        return GameInfo(status, winner=winner, reason="Stalemate (loss for stalemated side)")
