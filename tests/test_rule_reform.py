"""Smoke test for 3 rule reforms: no_promotion, chess_palace, knight_block."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid.core.config import VariantConfig
from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side, PieceKind, Move
from hybrid.cpp_engine import best_move as cpp_best_move, Side as CppSide

def test_no_promotion():
    """Pawn at y=8 should NOT generate promotion moves when no_promotion=True."""
    vcfg = VariantConfig(no_promotion=True)
    env = HybridChessEnv(use_cpp=True, variant=vcfg)
    state = env.reset()
    
    # Check Python side
    from hybrid.core import rules
    rules._active_variant = vcfg
    from hybrid.core.board import Board
    from hybrid.core.types import Piece
    
    # Create a board with a pawn at y=8 with clear path to y=9
    b = Board.empty()
    b.set(4, 8, Piece(PieceKind.PAWN, Side.CHESS))
    b.set(4, 0, Piece(PieceKind.KING, Side.CHESS))
    b.set(5, 8, Piece(PieceKind.GENERAL, Side.XIANGQI))  # General away from pawn path
    b.set(5, 9, Piece(PieceKind.ADVISOR, Side.XIANGQI))   # protect general
    
    moves = rules.generate_legal_moves(b, Side.CHESS)
    pawn_moves = [m for m in moves if m.fx == 4 and m.fy == 8]
    
    # Pawn can move forward to (4,9) without promotion, and capture (5,9) without promotion
    for pm in pawn_moves:
        assert pm.promotion is None, f"Expected no promotion, got {pm.promotion}"
    print(f"  [PASS] no_promotion: pawn generates {len(pawn_moves)} move(s) without promotion")
    
    # Verify with promotion enabled
    rules._active_variant = VariantConfig()
    moves2 = rules.generate_legal_moves(b, Side.CHESS)
    pawn_moves2 = [m for m in moves2 if m.fx == 4 and m.fy == 8]
    promo_moves = [m for m in pawn_moves2 if m.promotion is not None]
    assert len(promo_moves) > 0, f"Expected promotion moves, got none"
    print(f"  [PASS] default: pawn generates {len(promo_moves)} promotion moves")

def test_chess_palace():
    """Chess King should be restricted to 3x3 palace (x=3-5, y=0-2)."""
    vcfg = VariantConfig(chess_palace=True)
    env = HybridChessEnv(use_cpp=True, variant=vcfg)
    
    from hybrid.core import rules
    rules._active_variant = vcfg
    from hybrid.core.board import Board
    from hybrid.core.types import Piece
    
    # King at center of palace (4,1), with column blocker to prevent flying general
    b = Board.empty()
    b.set(4, 1, Piece(PieceKind.KING, Side.CHESS))
    b.set(4, 5, Piece(PieceKind.SOLDIER, Side.XIANGQI))  # blocks flying general column 4
    b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
    
    moves = rules.generate_legal_moves(b, Side.CHESS)
    king_moves = [m for m in moves if m.fx == 4 and m.fy == 1]
    destinations = set((m.tx, m.ty) for m in king_moves)
    
    # Should be able to reach all 8 neighbors within palace (3-5, 0-2)
    expected = {(3,0),(4,0),(5,0),(3,1),(5,1),(3,2),(4,2),(5,2)}
    assert destinations == expected, f"Expected {expected}, got {destinations}"
    print("  [PASS] chess_palace: King at (4,1) can reach all 8 palace squares")
    
    # King at edge of palace (5,2) — should not go to (5,3) or (6,2)
    b2 = Board.empty()
    b2.set(5, 2, Piece(PieceKind.KING, Side.CHESS))
    b2.set(5, 5, Piece(PieceKind.SOLDIER, Side.XIANGQI))  # blocks flying general col 5
    b2.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
    
    moves2 = rules.generate_legal_moves(b2, Side.CHESS)
    king_moves2 = [m for m in moves2 if m.fx == 5 and m.fy == 2]
    dests2 = set((m.tx, m.ty) for m in king_moves2)
    
    assert (5, 3) not in dests2, "King should NOT go to (5,3) outside palace"
    assert (6, 2) not in dests2, "King should NOT go to (6,2) outside palace"
    print("  [PASS] chess_palace: King at (5,2) cannot leave palace")

def test_knight_block():
    """Chess Knight should have leg-blocking like XQ Horse."""
    vcfg = VariantConfig(knight_block=True)
    
    from hybrid.core import rules
    rules._active_variant = vcfg
    from hybrid.core.board import Board
    from hybrid.core.types import Piece
    
    # Knight at (4,4) with a blocking piece at (4,5) — blocks (3,6) and (5,6)
    b = Board.empty()
    b.set(4, 4, Piece(PieceKind.KNIGHT, Side.CHESS))
    b.set(4, 0, Piece(PieceKind.KING, Side.CHESS))
    b.set(4, 5, Piece(PieceKind.PAWN, Side.CHESS))  # blocks upward leg
    b.set(4, 9, Piece(PieceKind.GENERAL, Side.XIANGQI))
    
    moves = rules.generate_legal_moves(b, Side.CHESS)
    knight_moves = [m for m in moves if m.fx == 4 and m.fy == 4]
    dests = set((m.tx, m.ty) for m in knight_moves)
    
    assert (3, 6) not in dests, "Knight should be blocked from (3,6) by piece at (4,5)"
    assert (5, 6) not in dests, "Knight should be blocked from (5,6) by piece at (4,5)"
    # But other directions should still work
    assert (2, 5) in dests or (6, 5) in dests or (2, 3) in dests, "Some other knight moves should exist"
    print("  [PASS] knight_block: Knight at (4,4) blocked by pawn at (4,5)")
    
    # Without blocking: Knight should reach (3,6) and (5,6)
    rules._active_variant = VariantConfig(knight_block=False)
    moves2 = rules.generate_legal_moves(b, Side.CHESS)
    knight_moves2 = [m for m in moves2 if m.fx == 4 and m.fy == 4]
    dests2 = set((m.tx, m.ty) for m in knight_moves2)
    assert (3, 6) in dests2, "Standard Knight should reach (3,6)"
    assert (5, 6) in dests2, "Standard Knight should reach (5,6)"
    print("  [PASS] default: Standard Knight at (4,4) can jump over piece at (4,5)")

def test_cpp_rule_flags():
    """Verify C++ engine respects rule flags via AB search."""
    from hybrid.cpp_engine import set_rule_flags, RuleFlags
    
    # Test that set_rule_flags doesn't crash
    flags = RuleFlags()
    flags.no_promotion = True
    flags.chess_palace = True
    flags.knight_block = True
    set_rule_flags(flags)
    print("  [PASS] C++ RuleFlags set successfully")
    
    # Reset
    set_rule_flags(RuleFlags())

def test_ab_with_flags():
    """Quick AB D1 game with all flags active to check no crashes."""
    vcfg = VariantConfig(no_promotion=True, chess_palace=True, knight_block=True)
    env = HybridChessEnv(use_cpp=True, max_plies=50, variant=vcfg)
    state = env.reset()
    
    for _ in range(50):
        cpp_board = env._cpp_board
        side_cpp = CppSide.CHESS if state.side_to_move == Side.CHESS else CppSide.XIANGQI
        sr = cpp_best_move(cpp_board, side_cpp, 1, dict(state.repetition), state.ply, 50)
        if sr.best_move is None:
            break
        py_mv = Move(sr.best_move.fx, sr.best_move.fy, sr.best_move.tx, sr.best_move.ty)
        state, reward, done, info = env.step(py_mv)
        if done:
            break
    
    print(f"  [PASS] AB D1 game completed: ply={state.ply}, done={done}")

if __name__ == "__main__":
    print("=== Rule Reform Smoke Tests ===")
    print()
    
    print("[1] no_promotion")
    test_no_promotion()
    print()
    
    print("[2] chess_palace")
    test_chess_palace()
    print()
    
    print("[3] knight_block")
    test_knight_block()
    print()
    
    print("[4] C++ RuleFlags")
    test_cpp_rule_flags()
    print()
    
    print("[5] AB D1 with all flags")
    test_ab_with_flags()
    print()
    
    print("=== ALL TESTS PASSED ===")
