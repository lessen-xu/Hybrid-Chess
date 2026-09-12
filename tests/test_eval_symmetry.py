"""Tests for evaluation function zero-sum symmetry and endgame semantics."""
import pytest
from hybrid.core.board import Board
from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side, PieceKind as K, Piece, Move
from hybrid.core.config import VariantConfig
from hybrid.agents.eval import evaluate, material_score, mobility_score, EvalWeights
from hybrid.web_variants import PRESETS, parse_variant


def test_initial_board_symmetry_all_presets():
    """All preset opening positions must be strictly zero-sum symmetric."""
    for spec in PRESETS:
        vcfg = parse_variant(spec["id"])
        env = HybridChessEnv(variant=vcfg)
        state = env.reset()
        score_chess = evaluate(state, Side.CHESS)
        score_xiangqi = evaluate(state, Side.XIANGQI)
        assert score_chess == pytest.approx(-score_xiangqi, abs=1e-6), (
            f"Preset {spec['id']} failed zero-sum symmetry: "
            f"Chess={score_chess}, Xiangqi={score_xiangqi}, sum={score_chess + score_xiangqi}"
        )



def test_opening_does_not_trigger_endgame_conversion():
    """Standard opening has full armies (32 pieces); it must not trigger material tripling or king confinement."""
    env = HybridChessEnv()
    state = env.reset()
    mat_chess = material_score(state, Side.CHESS)
    mat_xq = material_score(state, Side.XIANGQI)
    # Chess initial material = 40 (with extra pawn), Xiangqi = 49 -> diff = -9.0 for Chess, +9.0 for Xiangqi
    assert mat_chess == -9.0
    assert mat_xq == 9.0

    score_chess = evaluate(state, Side.CHESS)
    score_xq = evaluate(state, Side.XIANGQI)
    # If tripling happened, material alone would be 27, yielding |score| > 25.
    # Without tripling, material is 9.0, mobility is small, check is 0 -> |score| ~ 10.15.
    assert abs(score_chess) < 15.0
    assert abs(score_xq) < 15.0
    assert score_chess == pytest.approx(-score_xq, abs=1e-6)


def test_stalemate_does_not_penalize_winning_attacker():
    """Driving opponent into near-stalemate (0 or 1 legal moves) must NOT penalize attacker with -8.0."""
    board = Board.empty()
    # Chess has King + 2 Rooks + Queen; Xiangqi has only General on (4,9)
    board.set(0, 0, Piece(K.KING, Side.CHESS))
    board.set(3, 0, Piece(K.ROOK, Side.CHESS))
    board.set(5, 0, Piece(K.ROOK, Side.CHESS))
    board.set(8, 0, Piece(K.QUEEN, Side.CHESS))
    board.set(4, 9, Piece(K.GENERAL, Side.XIANGQI))

    env = HybridChessEnv()
    state = env.reset_from_board(board, Side.XIANGQI)
    legal_opp = env.legal_moves()
    assert len(legal_opp) == 1, "Expected single legal escape for defender"

    score_chess = evaluate(state, Side.CHESS)
    score_xq = evaluate(state, Side.XIANGQI)

    assert score_chess > 50.0, "Chess should have a overwhelmingly winning evaluation"
    assert score_chess == pytest.approx(-score_xq, abs=1e-6)


def test_random_playout_positions_are_all_symmetric():
    """Positions generated along a game playout must strictly satisfy zero-sum antisymmetry."""
    env = HybridChessEnv()
    state = env.reset()

    import random
    rng = random.Random(42)

    for step in range(30):
        legal = env.legal_moves()
        if not legal:
            break
        c_score = evaluate(state, Side.CHESS)
        x_score = evaluate(state, Side.XIANGQI)
        assert c_score == pytest.approx(-x_score, abs=1e-5), (
            f"Step {step} failed symmetry: Chess={c_score}, XQ={x_score}"
        )
        mv = rng.choice(legal)
        state, _, done, _ = env.step(mv)
        if done:
            break

