"""Search-level tactical and rule-context acceptance positions."""
import json
from pathlib import Path
import pytest
import torch

from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig, PolicyValueModel, TorchPolicyValueModel
from hybrid.core.board import Board
from hybrid.core.env import HybridChessEnv
from hybrid.core.rules import apply_move, board_hash
from hybrid.core.types import Piece, PieceKind as K, Side, Move
from hybrid.rl.general_model import new_model
from hybrid.web_variants import parse_variant


def restore(record, cpp=False):
    board = Board.empty()
    for x, y, kind, side in record["pieces"]:
        board.set(x, y, Piece(K[kind], Side[side]))
    env = HybridChessEnv(variant=parse_variant(record["variant"]), use_cpp=cpp,
                         max_plies=record.get("max_plies", 400))
    state = env.reset_from_board(board, Side[record["side"]])
    state.ply = record["ply"]
    state.repetition = dict(record["repetition"])
    env.state = state
    return env, state


def test_heldout_combinations_are_outside_the_training_sampler():
    from hybrid.rl.general_model import RULE_FIELDS
    default = parse_variant("none")
    variants = json.loads((Path(__file__).resolve().parents[1] / "configs/heldout-variants.json").read_text())
    for entry in variants:
        variant = parse_variant(entry["variant"])
        assert sum(getattr(variant, k) != getattr(default, k) for k in RULE_FIELDS) >= 7


def position(entries, side, flags=None, cpp=False):
    if cpp:
        pytest.importorskip("hybrid.cpp_engine.hybrid_cpp_engine")
    board = Board.empty()
    for x, y, kind, army in entries:
        board.set(x, y, Piece(kind, army))
    env = HybridChessEnv(variant=parse_variant(flags or {}), use_cpp=cpp)
    state = env.reset_from_board(board, side)
    return env, state


@pytest.mark.parametrize("cpp", [False, True])
@pytest.mark.parametrize("kind,side,source,target,screen", [
    (K.QUEEN, Side.CHESS, (4, 7), (4, 9), []),
    (K.CHARIOT, Side.XIANGQI, (0, 2), (0, 0), []),
    (K.CANNON, Side.XIANGQI, (0, 3), (0, 0), [(0, 2, K.SOLDIER, Side.XIANGQI)]),
])
def test_search_finds_immediate_win(cpp, kind, side, source, target, screen):
    torch.set_num_threads(1)
    torch.manual_seed(10)
    env, state = position([(0, 0, K.KING, Side.CHESS), (4, 9, K.GENERAL, Side.XIANGQI),
        (*source, kind, side)] + screen, side, cpp=cpp)
    agent = AlphaZeroMiniAgent(TorchPolicyValueModel(new_model(8, 1)),
        MCTSConfig(simulations=128, dirichlet_eps=0., discount_factor=1.), use_cpp=cpp)
    legal = env.legal_moves()
    assert Move(*source, *target) in legal
    chosen = agent.select_move(state, legal)
    # Capturing the royal and giving a one-move mate/stalemate are equal wins.
    _, _, done, info = env.step(chosen)
    assert done and info.winner == side


class _BadPolicyModel(PolicyValueModel):
    def __init__(self, preferred_move: Move):
        self.preferred_move = preferred_move

    def predict(self, state, moves):
        policy = {mv: 1.0 if mv == self.preferred_move else 0.001 for mv in moves}
        return policy, 0.98331

    def predict_batch(self, inputs):
        return [self.predict(state, moves) for state, moves in inputs]


def test_ai_prefers_immediate_tactical_win():
    record = {
        "pieces": [
            [4, 0, "KING", "CHESS"],
            [7, 1, "CHARIOT", "XIANGQI"],
            [1, 2, "CHARIOT", "XIANGQI"],
            [5, 2, "PAWN", "CHESS"],
            [8, 2, "SOLDIER", "XIANGQI"],
            [0, 3, "PAWN", "CHESS"],
            [4, 3, "PAWN", "CHESS"],
            [3, 5, "PAWN", "CHESS"],
            [0, 6, "SOLDIER", "XIANGQI"],
            [2, 6, "SOLDIER", "XIANGQI"],
            [4, 6, "SOLDIER", "XIANGQI"],
            [6, 6, "SOLDIER", "XIANGQI"],
            [4, 8, "ADVISOR", "XIANGQI"],
            [1, 9, "HORSE", "XIANGQI"],
            [2, 9, "ELEPHANT", "XIANGQI"],
            [3, 9, "GENERAL", "XIANGQI"],
            [5, 9, "ADVISOR", "XIANGQI"],
            [6, 9, "ELEPHANT", "XIANGQI"],
        ],
        "side": "XIANGQI",
        "ply": 47,
        "repetition": {},
        "variant": {
            "extra_pawn_i_file": True,
            "no_queen": False,
            "no_bishop": False,
            "one_rook": False,
            "remove_extra_pawn": False,
            "extra_cannon": False,
            "extra_soldier": False,
            "xq_queen": False,
            "flying_general": True,
            "no_promotion": False,
            "chess_palace": True,
            "knight_block": True,
            "no_queen_promotion": False,
        },
        "max_plies": 400,
    }
    env, state = restore(record)
    legal = env.legal_moves()
    assert legal, "expected legal moves from diagnostic position"

    forced_win = Move(1, 2, 1, 0)
    assert forced_win in legal

    # Give the model a strong incorrect preference for a non-winning move to
    # reproduce the shallow-search failure mode.
    wrong = next(mv for mv in legal if mv != forced_win)
    agent = AlphaZeroMiniAgent(_BadPolicyModel(wrong), MCTSConfig(simulations=32, dirichlet_eps=0.), use_cpp=False)
    chosen = agent.select_move(state, legal)

    assert chosen == forced_win
    _, _, done, info = env.step(chosen)
    assert done and info.winner == Side.XIANGQI


@pytest.mark.parametrize("cpp", [False, True])
def test_search_scores_a_third_repetition_as_draw(cpp):
    if cpp:
        pytest.importorskip("hybrid.cpp_engine.hybrid_cpp_engine")
    env = HybridChessEnv(use_cpp=cpp)
    state = env.reset()
    legal = env.legal_moves()
    chosen = legal[0]
    key = board_hash(apply_move(state.board, chosen), state.side_to_move.opponent())
    state.repetition[key] = 2
    class Model:
        def predict(self, state, moves):
            return {m: float(m == chosen) for m in moves}, .75
    agent = AlphaZeroMiniAgent(Model(), MCTSConfig(simulations=2, leaf_batch_size=1,
        dirichlet_eps=0., discount_factor=1.), use_cpp=cpp)
    root = agent._run_mcts_search(state, legal, add_noise=False)
    assert root.children[chosen].N == 2
    assert root.children[chosen].Q == 0.
    assert root.Q == 0.
    assert state.repetition[key] == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_cpp_flying_general_switch_on_exposed_kings(enabled):
    entries = [(4, 0, K.KING, Side.CHESS), (4, 9, K.GENERAL, Side.XIANGQI)]
    py, a = position(entries, Side.XIANGQI, {"flying_general": enabled})
    cpp, b = position(entries, Side.XIANGQI, {"flying_general": enabled}, True)
    assert set(py.legal_moves()) == set(cpp.legal_moves())
    assert (Move(4, 9, 4, 0) in cpp.legal_moves()) == enabled


@pytest.mark.parametrize("flags,promotions", [({}, {K.QUEEN, K.ROOK, K.BISHOP, K.KNIGHT}),
    ({"no_queen_promotion": True}, {K.ROOK, K.BISHOP, K.KNIGHT}),
    ({"no_promotion": True}, {None})])
def test_search_respects_promotion_rules(flags, promotions):
    torch.set_num_threads(1)
    env, state = position([(4, 0, K.KING, Side.CHESS), (4, 9, K.GENERAL, Side.XIANGQI),
        (4, 5, K.PAWN, Side.CHESS), (0, 8, K.PAWN, Side.CHESS)], Side.CHESS, flags, True)
    legal = env.legal_moves()
    assert {m.promotion for m in legal if m.fx == 0} == promotions
    agent = AlphaZeroMiniAgent(TorchPolicyValueModel(new_model(8, 1)),
        MCTSConfig(simulations=8, dirichlet_eps=0., discount_factor=1.), use_cpp=True)
    assert agent.select_move(state, legal) in legal
