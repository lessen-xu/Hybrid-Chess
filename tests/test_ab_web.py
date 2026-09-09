"""Search regressions: viewpoint, repetition and bounded completion."""
from time import perf_counter
from types import SimpleNamespace
import pytest
import hybrid.agents.alphabeta_agent as search
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.core.env import GameState, HybridChessEnv
from hybrid.core.rules import board_hash, GameInfo, TerminalStatus
from hybrid.core.types import Side, Move


@pytest.mark.parametrize("root_side", [Side.CHESS, Side.XIANGQI])
def test_two_ply_search_values_the_correct_player(monkeypatch, root_side):
    a, b, reply = Move(0, 0, 0, 1), Move(1, 0, 1, 1), Move(2, 0, 2, 1)
    tree = {"root": [(a, "a"), (b, "b")], "a": [(reply, "good")], "b": [(reply, "bad")]}
    scores = {"good": 4.0, "bad": -7.0}
    monkeypatch.setattr(search, "apply_move", lambda board, move: dict(tree[board])[move])
    monkeypatch.setattr(search, "generate_legal_moves", lambda board, side: [m for m, _ in tree.get(board, [])])
    monkeypatch.setattr(search, "terminal_info", lambda *args: GameInfo(TerminalStatus.ONGOING))
    monkeypatch.setattr(search, "board_hash", lambda board, side: str(board) + side.name)
    monkeypatch.setattr(search, "evaluate", lambda state, perspective, weights: scores[state.board] * (1 if perspective == root_side else -1))
    agent = AlphaBetaAgent(SearchConfig(depth=2))
    monkeypatch.setattr(agent, "_move_order_key", lambda *args: 0)
    state = GameState("root", root_side)
    assert agent.select_move(state, [a, b]) == a
    assert agent.last_completed_depth == 2


def test_search_child_counts_repetition_without_mutating_parent():
    env = HybridChessEnv()
    state = env.reset()
    agent = AlphaBetaAgent()
    move = env.legal_moves()[0]
    first = agent._child(state, move)
    key = board_hash(first.board, first.side_to_move)
    state.repetition[key] = 2
    before = dict(state.repetition)
    child = agent._child(state, move)
    assert child.repetition[key] == 3
    assert state.repetition == before
    assert child.ply == state.ply + 1
    assert agent._negamax(child, 1, -1e18, 1e18, child.side_to_move) == 0


def test_timeout_keeps_last_complete_iteration(monkeypatch):
    env = HybridChessEnv()
    state = env.reset()
    moves = env.legal_moves()[:2]
    agent = AlphaBetaAgent(SearchConfig(depth=4, time_limit_seconds=10))
    monkeypatch.setattr(agent, "_ordered_moves", lambda state, legal: legal)
    calls = []
    def value(child, depth, *args):
        calls.append(depth)
        if depth:
            raise search._SearchTimeout
        return 2 if len(calls) == 1 else -3
    monkeypatch.setattr(agent, "_negamax", value)
    assert agent.select_move(state, moves) == moves[1]
    assert agent.last_completed_depth == 1
    assert calls == [0, 0, 1]
    assert agent._deadline is None


def test_timeout_before_any_iteration_returns_a_legal_move(monkeypatch):
    env = HybridChessEnv()
    state = env.reset()
    moves = env.legal_moves()
    agent = AlphaBetaAgent(SearchConfig(time_limit_seconds=1))
    def timeout(*args):
        raise search._SearchTimeout
    monkeypatch.setattr(agent, "_ordered_moves", timeout)
    assert agent.select_move(state, moves) == moves[0]
    assert agent.last_completed_depth == 0


def test_real_search_respects_small_budget():
    env = HybridChessEnv()
    state = env.reset()
    moves = env.legal_moves()
    agent = AlphaBetaAgent(SearchConfig(depth=4, time_limit_seconds=0.05))
    start = perf_counter()
    assert agent.select_move(state, moves) in moves
    assert perf_counter() - start < 2.0  # allow slow CI and one in-flight leaf


def test_web_budgets_do_not_change_offline_defaults():
    from hybrid.server import create_agent
    assert SearchConfig().time_limit_seconds is None
    for name, depth, seconds in [("ab_d1", 1, 1), ("ab_d2", 2, 3), ("ab_d4", 4, 6)]:
        config = create_agent(name).cfg
        assert config.depth == depth and config.time_limit_seconds == seconds
    assert create_agent("random").name == "random"
    with pytest.raises(ValueError):
        SearchConfig(time_limit_seconds=0)
