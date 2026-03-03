# -*- coding: utf-8 -*-
"""Unified evaluation module: standardized match play and statistical testing.

Self-play mode uses Dirichlet noise + temperature for exploration.
Eval mode disables all randomness (eps=0, argmax) for fair measurement.
swap_sides alternates agent assignments to neutralize side bias.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.types import Side, Move
from hybrid.core.render import render_board
from hybrid.agents.alphazero_stub import (
    AlphaZeroMiniAgent,
    MCTSConfig,
    TorchPolicyValueModel,
)
from hybrid.agents.random_agent import RandomAgent
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig


# ====================================================================
# Wilson CI — for adaptive gating
# ====================================================================

def wilson_ci(
    wins: int,
    losses: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Wilson score confidence interval for win rate (ignoring draws).

    Returns (p_hat, ci_low, ci_high). If wins+losses==0, returns (0.5, 0.0, 1.0).
    """
    import math

    n = wins + losses
    if n == 0:
        return (0.5, 0.0, 1.0)

    _z_table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    if confidence in _z_table:
        z = _z_table[confidence]
    else:
        p = (1.0 - confidence) / 2.0
        t = math.sqrt(-2.0 * math.log(p))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / \
                (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)

    p_hat = wins / n
    z2 = z * z

    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))

    ci_low = max(0.0, centre - spread)
    ci_high = min(1.0, centre + spread)

    return (p_hat, ci_low, ci_high)


def score_ci(
    wins: int,
    draws: int,
    losses: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Score-based CI (W=1, D=0.5, L=0). Unlike wilson_ci, draws provide information.

    Returns (mean_score, ci_low, ci_high). Normal approximation, accurate for n>=10.
    """
    import math

    n = wins + draws + losses
    if n == 0:
        return (0.5, 0.0, 1.0)

    total_score = wins * 1.0 + draws * 0.5
    mean = total_score / n

    # Trinomial variance
    sum_sq = wins * 1.0 + draws * 0.25
    variance = max(sum_sq / n - mean * mean, 0.0)

    _z_table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    if confidence in _z_table:
        z = _z_table[confidence]
    else:
        p = (1.0 - confidence) / 2.0
        t = math.sqrt(-2.0 * math.log(p))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / \
                (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t)

    se = math.sqrt(variance / n) if n > 0 else 0.0

    ci_low = max(0.0, mean - z * se)
    ci_high = min(1.0, mean + z * se)

    return (mean, ci_low, ci_high)


@dataclass
class MatchStats:
    """Match result statistics."""
    win_a: int = 0
    draw: int = 0
    win_b: int = 0
    total_plies: int = 0
    games: int = 0

    @property
    def avg_plies(self) -> float:
        return self.total_plies / self.games if self.games > 0 else 0.0

    @property
    def score(self) -> float:
        """Agent A's mean score (W=1, D=0.5, L=0)."""
        if self.games == 0:
            return 0.5
        return (self.win_a + 0.5 * self.draw) / self.games

    def to_dict(self) -> dict:
        return {
            "win_a": self.win_a,
            "draw": self.draw,
            "win_b": self.win_b,
            "avg_plies": round(self.avg_plies, 1),
            "score": round(self.score, 4),
            "games": self.games,
        }


def play_one_game(
    agent_chess,
    agent_xiangqi,
    seed: int = 0,
    record: bool = False,
    use_cpp: bool = False,
) -> Tuple[Optional[Side], int, Optional[dict]]:
    """Play one game. Returns (winner, plies, game_record_or_None)."""
    env = HybridChessEnv(use_cpp=use_cpp)
    state = env.reset()
    agents = {Side.CHESS: agent_chess, Side.XIANGQI: agent_xiangqi}

    states_ascii = []
    moves_str = []
    if record:
        states_ascii.append(render_board(state.board))

    while True:
        legal = env.legal_moves()
        if len(legal) == 0:
            break
        agent = agents[state.side_to_move]
        mv = agent.select_move(state, legal)

        if record:
            moves_str.append(str(mv))

        state, reward, done, info = env.step(mv)

        if record:
            states_ascii.append(render_board(state.board))

        if done:
            break

    game_record_dict = None
    if record:
        result_str = "1/2-1/2"
        if info.winner == Side.CHESS:
            result_str = "Chess wins"
        elif info.winner == Side.XIANGQI:
            result_str = "Xiangqi wins"

        game_record_dict = {
            "result": result_str,
            "moves": moves_str,
            "states_ascii": states_ascii,
            "meta": {
                "plies": state.ply,
                "reason": info.reason if hasattr(info, 'reason') else "",
                "seed": seed,
            },
        }

    return info.winner, state.ply, game_record_dict


def play_match(
    agent_a,
    agent_b,
    games: int,
    swap_sides: bool = True,
    seed: int = 0,
    record_first_n: int = 0,
    use_cpp: bool = False,
) -> Tuple[MatchStats, List[dict]]:
    """Play a match (multiple games). swap_sides=True alternates side assignments."""
    stats = MatchStats()
    recordings: List[dict] = []

    for game_i in range(games):
        if swap_sides and game_i >= games // 2:
            agent_chess = agent_b
            agent_xiangqi = agent_a
            a_is_chess = False
        else:
            agent_chess = agent_a
            agent_xiangqi = agent_b
            a_is_chess = True

        should_record = game_i < record_first_n

        winner, plies, game_rec = play_one_game(
            agent_chess, agent_xiangqi,
            seed=seed + game_i,
            record=should_record,
            use_cpp=use_cpp,
        )

        stats.total_plies += plies
        stats.games += 1

        if winner is None:
            stats.draw += 1
        elif (winner == Side.CHESS and a_is_chess) or \
             (winner == Side.XIANGQI and not a_is_chess):
            stats.win_a += 1
        else:
            stats.win_b += 1

        if game_rec is not None:
            game_rec["meta"]["game_index"] = game_i
            game_rec["meta"]["a_is_chess"] = a_is_chess
            recordings.append(game_rec)

    return stats, recordings


def make_eval_az_agent(
    model: TorchPolicyValueModel,
    simulations: int = 50,
    seed: int = 0,
    use_cpp: bool = False,
) -> AlphaZeroMiniAgent:
    """Create an eval-mode AZ agent (no Dirichlet noise, argmax selection)."""
    return AlphaZeroMiniAgent(
        model=model,
        cfg=MCTSConfig(
            simulations=simulations,
            dirichlet_eps=0.0,
        ),
        seed=seed,
        use_cpp=use_cpp,
    )
