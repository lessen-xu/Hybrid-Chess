"""Multi-agent paired tournament runner for rule screening.

Decouples agent competence from army advantage by executing paired games:
In each seed, Agent A plays Chess vs Agent B as Xiangqi, and then Agent B
plays Chess vs Agent A as Xiangqi.

Tracks complete game outcomes and termination distributions:
- Checkmate
- Stalemate (loss or draw)
- Threefold Repetition
- Perpetual Check
- Max Plies Truncation
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import math
import time

from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.types import Side, Move
from hybrid.core.rules import terminal_info, TerminalStatus
from hybrid.core.config import VariantConfig
from hybrid.agents.base import Agent
from hybrid.agents.random_agent import RandomAgent
from hybrid.agents.greedy_agent import GreedyAgent
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig


@dataclass
class TournamentConfig:
    """Tournament execution configuration."""
    num_pairs: int = 10                  # Each pair = 2 games (sides swapped)
    max_plies: int = 200                 # Truncate at 200 for fast screening
    agent_a: str = "ab_fast"             # AlphaBeta d=1
    agent_b: str = "greedy"              # Greedy 1-ply capture
    sprt_enabled: bool = False           # Early stopping on severe asymmetry
    sprt_min_pairs: int = 5
    sprt_disparity_cutoff: float = 0.50  # Disparity >= 50% triggers early rejection


@dataclass
class GameResult:
    """Telemetry for a single played game."""
    seed: int
    chess_agent: str
    xiangqi_agent: str
    winner: Optional[str]               # "chess", "xiangqi", or None (draw)
    status: str                         # TerminalStatus enum name
    reason: str                         # Reason string
    plies: int
    duration_sec: float


@dataclass
class TournamentResult:
    """Aggregated tournament results for a single variant configuration."""
    variant_name: str
    factors: Dict[str, Any]
    total_games: int
    chess_wins: int
    xiangqi_wins: int
    draws: int
    chess_score: float                  # chess_wins + 0.5 * draws
    xiangqi_score: float                # xiangqi_wins + 0.5 * draws
    chess_score_rate: float             # chess_score / total_games
    xiangqi_score_rate: float           # xiangqi_score / total_games
    balance_disparity: float            # |chess_score_rate - xiangqi_score_rate|
    mean_plies: float
    termination_distribution: Dict[str, float]  # fraction per reason
    sprt_stopped_early: bool
    games: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_agent(name: str, seed: int = 0) -> Agent:
    """Instantiate agent by name."""
    if name == "random":
        return RandomAgent(seed=seed)
    elif name == "greedy":
        return GreedyAgent(seed=seed)
    elif name == "ab_fast":
        from hybrid.agents.eval import EvalWeights
        return AlphaBetaAgent(cfg=SearchConfig(depth=1, eval_weights=EvalWeights(mobility=0.0)))
    elif name == "ab_d2":
        from hybrid.agents.eval import EvalWeights
        return AlphaBetaAgent(cfg=SearchConfig(depth=2, eval_weights=EvalWeights(mobility=0.0)))
    raise ValueError(f"Unknown agent name: {name}")


def play_single_game(
    variant: VariantConfig,
    chess_agent_name: str,
    xiangqi_agent_name: str,
    seed: int,
    max_plies: int = 200,
) -> GameResult:
    """Play a single game between two specified agents."""
    env = HybridChessEnv(max_plies=max_plies, use_cpp=False, variant=variant)
    state = env.reset()

    agent_chess = _build_agent(chess_agent_name, seed=seed)
    agent_xq = _build_agent(xiangqi_agent_name, seed=seed + 10000)

    start_t = time.perf_counter()

    while True:
        legal_moves = env.legal_moves()
        if not legal_moves:
            info = terminal_info(state.board, state.side_to_move, state.repetition, state.ply, max_plies)
            winner_str = "chess" if info.winner == Side.CHESS else ("xiangqi" if info.winner == Side.XIANGQI else None)
            return GameResult(
                seed=seed,
                chess_agent=chess_agent_name,
                xiangqi_agent=xiangqi_agent_name,
                winner=winner_str,
                status=str(info.status),
                reason=info.reason or "No legal moves",
                plies=state.ply,
                duration_sec=time.perf_counter() - start_t,
            )

        current_agent = agent_chess if state.side_to_move == Side.CHESS else agent_xq
        mv = current_agent.select_move(state, legal_moves)
        state, _, done, info = env.step(mv)

        if done:
            winner_str = None
            if info.winner == Side.CHESS:
                winner_str = "chess"
            elif info.winner == Side.XIANGQI:
                winner_str = "xiangqi"

            return GameResult(
                seed=seed,
                chess_agent=chess_agent_name,
                xiangqi_agent=xiangqi_agent_name,
                winner=winner_str,
                status=str(info.status),
                reason=info.reason,
                plies=state.ply,
                duration_sec=time.perf_counter() - start_t,
            )


def run_paired_tournament(
    variant: VariantConfig,
    config: TournamentConfig,
    variant_name: str = "variant",
    factors: Optional[Dict[str, Any]] = None,
) -> TournamentResult:
    """Run a paired tournament (swapping sides per seed) for a single variant."""
    games: List[GameResult] = []
    chess_wins = 0
    xiangqi_wins = 0
    draws = 0
    total_plies = 0
    termination_counts: Dict[str, int] = {}
    sprt_stopped = False

    for pair_idx in range(config.num_pairs):
        seed = pair_idx * 2

        # Game 1: Agent A as Chess, Agent B as Xiangqi
        g1 = play_single_game(
            variant,
            chess_agent_name=config.agent_a,
            xiangqi_agent_name=config.agent_b,
            seed=seed,
            max_plies=config.max_plies,
        )
        games.append(g1)

        # Game 2: Agent B as Chess, Agent A as Xiangqi
        g2 = play_single_game(
            variant,
            chess_agent_name=config.agent_b,
            xiangqi_agent_name=config.agent_a,
            seed=seed + 1,
            max_plies=config.max_plies,
        )
        games.append(g2)

        # Update running tallies
        for g in (g1, g2):
            if g.winner == "chess":
                chess_wins += 1
            elif g.winner == "xiangqi":
                xiangqi_wins += 1
            else:
                draws += 1
            total_plies += g.plies

            # Normalize reason category
            cat = "checkmate"
            low_r = g.reason.lower()
            if "stalemate" in low_r:
                cat = "stalemate_draw" if "draw" in low_r else "stalemate_loss"
            elif "repetition" in low_r:
                cat = "threefold_repetition"
            elif "perpetual" in low_r:
                cat = "perpetual_check"
            elif "max plies" in low_r:
                cat = "max_plies"
            elif "captured" in low_r:
                cat = "royal_capture"

            termination_counts[cat] = termination_counts.get(cat, 0) + 1

        # Check SPRT early stopping
        current_games = len(games)
        c_score = chess_wins + 0.5 * draws
        x_score = xiangqi_wins + 0.5 * draws
        disparity = abs(c_score - x_score) / current_games

        if config.sprt_enabled and (pair_idx + 1) >= config.sprt_min_pairs:
            if disparity >= config.sprt_disparity_cutoff:
                sprt_stopped = True
                break

    total_games = len(games)
    chess_score = chess_wins + 0.5 * draws
    xiangqi_score = xiangqi_wins + 0.5 * draws
    c_rate = chess_score / max(1, total_games)
    x_rate = xiangqi_score / max(1, total_games)
    final_disparity = abs(c_rate - x_rate)

    term_dist = {
        cat: round(cnt / max(1, total_games), 4)
        for cat, cnt in termination_counts.items()
    }

    return TournamentResult(
        variant_name=variant_name,
        factors=factors or {},
        total_games=total_games,
        chess_wins=chess_wins,
        xiangqi_wins=xiangqi_wins,
        draws=draws,
        chess_score=chess_score,
        xiangqi_score=xiangqi_score,
        chess_score_rate=round(c_rate, 4),
        xiangqi_score_rate=round(x_rate, 4),
        balance_disparity=round(final_disparity, 4),
        mean_plies=round(total_plies / max(1, total_games), 2),
        termination_distribution=term_dist,
        sprt_stopped_early=sprt_stopped,
        games=[asdict(g) for g in games],
    )
