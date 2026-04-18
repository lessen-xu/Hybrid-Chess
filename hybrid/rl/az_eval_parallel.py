"""Multi-process parallel evaluation and gating matches.

play_match_parallel(): AZ agent vs fixed opponent (Random / AB depth=1).
gating_match_parallel(): candidate AZ vs best AZ (two checkpoints).

Each worker loads model from checkpoint (CPU inference), plays games independently,
and returns MatchStats via mp.Queue.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import multiprocessing as mp
from dataclasses import dataclass

from hybrid.rl.az_eval import MatchStats
# Worker functions

def _eval_worker(
    worker_id: int,
    games: int,
    model_ckpt_path: str,
    opponent_type: str,       # "random" or "ab_d1"
    simulations: int,
    seed: int,
    ablation: str,
    swap_sides: bool,
    game_offset: int,
    total_games: int,
    result_queue: mp.Queue,
    use_cpp: bool = False,
) -> None:
    """Worker: load model, create agent, play games, put MatchStats in queue."""
    import torch
    from hybrid.rl.az_runner import _apply_ablation
    from hybrid.rl.az_network import PolicyValueNet
    from hybrid.agents.alphazero_stub import (
        AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel,
    )
    from hybrid.agents.random_agent import RandomAgent
    from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
    from hybrid.rl.az_eval import play_one_game, MatchStats
    from hybrid.core.types import Side

    variant_cfg = _apply_ablation(ablation)

    from hybrid.rl.az_runner import build_net_from_checkpoint
    net = build_net_from_checkpoint(model_ckpt_path, device="cpu")
    model = TorchPolicyValueModel(net, device="cpu")

    az_agent = AlphaZeroMiniAgent(
        model=model,
        cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.0),
        seed=seed,
        use_cpp=use_cpp,
    )

    if opponent_type == "random":
        opponent = RandomAgent(seed=seed + 999)
    elif opponent_type == "ab_d1":
        opponent = AlphaBetaAgent(SearchConfig(depth=1))
    else:
        raise ValueError(f"Unknown opponent: {opponent_type}")

    stats = MatchStats()
    for local_i in range(games):
        global_i = game_offset + local_i

        if swap_sides and global_i >= total_games // 2:
            agent_chess = opponent
            agent_xiangqi = az_agent
            az_is_chess = False
        else:
            agent_chess = az_agent
            agent_xiangqi = opponent
            az_is_chess = True

        winner, plies, _ = play_one_game(agent_chess, agent_xiangqi, seed=seed + local_i, use_cpp=use_cpp, variant=variant_cfg)
        stats.total_plies += plies
        stats.games += 1

        if winner is None:
            stats.draw += 1
        elif (winner == Side.CHESS and az_is_chess) or \
             (winner == Side.XIANGQI and not az_is_chess):
            stats.win_a += 1
        else:
            stats.win_b += 1

    result_queue.put((worker_id, stats))


def _gating_worker(
    worker_id: int,
    games: int,
    candidate_ckpt: str,
    best_ckpt: str,
    simulations: int,
    seed: int,
    ablation: str,
    swap_sides: bool,
    game_offset: int,
    total_games: int,
    result_queue: mp.Queue,
    use_cpp: bool = False,
) -> None:
    """Worker: load candidate and best models, play gating games."""
    import torch
    from hybrid.rl.az_runner import _apply_ablation
    from hybrid.rl.az_network import PolicyValueNet
    from hybrid.agents.alphazero_stub import (
        AlphaZeroMiniAgent, MCTSConfig, TorchPolicyValueModel,
    )
    from hybrid.rl.az_eval import play_one_game, MatchStats
    from hybrid.core.types import Side

    variant_cfg = _apply_ablation(ablation)

    from hybrid.rl.az_runner import build_net_from_checkpoint

    cand_net = build_net_from_checkpoint(candidate_ckpt, device="cpu")
    cand_model = TorchPolicyValueModel(cand_net, device="cpu")
    cand_agent = AlphaZeroMiniAgent(
        model=cand_model,
        cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.0),
        seed=seed,
        use_cpp=use_cpp,
    )

    best_net = build_net_from_checkpoint(best_ckpt, device="cpu")
    best_model = TorchPolicyValueModel(best_net, device="cpu")
    best_agent = AlphaZeroMiniAgent(
        model=best_model,
        cfg=MCTSConfig(simulations=simulations, dirichlet_eps=0.0),
        seed=seed + 1,
        use_cpp=use_cpp,
    )

    stats = MatchStats()
    for local_i in range(games):
        global_i = game_offset + local_i

        if swap_sides and global_i >= total_games // 2:
            agent_chess = best_agent
            agent_xiangqi = cand_agent
            cand_is_chess = False
        else:
            agent_chess = cand_agent
            agent_xiangqi = best_agent
            cand_is_chess = True

        winner, plies, _ = play_one_game(agent_chess, agent_xiangqi, seed=seed + local_i, use_cpp=use_cpp, variant=variant_cfg)
        stats.total_plies += plies
        stats.games += 1

        if winner is None:
            stats.draw += 1
        elif (winner == Side.CHESS and cand_is_chess) or \
             (winner == Side.XIANGQI and not cand_is_chess):
            stats.win_a += 1  # candidate wins
        else:
            stats.win_b += 1  # best wins

    result_queue.put((worker_id, stats))
# Main-process orchestration

def play_match_parallel(
    model_ckpt_path: str,
    opponent_type: str,
    games: int,
    num_workers: int,
    simulations: int = 50,
    seed: int = 0,
    ablation: str = "extra_cannon",
    swap_sides: bool = True,
    use_cpp: bool = False,
) -> MatchStats:
    """Parallel evaluation: AZ vs fixed opponent."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    actual_workers = min(num_workers, games)
    if actual_workers <= 1:
        # Fallback to single process
        from hybrid.rl.az_eval import play_match, make_eval_az_agent
        import torch
        from hybrid.rl.az_network import PolicyValueNet
        from hybrid.agents.alphazero_stub import TorchPolicyValueModel
        from hybrid.agents.random_agent import RandomAgent
        from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig

        from hybrid.rl.az_runner import build_net_from_checkpoint

        net = build_net_from_checkpoint(model_ckpt_path, device="cpu")
        model = TorchPolicyValueModel(net, device="cpu")
        az = make_eval_az_agent(model, simulations=simulations, seed=seed, use_cpp=use_cpp)

        if opponent_type == "random":
            opp = RandomAgent(seed=seed + 999)
        else:
            opp = AlphaBetaAgent(SearchConfig(depth=1))
        stats, _ = play_match(az, opp, games=games, swap_sides=swap_sides, seed=seed, use_cpp=use_cpp)
        return stats

    result_queue = mp.Queue()

    games_per_worker = games // actual_workers
    remainder = games % actual_workers

    workers = []
    offset = 0
    for wid in range(actual_workers):
        n = games_per_worker + (1 if wid < remainder else 0)
        p = mp.Process(
            target=_eval_worker,
            kwargs=dict(
                worker_id=wid, games=n, model_ckpt_path=model_ckpt_path,
                opponent_type=opponent_type, simulations=simulations,
                seed=seed + wid * 10000, ablation=ablation,
                swap_sides=swap_sides, game_offset=offset,
                total_games=games, result_queue=result_queue,
                use_cpp=use_cpp,
            ),
        )
        p.start()
        workers.append(p)
        offset += n

    for p in workers:
        p.join()

    for i, p in enumerate(workers):
        if p.exitcode != 0:
            raise RuntimeError(f"eval_worker {i} exited with code {p.exitcode}")

    total = MatchStats()
    for _ in range(actual_workers):
        _, stats = result_queue.get()
        total.win_a += stats.win_a
        total.draw += stats.draw
        total.win_b += stats.win_b
        total.total_plies += stats.total_plies
        total.games += stats.games

    return total


def gating_match_parallel(
    candidate_ckpt: str,
    best_ckpt: str,
    games: int,
    num_workers: int,
    simulations: int = 20,
    seed: int = 0,
    ablation: str = "extra_cannon",
    swap_sides: bool = True,
    use_cpp: bool = False,
) -> MatchStats:
    """Parallel gating: candidate vs best. win_a = candidate wins, win_b = best wins."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    actual_workers = min(num_workers, games)
    if actual_workers <= 1:
        from hybrid.rl.az_eval import play_match, make_eval_az_agent
        import torch
        from hybrid.rl.az_network import PolicyValueNet
        from hybrid.agents.alphazero_stub import TorchPolicyValueModel

        from hybrid.rl.az_runner import build_net_from_checkpoint

        cnet = build_net_from_checkpoint(candidate_ckpt, device="cpu")
        cm = TorchPolicyValueModel(cnet, device="cpu")
        ca = make_eval_az_agent(cm, simulations=simulations, seed=seed, use_cpp=use_cpp)

        bnet = build_net_from_checkpoint(best_ckpt, device="cpu")
        bm = TorchPolicyValueModel(bnet, device="cpu")
        ba = make_eval_az_agent(bm, simulations=simulations, seed=seed + 1, use_cpp=use_cpp)

        stats, _ = play_match(ca, ba, games=games, swap_sides=swap_sides, seed=seed, use_cpp=use_cpp)
        return stats

    result_queue = mp.Queue()

    games_per_worker = games // actual_workers
    remainder = games % actual_workers

    workers = []
    offset = 0
    for wid in range(actual_workers):
        n = games_per_worker + (1 if wid < remainder else 0)
        p = mp.Process(
            target=_gating_worker,
            kwargs=dict(
                worker_id=wid, games=n, candidate_ckpt=candidate_ckpt,
                best_ckpt=best_ckpt, simulations=simulations,
                seed=seed + wid * 10000, ablation=ablation,
                swap_sides=swap_sides, game_offset=offset,
                total_games=games, result_queue=result_queue,
                use_cpp=use_cpp,
            ),
        )
        p.start()
        workers.append(p)
        offset += n

    for p in workers:
        p.join()

    for i, p in enumerate(workers):
        if p.exitcode != 0:
            raise RuntimeError(f"gating_worker {i} exited with code {p.exitcode}")

    total = MatchStats()
    for _ in range(actual_workers):
        _, stats = result_queue.get()
        total.win_a += stats.win_a
        total.draw += stats.draw
        total.win_b += stats.win_b
        total.total_plies += stats.total_plies
        total.games += stats.games

    return total
