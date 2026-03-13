"""AlphaZero-Mini: MCTS + policy/value network agent.

Value convention (critical for correctness):
- Network `predict` returns value from `state.side_to_move`'s perspective, in [-1, 1].
- Node.W/Q store cumulative/mean value from that node's side_to_move perspective.
- In _select_child, parent maximizes (-child.Q) because child.Q is the opponent's perspective.
- In _backup, value sign flips at each level.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random

from .base import Agent
from hybrid.core.env import GameState
from hybrid.core.types import Move, Side
from hybrid.core.rules import apply_move, generate_legal_moves, terminal_info, TerminalStatus
from hybrid.core.config import MAX_PLIES


@dataclass
class MCTSConfig:
    simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    discount_factor: float = 0.99   # γ: value decay per tree depth to prefer shorter wins
    leaf_batch_size: int = 8        # K: virtual-loss leaf gathering batch size


@dataclass
class Node:
    state: GameState
    prior: float = 0.0
    parent: Optional["Node"] = None
    children: Dict[Move, "Node"] = field(default_factory=dict)

    N: int = 0       # visit count
    W: float = 0.0   # total value
    Q: float = 0.0   # mean value
    virtual_loss: int = 0  # in-flight penalty counter for leaf batching

    # C++ engine fields (only set when use_cpp=True)
    cpp_board: object = None   # CppBoard or None
    cpp_side: object = None    # CppSide or None

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class PolicyValueModel:
    """Policy-value network interface."""

    def predict(self, state: GameState, legal_moves: List[Move]) -> Tuple[Dict[Move, float], float]:
        """Return (policy_dict, value) where policy sums to 1 and value is in [-1, 1]."""
        raise NotImplementedError


class AlphaZeroMiniAgent(Agent):
    """AlphaZero-Mini agent: MCTS + neural network search."""
    name = "alphazero_mini"

    def __init__(self, model: PolicyValueModel, cfg: MCTSConfig = MCTSConfig(),
                 seed: int = 0, use_cpp: bool = False):
        self.model = model
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.use_cpp = use_cpp

        # Lazy-init C++ helpers
        self._cpp = None
        if use_cpp:
            self._init_cpp()

    def _init_cpp(self):
        """Lazy-import C++ engine bindings and type maps."""
        import hybrid.core.env as _env
        _env._ensure_cpp_maps()
        # Access module-level globals AFTER _ensure_cpp_maps() has set them
        from types import SimpleNamespace
        self._cpp = SimpleNamespace(
            module=_env._cpp_module,
            PY_TO_CPP_SIDE=_env._PY_TO_CPP_SIDE,
            CPP_TO_PY_KIND=_env._CPP_TO_PY_KIND,
            PY_TO_CPP_KIND=_env._PY_TO_CPP_KIND,
            sync_to_cpp=_env._sync_to_cpp,
            sync_to_py=_env._sync_to_py,
            cpp_to_py_move=_env._cpp_to_py_move,
            py_to_cpp_move=_env._py_to_cpp_move,
        )
    # Core MCTS

    def _run_mcts_search(self, state: GameState, legal_moves: List[Move],
                         add_noise: bool = True) -> Node:
        """Run MCTS and return the root node."""
        if self.use_cpp:
            return self._run_mcts_search_cpp(state, legal_moves, add_noise)

        root = Node(state=state)

        # Expand root with optional Dirichlet noise
        policy, _ = self.model.predict(state, legal_moves)
        priors = {m: policy.get(m, 0.0) for m in legal_moves}
        if add_noise:
            self._add_dirichlet_noise(priors)
        self._expand(root, priors)

        for _ in range(self.cfg.simulations):
            node = root
            path = [node]

            # Selection
            while node.is_expanded():
                mv, node = self._select_child(node)
                path.append(node)

            # Evaluation
            info = terminal_info(node.state.board, node.state.side_to_move,
                                 node.state.repetition, node.state.ply, MAX_PLIES)
            if info.status != TerminalStatus.ONGOING:
                if info.status == TerminalStatus.DRAW:
                    value = 0.0
                else:
                    value = 1.0 if info.winner == node.state.side_to_move else -1.0
            else:
                moves = generate_legal_moves(node.state.board, node.state.side_to_move)
                policy, value = self.model.predict(node.state, moves)
                priors = {m: policy.get(m, 0.0) for m in moves}
                self._expand(node, priors)

            # Backup
            self._backup(path, value)

        return root
    # C++ MCTS path

    def _run_mcts_search_cpp(self, state: GameState, legal_moves: List[Move],
                              add_noise: bool = True) -> Node:
        """Run MCTS using C++ engine with virtual-loss leaf batching.

        Gathers up to K=leaf_batch_size leaves per round before calling
        model.predict_batch() once, dramatically reducing IPC round-trips.
        """
        cpp = self._cpp
        module = cpp.module
        K = self.cfg.leaf_batch_size

        # Build root: sync Python board → C++ board once
        cpp_board = cpp.sync_to_cpp(state.board)
        cpp_side = cpp.PY_TO_CPP_SIDE[state.side_to_move]
        root = Node(state=state, cpp_board=cpp_board, cpp_side=cpp_side)

        # Expand root: need NN inference → use Python state (already available)
        policy, _ = self.model.predict(state, legal_moves)
        priors = {m: policy.get(m, 0.0) for m in legal_moves}
        if add_noise:
            self._add_dirichlet_noise(priors)
        self._expand_cpp(root, priors)

        sims_done = 0
        total_sims = self.cfg.simulations

        while sims_done < total_sims:
            current_k = min(K, total_sims - sims_done)
            leaves_data = []   # (leaf_state, py_moves, path) for NN eval
            paths_for_vl = []  # paths that have virtual loss applied

            # ── Phase 1: Gather up to K leaves ──
            for _ in range(current_k):
                node = root
                path = [node]

                # Selection: traverse using VL-adjusted PUCT
                while node.is_expanded():
                    mv, node = self._select_child(node)
                    path.append(node)

                # Terminal check via C++
                cpp_info = module.terminal_info(
                    node.cpp_board, node.cpp_side,
                    node.state.repetition, node.state.ply, MAX_PLIES,
                )

                if cpp_info.status != TerminalStatus.ONGOING:
                    # Terminal: backup immediately, no VL needed
                    if cpp_info.status == TerminalStatus.DRAW:
                        value = 0.0
                    else:
                        cpp_winner = cpp_info.winner
                        if cpp_winner == 1:
                            winner = Side.CHESS
                        elif cpp_winner == 2:
                            winner = Side.XIANGQI
                        else:
                            winner = None
                        value = 1.0 if winner == node.state.side_to_move else -1.0
                    self._backup(path, value)
                    sims_done += 1
                    continue

                # Non-terminal leaf: apply virtual loss to divert next selection
                for n in path:
                    n.virtual_loss += 1
                paths_for_vl.append(path)

                # Generate legal moves via C++ and sync board for encoding
                cpp_moves = module.gen_legal(node.cpp_board, node.cpp_side)
                py_moves = [cpp.cpp_to_py_move(cm) for cm in cpp_moves]
                py_board = cpp.sync_to_py(node.cpp_board)
                leaf_state = GameState(
                    board=py_board,
                    side_to_move=node.state.side_to_move,
                    ply=node.state.ply,
                    repetition=node.state.repetition,
                )
                node.state = leaf_state
                leaves_data.append((leaf_state, py_moves, path))

            # ── Phase 2: Batch NN evaluation ──
            if leaves_data:
                if hasattr(self.model, 'predict_batch') and len(leaves_data) > 1:
                    results = self.model.predict_batch(
                        [(ld[0], ld[1]) for ld in leaves_data]
                    )
                else:
                    # Fallback: serial predict (for TorchPolicyValueModel / K=1)
                    results = [
                        self.model.predict(ld[0], ld[1])
                        for ld in leaves_data
                    ]

                # ── Phase 3: Remove VL, expand, backup ──
                for (leaf_state, py_moves, path), (policy, value) in zip(leaves_data, results):
                    leaf_node = path[-1]

                    # Remove virtual loss from entire path
                    for n in path:
                        n.virtual_loss -= 1

                    # Expand leaf (guard against duplicate expansion from collision)
                    if not leaf_node.is_expanded():
                        priors = {m: policy.get(m, 0.0) for m in py_moves}
                        self._expand_cpp(leaf_node, priors)

                    self._backup(path, value)
                    sims_done += 1

        # Safety: DFS verify zero virtual loss leakage
        self._assert_no_vl_leak(root)

        return root

    def _assert_no_vl_leak(self, root: Node) -> None:
        """DFS the entire tree asserting all virtual_loss == 0."""
        stack = [root]
        while stack:
            node = stack.pop()
            assert node.virtual_loss == 0, (
                f"VL leak! node.virtual_loss={node.virtual_loss}, "
                f"N={node.N}, children={len(node.children)}"
            )
            for ch in node.children.values():
                stack.append(ch)

    def _expand_cpp(self, node: Node, priors: Dict[Move, float]) -> None:
        """Expand node using C++ apply_move for child boards."""
        cpp = self._cpp
        module = cpp.module

        parent_cpp_board = node.cpp_board
        parent_side = node.state.side_to_move
        child_side_py = parent_side.opponent()
        child_cpp_side = cpp.PY_TO_CPP_SIDE[child_side_py]

        for mv, p in priors.items():
            cpp_mv = cpp.py_to_cpp_move(mv)
            child_cpp_board = module.apply_move(parent_cpp_board, cpp_mv)

            # Lightweight child state: board is None (deferred sync)
            # We only need side_to_move, ply, and repetition for terminal_info
            child_state = GameState(
                board=None,  # deferred — synced only if this node becomes a leaf
                side_to_move=child_side_py,
                ply=node.state.ply + 1,
                repetition=node.state.repetition,
            )
            node.children[mv] = Node(
                state=child_state,
                prior=float(p),
                parent=node,
                cpp_board=child_cpp_board,
                cpp_side=child_cpp_side,
            )
    # Shared methods (used by both Python and C++ paths)

    def select_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """Return the most-visited move after MCTS."""
        root = self._run_mcts_search(state, legal_moves)
        best_mv = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_mv

    def run_mcts(self, state: GameState, legal_moves: List[Move],
                 add_noise: bool = True) -> Tuple[Dict[Move, float], float]:
        """Run MCTS, return (pi_dict, root_value). pi_dict is the visit-count distribution."""
        root = self._run_mcts_search(state, legal_moves, add_noise=add_noise)

        total_visits = sum(ch.N for ch in root.children.values())
        if total_visits == 0:
            n = len(legal_moves)
            pi_dict = {mv: 1.0 / n for mv in legal_moves} if n > 0 else {}
        else:
            pi_dict = {mv: ch.N / total_visits for mv, ch in root.children.items()}

        root_value = root.Q
        return pi_dict, root_value

    def select_move_with_pi(
        self, state: GameState, legal_moves: List[Move],
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> Tuple[Move, Dict[Move, float], float]:
        """Run MCTS, sample a move by temperature, return (chosen, pi_dict, root_value).

        temperature > 0: sample proportional to N^(1/T).
        temperature ≈ 0: argmax (most-visited move).
        """
        root = self._run_mcts_search(state, legal_moves, add_noise=add_noise)

        moves = list(root.children.keys())
        visits = [root.children[mv].N for mv in moves]
        total_visits = sum(visits)

        if total_visits == 0:
            pi_dict = {mv: 1.0 / len(moves) for mv in moves}
        else:
            pi_dict = {mv: v / total_visits for mv, v in zip(moves, visits)}

        if temperature < 1e-8:
            chosen = moves[max(range(len(visits)), key=lambda i: visits[i])]
        else:
            adjusted = [v ** (1.0 / temperature) for v in visits]
            s = sum(adjusted)
            if s < 1e-12:
                chosen = self.rng.choice(moves)
            else:
                probs = [a / s for a in adjusted]
                chosen = self.rng.choices(moves, weights=probs, k=1)[0]

        return chosen, pi_dict, root.Q

    def _expand(self, node: Node, priors: Dict[Move, float]) -> None:
        for mv, p in priors.items():
            nb = apply_move(node.state.board, mv)
            child_state = GameState(board=nb, side_to_move=node.state.side_to_move.opponent(), ply=node.state.ply+1, repetition=node.state.repetition)
            node.children[mv] = Node(state=child_state, prior=float(p), parent=node)

    def _select_child(self, node: Node) -> Tuple[Move, Node]:
        """PUCT selection with virtual loss support.

        Uses -effective_Q (opponent's value flipped) where effective_Q accounts
        for in-flight virtual losses to divert parallel selections.
        """
        best_score = -1e18
        best = None
        total_N = sum(ch.N + ch.virtual_loss for ch in node.children.values()) + 1
        c_puct = self.cfg.c_puct
        for mv, ch in node.children.items():
            effective_N = ch.N + ch.virtual_loss
            if effective_N > 0:
                effective_W = ch.W - ch.virtual_loss  # -1 penalty per VL
                Q = effective_W / effective_N
            else:
                Q = 0.0
            U = c_puct * ch.prior * math.sqrt(total_N) / (1 + effective_N)
            score = (-Q) + U
            if score > best_score:
                best_score = score
                best = (mv, ch)
        assert best is not None
        return best

    def _backup(self, path: List[Node], value: float) -> None:
        """Propagate value up the tree, flipping sign and applying
        discount γ at each level so shorter wins are strictly preferred.

        With γ=0.99 a mate-in-3 returns ≈0.97 while mate-in-15 returns ≈0.86,
        giving the agent a strong "sense of urgency" that breaks king-chase loops.
        """
        gamma = self.cfg.discount_factor
        v = value
        for node in reversed(path):
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -(v * gamma)

    def _add_dirichlet_noise(self, priors: Dict[Move, float]) -> None:
        alpha = self.cfg.dirichlet_alpha
        eps = self.cfg.dirichlet_eps
        if eps <= 0:
            return
        moves = list(priors.keys())
        if not moves:
            return
        noise = [self.rng.gammavariate(alpha, 1.0) for _ in moves]
        s = sum(noise)
        noise = [n / s for n in noise]
        for mv, n in zip(moves, noise):
            priors[mv] = (1 - eps) * priors[mv] + eps * n
# TorchPolicyValueModel

import torch

from hybrid.rl.az_encoding import encode_state, extract_policy_logits
from hybrid.rl.az_network import PolicyValueNet


class TorchPolicyValueModel(PolicyValueModel):
    """Wraps a PolicyValueNet for MCTS consumption.

    Encodes state → forward pass → extracts legal-move logits → softmax → policy dict.
    """

    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.net.eval()

    def predict(
        self, state: GameState, legal_moves: List[Move]
    ) -> Tuple[Dict[Move, float], float]:
        """Return (policy_dict, value) for the given state and legal moves."""
        if len(legal_moves) == 0:
            return {}, 0.0

        with torch.no_grad():
            x = encode_state(state).unsqueeze(0).to(self.device)  # (1, C, 10, 9)
            policy_planes, value_tensor = self.net(x)
            policy_planes = policy_planes.squeeze(0)  # (92, 10, 9)
            value = value_tensor.item()

            logits = extract_policy_logits(policy_planes, legal_moves)
            logits = logits - logits.max()  # numerical stability
            probs = torch.softmax(logits, dim=0)

            policy_dict = {mv: probs[i].item() for i, mv in enumerate(legal_moves)}

        return policy_dict, value

    def predict_batch(
        self, inputs: List[Tuple[GameState, List[Move]]]
    ) -> List[Tuple[Dict[Move, float], float]]:
        """Batch prediction: K leaf states → 1 GPU forward pass → K results."""
        if not inputs:
            return []

        with torch.no_grad():
            batch = torch.stack(
                [encode_state(s) for s, _ in inputs]
            ).to(self.device)                              # (K, C, 10, 9)
            policy_batch, value_batch = self.net(batch)    # (K, 92, 10, 9), (K, 1)

            results: List[Tuple[Dict[Move, float], float]] = []
            for idx, (state, legal_moves) in enumerate(inputs):
                if not legal_moves:
                    results.append(({}, 0.0))
                    continue
                pp = policy_batch[idx]                     # (92, 10, 9)
                value = value_batch[idx].item()

                logits = extract_policy_logits(pp, legal_moves)
                logits = logits - logits.max()
                probs = torch.softmax(logits, dim=0)

                policy_dict = {mv: probs[i].item() for i, mv in enumerate(legal_moves)}
                results.append((policy_dict, value))

        return results
