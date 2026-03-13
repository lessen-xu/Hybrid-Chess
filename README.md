<div align="center">

# ♔ Hybrid Chess 將

### A Production-Grade AlphaZero Implementation for Asymmetric Board Games

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lessen-xu/Hybrid-Chess/blob/main/notebooks/01_game_rules_and_env.ipynb)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-0081a7)](https://gymnasium.farama.org)

**Learn AlphaZero from a real-world codebase. Train your own agents. Play in the browser.**

[Get Started](#quick-start) · [Tutorials](notebooks/) · [Architecture](docs/ARCHITECTURE.md) · [Game Rules](RULES.md)

</div>

---

## Why This Project?

Most AlphaZero implementations on GitHub are either **too simplified** (tic-tac-toe, Connect4) or **too complex to read** (Leela Chess Zero). This project sits in the sweet spot:

| | Toy Implementations | **This Project** | Production Engines |
|---|---|---|---|
| **Game Complexity** | Trivial | ✅ Real-world (≈8,280 action space) | Very high |
| **Code Readability** | Good | ✅ Clean, documented | Hard to follow |
| **Full Pipeline** | Usually missing | ✅ Self-play → Train → Gate → Eval | Yes |
| **Performance** | Pure Python | ✅ C++ engine, GPU batch, shared memory | Heavily optimized |
| **Hackability** | Easy | ✅ Pluggable: network, rewards, rules | Difficult |

### What You Get

🧠 **Complete AlphaZero pipeline** — MCTS + dual-head ResNet, self-play, gating, curriculum learning

⚡ **Production-grade performance** — C++ engine (~50× speedup), GPU batch inference, shared-memory IPC, FP16 AMP

🧩 **Fully extensible** — Swap network architectures (`BaseModel` ABC), customize rules (`VariantConfig`), shape rewards

🎮 **Play in the browser** — Zero-dependency HTTP server, human vs AI, game replay viewer

🏋️ **Gymnasium compatible** — `gym.make("HybridChess-v0")`, plug into any RL framework

📊 **Experiment tracking** — WandB / TensorBoard integration (lazy import, zero config)

---

## The Game

**Hybrid Chess** pits International Chess pieces against Chinese Chess (Xiangqi) pieces on a shared 9×10 board — creating a fascinating asymmetric game where each side retains its native piece movement rules.

```
Xiangqi ──  10  c  h  e  a  g  a  e  h  c     ← Chariot, Horse, Elephant, Advisor, General
             9  .  .  .  .  .  .  .  .  .
             8  .  n  .  .  .  .  .  n  .     ← Cannons (jump to capture!)
             7  s  .  s  .  s  .  s  .  s     ← Soldiers
             6  ─  ─  ─  ─  ─  ─  ─  ─  ─     ← River
             5  .  .  .  .  .  .  .  .  .
             4  .  .  .  .  .  .  .  .  .
             3  .  .  .  .  .  .  .  .  .
             2  P  P  P  P  P  P  P  P  P     ← Pawns (promote at top)
 Chess  ──   1  R  N  B  Q  K  B  N  R  .     ← Rook, Knight, Bishop, Queen, King
                a  b  c  d  e  f  g  h  i
```

> **Why this game?** The asymmetry makes it a challenging and **novel** test bed for RL — no existing datasets, no tablebases, no openings books. The AI must learn everything from scratch through self-play.

📖 [Full game rules →](RULES.md)

---

## Quick Start

### Installation

```bash
git clone https://github.com/lessen-xu/Hybrid-Chess.git
cd Hybrid-Chess
pip install -e .

# Optional: C++ engine for ~50× speedup (recommended for training)
pip install pybind11 && bash cpp/build.sh   # macOS/Linux
pip install pybind11 && .\cpp\build.ps1     # Windows (MSYS2)
```

### ▶️ Play in the Browser

```bash
python -m hybrid server
# Opens http://localhost:8000 — choose your side and play!
```

### 🧠 Train AlphaZero

```bash
# Quick test (CPU, ~5 min)
python -m hybrid train --iterations 5 --games 20 --simulations 50

# Full training (GPU, parallel self-play)
python -m hybrid train \
    --iterations 100 --games 200 --simulations 200 \
    --device cuda --workers 8 --use-cpp \
    --output runs/my_experiment
```

### 📊 Evaluate

```bash
# AlphaZero checkpoint vs AlphaBeta
python -m hybrid eval --model runs/my_experiment/best_model.pt --vs ab_d2 --games 50
```

---

## 📓 Tutorials

The best way to learn is through our interactive notebooks:

| # | Notebook | What You'll Learn |
|---|----------|-------------------|
| 01 | [**Game Rules & Environment**](notebooks/01_game_rules_and_env.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lessen-xu/Hybrid-Chess/blob/main/notebooks/01_game_rules_and_env.ipynb) | Create environments, make moves, visualize boards, custom variants |
| 02 | AI Agents & Search *(coming soon)* | AlphaBeta search, agent comparison, evaluation functions |
| 03 | MCTS Explained *(coming soon)* | Build MCTS from scratch, visualize the search tree |
| 04 | AlphaZero Training *(coming soon)* | Full training pipeline walkthrough |
| 05 | Experiments *(coming soon)* | Reward shaping, custom architectures, variant ablations |

---

## Use as a Library

### Gymnasium API

```python
import gymnasium as gym
import hybrid.gym_env  # registers HybridChess-v0

env = gym.make("HybridChess-v0")
obs, info = env.reset()

action = info["legal_actions"][0]  # pick a legal action
obs, reward, terminated, truncated, info = env.step(action)
```

### Custom Rule Variants

```python
from hybrid.core.config import VariantConfig
from hybrid.core.env import HybridChessEnv

# No queen, extra cannon, disable flying general
variant = VariantConfig(no_queen=True, extra_cannon=True, flying_general=False)
env = HybridChessEnv(variant=variant)
state = env.reset()
```

### Custom Network Architecture

```python
from hybrid.rl.az_network import BaseModel
import torch, torch.nn as nn

class MyTransformerNet(BaseModel):
    def __init__(self):
        super().__init__()
        # Your architecture here
        ...

    def forward(self, x):
        # x: (B, 14, 10, 9) encoded state
        # Return: (policy_logits: (B,92,10,9), value: (B,1))
        ...
```

### Load Custom Positions via FEN

```python
env = HybridChessEnv()
state = env.reset_from_fen("cheagaehc/9/1n5n1/s1s1s1s1s/9/9/9/9/PPPPPPPPP/RNBQKBNR1 c")
print(f"{len(env.legal_moves())} legal moves")
```

### Write a Custom Agent

```python
from hybrid.agents.base import Agent
from hybrid.core.env import GameState
from hybrid.core.types import Move

class MyAgent(Agent):
    name = "my_agent"

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move:
        return legal_moves[0]  # your logic here
```

---

## Architecture

For detailed architecture documentation with Mermaid diagrams, see [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md).

```
hybrid-chess/
├── hybrid/                  # Core Python package
│   ├── core/                # Game engine
│   │   ├── types.py         #   Side, PieceKind, Move, Piece
│   │   ├── board.py         #   Board representation (9×10 grid)
│   │   ├── rules.py         #   Legal move generation, terminal detection
│   │   ├── config.py        #   VariantConfig + game constants
│   │   ├── env.py           #   HybridChessEnv (gym-like API)
│   │   ├── fen.py           #   FEN parser/serializer
│   │   └── render.py        #   ASCII board renderer
│   ├── agents/              # AI players
│   │   ├── base.py          #   Agent ABC
│   │   ├── random_agent.py  #   Random baseline
│   │   ├── greedy_agent.py  #   1-ply capture maximizer
│   │   ├── alphabeta_agent.py  # Negamax α-β with hand-crafted eval
│   │   └── alphazero_stub.py   # MCTS + policy/value network
│   ├── rl/                  # AlphaZero training pipeline
│   │   ├── az_network.py    #   BaseModel ABC + PolicyValueNet (ResNet)
│   │   ├── az_encoding.py   #   State/action encoding (14ch → 92 planes)
│   │   ├── az_selfplay.py   #   Self-play + reward shaping hook
│   │   ├── az_train.py      #   Training loop (policy CE + value MSE)
│   │   ├── az_eval.py       #   Match evaluation (Wilson CI gating)
│   │   ├── az_runner.py     #   Iterative runner + WandB/TB logging
│   │   ├── az_inference_server.py  # GPU batch inference server
│   │   ├── az_shm_pool.py   #   Zero-copy shared memory IPC
│   │   └── ...              #   Parallel workers, replay buffer, endgame spawner
│   ├── server.py            # Zero-dep HTTP game server
│   ├── gym_env.py           # Gymnasium wrapper (HybridChess-v0)
│   └── __main__.py          # CLI entry point
├── cpp/                     # C++ engine (pybind11, ~50× faster)
├── ui/                      # Browser UI (play, replay, shared renderer)
├── notebooks/               # Interactive tutorials
├── docs/                    # Architecture documentation
├── tests/                   # pytest suite
└── runs/                    # Experiment outputs (gitignored)
```

---

## Performance

The training pipeline includes multiple layers of acceleration:

| Optimization | Description |
|-------------|-------------|
| **C++ engine** | Board, legal-move gen, terminal detection, full α-β search in C++ via pybind11 |
| **GPU batch inference** | Centralized inference server batches N workers' requests into one forward pass |
| **Zero-copy shared memory** | Workers ↔ server communicate via `SharedMemoryPool`, no pickle, only 8-byte signals |
| **MCTS leaf batching** | Gathers up to K leaves per MCTS round with virtual loss, evaluates in a single batch |
| **FP16 / TF32 AMP** | Automatic mixed precision on Ampere+ GPUs (TF32 matmul + FP16 autocast) |
| **GPU state encoding** | Vectorized scatter-based one-hot encoding on GPU, zero-allocation hot path |
| **Parallel self-play** | Multi-process `spawn` workers with independent MCTS trees |
| **Pinned memory DMA** | Pre-allocated pinned CPU buffers for async GPU transfer |

---

## Rule Variants

The asymmetric design creates a natural balance challenge. Built-in variants for fairness tuning:

| Variant | Effect | Balance Impact |
|---------|--------|----------------|
| `none` (standard) | Full Chess army vs full Xiangqi army | Chess favored (Queen dominant) |
| `extra_cannon` | Xiangqi gets a 3rd Cannon | More balanced |
| `no_queen` | Chess loses Queen | Xiangqi favored |
| `no_bishop` | Chess loses left Bishop | Slight Chess nerf |
| `extra_soldier` | Xiangqi gets 6th Soldier | Slight Xiangqi buff |
| `one_rook` | Chess loses right Rook | Significant Chess nerf |
| `no_flying_general` | Disables Flying General rule | Weakens Xiangqi General |
| `no_queen_promo` | Pawns can't promote to Queen | Reduces late-game Chess power |

Variants can be **combined**: `--ablation extra_cannon,no_bishop`

```python
from hybrid.core.config import VariantConfig
env = HybridChessEnv(variant=VariantConfig(extra_cannon=True, no_bishop=True))
```

---

## CLI Reference

```
python -m hybrid server   [--port 8000] [--host 127.0.0.1] [--no-browser]
python -m hybrid train    [--iterations N] [--games N] [--simulations N]
                          [--device auto|cpu|cuda] [--workers N] [--use-cpp]
                          [--ablation none|extra_cannon|no_queen|...]
                          [--lr 1e-3] [--batch-size 256] [--output DIR]
                          [--res-blocks 3] [--channels 64]
                          [--use-inference-server] [--inference-batch-size 32]
                          [--curriculum none|3phase|3phase_v2]
                          [--use-wandb] [--use-tensorboard]
python -m hybrid eval     [--model PATH] [--vs random|ab_d1|ab_d2|ab_d4]
                          [--games N] [--simulations N] [--device auto]
```

---

## Contributing

Pull requests welcome! Some ideas:

- 🧠 **New network architectures** — subclass `BaseModel` (Transformer? MobileNet?)
- ⚖️ **Novel rule variants** — extend `VariantConfig`
- 📊 **Reward shaping experiments** — use the `reward_shaper` hook
- 📓 **Tutorial notebooks** — help others learn AlphaZero
- 🌐 **Translations** of game rules and documentation

---

## License

MIT
