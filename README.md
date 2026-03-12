<div align="center">

# ♔ Hybrid Chess 將

**An extensible RL framework for asymmetric chess variants**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-compatible-0081a7)](https://gymnasium.farama.org)

♖ ♘ ♗ ♕ ♔ &nbsp;&nbsp;⚔️&nbsp;&nbsp; 將 車 馬 象 砲

*Define your own rules, train AlphaZero agents, and experiment — all in a few lines of Python.*

</div>

---

### ✨ Features

🧩 **Extensible**: Composable `VariantConfig` API — mix & match rule tweaks without touching source code

🎮 **Play**: Browser UI, play as Chess or Xiangqi against AI (Random / Greedy / AlphaBeta)

🧠 **Train**: Full AlphaZero pipeline, MCTS + ResNet, self-play, gating, curriculum learning

🔌 **Pluggable**: `BaseModel` ABC for custom networks (Transformer, MobileNet, ...); reward shaping hooks

📊 **Track**: WandB / TensorBoard integration (lazy import, zero config)

⚡ **Fast**: C++ engine, GPU batch inference, shared-memory IPC, FP16 AMP

🏋️ **Gymnasium**: `gym.make("HybridChess-v0")`, drop into any RL framework

📝 **FEN Support**: Load/save arbitrary positions via FEN notation

---

## Installation

```bash
# Clone
git clone https://github.com/lessen-xu/Hybrid-Chess.git
cd Hybrid-Chess

# Install (editable, core only)
pip install -e .

# With dev tools (pytest, matplotlib, pandas)
pip install -e ".[dev]"

# With Gymnasium support
pip install -e ".[gym]"
```

### C++ Engine (Recommended)

The pure-Python engine can run simple games, but is **significantly slower** in practice.  
For any serious training or AlphaBeta search beyond depth 2, the C++ engine (~50× faster) is strongly recommended:

```bash
# Requires: pip install pybind11
# Windows (PowerShell, needs MSYS2 g++)
.\cpp\build.ps1
# macOS / Linux (auto-detects g++ or clang++)
bash cpp/build.sh
```

### Built-in Performance Optimizations

The training pipeline includes multiple layers of acceleration:

| Optimization | Description |
|-------------|-------------|
| **C++ engine** | Board, legal-move gen, terminal detection, full α-β search in C++ via pybind11 |
| **GPU batch inference** | Centralized inference server batches N workers' requests into one forward pass |
| **Zero-copy shared memory** | Workers ↔ server communicate via `SharedMemoryPool`, no pickle, only 8-byte signals through Queue |
| **MCTS leaf batching** | Gathers up to K leaves per MCTS round with virtual loss, then evaluates in a single batch |
| **FP16 / TF32 AMP** | Automatic mixed precision on Ampere+ GPUs (TF32 matmul + FP16 autocast) |
| **GPU state encoding** | `encode_batch_gpu()`, vectorized scatter-based one-hot encoding on GPU, zero-allocation hot path |
| **Parallel self-play** | Multi-process `spawn` workers with independent MCTS trees |
| **Parallel evaluation** | Gating and eval matches distributed across CPU workers |
| **Pinned memory DMA** | Pre-allocated pinned CPU buffers for async GPU transfer |
| **Static batching** | Fixed batch-size forward pass eliminates CUDA graph recompilation |

---

## Quick Start

### Play in the Browser

```bash
python -m hybrid server
# Opens http://localhost:8000, choose your side and play!
```

### Train AlphaZero

```bash
# Basic training (CPU, 10 iterations)
python -m hybrid train --iterations 10 --games 50 --simulations 50

# Full training (GPU, parallel, C++ engine)
python -m hybrid train \
    --iterations 100 --games 200 --simulations 200 \
    --device cuda --workers 8 --use-cpp \
    --output runs/my_experiment
```

### Evaluate Agents

```bash
# AlphaZero checkpoint vs AlphaBeta depth 2
python -m hybrid eval --model runs/my_experiment/best_model.pt --vs ab_d2 --games 50

# AlphaBeta vs Random
python -m hybrid eval --vs random --games 100
```

### Use as a Gymnasium Env

```python
import gymnasium as gym
import hybrid.gym_env  # registers HybridChess-v0

env = gym.make("HybridChess-v0")
obs, info = env.reset()

# info["legal_actions"] = list of valid action indices
action = info["legal_actions"][0]
obs, reward, terminated, truncated, info = env.step(action)
```

### Define a Custom Variant (new!)

```python
from hybrid.core.config import VariantConfig
from hybrid.core.env import HybridChessEnv

# No queen, extra cannon, disable flying general
my_rules = VariantConfig(no_queen=True, extra_cannon=True, flying_general=False)
env = HybridChessEnv(variant=my_rules)
state = env.reset()
print(env.legal_moves())  # rules applied automatically
```

### Load a Custom Position via FEN (new!)

```python
env = HybridChessEnv()
state = env.reset_from_fen("cheagaehc/9/1n5n1/s1s1s1s1s/9/9/9/9/PPPPPPPPP/RNBQKBNR1 c")
print(f"{len(env.legal_moves())} legal moves")
```

### Custom Network Architecture (new!)

```python
from hybrid.rl.az_network import BaseModel
import torch, torch.nn as nn

class MyTinyNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(14, 92, 1)
        self.value = nn.Linear(92 * 10 * 9, 1)

    def forward(self, x):
        p = self.conv(x)
        v = torch.tanh(self.value(p.flatten(1)))
        return p, v.unsqueeze(-1)
```

### Write a Custom Agent

```python
from hybrid.agents.base import Agent
from hybrid.core.env import GameState
from hybrid.core.types import Move

class MyAgent(Agent):
    name = "my_agent"

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move:
        # Your logic here: MCTS, neural net, heuristic, ...
        return legal_moves[0]
```

---

## Architecture

```
hybrid-chess/
├── hybrid/                  # Core Python package
│   ├── core/                # Game engine
│   │   ├── types.py         #   Side, PieceKind, Move, Piece
│   │   ├── board.py         #   Board representation
│   │   ├── rules.py         #   Legal move generation, terminal detection
│   │   ├── config.py        #   VariantConfig + legacy flags
│   │   ├── env.py           #   HybridChessEnv (gym-like API)
│   │   ├── fen.py           #   FEN parser/serializer
│   │   └── render.py        #   ASCII board renderer
│   ├── agents/              # AI players
│   │   ├── base.py          #   Agent ABC
│   │   ├── random_agent.py  #   Random baseline
│   │   ├── greedy_agent.py  #   1-ply capture maximizer
│   │   ├── alphabeta_agent.py  # Negamax α-β with hand-crafted eval
│   │   ├── alphazero_stub.py   # MCTS + policy/value network
│   │   └── eval.py          #   Hand-crafted evaluation function
│   ├── rl/                  # AlphaZero training pipeline
│   │   ├── az_network.py    #   BaseModel ABC + PolicyValueNet
│   │   ├── az_encoding.py   #   State/action encoding (92 move planes)
│   │   ├── az_selfplay.py   #   Self-play + reward shaping hook
│   │   ├── az_train.py      #   Training loop (policy CE + value MSE)
│   │   ├── az_eval.py       #   Match evaluation
│   │   ├── az_runner.py     #   Iterative runner + WandB/TB logging
│   │   └── ...              #   Inference server, parallel workers, replay buffer
│   ├── server.py            # Zero-dep HTTP game server
│   ├── gym_env.py           # Gymnasium wrapper
│   └── __main__.py          # CLI entry point
├── cpp/                     # C++ engine (pybind11)
│   └── src/                 #   board, rules, ab_search, bindings
├── ui/                      # Web UI
│   ├── index.html           #   Landing page
│   ├── play/                #   Interactive play (human vs AI)
│   ├── replay/              #   Game replay viewer
│   └── shared/              #   Shared board renderer & CSS
├── scripts/                 # CLI tools
│   ├── train_az_iter.py     #   AZ training launcher
│   └── eval_arena.py        #   Side-switching evaluation
├── examples/                # Usage examples
│   └── custom_variant.py    #   Creating & training a custom variant
├── tests/                   # pytest suite
├── RULES.md                 # Game rules & piece reference
└── runs/                    # Experiment outputs (gitignored)
```

---

## Rule Variants & Balance

The asymmetric design creates a natural balance challenge.  
Built-in rule variants for fairness tuning:

| Variant | Effect | Balance impact |
|---------|--------|----------------|
| `none` (standard) | Full Chess army vs full Xiangqi army | Chess favored (Queen is dominant) |
| `extra_cannon` | Xiangqi gets a 3rd Cannon at center (4,7) | More balanced |
| `no_queen` | Chess loses the Queen | Xiangqi favored |
| `no_bishop` | Chess loses the left Bishop (c1) | Slight Chess nerf |
| `extra_soldier` | Xiangqi gets a 6th Soldier at center (4,5) | Slight Xiangqi buff |
| `one_rook` | Chess loses the right Rook (h1) | Significant Chess nerf |
| `no_flying_general` | Disables Flying-General capture rule | Weakens Xiangqi General |
| `remove_pawn` | Removes the extra 9th-file Chess Pawn | Slight Chess nerf |
| `no_queen_promo` | Pawns cannot promote to Queen (only R/B/N) | Reduces late-game Chess power |

Variants can be **combined** with commas: `--ablation extra_cannon,no_bishop`

Applied via CLI: `--ablation extra_cannon` or in the UI's "Rule Variant" dropdown.

**Programmatic API** (preferred for new code):

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
                          [--ablation none|extra_cannon|no_queen|no_bishop|
                                      extra_soldier|one_rook|no_flying_general|
                                      remove_pawn|no_queen_promo]
                          [--lr 1e-3] [--batch-size 256] [--output DIR]
                          [--res-blocks 3] [--channels 64]
                          [--use-inference-server] [--inference-batch-size 32]
                          [--eval-games 20] [--eval-simulations 200]
                          [--gating-simulations 20]
                          [--curriculum none|3phase|3phase_v2]
                          [--endgame-ratio 0.0]
                          [--buffer-capacity 50000] [--train-epochs 1]
                          [--seed 0]
python -m hybrid eval     [--model PATH] [--vs random|ab_d1|ab_d2|ab_d4]
                          [--games N] [--simulations N] [--device auto]
```

---

## Contributing

Pull requests welcome! Some ideas:

- 🧠 New network architectures (subclass `BaseModel`)
- ⚖️ Novel rule variants via `VariantConfig`
- 📊 Reward shaping experiments
- 🌐 Translations of game rules

---

## License

MIT
