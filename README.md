# Hybrid Chess

AlphaZero for asymmetric board games (International Chess vs Chinese Chess on a shared 9×10 board).

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Hybrid Chess places International Chess pieces against Xiangqi (Chinese Chess) pieces on a 9×10 board. Each side follows its own movement rules, creating an asymmetric game with no existing datasets or opening theory. Agents must learn entirely through self-play.

```
Xiangqi    10  c  h  e  a  g  a  e  h  c      Chariot, Horse, Elephant, Advisor, General
            9  .  .  .  .  .  .  .  .  .
            8  .  n  .  .  .  .  .  n  .      Cannons
            7  s  .  s  .  s  .  s  .  s      Soldiers
            6  ─  ─  ─  ─  ─  ─  ─  ─  ─      River
            5  .  .  .  .  .  .  .  .  .
            4  .  .  .  .  .  .  .  .  .
            3  .  .  .  .  .  .  .  .  .
            2  P  P  P  P  P  P  P  P  P      Pawns
Chess       1  R  N  B  Q  K  B  N  R  .      Rook, Knight, Bishop, Queen, King
               a  b  c  d  e  f  g  h  i
```

See [RULES.md](RULES.md) for the complete rule specification.

## Setup

```bash
git clone https://github.com/lessen-xu/Hybrid-Chess.git
cd Hybrid-Chess
pip install -e .

# Optional: build C++ engine for faster move generation
pip install pybind11 && bash cpp/build.sh   # macOS/Linux
pip install pybind11 && .\cpp\build.ps1     # Windows
```

## Usage

### Play

```bash
python -m hybrid server
# Opens http://localhost:8000
```

### Train

```bash
# Quick test on CPU
python -m hybrid train --iterations 5 --games 20 --simulations 50

# Full training on GPU
python -m hybrid train \
    --iterations 100 --games 200 --simulations 200 \
    --device cuda --workers 8 --use-cpp \
    --output runs/my_experiment
```

### Evaluate

```bash
python -m hybrid eval --model runs/my_experiment/best_model.pt --vs ab_d2 --games 50
```

## API Examples

### Gymnasium

```python
import gymnasium as gym
import hybrid.gym_env  # registers HybridChess-v0

env = gym.make("HybridChess-v0")
obs, info = env.reset()

action = info["legal_actions"][0]
obs, reward, terminated, truncated, info = env.step(action)
```

### Custom Variants

```python
from hybrid.core.config import VariantConfig
from hybrid.core.env import HybridChessEnv

variant = VariantConfig(no_queen=True, extra_cannon=True, flying_general=False)
env = HybridChessEnv(variant=variant)
state = env.reset()
```

### Custom Network

```python
from hybrid.rl.az_network import BaseModel
import torch, torch.nn as nn

class MyTransformerNet(BaseModel):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x):
        # x: (B, 14, 10, 9) encoded state
        # Return: (policy_logits: (B,92,10,9), value: (B,1))
        ...
```

### Custom Agent

```python
from hybrid.agents.base import Agent
from hybrid.core.env import GameState
from hybrid.core.types import Move

class MyAgent(Agent):
    name = "my_agent"

    def select_move(self, state: GameState, legal_moves: list[Move]) -> Move:
        return legal_moves[0]  # your logic here
```

### FEN Loading

```python
env = HybridChessEnv()
state = env.reset_from_fen("cheagaehc/9/1n5n1/s1s1s1s1s/9/9/9/9/PPPPPPPPP/RNBQKBNR1 c")
print(f"{len(env.legal_moves())} legal moves")
```

## Project Structure

```
hybrid-chess/
├── hybrid/                     # Python package
│   ├── core/                   # Game engine
│   │   ├── types.py            #   Side, PieceKind, Move, Piece
│   │   ├── board.py            #   Board representation (9x10)
│   │   ├── rules.py            #   Move generation, terminal detection
│   │   ├── config.py           #   VariantConfig, constants
│   │   ├── env.py              #   HybridChessEnv
│   │   ├── fen.py              #   FEN parser/serializer
│   │   └── render.py           #   ASCII renderer
│   ├── agents/                 # AI agents
│   │   ├── base.py             #   Agent ABC
│   │   ├── random_agent.py     #   Random baseline
│   │   ├── greedy_agent.py     #   1-ply capture maximizer
│   │   ├── alphabeta_agent.py  #   Negamax with alpha-beta pruning
│   │   └── alphazero_stub.py   #   MCTS + policy/value network
│   ├── rl/                     # Training pipeline
│   │   ├── az_network.py       #   BaseModel ABC, PolicyValueNet
│   │   ├── az_encoding.py      #   State/action encoding
│   │   ├── az_selfplay.py      #   Self-play with reward shaping
│   │   ├── az_train.py         #   Training loop
│   │   ├── az_eval.py          #   Evaluation and gating
│   │   ├── az_runner.py        #   Iterative runner
│   │   ├── az_inference_server.py  # GPU batch inference
│   │   ├── az_shm_pool.py     #   Shared memory IPC
│   │   └── ...
│   ├── server.py               # HTTP game server
│   ├── gym_env.py              # Gymnasium wrapper
│   └── __main__.py             # CLI entry point
├── cpp/                        # C++ engine (pybind11)
├── ui/                         # Browser frontend
├── docs/                       # Architecture docs
├── tests/                      # Test suite
└── runs/                       # Experiment outputs (gitignored)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Performance

| Optimization | Description |
|---|---|
| C++ engine | Board, move generation, terminal detection, alpha-beta in C++ via pybind11 |
| GPU batch inference | Centralized server batches worker requests into one forward pass |
| Shared memory IPC | Workers and server communicate via `SharedMemoryPool`, no serialization |
| MCTS leaf batching | Gathers up to K leaves per round with virtual loss, single batch eval |
| FP16 / TF32 AMP | Mixed precision on Ampere+ GPUs |
| Parallel self-play | Multi-process workers with independent MCTS trees |

## Balance and Ablation

Chess is naturally favored in this asymmetric setup due to the Queen's power and unrestricted piece movement. The project provides `VariantConfig` flags to explore rebalancing:

**Weaken Chess**

| Flag | Effect |
|---|---|
| `no_queen` | Remove the Queen |
| `no_bishop` | Remove one Bishop |
| `one_rook` | Remove one Rook |
| `no_queen_promo` | Pawns promote to R/B/N only |
| `remove_extra_pawn` | Remove the 9th Pawn |

**Buff Xiangqi**

| Flag | Effect |
|---|---|
| `extra_cannon` | Add a 3rd Cannon |
| `extra_soldier` | Add a 6th Soldier |
| `flying_general` (default on) | General can capture King across the board |

```python
balanced = VariantConfig(no_queen=True, extra_cannon=True)
env = HybridChessEnv(variant=balanced)

# Or via CLI:
# python -m hybrid train --ablation extra_cannon,no_bishop
```

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

## License

MIT
