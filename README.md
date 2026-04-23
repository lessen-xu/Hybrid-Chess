# Hybrid Chess

AlphaZero for asymmetric board games: International Chess pieces vs. Xiangqi (Chinese Chess) pieces on a shared 9×10 board.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Hybrid Chess places International Chess pieces against Xiangqi pieces on a 9×10 board. Each side follows its own movement rules, creating an asymmetric two-player zero-sum game with no existing datasets or opening theory. Agents must learn entirely through self-play.

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

### Train

```bash
# Quick test (CPU)
python -m hybrid train --iterations 5 --games 20 --simulations 50

# Full training with rule variants (GPU)
python scripts/train_az_iter.py \
    --iterations 50 --selfplay-games-per-iter 100 --simulations 50 \
    --ablation "chess_palace,knight_block,xq_queen" \
    --use-cpp --num-workers 4 \
    --outdir runs/my_experiment
```

### Evaluate

```bash
python -m hybrid eval --model runs/my_experiment/best_model.pt --vs ab_d2 --games 50
```

### Rule Variants

The project supports rule variants via `VariantConfig` to study game balance:

```python
from hybrid.core.config import VariantConfig
from hybrid.core.env import HybridChessEnv

# Default rules
env = HybridChessEnv()

# Structural reform: palace + knight blocking
env = HybridChessEnv(variant=VariantConfig(chess_palace=True, knight_block=True))

# Best balance: palace + knight block + give XQ a Queen
env = HybridChessEnv(variant=VariantConfig(
    chess_palace=True, knight_block=True, xq_queen=True
))
```

Available flags: `no_queen`, `chess_palace`, `knight_block`, `xq_queen`, `no_promotion`, `extra_cannon`, etc. See [RULES.md](RULES.md).

### Gymnasium Interface

```python
import gymnasium as gym
import hybrid.gym_env  # registers HybridChess-v0

env = gym.make("HybridChess-v0")
obs, info = env.reset()
action = info["legal_actions"][0]
obs, reward, terminated, truncated, info = env.step(action)
```

## Project Structure

```
hybrid-chess/
├── hybrid/                     # Python package
│   ├── core/                   # Game engine
│   │   ├── types.py            #   Side, PieceKind, Move, Piece
│   │   ├── board.py            #   Board representation (9×10)
│   │   ├── rules.py            #   Move generation, terminal detection
│   │   ├── config.py           #   VariantConfig, game constants
│   │   ├── env.py              #   HybridChessEnv
│   │   └── fen.py              #   FEN parser/serializer
│   ├── agents/                 # AI agents
│   │   ├── random_agent.py     #   Uniform random baseline
│   │   ├── greedy_agent.py     #   1-ply capture maximiser
│   │   ├── alphabeta_agent.py  #   Negamax with alpha-beta pruning
│   │   └── alphazero_stub.py   #   MCTS + neural network agent
│   └── rl/                     # AlphaZero training pipeline
│       ├── az_network.py       #   Dual-head residual CNN
│       ├── az_encoding.py      #   14-plane state / 92-plane action encoding
│       ├── az_selfplay.py      #   Self-play data generation
│       ├── az_train.py         #   Training loop
│       ├── az_eval.py          #   Evaluation, gating, Wilson CI
│       └── az_runner.py        #   Iterative AlphaZero runner
├── cpp/                        # C++ engine (pybind11, ~50× speedup)
├── scripts/                    # Experiment & evaluation scripts
│   ├── train_az_iter.py        #   Standalone AZ training CLI
│   ├── cross_variant_tournament.py  # Cross-variant round-robin
│   └── eval_arena.py           #   Side-switching evaluation
├── ui/                         # Browser-based game UI
├── tests/                      # Test suite
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         #   System architecture
│   └── EXPERIMENTS.md          #   Experiment log & results
└── runs/                       # Training outputs (gitignored)
```

## License

MIT
