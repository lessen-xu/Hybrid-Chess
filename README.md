# Hybrid Chess ♔♚

**Topological Phase Transitions in Asymmetric Zero-Sum Games: An Empirical Game-Theoretic Analysis**

> A two-player zero-sum game where one side plays with International Chess pieces and the other with Chinese Chess (Xiangqi) pieces on a shared 9×10 board. We train AlphaZero-style agents via self-play and apply Empirical Game-Theoretic Analysis (EGTA) to study how asymmetric rule perturbations reshape the topology of multi-agent strategy spaces.

---

## Project Structure

```
hybrid-chess/
├── README.md                       # This file
├── docs/                           # Technical documentation
│   ├── game_rules.md               #   Rules, pieces, ablation variants
│   ├── methodology.md              #   Engine, agents, training pipeline
│   ├── results/                    #   Experiment results (one file per study)
│   │   ├── ab_tournament.md        #     AB balance matrix (2×3)
│   │   ├── az_training.md          #     V1–V4 training + champion eval
│   │   └── egta_pilot.md           #     EGTA pilot results (pending)
│   └── timeline.md                 #   Development log (Phase 0–50)
├── paper/                          # LaTeX drafts
├── notebooks/                      # Analysis scripts
├── hybrid/                         # Core library
│   ├── core/                       #   Game engine (Python + C++ backend)
│   ├── agents/                     #   Agent implementations
│   └── rl/                         #   AlphaZero training pipeline
├── cpp/                            # C++ engine (pybind11)
├── scripts/                        # CLI tools (train, eval, tournament)
├── tests/                          # pytest suite (~150 tests)
└── runs/                           # Experiment outputs (not in repo)
```

## Quick Start

```bash
pip install -r requirements.txt

# Build C++ engine
.\cpp\build.ps1

# Run tests
pytest -q

# Train AlphaZero
python -m scripts.train_az_iter --iterations 20 \
    --curriculum-schedule 3phase_v2 \
    --simulations 200 --eval-simulations 400 \
    --num-workers 8 --use-inference-server --inference-device cuda \
    --use-cpp --ablation extra_cannon \
    --outdir runs/az_run

# Side-switching evaluation arena
python -m scripts.eval_arena --model-a ab_d4 --model-b ab_d1 \
    --games 20 --use-cpp

# EGTA dual-matrix tournament
python -m scripts.egta_tournament --preset v4 --outdir runs/egta
```

## Documentation

| Document | Content |
|---|---|
| [Game Rules](docs/game_rules.md) | Board, pieces, termination, ablation variants |
| [Methodology](docs/methodology.md) | C++ engine, agents, training pipeline, GPU scaling |
| [Results: AB Tournament](docs/results/ab_tournament.md) | AB balance 2×3 matrix, termination analysis |
| [Results: AZ Training](docs/results/az_training.md) | V1–V4 runs, champion eval, sim scaling |
| [Timeline](docs/timeline.md) | Development phases 0–50 |
