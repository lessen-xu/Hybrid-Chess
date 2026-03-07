# Hybrid Chess ♔♚

## Project Structure

```
hybrid-chess/
├── README.md
├── requirements.txt
├── docs/
│   ├── game_rules.md               # Rules, pieces, termination, ablation variants
│   ├── methodology.md              # Engine, agents, training pipeline
│   ├── results/
│   │   ├── ab_tournament.md        # AB balance matrix (2×3)
│   │   ├── az_training.md          # V1–V4 training + champion eval
│   │   ├── egta_pilot.md           # EGTA pilot (720 games)
│   │   └── egta_n100.md            # EGTA N=100 dual-universe results
│   └── timeline.md                 # Development log (Phase 0–56)
├── hybrid/                         # Core library
│   ├── core/                       #   Game engine (Python + C++ backend)
│   ├── agents/                     #   Agent implementations
│   ├── cpp_engine/                 #   Compiled C++ pyd
│   └── rl/                         #   AlphaZero training pipeline
├── cpp/                            # C++ engine source (pybind11)
├── scripts/                        # CLI tools (train, eval, tournament)
├── tests/                          # pytest suite
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
| [Game Spec V2](docs/game_spec_v2.md) | Canonical game definition (protocol lock) |
| [Methodology](docs/methodology.md) | C++ engine, agents, training pipeline, GPU scaling |
| [Results: AB Tournament](docs/results/ab_tournament.md) | AB balance 2×3 matrix, termination analysis |
| [Results: AZ Training](docs/results/az_training.md) | V1–V4 runs, champion eval, sim scaling |
| [Legacy EGTA](docs/results/legacy_pre_v2/) | Pre-V2 results (invalidated — see INVALIDATED.md) |
| [Timeline](docs/timeline.md) | Development phases 0–56 |
