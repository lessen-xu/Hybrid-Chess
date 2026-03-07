# Hybrid Chess ♔♚

## Project Structure

```
hybrid-chess/
├── docs/          # Game rules, methodology, results, timeline
├── hybrid/        # Core library (game engine, agents, RL pipeline)
├── cpp/           # C++ engine source (pybind11)
├── scripts/       # CLI tools (train, eval, tournament, diagnostics)
├── tests/         # pytest suite
└── runs/          # Experiment outputs (not in repo)
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

