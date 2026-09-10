# Rule-aware AI

The general AI learns both armies and multiple rule combinations with one
policy/value network. Its first recipe uses Alpha-Beta demonstrations, followed
by self-play. A checkpoint is a candidate, not evidence of a particular strength.

## Play locally

```bash
python -m hybrid server --model /path/to/candidate.pt
```

Compatible weights add Neural Quick, Standard and Deep opponents with approximate
1, 3 and 6 second budgets. Inference runs on one CPU thread. The C++ extension is
optional for local play. Without a model, the existing opponents remain available.
Model files use PyTorch's weights-only loader; full training checkpoints are not
model downloads. Model and replay encoding version 2 must match the loader.

## Model and data

The default network has four residual blocks with 96 channels. Version 2 inputs
contain 29 planes: 14 piece types, side to move, 12 canonical rule flags, remaining
plies / move limit, and current repetition count / 3. Feature order is defined in
`hybrid/rl/general_model.py`. Search carries the full repetition dictionary; the
network sees a summary. The policy output remains 92 x 10 x 9. Version 1 models
retain their legacy 15-plane encoding and are not accepted by the web model option.

Per-game replay shards store float16 features, every legal action (including zero
visit targets), search or teacher probabilities, and the final result from the
moving army's perspective. The final values are +1 / -1 / 0. Automatic resignation,
heuristic draw adjudication, material-based results and discounting are disabled
in this recipe. Games stop at the same 400-ply rule limit used by web play.

Every block of 30 scheduled games contains four games per preset and six custom
games. Custom games flip 1-3 rule flags and canonicalize dependent flags. Each game
has an independent seed and identity. The supervised split is by complete game,
using a separately seeded 10% validation assignment. Shared positions across
distinct opening games can occur; validation is not a claim of novel-position accuracy.

## Train on UBELIX

The recipe is `configs/general-ai.json`. Install `requirements-cluster.txt` in an
isolated Linux environment with Python 3.11 and a compatible NVIDIA driver. Build
the existing extension with `bash cpp/build.sh` on a compute node. Model, data,
optimizer and CUDA checks also run inside Slurm allocations. The CLI refuses to
train on the login node or outside Slurm.

```bash
# In a CPU compute allocation:
python -m hybrid.rl.general_train teacher --config configs/general-ai.json \
  --output /persistent/run/teacher --source-version SOURCE_SHA --wall-seconds 7000

# In a GPU compute allocation (repeat with the same output to resume):
python -m hybrid.rl.general_train train --config configs/general-ai.json \
  --teacher-manifest /persistent/run/teacher/dataset.json \
  --output /persistent/run/train --source-version SOURCE_SHA \
  --device cuda --wall-seconds 13500

# In a CPU compute allocation:
python -m hybrid.rl.general_eval --model /persistent/run/train/candidate.pt \
  --output /persistent/run/evaluation --workers 14 --games 20 --wall-seconds 7000
```

The source packager records a hash of each file, a source-tree hash and the base
Git commit. It normalizes text line endings and excludes ignored training outputs:

```bash
python scripts/package_source.py --output runs/source.tar.gz
```

Use the recorded source hash as `SOURCE_SHA`; upload and unpack into a new release
directory. Do not overwrite a release used by running jobs. Reusing a run with
different source or configuration is rejected. `scripts/cluster_submit.py` submits
finite jobs to `gratis` and keeps submission intent/job IDs in a locked ledger.
This first-round ledger caps GPU allocations at 8 card-hours and auxiliary CPU
allocations at 64 core-hours, charging finished jobs from Slurm elapsed time and
reserving full wall limits for pending/running jobs. GPU jobs include their own
16-CPU allocation; those CPUs are not part of the auxiliary CPU cap.

## Recovery

Complete game shards are atomically committed. Incomplete games may be replayed
from their seed, but never enter the buffer with invented draw labels. Collection
uses a frozen model per iteration. Two checksummed full checkpoints contain the
optimizer, random generators, training phase, minibatch position, ordered replay
references and metrics. The replay content remains in checksummed immutable shards
on persistent storage. Keep shards when archiving full recovery state.

Periodic checkpoints occur at safe boundaries, at least after each training epoch
and around each collection/training transition. During collection the ten-minute
timer is checked when a game completes; a slow game may delay it. Every completed
game is already durable independently. USR1 requests a safe save and exit. The
Slurm launcher forwards the signal through srun. SIGKILL recovery uses the latest
complete checkpoint and complete game files. A corrupt latest checkpoint falls
back to the previous checksum-verified version and writes `recovery.json`.

CPU training with matching hardware/backends and the same data/configuration supports
exact interrupted versus continuous comparisons. Different CPU instruction paths,
GPU kernels and batches assembled by parallel workers
can differ numerically or in scheduling; interrupted self-play is not advertised
as bit-identical to an uninterrupted GPU run. Game IDs still prevent double counting.

## Evaluation and distribution

The default evaluation has 360 games: six presets, three opponents, twenty games
per group, with armies swapped for each of ten independently seeded openings.
Four opening plies provide controlled diversity. Opponents run in the same CPU
worker with one thread and the same one-second upper search budget; Random and
Greedy naturally use less time. The AB baseline is the web Quick opponent (depth 1).

Use `--seed` to choose the base opening seed (default `876000`). Use a different
base seed for final acceptance than for interim candidate screening, and record
both. Resuming with a different seed requires a new evaluation output directory.

Reports include army-specific scores, W/D/L, move latency and conservative 95%
Hoeffding intervals over complete opening-pair mean scores. Twenty games per group
are a screening sample. Partial runs report expected/completed counts explicitly.
The first opening pair in each group includes board snapshots and can be imported
into the replay viewer. Other game files retain moves and results for analysis.
Custom rule strength must be assessed separately; training on random combinations
does not establish strength on every possible combination.

`--variants configs/heldout-variants.json --games 2` evaluates six deliberately
unseen combinations as a small stress test (36 games). Each differs from the
default in seven effective switches; the training sampler can change at most six
from the default. These configurations can strongly favor one army, so report
both assignments and do not describe their raw army win rates as model strength.

Distribute weights with their SHA256, source archive/hash, configuration, model
card, evaluation summary and selected game recordings. Keep weights and full
training state outside ordinary Git history.
