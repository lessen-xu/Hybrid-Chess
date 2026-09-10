# Army imbalance diagnostics

The opt-in diagnostic pipeline separates rule asymmetry, baseline evaluation,
search behaviour and replay sampling. It keeps the public game and model
defaults unchanged. Observing an army score difference does not establish that
the rules favour that army under optimal play.

`configs/diagnose-ai.json` fixes the first diagnostic experiment: frozen round
104 model and recovery-state hashes, fresh seeds, 400 plies, gamma 1, 120 baseline
games and 48 sampling-ablation evaluation games. The maximum additional budget
is two GPU card-hours and 24 auxiliary CPU core-hours, including validation.

Run from an immutable source archive in a Slurm compute allocation:

```bash
python -m hybrid.rl.diagnose audit \
  --config configs/diagnose-ai.json --source-version SOURCE_SHA \
  --output /persistent/diagnosis --model /persistent/candidate-0104.pt \
  --teacher /persistent/teacher --training /persistent/train \
  --evaluation /persistent/eval-final --wall-seconds 3300 --workers 4
```

Use the same arguments for `probe`, `arena`, `ablate`, and `report`. After
`ablate`, use `arena --after`. `ablate` requires a GPU; other stages use CPUs.
Audit must finish before probe or ablate. Stages have independent locks, so
baseline games and ablation fitting may run concurrently. All four ablation
models must finish before their arena starts.

A corrected probe can run in a new output directory with `--reference-run`
pointing to an earlier run with identical configuration and frozen inputs. This
is supported only by `probe` and `report`: the source identity and identity hash
of the referenced run are recorded, and its files are never rewritten. Reports
record each stage's actual origin rather than attributing old results to new code.

Submit with `scripts/cluster_submit.py --budget-config configs/diagnose-ai.json`
and a **new** ledger. CPU jobs also select `--cpu-group checks`, `baseline`, or
`after`; limits are 5, 12 and 7 core-hours respectively. The submitter reserves
pending allocations and charges completed allocations, including failed jobs.
Existing ledger limits cannot be changed by passing a different configuration.
The GPU allocation's CPUs are accounted with its GPU job, as in the first run.

When ordinary free resources are unavailable, `--kind preempt` requests one
H100 on `gpu-invest` with the free `job_gpu_preemptable` QoS. It sets
`--no-requeue`: an interrupted allocation is accounted separately before a new
job resumes the same checkpoint. It never silently restarts under one job ID.

The audit reads every game's metadata, verifies teacher repetition cycles and
reconstructs the exact last 100,000-position pool from the frozen checkpoint.
The 90/10 split keeps game IDs disjoint, including the partial oldest shard at
the replay boundary. The starting model has already seen these positions;
validation loss describes this intervention, not novel-position accuracy.

The probes include 24 independently specified rule/tactical contracts and 192
real positions with full repetition histories. Rule-safety cases are not counted
as winning-move successes. Strategy interventions retain exact terminal values
while replacing only neural priors or nonterminal neural values. CPU/C++ batch-1
and batch-8 results are recorded separately. Search instrumentation never changes
the default agent or its move-selection rule.

The synthetic suite includes direct royal-capture positions that exercise the
engine interface; these need not be reachable from a legal game. They cannot
alone establish practical tactical strength. `python -m
hybrid.rl.diagnostics.confirm --original RUN --output NEW_DIRECTORY --model
MODEL --workers 4` adds 24 reachable mate-in-one positions (12 per army) from
completed fresh arena games, with complete move histories and all immediately
winning answers. Its 168 separate probes and one-move replay excerpts preserve
their own source identity. Choosing a nonterminal move is not automatically a
lost game.

`python -m hybrid.rl.diagnostics.evidence --original RUN --corrected PROBE_RUN
--output NEW_DIRECTORY --final` derives detailed tables and independently
replays every completed arena game through the C++ rules engine. It verifies
legal moves, every saved board, final results and fixed task identities. Omit
`--final` for baseline tables before the post-training arena finishes.

Both sampling arms start from identical frozen weights and fresh AdamW state.
Uniform uses the existing replay sampler; balanced chooses one of six army/result
strata uniformly, then a position uniformly within it. Both use 128 updates,
batch 256, learning rate 1e-4 and weight decay 1e-4, repeated with two seeds.
All legal actions remain in the policy loss, including zero-target actions.

The output records immutable identities, source/model hashes, per-case search
traces, whole-game replays, atomic optimizer/RNG checkpoints (retaining up to the latest two per arm),
JSON summaries and a Chinese `REPORT.md`. Resume rejects incompatible identity
or changed shards. Incomplete games never become draw labels; paired statistics
include only complete pairs. A small positive score difference whose uncertainty
includes zero is not reported as an established improvement.
