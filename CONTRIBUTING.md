# Contributing

Contributions can improve play, explore new rules, strengthen an agent or make
training easier to reproduce. For a bug report, include the rule configuration,
army, opponent and moves needed to reproduce it. An exported replay or a small
position is especially useful for a rules or search issue.

## Development setup

From a checkout, install the Python development dependencies:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The optional C++ extension can be built after installing `pybind11`:

```bash
# Linux / macOS
bash cpp/build.sh
```

```powershell
# Windows
.\cpp\build.ps1
```

Some engine and GPU checks need their corresponding optional runtime. Check the
test summary for skips; a Python-only run does not verify the native engine or
GPU inference. The cluster check scripts provide the allocated-node workflow
used by the [AI recipe](docs/GENERAL_AI.md).

## Interface changes

Start `python -m hybrid server` and edit the plain files in `ui/`. Check both
languages, narrow and wide screens, keyboard focus, a game with each army and
replay import. Text shared by the interface belongs in `ui/shared/i18n.js`;
rule descriptions belong in `hybrid/web_variants.py`.

The UI behavior checks use Node.js:

```bash
npm ci --prefix tests/ui
npm test --prefix tests/ui
```

In PowerShell, use `npm.cmd` if script execution policy blocks the `npm` wrapper.
These checks cover behavior; inspect layout changes in a browser as well.

## Rules and agents

Keep [RULES.md](RULES.md), starting-board previews and the implementation aligned.
When changing a rule, add a small position that demonstrates the intended legal
moves and result. Cover the Python and C++ paths where both implement it.
Changes to state or action encoding need explicit version handling for saved data
and weights.

An agent receives the current state and legal moves and returns a legal move.
Search must preserve the caller's board and repetition history. Check evaluation
direction with positions for both armies; winning a game as one army alone does
not establish that an agent is stronger.

## Training and model contributions

Keep checkpoints, generated datasets and weights outside Git. Share a model with
its configuration, source version, SHA256, training budget and evaluation results.
Report the opponents, time limits, rules, army assignments and sample sizes so
others can reproduce the comparison. Small evaluations are useful for screening,
but should not become broad claims about balance or playing strength.

The [architecture guide](docs/ARCHITECTURE.md) maps the relevant modules;
the [AI guide](docs/GENERAL_AI.md) describes the current general-model pipeline.
