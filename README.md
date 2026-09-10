# Hybrid Chess

Play International Chess against Xiangqi on a shared 9 × 10 board. Each army
keeps its own movement rules. Choose a preset, combine your own rules, and play
either side against an AI.

Hybrid Chess is an open-source game and a place to explore asymmetric rules and
game AI. Rule combinations change the game substantially; no preset is promised
to give the two armies equal chances.

## Start playing

Use Python 3.9 or newer from a local checkout:

```bash
git clone https://github.com/lessen-xu/Hybrid-Chess.git
cd Hybrid-Chess
python -m pip install -e .
python -m hybrid server
```

Open **http://127.0.0.1:8000**. Choose the rules, your army and an opponent, then
start a game. The interface supports Chinese and English, opening previews,
legal-move hints, promotion choices, undo and board rotation. Chess moves first;
when you choose Xiangqi, the AI makes the opening move.

Quick, Standard and Deep opponents use Python Alpha-Beta with approximate
1, 3 and 6 second budgets. Random and Greedy opponents are also available.
Playing requires neither model weights nor a compiled C++ engine.

The local server keeps one shared game in memory. Refreshing the page restores
that game; restarting the server clears it. The secondary Replays page imports
JSON and JSONL recordings for stepping through or autoplaying a game.

## Choose the rules

| Preset | Changes from the original setup |
| --- | --- |
| Original rules | Both armies use their usual movement rules, with the shared-board adaptations below. |
| Palace & blocked knights | Confine the Chess king to a palace and make Chess knights subject to leg blocking. |
| A queen for Xiangqi | Replace Xiangqi's left advisor with a queen. |
| Palace, knights & queen | Combine those movement restrictions with the Xiangqi queen. |
| Chess without a queen | Remove the starting Chess queen. |
| An extra cannon | Add a third Xiangqi cannon. |

Custom settings also cover the ninth pawn, individual pieces, pawn promotion and
the flying general. [Game rules](RULES.md) explains the board, every piece,
termination and all rule switches.

## Explore the AI

The built-in opponents provide starting points for developing your own agents.
The rule-aware training pipeline uses one small policy/value network for both
armies, learns from Alpha-Beta games, then continues through mixed-rule self-play
with MCTS. See the [AI guide](docs/GENERAL_AI.md) for the model format, training,
recovery and evaluation.

To play with a compatible exported model:

```bash
python -m hybrid server --model /path/to/candidate.pt
```

Learned opponents run on a local CPU. Model files are distributed separately;
their accompanying evaluations describe which rules and opponents were tested.

## Develop

Create an environment directly to explore a rule combination:

```python
from hybrid.core.config import VariantConfig
from hybrid.core.env import HybridChessEnv

env = HybridChessEnv(variant=VariantConfig(
    chess_palace=True,
    knight_block=True,
    xq_queen=True,
))
state = env.reset()
state, reward, done, info = env.step(env.legal_moves()[0])
```

The engine is written in Python, with an optional C++ extension. The web UI uses
plain HTML, CSS, JavaScript and a shared SVG board; it has no frontend build step.

| Area | Location |
| --- | --- |
| Rules, board and game state | `hybrid/core/` |
| Random, Greedy, Alpha-Beta and MCTS agents | `hybrid/agents/` |
| Training, inference and evaluation | `hybrid/rl/` |
| Local HTTP server and rule catalog | `hybrid/server.py`, `hybrid/web_variants.py` |
| Play and replay interfaces | `ui/` |
| Optional native engine | `cpp/` |

See [Architecture](docs/ARCHITECTURE.md) for the data flow and
[Contributing](CONTRIBUTING.md) for setup, checks and extension points.

## License

MIT.
