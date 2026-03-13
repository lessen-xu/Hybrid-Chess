#!/usr/bin/env python3
"""Game replay tool.

Usage:
  python scripts/visualize_game.py --in game.json --mode ascii
  python scripts/visualize_game.py --in game.json --mode html --out game.html
  python scripts/visualize_game.py --in games.jsonl --mode html --out games.html
"""

from __future__ import annotations
import scripts._fix_encoding  # noqa: F401
import argparse
import json
import html as html_mod
import sys
from pathlib import Path
from typing import List


def load_games(path: str) -> List[dict]:
    """Load JSON or JSONL game file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.strip().split("\n") if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]


def ascii_replay(game: dict) -> None:
    """Step-by-step terminal replay."""
    states = game.get("states_ascii", [])
    moves = game.get("moves", [])
    meta = game.get("meta", {})
    result = game.get("result", "?")

    print(f"\n{'='*40}")
    print(f"Result: {result}")
    if meta:
        print(f"Plies: {meta.get('plies', '?')}  "
              f"Reason: {meta.get('reason', '?')}")
    print(f"{'='*40}\n")

    for i, board_str in enumerate(states):
        if i == 0:
            label = "Initial position"
        elif i <= len(moves):
            label = f"After move {i}: {moves[i-1]}"
        else:
            label = f"State {i}"

        print(f"--- {label} ---")
        print(board_str)

        if i < len(states) - 1:
            try:
                input("Press Enter for next move (Ctrl-C to quit) ...")
            except (KeyboardInterrupt, EOFError):
                print("\nAborted.")
                return

    print(f"\n{'='*40}")
    print(f"Game over: {result}")
    print(f"{'='*40}")


def generate_html(games: List[dict], title: str = "Game Replay") -> str:
    """Generate interactive HTML replay page."""
    # Serialize games to JSON for embedding
    games_json = json.dumps(games, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_mod.escape(title)}</title>
<style>
  :root {{
    --bg: #1a1a2e; --surface: #16213e; --text: #e0e0e0;
    --accent: #0f3460; --highlight: #e94560; --border: #333;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Consolas', 'Courier New', monospace;
    background: var(--bg); color: var(--text);
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 20px;
  }}
  h1 {{ color: var(--highlight); margin-bottom: 10px; font-size: 1.4em; }}
  .controls {{
    display: flex; gap: 10px; align-items: center; margin: 10px 0;
    flex-wrap: wrap; justify-content: center;
  }}
  button {{
    background: var(--accent); color: var(--text); border: 1px solid var(--border);
    padding: 8px 16px; border-radius: 4px; cursor: pointer;
    font-family: inherit; font-size: 0.9em; transition: background 0.2s;
  }}
  button:hover {{ background: var(--highlight); }}
  button:disabled {{ opacity: 0.4; cursor: default; }}
  input[type="range"] {{ width: 200px; accent-color: var(--highlight); }}
  .info {{
    background: var(--surface); padding: 10px 20px; border-radius: 6px;
    margin: 8px 0; text-align: center; border: 1px solid var(--border);
    min-width: 300px;
  }}
  .board {{
    background: var(--surface); padding: 15px; border-radius: 8px;
    border: 1px solid var(--border); white-space: pre;
    font-size: 14px; line-height: 1.4; margin: 10px 0;
    min-width: 300px;
  }}
  .game-nav {{
    margin-bottom: 15px; display: flex; gap: 8px;
    flex-wrap: wrap; justify-content: center;
  }}
  .game-nav button.active {{ background: var(--highlight); }}
  .meta {{ font-size: 0.85em; color: #999; margin-top: 5px; }}
</style>
</head>
<body>
<h1>{html_mod.escape(title)}</h1>

<div class="game-nav" id="gameNav"></div>
<div class="info" id="info">Loading...</div>
<div class="controls">
  <button id="btnFirst" onclick="goFirst()">⏮ First</button>
  <button id="btnPrev" onclick="goPrev()">◀ Prev</button>
  <input type="range" id="slider" min="0" max="0" value="0"
         oninput="goTo(parseInt(this.value))">
  <button id="btnNext" onclick="goNext()">Next ▶</button>
  <button id="btnLast" onclick="goLast()">Last ⏭</button>
</div>
<div class="board" id="board"></div>
<div class="meta" id="meta"></div>

<script>
const GAMES = {games_json};
let currentGame = 0;
let currentStep = 0;

function initGameNav() {{
  const nav = document.getElementById('gameNav');
  if (GAMES.length <= 1) {{ nav.style.display = 'none'; return; }}
  nav.innerHTML = '';
  GAMES.forEach((g, i) => {{
    const btn = document.createElement('button');
    btn.textContent = 'Game ' + (i + 1);
    btn.onclick = () => selectGame(i);
    if (i === currentGame) btn.classList.add('active');
    nav.appendChild(btn);
  }});
}}

function selectGame(idx) {{
  currentGame = idx; currentStep = 0;
  initGameNav(); updateSlider(); render();
}}

function updateSlider() {{
  const g = GAMES[currentGame];
  const slider = document.getElementById('slider');
  slider.max = (g.states_ascii || []).length - 1;
  slider.value = currentStep;
}}

function render() {{
  const g = GAMES[currentGame];
  const states = g.states_ascii || [];
  const moves = g.moves || [];
  const meta = g.meta || {{}};

  const info = document.getElementById('info');
  let label = currentStep === 0 ? 'Initial position'
    : (currentStep <= moves.length
       ? 'Move ' + currentStep + ': ' + moves[currentStep - 1]
       : 'State ' + currentStep);
  info.innerHTML = '<b>' + label + '</b> &nbsp; (' +
    (currentStep) + '/' + (states.length - 1) +
    ') &nbsp; Result: ' + (g.result || '?');

  document.getElementById('board').textContent =
    states[currentStep] || '(no data)';

  const metaEl = document.getElementById('meta');
  metaEl.textContent = 'Plies: ' + (meta.plies || '?') +
    '  Reason: ' + (meta.reason || '?') +
    '  Seed: ' + (meta.seed || '?');

  document.getElementById('slider').value = currentStep;
  document.getElementById('btnPrev').disabled = (currentStep <= 0);
  document.getElementById('btnFirst').disabled = (currentStep <= 0);
  document.getElementById('btnNext').disabled = (currentStep >= states.length - 1);
  document.getElementById('btnLast').disabled = (currentStep >= states.length - 1);
}}

function goTo(n) {{ currentStep = n; render(); }}
function goFirst() {{ currentStep = 0; render(); }}
function goLast() {{
  const g = GAMES[currentGame];
  currentStep = (g.states_ascii || []).length - 1;
  render();
}}
function goPrev() {{ if (currentStep > 0) {{ currentStep--; render(); }} }}
function goNext() {{
  const g = GAMES[currentGame];
  if (currentStep < (g.states_ascii || []).length - 1) {{
    currentStep++; render();
  }}
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') goPrev();
  else if (e.key === 'ArrowRight') goNext();
  else if (e.key === 'Home') goFirst();
  else if (e.key === 'End') goLast();
}});

initGameNav(); updateSlider(); render();
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Hybrid Chess Game Replay")
    parser.add_argument("--in", dest="input", required=True,
                        help="Input game file (.json or .jsonl)")
    parser.add_argument("--mode", choices=["ascii", "html"], default="ascii",
                        help="Replay mode")
    parser.add_argument("--out", default=None,
                        help="Output HTML file (required for --mode html)")
    args = parser.parse_args()

    games = load_games(args.input)
    if not games:
        print("No games found in input file.", file=sys.stderr)
        sys.exit(1)

    if args.mode == "ascii":
        for i, game in enumerate(games):
            if len(games) > 1:
                print(f"\n{'#'*40}  Game {i+1}/{len(games)}  {'#'*40}")
            ascii_replay(game)
    else:
        if not args.out:
            print("--out is required for HTML mode", file=sys.stderr)
            sys.exit(1)
        html_content = generate_html(games, title="Hybrid Chess Replay")
        Path(args.out).write_text(html_content, encoding="utf-8")
        print(f"HTML replay saved to: {args.out}")


if __name__ == "__main__":
    main()
