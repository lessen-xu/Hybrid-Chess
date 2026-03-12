#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hybrid Chess — Game Server

Zero‐dependency HTTP server (Python stdlib only).
Serves the web UI and exposes a REST API for interactive play.

Usage:
    python -m hybrid.server                    # http://localhost:8000
    python -m hybrid.server --port 9000        # custom port
    python -m hybrid.server --no-browser       # don't auto-open
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── project imports ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hybrid.core.board import Board, initial_board
from hybrid.core.env import HybridChessEnv, GameState
from hybrid.core.types import Side, PieceKind, Move, Piece
from hybrid.core.rules import generate_legal_moves
from hybrid.core.render import render_board


# ═══════════════════════════════════════════════
# Agent factory
# ═══════════════════════════════════════════════

AVAILABLE_AGENTS = [
    {"id": "random",  "label": "Random"},
    {"id": "greedy",  "label": "Greedy"},
    {"id": "ab_d1",   "label": "AlphaBeta d=1"},
    {"id": "ab_d2",   "label": "AlphaBeta d=2"},
    {"id": "ab_d4",   "label": "AlphaBeta d=4"},
]

def create_agent(agent_id: str):
    """Create an agent instance by its ID string."""
    if agent_id == "random":
        from hybrid.agents.random_agent import RandomAgent
        return RandomAgent(seed=None)
    elif agent_id == "greedy":
        from hybrid.agents.greedy_agent import GreedyAgent
        return GreedyAgent()
    elif agent_id.startswith("ab_d"):
        depth = int(agent_id.split("d")[1])
        from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
        return AlphaBetaAgent(cfg=SearchConfig(depth=depth))
    else:
        raise ValueError(f"Unknown agent: {agent_id}")



# ═══════════════════════════════════════════════
# Game Session
# ═══════════════════════════════════════════════

class GameSession:
    """Holds the state of a single game."""

    def __init__(self, human_side: str, ai_agent_id: str, variant: str = "none"):
        self.human_side = Side.CHESS if human_side == "chess" else Side.XIANGQI
        self.ai_side = self.human_side.opponent()
        self.ai_agent = create_agent(ai_agent_id)
        self.variant = variant

        self.env = HybridChessEnv(max_plies=400, use_cpp=False)
        self.env.reset()

        # Apply variant
        if variant == "no_queen":
            self._apply_no_queen()
        elif variant == "extra_cannon":
            self._apply_extra_cannon()

        self.history: List[GameState] = [self._clone_state()]
        self.move_history: List[Dict] = []

    def _apply_no_queen(self):
        """Remove the Chess Queen from starting position."""
        board = self.env.state.board
        for x in range(9):
            for y in range(10):
                p = board.get(x, y)
                if p and p.kind == PieceKind.QUEEN and p.side == Side.CHESS:
                    board.set(x, y, None)

    def _apply_extra_cannon(self):
        """Add an extra Cannon for Xiangqi side at (4, 7)."""
        board = self.env.state.board
        if board.get(4, 7) is None:
            board.set(4, 7, Piece(PieceKind.CANNON, Side.XIANGQI))

    def _clone_state(self):
        return GameState(
            board=self.env.state.board.clone(),
            side_to_move=self.env.state.side_to_move,
            ply=self.env.state.ply,
            repetition=dict(self.env.state.repetition),
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """Return current state as a JSON-friendly dict."""
        state = self.env.state
        legal = self.env.legal_moves()
        board_ascii = render_board(state.board)

        return {
            "board_ascii": board_ascii,
            "side_to_move": state.side_to_move.name.lower(),
            "ply": state.ply,
            "legal_moves": [
                {
                    "fx": m.fx, "fy": m.fy,
                    "tx": m.tx, "ty": m.ty,
                    "promotion": m.promotion.name if m.promotion else None,
                }
                for m in legal
            ],
            "game_over": False,
            "result": "",
            "reason": "",
        }

    def apply_human_move(self, fx: int, fy: int, tx: int, ty: int,
                          promotion: Optional[str] = None) -> Dict[str, Any]:
        """Apply a human move and return new state."""
        promo = None
        if promotion:
            promo = PieceKind[promotion.upper()]

        move = Move(fx, fy, tx, ty, promo)

        # Validate
        legal = self.env.legal_moves()
        if not any(m.fx == fx and m.fy == fy and m.tx == tx and m.ty == ty and
                   (m.promotion == promo or (m.promotion is None and promo is None))
                   for m in legal):
            raise ValueError(f"Illegal move: ({fx},{fy})->({tx},{ty})")

        state, reward, done, info = self.env.step(move)
        self.history.append(self._clone_state())

        result = self._build_result(done, info)
        return result

    def ai_move(self) -> Dict[str, Any]:
        """Let the AI make a move."""
        legal = self.env.legal_moves()
        if not legal:
            return self.get_state_dict()

        move = self.ai_agent.select_move(self.env.state, legal)
        state, reward, done, info = self.env.step(move)
        self.history.append(self._clone_state())

        result = self._build_result(done, info)
        result["move"] = {
            "fx": move.fx, "fy": move.fy,
            "tx": move.tx, "ty": move.ty,
        }
        return result

    def undo(self) -> Dict[str, Any]:
        """Undo the last two moves (human + AI)."""
        undo_count = min(2, len(self.history) - 1)
        for _ in range(undo_count):
            self.history.pop()

        # Restore state
        saved = self.history[-1]
        self.env.state = GameState(
            board=saved.board.clone(),
            side_to_move=saved.side_to_move,
            ply=saved.ply,
            repetition=dict(saved.repetition),
        )
        return self.get_state_dict()

    def resign(self) -> Dict[str, Any]:
        """Human resigns."""
        winner = self.ai_side.name
        return {
            **self.get_state_dict(),
            "game_over": True,
            "result": f"{winner} wins",
            "reason": "Resignation",
        }

    def _build_result(self, done: bool, info) -> Dict[str, Any]:
        result = self.get_state_dict()
        if done:
            result["game_over"] = True
            winner = getattr(info, 'winner', None)
            reason = getattr(info, 'reason', '')
            if winner == Side.CHESS:
                result["result"] = "Chess wins"
            elif winner == Side.XIANGQI:
                result["result"] = "Xiangqi wins"
            else:
                result["result"] = "Draw"
            result["reason"] = reason
        return result


# ═══════════════════════════════════════════════
# HTTP Handler
# ═══════════════════════════════════════════════

# Global session (single-player for simplicity)
current_session: Optional[GameSession] = None

class HybridChessHandler(SimpleHTTPRequestHandler):
    """Serves static files from ui/ and handles API requests."""

    def __init__(self, *args, **kwargs):
        self.ui_dir = str(ROOT / "ui")
        super().__init__(*args, directory=self.ui_dir, **kwargs)

    def do_GET(self):
        if self.path == '/api/agents':
            self._json_response({"agents": AVAILABLE_AGENTS})
        elif self.path == '/api/state':
            if current_session:
                self._json_response(current_session.get_state_dict())
            else:
                self._json_response({"error": "No active game"}, 404)
        else:
            super().do_GET()

    def do_POST(self):
        global current_session

        body = self._read_body()

        if self.path == '/api/new':
            try:
                current_session = GameSession(
                    human_side=body.get("human_side", "chess"),
                    ai_agent_id=body.get("ai_agent", "ab_d1"),
                    variant=body.get("variant", "none"),
                )
                self._json_response(current_session.get_state_dict())
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif self.path == '/api/move':
            if not current_session:
                self._json_response({"error": "No active game"}, 400)
                return
            try:
                result = current_session.apply_human_move(
                    fx=body["fx"], fy=body["fy"],
                    tx=body["tx"], ty=body["ty"],
                    promotion=body.get("promotion"),
                )
                self._json_response(result)
            except Exception as e:
                self._json_response({"error": str(e)}, 400)

        elif self.path == '/api/ai_move':
            if not current_session:
                self._json_response({"error": "No active game"}, 400)
                return
            try:
                result = current_session.ai_move()
                self._json_response(result)
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif self.path == '/api/undo':
            if not current_session:
                self._json_response({"error": "No active game"}, 400)
                return
            try:
                result = current_session.undo()
                self._json_response(result)
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif self.path == '/api/resign':
            if not current_session:
                self._json_response({"error": "No active game"}, 400)
                return
            result = current_session.resign()
            self._json_response(result)

        else:
            self._json_response({"error": "Not found"}, 404)

    def _read_body(self) -> dict:
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode('utf-8'))

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        msg = str(args[0]) if args else ''
        if '/api/' in msg:
            sys.stderr.write(f"  [API] {msg}\n")


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Hybrid Chess Game Server")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), HybridChessHandler)
    url = f"http://{args.host}:{args.port}"

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║       Hybrid Chess Server            ║")
    print("  ╠══════════════════════════════════════╣")
    print(f"  ║  URL: {url:<30s} ║")
    print("  ║  Press Ctrl+C to stop               ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
