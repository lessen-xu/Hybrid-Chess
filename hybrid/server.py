"""Single-game local HTTP server for Hybrid Chess (Python standard library)."""
from __future__ import annotations
import argparse
import json
import threading
import uuid
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit
from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side, PieceKind, Move
from hybrid.core.rules import terminal_info, is_in_check, TerminalStatus
from hybrid.core.render import render_board
from hybrid.web_variants import bilingual, catalog, parse_variant, preview

ROOT = Path(__file__).resolve().parent.parent
AVAILABLE_AGENTS = [
    {"id": "ab_d1", "label": "Quick", "name": bilingual("快速", "Quick"), "seconds": 1},
    {"id": "ab_d2", "label": "Standard", "name": bilingual("标准", "Standard"), "seconds": 3},
    {"id": "ab_d4", "label": "Deep", "name": bilingual("深入", "Deep"), "seconds": 6},
    {"id": "greedy", "label": "Greedy", "name": bilingual("贪心练习", "Capture practice"), "seconds": 0},
    {"id": "random", "label": "Random", "name": bilingual("随机练习", "Random practice"), "seconds": 0},
]


class APIError(ValueError):
    def __init__(self, code, status=400):
        super().__init__(code)
        self.code, self.status = code, status


def create_agent(agent_id):
    if agent_id not in [a["id"] for a in AVAILABLE_AGENTS]:
        raise APIError("invalid_agent")
    if agent_id == "random":
        from hybrid.agents.random_agent import RandomAgent
        return RandomAgent(seed=None)
    if agent_id == "greedy":
        from hybrid.agents.greedy_agent import GreedyAgent
        return GreedyAgent()
    from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
    depth, seconds = {"ab_d1": (1, 1), "ab_d2": (2, 3), "ab_d4": (4, 6)}[agent_id]
    return AlphaBetaAgent(SearchConfig(depth=depth, time_limit_seconds=seconds))


def move_dict(move):
    return {"fx": move.fx, "fy": move.fy, "tx": move.tx, "ty": move.ty,
            "promotion": move.promotion.name if move.promotion else None}


def reason_code(reason):
    return {"": "", "Checkmate": "checkmate",
            "Stalemate (loss for stalemated side)": "stalemate",
            "Max plies reached": "move_limit", "Threefold repetition": "repetition",
            "Chess king captured": "royal_captured",
            "Xiangqi general captured": "royal_captured"}.get(reason, "")


class GameSession:
    def __init__(self, human_side, ai_agent_id, variant="none"):
        if human_side not in ("chess", "xiangqi"):
            raise APIError("invalid_side")
        self.variant = parse_variant(variant)
        self.human_side = Side.CHESS if human_side == "chess" else Side.XIANGQI
        self.ai_side = self.human_side.opponent()
        self.ai_agent = create_agent(ai_agent_id)
        self.ai_agent_id = ai_agent_id
        self.env = HybridChessEnv(variant=self.variant)
        self.env.reset()
        self.history = [self.env.state.clone()]
        self.move_history = []
        self.resigned = False
        self.session_id = uuid.uuid4().hex
        self.revision = 0

    def _info(self):
        self.env._set_active_variant()
        s = self.env.state
        return terminal_info(s.board, s.side_to_move, s.repetition, s.ply, self.env.max_plies)

    def get_state_dict(self):
        info = self._info()
        state = self.env.state
        done = self.resigned or info.status != TerminalStatus.ONGOING
        winner = self.ai_side if self.resigned else info.winner
        result = f"{winner.name.lower()}_win" if winner else ("draw" if done else "ongoing")
        return {
            "session_id": self.session_id, "revision": self.revision,
            "board_ascii": render_board(state.board),
            "side_to_move": state.side_to_move.name.lower(), "ply": state.ply,
            "human_side": self.human_side.name.lower(), "ai_agent": self.ai_agent_id,
            "variant": self.variant.to_dict(),
            "legal_moves": [] if done else [move_dict(m) for m in self.env.legal_moves()],
            "moves": list(self.move_history),
            "in_check": not done and is_in_check(state.board, state.side_to_move),
            "game_over": done, "result_code": result,
            "result": {"chess_win": "Chess wins", "xiangqi_win": "Xiangqi wins",
                       "draw": "Draw", "ongoing": ""}[result],
            "reason": "Resignation" if self.resigned else info.reason,
            "reason_code": "resignation" if self.resigned else reason_code(info.reason),
            "can_undo": not done and any(m["side"] == self.human_side.name.lower()
                                         for m in self.move_history),
        }

    def check_revision(self, body):
        if ("session_id" in body or "revision" in body) and (
            body.get("session_id") != self.session_id or
            type(body.get("revision")) is not int or body["revision"] != self.revision
        ):
            raise APIError("stale_state", 409)

    def _require_ongoing(self, side=None):
        if self.resigned or self._info().status != TerminalStatus.ONGOING:
            raise APIError("game_over", 409)
        if side is not None and self.env.state.side_to_move != side:
            raise APIError("wrong_turn", 409)

    def _apply(self, move):
        side = self.env.state.side_to_move.name.lower()
        self.env.step(move)
        record = {**move_dict(move), "side": side,
                  "notation": f"{'abcdefghi'[move.fx]}{move.fy + 1}-{'abcdefghi'[move.tx]}{move.ty + 1}"}
        if move.promotion:
            record["notation"] += "=" + {"QUEEN": "Q", "ROOK": "R", "BISHOP": "B", "KNIGHT": "N"}[move.promotion.name]
        self.move_history.append(record)
        self.history.append(self.env.state.clone())
        self.revision += 1
        return {**self.get_state_dict(), "move": move_dict(move)}

    def apply_human_move(self, fx, fy, tx, ty, promotion=None):
        self._require_ongoing(self.human_side)
        if any(type(c) is not int for c in (fx, fy, tx, ty)) or not (
            0 <= fx < 9 and 0 <= tx < 9 and 0 <= fy < 10 and 0 <= ty < 10
        ):
            raise APIError("invalid_move")
        if promotion is not None and promotion not in ("QUEEN", "ROOK", "BISHOP", "KNIGHT"):
            raise APIError("invalid_promotion")
        move = Move(fx, fy, tx, ty, PieceKind[promotion] if promotion else None)
        if move not in self.env.legal_moves():
            raise APIError("illegal_move")
        return self._apply(move)

    def ai_move(self):
        self._require_ongoing(self.ai_side)
        move = self.ai_agent.select_move(self.env.state.clone(), self.env.legal_moves())
        return self._apply(move)

    def undo(self):
        self._require_ongoing()
        indices = [i for i, m in enumerate(self.move_history)
                   if m["side"] == self.human_side.name.lower()]
        if not indices:
            raise APIError("nothing_to_undo", 409)
        index = indices[-1]
        self.env.state = self.history[index].clone()
        self.history = self.history[:index + 1]
        self.move_history = self.move_history[:index]
        self.revision += 1
        return self.get_state_dict()

    def resign(self):
        self._require_ongoing()
        self.resigned = True
        self.revision += 1
        return self.get_state_dict()


current_session = None


class HybridChessHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "ui"), **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/agents":
            self._json_response({"agents": AVAILABLE_AGENTS})
        elif path == "/api/variants":
            self._json_response(catalog())
        elif path == "/api/state":
            if current_session:
                self._json_response(current_session.get_state_dict())
            else:
                self._json_response({"error": "no_game"}, 404)
        elif path.startswith("/api/"):
            self._json_response({"error": "not_found"}, 404)
        else:
            super().do_GET()

    def do_POST(self):
        global current_session
        try:
            body = self._read_body()
            path = urlsplit(self.path).path
            if path == "/api/preview":
                result = preview(body.get("variant", "none"))
            elif path == "/api/new":
                if current_session:
                    current_session.check_revision(body)
                candidate = GameSession(body.get("human_side", "chess"),
                                        body.get("ai_agent", "ab_d1"), body.get("variant", "none"))
                current_session = candidate
                result = candidate.get_state_dict()
            elif path in ("/api/move", "/api/ai_move", "/api/undo", "/api/resign"):
                if current_session is None:
                    raise APIError("no_game", 404)
                current_session.check_revision(body)
                if path == "/api/move":
                    if not all(k in body for k in ("fx", "fy", "tx", "ty")):
                        raise APIError("invalid_move")
                    result = current_session.apply_human_move(
                        body["fx"], body["fy"], body["tx"], body["ty"], body.get("promotion"))
                else:
                    result = getattr(current_session, path.rsplit("/", 1)[1])()
            else:
                raise APIError("not_found", 404)
            self._json_response(result)
        except APIError as exc:
            self._json_response({"error": exc.code}, exc.status)
        except (ValueError, TypeError) as exc:
            code = "invalid_variant" if str(exc) == "invalid_variant" else "invalid_request"
            self._json_response({"error": code}, 400)
        except Exception:
            import traceback
            traceback.print_exc()
            self._json_response({"error": "server_error"}, 500)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not 0 <= length <= 65536:
            raise APIError("invalid_request")
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        if not isinstance(body, dict):
            raise APIError("invalid_request")
        return body

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="Hybrid Chess game server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), HybridChessHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Hybrid Chess: {url}\nPress Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
