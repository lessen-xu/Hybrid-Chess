"""Regression coverage for the local play API and rule-aware sessions."""
import json
from threading import Thread
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import hybrid.server as server
import hybrid.core.rules as rules
from hybrid.core.board import Board, initial_board
from hybrid.core.config import VariantConfig
from hybrid.core.render import render_board
from hybrid.core.types import Side, PieceKind as K, Piece, Move
from hybrid.server import APIError, GameSession
from hybrid.web_variants import PRESETS, RULES, catalog, parse_variant, preview


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p["id"])
def test_preset_preview_is_the_actual_opening(preset):
    data = preview(preset["id"])
    session = GameSession("chess", "random", preset["id"])
    state = session.get_state_dict()
    assert state["board_ascii"] == data["board_ascii"]
    assert state["variant"] == data["variant"]
    assert state["legal_moves"]
    key = rules.board_hash(session.env.state.board, Side.CHESS)
    assert session.env.state.repetition[key] == 1


@pytest.mark.parametrize("field", [r["key"] for r in RULES])
def test_each_rule_reaches_the_environment(field):
    default = VariantConfig().to_dict()
    default[field] = not default[field]
    config = parse_variant(default)
    session = GameSession("chess", "random", default)
    assert session.env.variant == config
    assert session.get_state_dict()["board_ascii"] == render_board(initial_board(config))


def test_catalog_and_normalization():
    assert len(catalog()["presets"]) == 6
    for entry in catalog()["rules"] + catalog()["presets"]:
        assert entry["name"]["zh"] and entry["name"]["en"]
        assert entry["description"]["zh"] and entry["description"]["en"]
    v = parse_variant({"extra_pawn_i_file": False, "no_promotion": True, "no_queen_promotion": True})
    assert v.remove_extra_pawn and v.extra_pawn_i_file
    assert v.no_promotion and not v.no_queen_promotion


@pytest.mark.parametrize("value", [None, [], 1, "unknown", {"unknown": True}, {"no_queen": 1}, {"no_queen": "true"}])
def test_invalid_variant_is_rejected(value):
    with pytest.raises(ValueError, match="invalid_variant"):
        parse_variant(value)


def test_preview_never_changes_the_active_game():
    session = GameSession("chess", "random", "pk")
    before = session.get_state_dict()
    active = rules._active_variant
    preview("none")
    assert rules._active_variant is active
    assert session.get_state_dict() == before


def human_move(session):
    return session.apply_human_move(**session.get_state_dict()["legal_moves"][0])


@pytest.mark.parametrize("side", ["chess", "xiangqi"])
def test_turn_order_undo_and_authoritative_history(side):
    session = GameSession(side, "random")
    if side == "xiangqi":
        with pytest.raises(APIError, match="wrong_turn"):
            human_move(session)
        session.ai_move()
        with pytest.raises(APIError, match="nothing_to_undo"):
            session.undo()
    initial = session.get_state_dict()
    human_move(session)
    with pytest.raises(APIError, match="wrong_turn"):
        human_move(session)
    session.ai_move()
    with pytest.raises(APIError, match="wrong_turn"):
        session.ai_move()
    restored = session.undo()
    assert restored["board_ascii"] == initial["board_ascii"]
    assert restored["moves"] == initial["moves"]
    assert restored["side_to_move"] == side
    assert restored["ply"] == (1 if side == "xiangqi" else 0)
    assert not restored["can_undo"]
    assert restored["revision"] > initial["revision"]


def test_undo_after_human_move_before_ai_response():
    session = GameSession("chess", "random")
    human_move(session)
    state = session.undo()
    assert state["ply"] == 0 and state["moves"] == []


@pytest.mark.parametrize("side", ["chess", "xiangqi"])
def test_resignation_persists_and_stops_actions(side):
    session = GameSession(side, "random")
    result = session.resign()
    assert session.get_state_dict() == result
    assert result["game_over"]
    assert result["result_code"] == ("xiangqi_win" if side == "chess" else "chess_win")
    assert result["legal_moves"] == [] and not result["can_undo"]
    for operation in (session.ai_move, session.undo, session.resign):
        with pytest.raises(APIError, match="game_over"):
            operation()


def position(session, entries, side=Side.CHESS):
    board = Board.empty()
    for x, y, kind, owner in entries:
        board.set(x, y, Piece(kind, owner))
    session.env.reset_from_board(board, side)
    session.history = [session.env.state.clone()]
    return session.get_state_dict()


def promotion_position(flags):
    session = GameSession("chess", "random", flags)
    position(session, [(4, 0, K.KING, Side.CHESS), (4, 9, K.GENERAL, Side.XIANGQI),
                       (4, 5, K.PAWN, Side.CHESS), (0, 8, K.PAWN, Side.CHESS)])
    return session


@pytest.mark.parametrize("flags,expected", [
    ({}, {"QUEEN", "ROOK", "BISHOP", "KNIGHT"}),
    ({"no_queen_promotion": True}, {"ROOK", "BISHOP", "KNIGHT"}),
    ({"no_promotion": True}, {None}),
    ({"no_promotion": True, "no_queen_promotion": True}, {None}),
])
def test_promotion_options_and_execution(flags, expected):
    session = promotion_position(flags)
    moves = [m for m in session.get_state_dict()["legal_moves"] if (m["fx"], m["fy"], m["tx"], m["ty"]) == (0, 8, 0, 9)]
    assert {m["promotion"] for m in moves} == expected
    result = session.apply_human_move(**moves[0])
    piece = session.env.state.board.get(0, 9)
    assert piece.kind == (K[moves[0]["promotion"]] if moves[0]["promotion"] else K.PAWN)
    assert result["moves"][-1]["promotion"] == moves[0]["promotion"]


def test_illegal_queen_promotion_is_rejected():
    session = promotion_position({"no_queen_promotion": True})
    with pytest.raises(APIError, match="illegal_move"):
        session.apply_human_move(0, 8, 0, 9, "QUEEN")
    assert session.env.state.ply == 0


def test_palace_and_knight_block_change_actual_moves():
    entries = [(4, 2, K.KING, Side.CHESS), (4, 9, K.GENERAL, Side.XIANGQI),
               (4, 5, K.PAWN, Side.CHESS), (1, 3, K.KNIGHT, Side.CHESS),
               (2, 3, K.PAWN, Side.CHESS)]
    standard = GameSession("chess", "random")
    original = position(standard, entries)["legal_moves"]
    restricted = GameSession("chess", "random", "pk")
    altered = position(restricted, entries)["legal_moves"]
    def contains(moves, xy):
        return any((m["fx"], m["fy"], m["tx"], m["ty"]) == xy for m in moves)
    assert contains(original, (4, 2, 4, 3)) and not contains(altered, (4, 2, 4, 3))
    assert contains(original, (1, 3, 3, 4)) and not contains(altered, (1, 3, 3, 4))
    # Switching between instances still re-applies their own rule configuration.
    assert standard.get_state_dict()["legal_moves"] == original


def test_xiangqi_queen_ownership_and_flying_general_switch():
    s = GameSession("xiangqi", "random", "xq_queen")
    piece = s.env.state.board.get(3, 9)
    assert piece.kind == K.XQ_QUEEN and piece.side == Side.XIANGQI
    assert "q" in s.get_state_dict()["board_ascii"]
    for enabled in (True, False):
        s = GameSession("xiangqi", "random", {"flying_general": enabled})
        state = position(s, [(4, 0, K.KING, Side.CHESS), (4, 9, K.GENERAL, Side.XIANGQI)], Side.XIANGQI)
        assert any((m["tx"], m["ty"]) == (4, 0) for m in state["legal_moves"]) is enabled


@pytest.mark.parametrize("extra,reason", [
    ([(0, 5, K.CHARIOT, Side.XIANGQI), (1, 5, K.CHARIOT, Side.XIANGQI)], "checkmate"),
    ([(1, 8, K.CHARIOT, Side.XIANGQI), (8, 1, K.CHARIOT, Side.XIANGQI)], "stalemate"),
])
def test_terminal_state_has_no_legal_moves(extra, reason):
    s = GameSession("chess", "random")
    state = position(s, [(0, 0, K.KING, Side.CHESS), (4, 9, K.GENERAL, Side.XIANGQI)] + extra)
    assert state["game_over"] and state["reason_code"] == reason
    assert state["result_code"] == "xiangqi_win" and not state["legal_moves"]


@pytest.mark.parametrize("cause", ["move_limit", "repetition"])
def test_draw_status_persists(cause):
    s = GameSession("chess", "random")
    if cause == "move_limit":
        s.env.state.ply = 400
    else:
        s.env.state.repetition[rules.board_hash(s.env.state.board, Side.CHESS)] = 3
    state = s.get_state_dict()
    assert state["result_code"] == "draw" and state["reason_code"] == cause
    assert state == s.get_state_dict()


@pytest.fixture
def http_api():
    server.current_session = None
    httpd = HTTPServer(("127.0.0.1", 0), server.HybridChessHandler)
    worker = Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    def call(path, body=None, raw=None):
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        request = Request(f"http://127.0.0.1:{httpd.server_port}/api/{path}", data=payload,
                          headers={"Content-Type": "application/json"})
        try:
            response = urlopen(request, timeout=15)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)
    yield call
    httpd.shutdown()
    httpd.server_close()
    worker.join(timeout=2)
    server.current_session = None


def test_http_open_preview_move_stale_request_and_refresh(http_api):
    assert http_api("state") == (404, {"error": "no_game"})
    assert len(http_api("variants")[1]["presets"]) == 6
    assert len(http_api("agents")[1]["agents"]) == 5
    status, state = http_api("new", {"human_side": "chess", "ai_agent": "random", "variant": "pk"})
    assert status == 200
    assert http_api("preview", {"variant": "xq_queen"})[0] == 200
    assert http_api("state")[1] == state
    guard = {"session_id": state["session_id"], "revision": state["revision"]}
    move = {**state["legal_moves"][0], **guard}
    status, after = http_api("move", move)
    assert status == 200 and after["ply"] == 1
    assert http_api("move", move) == (409, {"error": "stale_state"})
    assert http_api("ai_move", guard) == (409, {"error": "stale_state"})
    assert http_api("state")[1]["moves"] == after["moves"]
    status, ai = http_api("ai_move", {"session_id": after["session_id"], "revision": after["revision"]})
    assert status == 200 and ai["ply"] == 2
    assert http_api("resign", {})[1]["game_over"]
    assert http_api("state")[1]["reason_code"] == "resignation"


@pytest.mark.parametrize("path,body,code", [
    ("new", {"human_side": "bad"}, "invalid_side"),
    ("new", {"ai_agent": "ab_d999"}, "invalid_agent"),
    ("new", {"variant": {"no_queen": 1}}, "invalid_variant"),
    ("preview", {"variant": {"missing": True}}, "invalid_variant"),
    ("new", [], "invalid_request"),
])
def test_http_bad_configuration(http_api, path, body, code):
    assert http_api(path, body) == (400, {"error": code})


def test_http_malformed_json_does_not_break_server(http_api):
    assert http_api("new", raw=b"{broken") == (400, {"error": "invalid_request"})
    assert http_api("new", {})[0] == 200
    assert http_api("move", {"fx": True, "fy": 1, "tx": 0, "ty": 2}) == (400, {"error": "invalid_move"})
    assert http_api("move", {}) == (400, {"error": "invalid_move"})
    assert http_api("state")[1]["ply"] == 0
