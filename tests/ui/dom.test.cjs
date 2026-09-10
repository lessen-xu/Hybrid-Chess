/* DOM integration tests, not a substitute for visual browser inspection. */
const { test, before, after, beforeEach } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { JSDOM, VirtualConsole } = require("jsdom");
const root = path.resolve(__dirname, "../..");
let origin, python;
const windows = new Set();
const fixtureServer = String.raw`
import json
from http.server import HTTPServer
import hybrid.server as app
from hybrid.core.board import Board
from hybrid.core.types import Piece, PieceKind, Side
class Handler(app.HybridChessHandler):
    def log_message(self, *args):
        pass
    def do_POST(self):
        if self.path == "/__reset":
            app.current_session = None
            self._json_response({})
        elif self.path == "/__position":
            body = self._read_body()
            s = app.GameSession("chess", "random", body.get("variant", {}))
            b = Board.empty()
            for x,y,kind,side in body["pieces"]:
                b.set(x,y,Piece(PieceKind[kind],Side[side]))
            s.env.reset_from_board(b, Side.CHESS)
            s.history = [s.env.state.clone()]
            app.current_session = s
            self._json_response(s.get_state_dict())
        else:
            super().do_POST()
httpd = HTTPServer(("127.0.0.1", 0), Handler)
print("http://127.0.0.1:" + str(httpd.server_port), flush=True)
httpd.serve_forever()
`;
before(async () => {
  python = spawn(process.env.PYTHON || "python", ["-u", "-c", fixtureServer], {
    cwd: root,
    stdio: ["ignore", "pipe", "pipe"],
  });
  origin = await new Promise((resolve, reject) => {
    let output = "",
      errors = "";
    const timeout = setTimeout(
      () => reject(new Error("Fixture server did not start: " + errors)),
      10000,
    );
    python.stderr.on("data", (data) => (errors += data));
    python.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    python.once("exit", (code) => {
      if (!output) {
        clearTimeout(timeout);
        reject(new Error("Fixture server exited " + code + ": " + errors));
      }
    });
    python.stdout.on("data", (data) => {
      output += data;
      if (output.includes("\n")) {
        clearTimeout(timeout);
        resolve(output.trim().split(/\r?\n/)[0]);
      }
    });
  });
});
after(() => {
  for (const window of windows) window.close();
  python?.kill();
});
async function api(route, body) {
  const response = await fetch(
    origin + route,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  assert.equal(response.status, 200, JSON.stringify(data));
  return data;
}
beforeEach(async () => {
  for (const w of windows) w.close();
  windows.clear();
  await api("/__reset", {});
});
async function wait(predicate, message = "DOM did not settle") {
  const deadline = Date.now() + 7000;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await new Promise((r) => setTimeout(r, 15));
  }
  throw new Error(message);
}
async function domFor(route = "play", intercept, waitReady = true) {
  const failures = [];
  const console = new VirtualConsole();
  console.on("jsdomError", (error) => failures.push(error));
  const dom = new JSDOM(
    fs.readFileSync(path.join(root, "ui", route, "index.html"), "utf8"),
    {
      url: origin + "/" + route + "/",
      runScripts: "outside-only",
      pretendToBeVisual: true,
      virtualConsole: console,
      beforeParse(window) {
        window.fetch = (url, options) =>
          intercept
            ? intercept(new URL(url, origin).href, options)
            : fetch(new URL(url, origin), options);
        window.matchMedia = () => ({
          matches: true,
          addEventListener() {},
          removeEventListener() {},
        });
        window.HTMLElement.prototype.scrollIntoView = function () {};
        window.HTMLDialogElement.prototype.showModal = function () {
          this.open = true;
          this.returnValue = "";
        };
        window.HTMLDialogElement.prototype.close = function (value = "") {
          this.returnValue = value;
          this.open = false;
          this.dispatchEvent(new window.Event("close"));
        };
        window.SVGElement.prototype.getScreenCTM = () => ({
          inverse() {
            return this;
          },
        });
        window.DOMPoint = class {
          constructor(x, y) {
            this.x = x;
            this.y = y;
          }
          matrixTransform() {
            return this;
          }
        };
      },
    },
  );
  windows.add(dom.window);
  for (const script of dom.window.document.querySelectorAll("script[src]")) {
    const relative = new URL(script.src).pathname.replace(/^\//, "");
    dom.window.eval(fs.readFileSync(path.join(root, "ui", relative), "utf8"));
  }
  if (route === "play" && waitReady)
    await wait(() => {
      const d = dom.window.document;
      return (
        d.querySelectorAll(".preset").length === 6 &&
        (!d.getElementById("btnStart").disabled ||
          (!d.getElementById("btnNewGame").hidden &&
            !d.getElementById("btnNewGame").disabled))
      );
    });
  assert.deepEqual(failures, []);
  return { window: dom.window, d: dom.window.document, failures };
}
function chooseMove(window, move, flipped = false) {
  const svg = window.document.getElementById("boardSvg"),
    B = window.BoardRenderer;
  for (const [x, y] of [
    [move.fx, move.fy],
    [move.tx, move.ty],
  ]) {
    const p = B.boardToPixel(x, y, { flipped });
    svg.dispatchEvent(
      new window.MouseEvent("click", {
        bubbles: true,
        clientX: p.px,
        clientY: p.py,
      }),
    );
  }
}
async function settled(d, ply) {
  await wait(async () => {
    const response = await fetch(origin + "/api/state");
    if (response.status === 404) return false;
    const s = await response.json();
    return s.ply === ply && !d.getElementById("btnNewGame").disabled;
  });
}
test("an unavailable server leaves controls safe and reconnect restores setup", async () => {
  let offline = true;
  const { d, failures } = await domFor(
    "play",
    (url, options) => {
      if (offline) return Promise.reject(new Error("offline"));
      return fetch(url, options);
    },
    false,
  );
  await wait(() => !d.getElementById("errorNotice").hidden);
  assert.equal(d.getElementById("btnStart").disabled, true);
  assert.equal(d.querySelector('[data-side="xiangqi"]').disabled, true);
  d.querySelector("[data-language]").click();
  assert.equal(d.documentElement.lang, "en");
  offline = false;
  d.getElementById("btnRetry").click();
  await wait(() => !d.getElementById("btnStart").disabled);
  assert.equal(d.getElementById("errorNotice").hidden, true);
  assert.deepEqual(failures, []);
});

test("presets, custom rule preview, promotion dependencies and language", async () => {
  const { window, d } = await domFor();
  assert.equal(d.documentElement.lang, "zh-CN");
  assert.equal(d.querySelectorAll("[data-rule]").length, 12);
  d.querySelector('[data-preset="xq_queen"]').click();
  await wait(
    () =>
      !d.getElementById("btnStart").disabled &&
      d
        .querySelector('#pieces [data-bx="3"][data-by="9"] title')
        .textContent.includes("皇后"),
  );
  const before = d
    .getElementById("boardSvg")
    .querySelector("#pieces").textContent;
  const noPromo = d.querySelector('[data-rule="no_promotion"]');
  noPromo.checked = true;
  noPromo.dispatchEvent(new window.Event("change"));
  await wait(() => !d.getElementById("btnStart").disabled);
  assert.equal(
    d.querySelector('[data-rule="no_queen_promotion"]').disabled,
    true,
  );
  const leftBishop = d.querySelector('[data-rule="no_bishop"]');
  leftBishop.checked = true;
  leftBishop.dispatchEvent(new window.Event("change"));
  await wait(() => !d.getElementById("btnStart").disabled);
  assert.equal(d.querySelector('#pieces [data-bx="2"][data-by="0"]'), null);
  assert.ok(d.querySelector('#pieces [data-bx="5"][data-by="0"]'));
  d.querySelector("[data-language]").click();
  assert.equal(d.documentElement.lang, "en");
  assert.equal(
    d.getElementById("btnStart").textContent.trim(),
    "Start playing→",
  );
  assert.equal(
    JSON.parse(window.localStorage.getItem("hybrid.language")),
    "en",
  );
  assert.match(
    d.querySelector('#pieces [data-bx="3"][data-by="9"] title').textContent,
    /Xiangqi queen/,
  );
  assert.ok(before);
});
test("Xiangqi opening, human move, AI response, undo and refresh", async () => {
  const { window, d } = await domFor();
  d.querySelector('[data-preset="pk_xq_queen"]').click();
  await wait(() => !d.getElementById("btnStart").disabled);
  d.querySelector('[data-side="xiangqi"]').click();
  d.getElementById("aiSelect").value = "random";
  d.getElementById("aiSelect").dispatchEvent(new window.Event("change"));
  d.getElementById("btnStart").click();
  await settled(d, 1);
  assert.equal(d.getElementById("btnUndo").disabled, true);
  let state = await api("/api/state");
  assert.equal(state.human_side, "xiangqi");
  assert.equal(state.variant.xq_queen, true);
  chooseMove(window, state.legal_moves[0], true);
  await settled(d, 3);
  d.getElementById("btnUndo").click();
  await settled(d, 1);
  assert.equal(
    d.getElementById("moveHistory").querySelectorAll(".move-row").length,
    1,
  );
  const recorded = await api("/api/state");
  const refreshed = await domFor();
  assert.equal(refreshed.d.getElementById("sessionPanel").hidden, false);
  assert.equal((await api("/api/state")).board_ascii, recorded.board_ascii);
  assert.match(
    refreshed.d.getElementById("sessionRuleName").textContent,
    /象棋后/,
  );
});
test("duplicate clicks are serialized; flip and language remain usable while waiting", async () => {
  await api("/api/new", { human_side: "chess", ai_agent: "random" });
  let release,
    moveCalls = 0;
  const gate = new Promise((resolve) => (release = resolve));
  const { window, d } = await domFor("play", async (url, options) => {
    const response = await fetch(url, options);
    if (url.endsWith("/api/move")) {
      moveCalls++;
      await gate;
    }
    return response;
  });
  const state = await api("/api/state"),
    move = state.legal_moves[0];
  chooseMove(window, move);
  chooseMove(window, move);
  await wait(() => moveCalls === 1);
  assert.equal(d.getElementById("btnUndo").disabled, true);
  assert.equal(d.getElementById("btnNewGame").disabled, true);
  d.querySelector("[data-language]").click();
  d.getElementById("btnFlip").click();
  assert.equal(d.documentElement.lang, "en");
  assert.equal(d.getElementById("avatarBottom").textContent, "将");
  release();
  await settled(d, 2);
  assert.equal(moveCalls, 1);
});
test("a lost move response recovers without submitting the human move twice", async () => {
  await api("/api/new", { human_side: "chess", ai_agent: "random" });
  let moveCalls = 0;
  const { window, d } = await domFor("play", async (url, options) => {
    const response = await fetch(url, options);
    if (url.endsWith("/api/move")) {
      moveCalls++;
      throw new Error("Simulated lost response");
    }
    return response;
  });
  const state = await api("/api/state");
  chooseMove(window, state.legal_moves[0]);
  await wait(
    () =>
      !d.getElementById("errorNotice").hidden &&
      !d.getElementById("btnRetry").disabled,
  );
  assert.equal((await api("/api/state")).ply, 1);
  d.getElementById("btnRetry").click();
  await settled(d, 2);
  assert.equal(moveCalls, 1);
  assert.equal(d.getElementById("errorNotice").hidden, true);
});
test("promotion dialog contains only legal choices and sends the chosen piece", async () => {
  await api("/__position", {
    variant: { no_queen_promotion: true },
    pieces: [
      [4, 0, "KING", "CHESS"],
      [4, 9, "GENERAL", "XIANGQI"],
      [4, 5, "PAWN", "CHESS"],
      [0, 8, "PAWN", "CHESS"],
    ],
  });
  const { window, d } = await domFor();
  chooseMove(window, { fx: 0, fy: 8, tx: 0, ty: 9 });
  assert.equal(d.getElementById("promotionDialog").open, true);
  assert.equal(d.querySelectorAll(".promotion-choice").length, 3);
  assert.ok(!d.getElementById("promotionOptions").textContent.includes("皇后"));
  d.querySelector(".promotion-choice").click();
  await settled(d, 2);
  assert.equal((await api("/api/state")).moves[0].promotion, "ROOK");
});
test("keyboard play, resignation and new-game confirmation preserve state", async () => {
  await api("/api/new", { human_side: "chess", ai_agent: "random" });
  const { window, d } = await domFor();
  const svg = d.getElementById("boardSvg");
  for (const key of ["ArrowUp", "Enter", "ArrowUp", "Enter"])
    svg.dispatchEvent(
      new window.KeyboardEvent("keydown", { key, bubbles: true }),
    );
  await settled(d, 2);
  d.getElementById("btnNewGame").click();
  await wait(() => !d.getElementById("btnStart").disabled);
  const old = (await api("/api/state")).session_id;
  d.getElementById("btnStart").click();
  assert.equal(d.getElementById("confirmDialog").open, true);
  d.getElementById("confirmCancel").click();
  assert.equal((await api("/api/state")).session_id, old);
  d.getElementById("btnCancelSetup").click();
  d.getElementById("btnResign").click();
  d.getElementById("confirmAccept").click();
  await wait(() => !d.getElementById("resultPanel").hidden);
  assert.equal((await api("/api/state")).reason_code, "resignation");
  assert.equal(d.getElementById("btnUndo").disabled, true);
});
test("replay loads old JSONL, steps, switches language and rejects broken data", async () => {
  const initial = await api("/api/new", {
    human_side: "chess",
    ai_agent: "random",
    variant: "xq_queen",
  });
  const moved = await api("/api/move", initial.legal_moves[0]);
  const game = {
    states_ascii: [initial.board_ascii, moved.board_ascii],
    moves: [moved.moves[0].notation],
    result: "Draw",
  };
  const { window, d } = await domFor("replay");
  const input = d.getElementById("fileInput");
  async function load(text) {
    Object.defineProperty(input, "files", {
      configurable: true,
      value: [
        { name: "game.jsonl", size: text.length, text: async () => text },
      ],
    });
    input.dispatchEvent(new window.Event("change"));
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  await load(JSON.stringify(game) + "\n" + JSON.stringify(game));
  assert.equal(d.getElementById("gameSelect").options.length, 2);
  d.getElementById("btnNext").click();
  assert.equal(d.getElementById("stepSlider").value, "1");
  d.querySelector("[data-language]").click();
  assert.equal(d.documentElement.lang, "en");
  assert.equal(d.getElementById("stepSlider").value, "1");
  assert.equal(d.getElementById("infoResult").textContent, "Draw");
  await load("{broken");
  assert.equal(d.getElementById("errorNotice").hidden, false);
  assert.equal(d.getElementById("stepSlider").value, "1");
  d.getElementById("btnFirst").click();
  d.getElementById("btnAuto").click();
  await wait(() => d.getElementById("stepSlider").value === "1");
  assert.equal(d.getElementById("btnAuto").textContent, "▷");
  await load(JSON.stringify({ ...game, result: "chess_win" }));
  assert.equal(d.getElementById("infoResult").textContent, window.HybridI18n.t("chess_win"));
  d.querySelector("[data-language]").click();
  assert.equal(d.getElementById("infoResult").textContent, window.HybridI18n.t("chess_win"));
});
