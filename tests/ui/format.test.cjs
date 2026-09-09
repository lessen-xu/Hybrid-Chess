const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const F = require("../../ui/shared/replay-format.js");
const root = path.resolve(__dirname, "../..");
const opening = [
  "10 c h e q g a e h c",
  "9 . . . . . . . . .",
  "8 . n . . . . . n .",
  "7 s . s . s . s . s",
  "6 . . . . . . . . .",
  "5 . . . . . . . . .",
  "4 . . . . . . . . .",
  "3 . . . . . . . . .",
  "2 P P P P P P P P P",
  "1 R N B Q K B N R .",
  "   a b c d e f g h i",
].join("\n");
test("replay parser supports JSON, arrays, BOM and pretty JSONL", () => {
  const game = { states_ascii: [opening], moves: [], result: "Draw" };
  for (const source of [
    JSON.stringify(game),
    JSON.stringify([game, game]),
    "\uFEFF" + JSON.stringify(game, null, 2),
    JSON.stringify(game) + "\n\n  " + JSON.stringify(game),
  ]) {
    assert.equal(F.parse(source)[0].states_ascii[0], opening);
  }
});
test("invalid recordings reject before replacing current data", () => {
  for (const data of [
    { states_ascii: [] },
    { states_ascii: [""] },
    { states_ascii: [opening.replace("q", "<script>")] },
    { states_ascii: [opening], moves: [{}] },
    null,
    [],
  ])
    assert.throws(() => F.parse(JSON.stringify(data)));
  assert.throws(() => F.parse("{broken"));
});
test("move coordinates handle row ten, promotions and old dataclass records", () => {
  assert.deepEqual(F.parseMove("a9-a10=N"), {
    from: { x: 0, y: 8 },
    to: { x: 0, y: 9 },
  });
  assert.deepEqual(
    F.parseMove("Move(fx=3, fy=9, tx=4, ty=8, promotion=None)"),
    { from: { x: 3, y: 9 }, to: { x: 4, y: 8 } },
  );
  assert.equal(F.parseMove("z1-a20"), null);
});
test("every board coordinate round trips in both orientations", () => {
  const context = vm.createContext({ window: {} });
  vm.runInContext(
    fs.readFileSync(path.join(root, "ui/shared/board-renderer.js"), "utf8"),
    context,
  );
  const B = context.window.BoardRenderer;
  for (const flipped of [false, true])
    for (let x = 0; x < 9; x++)
      for (let y = 0; y < 10; y++) {
        const p = B.boardToPixel(x, y, { flipped }),
          q = B.pixelToBoard(p.px, p.py, { flipped });
        assert.equal(q.x, x);
        assert.equal(q.y, y);
      }
  assert.equal(B.pixelToBoard(-100, -100), null);
  assert.equal(B.parseAsciiBoard(opening)[9][3], "q");
  assert.equal(B.XIANGQI_PIECES.q.ch, "后");
});
