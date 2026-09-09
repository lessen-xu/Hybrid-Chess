/* Pure parsing shared with the Node regression checks. */
(() => {
  function validBoard(ascii) {
    if (typeof ascii !== "string") return false;
    const rows = new Map();
    for (const line of ascii.split("\n")) {
      const m = line.match(/^\s*(\d+)\s+(.+)/);
      if (!m) continue;
      const y = Number(m[1]),
        cells = m[2].trim().split(/\s+/);
      if (
        y < 1 ||
        y > 10 ||
        rows.has(y) ||
        cells.length !== 9 ||
        cells.some((c) => !/^[KQRBNPgaehcnsq.]$/.test(c))
      )
        return false;
      rows.set(y, cells);
    }
    return rows.size === 10;
  }
  function parse(text) {
    let data;
    const raw = String(text)
      .replace(/^\uFEFF/, "")
      .trim();
    try {
      data = JSON.parse(raw);
    } catch {
      data = raw
        .split(/\r?\n/)
        .filter((line) => line.trim())
        .map((line) => JSON.parse(line));
    }
    const games = Array.isArray(data) ? data : [data];
    if (
      !games.length ||
      games.some(
        (g) =>
          !g ||
          !Array.isArray(g.states_ascii) ||
          !g.states_ascii.length ||
          !g.states_ascii.every(validBoard) ||
          (g.moves !== undefined &&
            (!Array.isArray(g.moves) ||
              g.moves.some((m) => typeof m !== "string"))),
      )
    )
      throw new Error("replayBad");
    return games;
  }
  function parseMove(notation) {
    if (typeof notation !== "string") return null;
    const m = notation.match(
      /^([a-i])(10|[1-9])-([a-i])(10|[1-9])(?:=[QRBN])?$/,
    );
    if (m)
      return {
        from: { x: m[1].charCodeAt(0) - 97, y: Number(m[2]) - 1 },
        to: { x: m[3].charCodeAt(0) - 97, y: Number(m[4]) - 1 },
      };
    // Older training recordings used the dataclass representation of Move.
    const legacy = notation.match(
      /^Move\(fx=(\d), fy=(\d), tx=(\d), ty=(\d), promotion=.*\)$/,
    );
    if (legacy && Number(legacy[1]) < 9 && Number(legacy[3]) < 9)
      return {
        from: { x: Number(legacy[1]), y: Number(legacy[2]) },
        to: { x: Number(legacy[3]), y: Number(legacy[4]) },
      };
    return null;
  }
  const api = { parse, parseMove, validBoard };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.HybridReplayFormat = api;
})();
