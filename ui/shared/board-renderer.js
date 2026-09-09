/* Pure SVG board shared by play and replay. No game state or network calls. */
(() => {
  const BOARD_W = 9,
    BOARD_H = 10,
    SVG_NS = "http://www.w3.org/2000/svg";
  const CHESS_PIECES = {
    K: { icon: "♔", name: "King" },
    Q: { icon: "♕", name: "Queen" },
    R: { icon: "♖", name: "Rook" },
    B: { icon: "♗", name: "Bishop" },
    N: { icon: "♘", name: "Knight" },
    P: { icon: "♙", name: "Pawn" },
  };
  const XIANGQI_PIECES = {
    g: { ch: "将", name: "General" },
    a: { ch: "士", name: "Advisor" },
    e: { ch: "象", name: "Elephant" },
    h: { ch: "马", name: "Horse" },
    c: { ch: "车", name: "Chariot" },
    n: { ch: "炮", name: "Cannon" },
    s: { ch: "卒", name: "Soldier" },
    q: { ch: "后", name: "Xiangqi queen" },
  };
  const defaultOpts = () => ({
    marginX: 40,
    marginY: 40,
    cellW: 56,
    cellH: 56,
    flipped: false,
    variant: {},
  });
  function boardToPixel(bx, by, opts) {
    const o = { ...defaultOpts(), ...opts };
    return {
      px: o.marginX + (o.flipped ? 8 - bx : bx) * o.cellW,
      py: o.marginY + (o.flipped ? by : 9 - by) * o.cellH,
    };
  }
  function pixelToBoard(px, py, opts) {
    const o = { ...defaultOpts(), ...opts };
    const x = Math.round((px - o.marginX) / o.cellW),
      y = Math.round((py - o.marginY) / o.cellH);
    if (x < 0 || x >= BOARD_W || y < 0 || y >= BOARD_H) return null;
    return { x: o.flipped ? 8 - x : x, y: o.flipped ? y : 9 - y };
  }
  function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs))
      el.setAttribute(key, value);
    return el;
  }
  function clearGroup(svg, id) {
    let group = svg.querySelector("#" + id);
    if (!group) {
      group = svgEl("g", { id });
      svg.append(group);
    }
    group.replaceChildren();
    return group;
  }
  function text(parent, label, attrs) {
    const el = svgEl("text", attrs);
    el.textContent = label;
    parent.append(el);
    return el;
  }
  function drawBoard(svg, opts) {
    const o = { ...defaultOpts(), ...opts };
    svg.setAttribute(
      "viewBox",
      "0 0 " +
        (o.marginX * 2 + 8 * o.cellW) +
        " " +
        (o.marginY * 2 + 9 * o.cellH),
    );
    // Stable layer order keeps move markers below pieces and legal rings above them.
    for (const id of [
      "boardGrid",
      "hl-move",
      "hl-check",
      "pieces",
      "hl-selected",
      "hl-legal-move",
      "hl-focus",
      "coordLabels",
    ]) {
      let g = svg.querySelector("#" + id);
      if (!g) g = svgEl("g", { id });
      svg.append(g);
    }
    const g = clearGroup(svg, "boardGrid");
    const width = o.marginX * 2 + 8 * o.cellW,
      height = o.marginY * 2 + 9 * o.cellH;
    g.append(svgEl("rect", { width, height, fill: "#e8d7b8" }));
    const line = (a, b, extra = {}) => {
      const p = boardToPixel(...a, o),
        q = boardToPixel(...b, o);
      g.append(
        svgEl("line", {
          x1: p.px,
          y1: p.py,
          x2: q.px,
          y2: q.py,
          stroke: "#8d7b5d",
          "stroke-width": 1,
          ...extra,
        }),
      );
    };
    for (let y = 0; y < 10; y++) line([0, y], [8, y]);
    for (let x = 0; x < 9; x++) {
      if (x === 0 || x === 8) line([x, 0], [x, 9]);
      else {
        line([x, 0], [x, 4]);
        line([x, 5], [x, 9]);
      }
    }
    const palace = (a, b) => {
      line([3, a], [5, b], { stroke: "#aa9672" });
      line([5, a], [3, b], { stroke: "#aa9672" });
    };
    palace(7, 9);
    if (o.variant?.chess_palace) palace(0, 2);
    const riverY = o.marginY + 4.5 * o.cellH;
    text(g, "楚 河", {
      x: o.marginX + 1.5 * o.cellW,
      y: riverY + 7,
      fill: "#a18b66",
      "font-size": 21,
      "font-family": "Georgia, SimSun, serif",
      "text-anchor": "middle",
    });
    text(g, "汉 界", {
      x: o.marginX + 6.5 * o.cellW,
      y: riverY + 7,
      fill: "#a18b66",
      "font-size": 21,
      "font-family": "Georgia, SimSun, serif",
      "text-anchor": "middle",
    });
    for (const [x, y] of [
      [1, 7],
      [7, 7],
      [0, 6],
      [2, 6],
      [4, 6],
      [6, 6],
      [8, 6],
    ]) {
      const p = boardToPixel(x, y, o);
      for (const dx of [-1, 1])
        for (const dy of [-1, 1]) {
          const visualX = o.flipped ? 8 - x : x;
          if (visualX + dx < 0 || visualX + dx > 8) continue;
          const attrs = { stroke: "#a5906b", "stroke-width": 1 };
          g.append(
            svgEl("line", {
              x1: p.px + dx * 4,
              y1: p.py + dy * 4,
              x2: p.px + dx * 10,
              y2: p.py + dy * 4,
              ...attrs,
            }),
          );
          g.append(
            svgEl("line", {
              x1: p.px + dx * 4,
              y1: p.py + dy * 4,
              x2: p.px + dx * 4,
              y2: p.py + dy * 10,
              ...attrs,
            }),
          );
        }
    }
    g.append(
      svgEl("rect", {
        x: o.marginX - 7,
        y: o.marginY - 7,
        width: 8 * o.cellW + 14,
        height: 9 * o.cellH + 14,
        fill: "none",
        stroke: "#8d7958",
        "stroke-width": 1.7,
      }),
    );
  }
  function parseAsciiBoard(ascii) {
    const grid = Array.from({ length: 10 }, () => Array(9).fill(null));
    for (const line of String(ascii || "").split("\n")) {
      const match = line.match(/^\s*(\d+)\s+(.+)/);
      if (!match) continue;
      const y = Number(match[1]) - 1;
      if (y < 0 || y > 9) continue;
      match[2]
        .trim()
        .split(/\s+/)
        .slice(0, 9)
        .forEach((ch, x) => {
          if (ch !== ".") grid[y][x] = ch;
        });
    }
    return grid;
  }
  function drawPieces(svg, board, opts) {
    const o = { ...defaultOpts(), ...opts },
      g = clearGroup(svg, "pieces");
    const grid = typeof board === "string" ? parseAsciiBoard(board) : board;
    for (let y = 0; y < 10; y++)
      for (let x = 0; x < 9; x++) {
        const ch = grid?.[y]?.[x];
        if (!ch) continue;
        const p = boardToPixel(x, y, o);
        drawSinglePiece(g, ch, p.px, p.py, x, y);
      }
  }
  function drawSinglePiece(parent, ch, px, py, bx, by) {
    const chess = !!CHESS_PIECES[ch];
    const g = svgEl("g", {
      "data-bx": bx,
      "data-by": by,
      class: "piece " + (chess ? "chess-piece" : "xiangqi-piece"),
    });
    g.append(svgEl("circle", { cx: px, cy: py + 2, r: 22, fill: "#675c4233" }));
    g.append(
      svgEl("circle", {
        cx: px,
        cy: py,
        r: 22,
        fill: chess ? "#fffcf0" : "#a34434",
        stroke: chess ? "#baa982" : "#863729",
        "stroke-width": 1.5,
      }),
    );
    g.append(
      svgEl("circle", {
        cx: px,
        cy: py,
        r: 18.5,
        fill: "none",
        stroke: chess ? "#e8ddc1" : "#d08b74",
        "stroke-width": 0.8,
      }),
    );
    const label = chess ? CHESS_PIECES[ch].icon : XIANGQI_PIECES[ch]?.ch || ch;
    text(g, label, {
      x: px,
      y: py + 1,
      "font-size": chess ? 31 : 23,
      "font-family": chess
        ? '"Segoe UI Symbol", "DejaVu Sans", serif'
        : '"KaiTi", "STKaiti", "SimSun", serif',
      fill: chess ? "#303b32" : "#fff0d4",
      "text-anchor": "middle",
      "dominant-baseline": "central",
      "pointer-events": "none",
      "font-weight": chess ? "normal" : "bold",
    });
    const title = svgEl("title");
    title.textContent =
      (window.HybridI18n?.pieceName(ch) || label) +
      " · " +
      "abcdefghi"[bx] +
      (by + 1);
    g.append(title);
    parent.append(g);
  }
  function highlightSquares(svg, squares, className, opts) {
    const o = { ...defaultOpts(), ...opts },
      g = clearGroup(svg, "hl-" + className);
    for (const sq of squares) {
      const p = boardToPixel(sq.x, sq.y, o);
      const ring =
        sq.capture || ["selected", "check", "focus"].includes(className);
      g.append(
        svgEl("circle", {
          cx: p.px,
          cy: p.py,
          r: ring ? 25 : 7,
          fill: ring ? "none" : sq.fill || "#2d59487a",
          stroke: sq.stroke || (className === "check" ? "#a04435" : "#2d5948"),
          "stroke-width": ring ? 2.5 : 1,
          "stroke-dasharray": className === "focus" ? "4 3" : "none",
          "pointer-events": "none",
        }),
      );
    }
  }
  function highlightMove(svg, from, to, opts) {
    const g = clearGroup(svg, "hl-move");
    for (const sq of [from, to]) {
      const p = boardToPixel(sq.x, sq.y, opts);
      g.append(
        svgEl("rect", {
          x: p.px - 26,
          y: p.py - 26,
          width: 52,
          height: 52,
          rx: 5,
          fill: "#547c5540",
          "pointer-events": "none",
        }),
      );
    }
  }
  function clearHighlights(svg, className) {
    if (className) svg.querySelector("#hl-" + className)?.replaceChildren();
    else
      svg.querySelectorAll('[id^="hl-"]').forEach((g) => g.replaceChildren());
  }
  function drawCoordLabels(svg, opts) {
    const o = { ...defaultOpts(), ...opts },
      g = clearGroup(svg, "coordLabels");
    const attrs = {
      fill: "#857252",
      "font-size": 10,
      "font-family": "Consolas, monospace",
      "text-anchor": "middle",
    };
    for (let x = 0; x < 9; x++) {
      const p = boardToPixel(x, o.flipped ? 9 : 0, o);
      text(g, "abcdefghi"[x], {
        x: p.px,
        y: o.marginY + 9 * o.cellH + 27,
        ...attrs,
      });
    }
    for (let y = 0; y < 10; y++) {
      const p = boardToPixel(0, y, o);
      text(g, String(y + 1), { x: 15, y: p.py + 4, ...attrs });
    }
  }
  window.BoardRenderer = {
    BOARD_W,
    BOARD_H,
    SVG_NS,
    CHESS_PIECES,
    XIANGQI_PIECES,
    defaultOpts,
    boardToPixel,
    pixelToBoard,
    drawBoard,
    drawPieces,
    drawSinglePiece,
    parseAsciiBoard,
    highlightSquares,
    highlightMove,
    clearHighlights,
    drawCoordLabels,
  };
})();
