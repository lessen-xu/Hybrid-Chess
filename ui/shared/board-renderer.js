/**
 * Hybrid Chess — Shared Board Renderer
 *
 * Pure SVG rendering for the 9×10 board.  No game logic, no UI state.
 * Both play/ and replay/ import this module.
 */

/* ───── constants ───── */
const BOARD_W = 9, BOARD_H = 10;
const SVG_NS = 'http://www.w3.org/2000/svg';

/* ───── piece definitions ───── */
const CHESS_PIECES = {
  K: { icon: '♔', name: 'King' },
  Q: { icon: '♕', name: 'Queen' },
  R: { icon: '♖', name: 'Rook' },
  B: { icon: '♗', name: 'Bishop' },
  N: { icon: '♘', name: 'Knight' },
  P: { icon: '♙', name: 'Pawn' },
};

const XIANGQI_PIECES = {
  g: { ch: '將', name: 'General' },
  a: { ch: '士', name: 'Advisor' },
  e: { ch: '象', name: 'Elephant' },
  h: { ch: '馬', name: 'Horse' },
  c: { ch: '車', name: 'Chariot' },
  n: { ch: '砲', name: 'Cannon' },
  s: { ch: '卒', name: 'Soldier' },
};

/* ───── default options ───── */
function defaultOpts() {
  return {
    marginX: 40, marginY: 40,
    cellW: 56, cellH: 56,
    flipped: false,
  };
}

/* ───── coordinate helpers ───── */
function boardToPixel(bx, by, opts) {
  const o = { ...defaultOpts(), ...opts };
  const vx = o.flipped ? (BOARD_W - 1 - bx) : bx;
  const vy = o.flipped ? by : (BOARD_H - 1 - by);
  return {
    px: o.marginX + vx * o.cellW,
    py: o.marginY + vy * o.cellH,
  };
}

function pixelToBoard(px, py, opts) {
  const o = { ...defaultOpts(), ...opts };
  let vx = Math.round((px - o.marginX) / o.cellW);
  let vy = Math.round((py - o.marginY) / o.cellH);
  if (vx < 0 || vx >= BOARD_W || vy < 0 || vy >= BOARD_H) return null;
  const bx = o.flipped ? (BOARD_W - 1 - vx) : vx;
  const by = o.flipped ? vy : (BOARD_H - 1 - vy);
  return { x: bx, y: by };
}

/* ───── SVG helpers ───── */
function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function clearGroup(svg, groupId) {
  let g = svg.querySelector(`#${groupId}`);
  if (g) { g.innerHTML = ''; return g; }
  g = svgEl('g', { id: groupId });
  svg.appendChild(g);
  return g;
}

/* ═══════════════════════════════════════════════
   Draw board grid
   ═══════════════════════════════════════════════ */
function drawBoard(svg, opts) {
  const o = { ...defaultOpts(), ...opts };
  const totalW = o.marginX * 2 + (BOARD_W - 1) * o.cellW;
  const totalH = o.marginY * 2 + (BOARD_H - 1) * o.cellH;
  svg.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);

  const g = clearGroup(svg, 'boardGrid');

  /* background */
  g.appendChild(svgEl('rect', {
    x: 0, y: 0, width: totalW, height: totalH,
    fill: '#e8c97a', rx: 8,
  }));

  /* board area background */
  const bx0 = o.marginX, by0 = o.marginY;
  const bx1 = o.marginX + (BOARD_W - 1) * o.cellW;
  const by1 = o.marginY + (BOARD_H - 1) * o.cellH;
  g.appendChild(svgEl('rect', {
    x: bx0 - 8, y: by0 - 8,
    width: bx1 - bx0 + 16, height: by1 - by0 + 16,
    fill: '#d4a843', rx: 4,
  }));

  const lineStyle = { stroke: '#5a3a1a', 'stroke-width': 1.2, fill: 'none' };

  /* horizontal lines */
  for (let r = 0; r < BOARD_H; r++) {
    const y = o.marginY + r * o.cellH;
    g.appendChild(svgEl('line', { x1: bx0, y1: y, x2: bx1, y2: y, ...lineStyle }));
  }

  /* vertical lines (with river gap) */
  for (let c = 0; c < BOARD_W; c++) {
    const x = o.marginX + c * o.cellW;
    if (c === 0 || c === BOARD_W - 1) {
      g.appendChild(svgEl('line', { x1: x, y1: by0, x2: x, y2: by1, ...lineStyle }));
    } else {
      /* top half (rows 0-4 in visual = board rows 9-5) */
      const midTop = o.marginY + 4 * o.cellH;
      const midBot = o.marginY + 5 * o.cellH;
      g.appendChild(svgEl('line', { x1: x, y1: by0, x2: x, y2: midTop, ...lineStyle }));
      g.appendChild(svgEl('line', { x1: x, y1: midBot, x2: x, y2: by1, ...lineStyle }));
    }
  }

  /* palace diagonals */
  const palaces = [
    /* Xiangqi palace (board y=7..9) */
    { cols: [3, 5], rowsVisual: null, boardRows: [7, 9] },
    /* Chess palace (board y=0..2) */
    { cols: [3, 5], rowsVisual: null, boardRows: [0, 2] },
  ];
  for (const p of palaces) {
    const p1 = boardToPixel(p.cols[0], p.boardRows[0], o);
    const p2 = boardToPixel(p.cols[1], p.boardRows[1], o);
    const p3 = boardToPixel(p.cols[1], p.boardRows[0], o);
    const p4 = boardToPixel(p.cols[0], p.boardRows[1], o);
    g.appendChild(svgEl('line', { x1: p1.px, y1: p1.py, x2: p2.px, y2: p2.py, ...lineStyle, 'stroke-dasharray': '6 3' }));
    g.appendChild(svgEl('line', { x1: p3.px, y1: p3.py, x2: p4.px, y2: p4.py, ...lineStyle, 'stroke-dasharray': '6 3' }));
  }

  /* river text */
  const riverY = o.marginY + 4.5 * o.cellH;
  const riverLeft = svgEl('text', {
    x: o.marginX + 1.5 * o.cellW, y: riverY + 6,
    'font-size': 22, fill: '#7a5a2aaa', 'text-anchor': 'middle',
    'font-family': "'Noto Serif SC', serif", 'font-weight': 'bold',
  });
  riverLeft.textContent = '楚河';
  g.appendChild(riverLeft);

  const riverRight = svgEl('text', {
    x: o.marginX + 6.5 * o.cellW, y: riverY + 6,
    'font-size': 22, fill: '#7a5a2aaa', 'text-anchor': 'middle',
    'font-family': "'Noto Serif SC', serif", 'font-weight': 'bold',
  });
  riverRight.textContent = '漢界';
  g.appendChild(riverRight);

  /* cross markers at cannon & soldier starting positions */
  const crossPositions = [
    /* Xiangqi cannon (board y=7) */
    { x: 1, y: 7 }, { x: 7, y: 7 },
    /* Xiangqi soldiers (board y=6) */
    { x: 0, y: 6 }, { x: 2, y: 6 }, { x: 4, y: 6 }, { x: 6, y: 6 }, { x: 8, y: 6 },
    /* Chess pawns: symmetry would be y=1 or y=3 — skip for clarity */
  ];
  const cLen = 6, cGap = 3;
  for (const cp of crossPositions) {
    const { px, py } = boardToPixel(cp.x, cp.y, o);
    for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
      if (cp.x + dx < 0 || cp.x + dx >= BOARD_W) continue;
      g.appendChild(svgEl('line', {
        x1: px + dx * cGap, y1: py + dy * cGap,
        x2: px + dx * (cGap + cLen), y2: py + dy * cGap,
        stroke: '#5a3a1a', 'stroke-width': 1,
      }));
      g.appendChild(svgEl('line', {
        x1: px + dx * cGap, y1: py + dy * cGap,
        x2: px + dx * cGap, y2: py + dy * (cGap + cLen),
        stroke: '#5a3a1a', 'stroke-width': 1,
      }));
    }
  }

  /* outer border */
  g.appendChild(svgEl('rect', {
    x: bx0 - 12, y: by0 - 12,
    width: bx1 - bx0 + 24, height: by1 - by0 + 24,
    fill: 'none', stroke: '#5a3a1a', 'stroke-width': 2.5, rx: 6,
  }));
}

/* ═══════════════════════════════════════════════
   Draw pieces
   ═══════════════════════════════════════════════ */

/**
 * Parse an ASCII board string (from render.py) into a 2D array.
 * Format:  "10 c . n . g . n . c\n 9 . . . . . . . . .\n ... \n    a b c d e f g h i"
 */
function parseAsciiBoard(ascii) {
  const grid = Array.from({ length: BOARD_H }, () => Array(BOARD_W).fill(null));
  const lines = ascii.split('\n');
  for (const line of lines) {
    const m = line.match(/^\s*(\d+)\s+(.+)/);
    if (!m) continue;
    const row = parseInt(m[1]) - 1; // 1-indexed → 0-indexed (y)
    const cells = m[2].trim().split(/\s+/);
    for (let col = 0; col < Math.min(cells.length, BOARD_W); col++) {
      const ch = cells[col];
      if (ch !== '.') grid[row][col] = ch;
    }
  }
  return grid;
}

function drawPieces(svg, boardData, opts) {
  const o = { ...defaultOpts(), ...opts };
  const g = clearGroup(svg, 'pieces');

  let grid;
  if (typeof boardData === 'string') {
    grid = parseAsciiBoard(boardData);
  } else {
    grid = boardData; // already a 2D array
  }

  for (let y = 0; y < BOARD_H; y++) {
    for (let x = 0; x < BOARD_W; x++) {
      const ch = grid[y]?.[x];
      if (!ch) continue;
      const { px, py } = boardToPixel(x, y, o);
      drawSinglePiece(g, ch, px, py, x, y);
    }
  }
}

function drawSinglePiece(parent, ch, px, py, bx, by) {
  const isChess = ch === ch.toUpperCase();
  const r = 22;

  /* disk */
  const disk = svgEl('circle', {
    cx: px, cy: py, r,
    fill: isChess ? '#faf0d6' : '#6b2c2c',
    stroke: isChess ? '#bfa44e' : '#3a1515',
    'stroke-width': 2,
    'data-bx': bx, 'data-by': by,
    class: `piece-disk ${isChess ? 'chess-piece' : 'xiangqi-piece'}`,
  });
  parent.appendChild(disk);

  /* label */
  const lower = ch.toLowerCase();
  let label, fontSize, fontFamily, fill;

  if (isChess && CHESS_PIECES[ch]) {
    label = CHESS_PIECES[ch].icon;
    fontSize = 26;
    fontFamily = 'serif';
    fill = '#3a2a0a';
  } else if (!isChess && XIANGQI_PIECES[lower]) {
    label = XIANGQI_PIECES[lower].ch;
    fontSize = 20;
    fontFamily = "'Noto Serif SC', serif";
    fill = '#f0c860';
  } else {
    label = ch;
    fontSize = 18;
    fontFamily = 'monospace';
    fill = '#333';
  }

  const text = svgEl('text', {
    x: px, y: py + (isChess ? 2 : 1),
    'font-size': fontSize, fill,
    'text-anchor': 'middle', 'dominant-baseline': 'central',
    'font-family': fontFamily, 'font-weight': 'bold',
    'pointer-events': 'none',
  });
  text.textContent = label;
  parent.appendChild(text);
}

/* ═══════════════════════════════════════════════
   Highlights
   ═══════════════════════════════════════════════ */
function highlightSquares(svg, squares, className, opts) {
  const o = { ...defaultOpts(), ...opts };
  const g = clearGroup(svg, `hl-${className}`);
  for (const sq of squares) {
    const { px, py } = boardToPixel(sq.x, sq.y, o);
    g.appendChild(svgEl('circle', {
      cx: px, cy: py, r: 12,
      class: `highlight ${className}`,
      fill: sq.fill || 'rgba(76,175,80,0.5)',
      stroke: sq.stroke || 'rgba(76,175,80,0.8)',
      'stroke-width': 2,
      'pointer-events': 'none',
    }));
  }
}

function highlightMove(svg, from, to, opts) {
  const o = { ...defaultOpts(), ...opts };
  const g = clearGroup(svg, 'hl-move');
  for (const sq of [from, to]) {
    const { px, py } = boardToPixel(sq.x, sq.y, o);
    g.appendChild(svgEl('rect', {
      x: px - 26, y: py - 26, width: 52, height: 52, rx: 6,
      fill: 'none',
      stroke: sq === from ? '#ff9800' : '#4caf50',
      'stroke-width': 3,
      opacity: 0.7,
      'pointer-events': 'none',
    }));
  }
}

function clearHighlights(svg, className) {
  if (className) {
    const g = svg.querySelector(`#hl-${className}`);
    if (g) g.innerHTML = '';
  } else {
    svg.querySelectorAll('[id^="hl-"]').forEach(g => g.innerHTML = '');
  }
}

/* ═══════════════════════════════════════════════
   Column / coordinate labels
   ═══════════════════════════════════════════════ */
function drawCoordLabels(svg, opts) {
  const o = { ...defaultOpts(), ...opts };
  const g = clearGroup(svg, 'coordLabels');
  const cols = 'abcdefghi';
  const labelStyle = {
    'font-size': 11, fill: '#7a5a2a', 'text-anchor': 'middle',
    'font-family': "'Inter', sans-serif", 'pointer-events': 'none',
  };
  for (let c = 0; c < BOARD_W; c++) {
    const vc = o.flipped ? (BOARD_W - 1 - c) : c;
    const x = o.marginX + vc * o.cellW;
    const label = svgEl('text', {
      x, y: o.marginY + (BOARD_H - 1) * o.cellH + 28, ...labelStyle,
    });
    label.textContent = cols[c];
    g.appendChild(label);
  }
  for (let r = 0; r < BOARD_H; r++) {
    const vr = o.flipped ? r : (BOARD_H - 1 - r);
    const y = o.marginY + vr * o.cellH;
    const label = svgEl('text', {
      x: o.marginX - 20, y: y + 4, ...labelStyle,
    });
    label.textContent = r + 1;
    g.appendChild(label);
  }
}

/* ───── export ───── */
window.BoardRenderer = {
  BOARD_W, BOARD_H, SVG_NS,
  CHESS_PIECES, XIANGQI_PIECES,
  defaultOpts,
  boardToPixel, pixelToBoard,
  drawBoard, drawPieces, drawSinglePiece,
  parseAsciiBoard,
  highlightSquares, highlightMove, clearHighlights,
  drawCoordLabels,
};
