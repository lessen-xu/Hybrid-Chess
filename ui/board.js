/* ═══════════════════════════════════════════
   Hybrid Chess UI — Board Renderer & Game Replay
   ═══════════════════════════════════════════ */

(() => {
  'use strict';

  // ── Constants ──
  const COLS = 9, ROWS = 10;
  const CELL = 60;
  const PAD  = 30;                // padding inside SVG
  const SVG_W = PAD * 2 + (COLS - 1) * CELL;   // 540
  const SVG_H = PAD * 2 + (ROWS - 1) * CELL;   // 600

  // ── Piece definitions ──
  // Each piece: { label (for xiangqi text), unicode (for chess symbol), name }
  const CHESS_PIECES = {
    K: { unicode: '♔', name: 'King',   desc: 'All 8 dirs, 1 step' },
    Q: { unicode: '♕', name: 'Queen',  desc: 'Orth + diag slide' },
    R: { unicode: '♖', name: 'Rook',   desc: 'Orth slide' },
    B: { unicode: '♗', name: 'Bishop', desc: 'Diag slide' },
    N: { unicode: '♘', name: 'Knight', desc: 'L-shape, no block' },
    P: { unicode: '♙', name: 'Pawn',   desc: 'Forward, promotes at y=9' },
  };

  const XIANGQI_PIECES = {
    g: { label: '將', name: 'General',  desc: 'Orth 1-step, palace' },
    a: { label: '士', name: 'Advisor',  desc: 'Diag 1-step, palace' },
    e: { label: '象', name: 'Elephant', desc: 'Diag 2-step, eye block' },
    h: { label: '馬', name: 'Horse',    desc: 'L-shape, leg block' },
    c: { label: '車', name: 'Chariot',  desc: 'Orth slide' },
    n: { label: '砲', name: 'Cannon',   desc: 'Screen jump capture' },
    s: { label: '卒', name: 'Soldier',  desc: 'Forward, +sideways after river' },
  };

  // ── SVG Namespace ──
  const NS = 'http://www.w3.org/2000/svg';

  function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      el.setAttribute(k, v);
    }
    return el;
  }

  // ── Board coordinate helpers ──
  // Board grid: x 0-8 (columns a-i), y 0-9 (row 1-10, bottom to top)
  // SVG coords: board position → pixel
  function bx(x) { return PAD + x * CELL; }
  function by(y) { return PAD + (ROWS - 1 - y) * CELL; }  // y=0 at bottom → large SVG y

  // ── State ──
  let games = [];
  let currentGameIdx = 0;
  let currentStep = 0;
  let autoPlayInterval = null;
  let lastMoveFrom = null;
  let lastMoveTo = null;

  // ── Initialize ──
  const svg = document.getElementById('boardSvg');
  svg.setAttribute('viewBox', `0 0 ${SVG_W} ${SVG_H}`);

  // Paint initial board on load
  drawBoard();
  renderPosition(getInitialGrid());
  buildLegend();
  bindControls();

  // ══════════════════════════════════════════════════
  //  BOARD DRAWING
  // ══════════════════════════════════════════════════

  function drawBoard() {
    // Clear
    svg.innerHTML = '';

    // Defs (gradients, filters)
    const defs = svgEl('defs');

    // Board wood gradient
    const boardGrad = svgEl('linearGradient', { id: 'boardGrad', x1: '0%', y1: '0%', x2: '100%', y2: '100%' });
    boardGrad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#e2b968' }));
    boardGrad.appendChild(svgEl('stop', { offset: '50%', 'stop-color': '#d4a55a' }));
    boardGrad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#c89843' }));
    defs.appendChild(boardGrad);

    // Chess piece gradient (white)
    const whiteGrad = svgEl('radialGradient', { id: 'chessPieceGrad', cx: '40%', cy: '35%', r: '60%' });
    whiteGrad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#fffef8' }));
    whiteGrad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#e8dcc8' }));
    defs.appendChild(whiteGrad);

    // Xiangqi piece gradient (dark wood)
    const darkGrad = svgEl('radialGradient', { id: 'xiangqiPieceGrad', cx: '40%', cy: '35%', r: '60%' });
    darkGrad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#6a3030' }));
    darkGrad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#3d1a1a' }));
    defs.appendChild(darkGrad);

    // Drop shadow for pieces
    const shadow = svgEl('filter', { id: 'pieceShadow', x: '-30%', y: '-30%', width: '160%', height: '160%' });
    const blur = svgEl('feDropShadow', { dx: '1', dy: '2', stdDeviation: '2', 'flood-color': 'rgba(0,0,0,0.4)' });
    shadow.appendChild(blur);
    defs.appendChild(shadow);

    // Highlight glow
    const glow = svgEl('filter', { id: 'highlightGlow', x: '-50%', y: '-50%', width: '200%', height: '200%' });
    const glBlur = svgEl('feGaussianBlur', { stdDeviation: '4', result: 'blur' });
    glow.appendChild(glBlur);
    const merge = svgEl('feMerge');
    merge.appendChild(svgEl('feMergeNode', { 'in': 'blur' }));
    merge.appendChild(svgEl('feMergeNode', { 'in': 'SourceGraphic' }));
    glow.appendChild(merge);
    defs.appendChild(glow);

    svg.appendChild(defs);

    // Board background
    svg.appendChild(svgEl('rect', {
      x: 0, y: 0, width: SVG_W, height: SVG_H,
      fill: 'url(#boardGrad)', rx: 8
    }));

    // ── Grid lines ──
    const lineGroup = svgEl('g', { stroke: '#5a3a1a', 'stroke-width': '1.2', fill: 'none' });

    // Horizontal lines
    for (let row = 0; row < ROWS; row++) {
      lineGroup.appendChild(svgEl('line', {
        x1: bx(0), y1: by(row), x2: bx(COLS - 1), y2: by(row)
      }));
    }

    // Vertical lines — full height only for edge columns
    // Inner columns break at the river (between y=4 and y=5)
    for (let col = 0; col < COLS; col++) {
      if (col === 0 || col === COLS - 1) {
        // Full line
        lineGroup.appendChild(svgEl('line', {
          x1: bx(col), y1: by(0), x2: bx(col), y2: by(ROWS - 1)
        }));
      } else {
        // Bottom half (y=0 to y=4)
        lineGroup.appendChild(svgEl('line', {
          x1: bx(col), y1: by(0), x2: bx(col), y2: by(4)
        }));
        // Top half (y=5 to y=9)
        lineGroup.appendChild(svgEl('line', {
          x1: bx(col), y1: by(5), x2: bx(col), y2: by(ROWS - 1)
        }));
      }
    }

    // ── Palace diagonals ──
    // Xiangqi palace: x=3-5, y=7-9
    lineGroup.appendChild(svgEl('line', { x1: bx(3), y1: by(9), x2: bx(5), y2: by(7) }));
    lineGroup.appendChild(svgEl('line', { x1: bx(5), y1: by(9), x2: bx(3), y2: by(7) }));

    // Chess "palace": x=3-5, y=0-2 (mirrored concept)
    lineGroup.appendChild(svgEl('line', { x1: bx(3), y1: by(2), x2: bx(5), y2: by(0) }));
    lineGroup.appendChild(svgEl('line', { x1: bx(5), y1: by(2), x2: bx(3), y2: by(0) }));

    svg.appendChild(lineGroup);

    // ── River ──
    // River is between y=4 and y=5
    const riverY1 = by(4);
    const riverY2 = by(5);
    const riverRect = svgEl('rect', {
      x: bx(0) + 1, y: riverY2,
      width: (COLS - 1) * CELL - 2,
      height: riverY1 - riverY2,
      fill: 'rgba(180, 140, 70, 0.3)'
    });
    svg.appendChild(riverRect);

    // River text
    const riverMidY = (riverY1 + riverY2) / 2;
    const riverTextL = svgEl('text', {
      x: bx(2), y: riverMidY + 2,
      'font-family': "'Noto Serif SC', serif",
      'font-size': '22',
      'font-weight': '700',
      fill: '#7a5a2a',
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      opacity: '0.6'
    });
    riverTextL.textContent = '楚 河';
    svg.appendChild(riverTextL);

    const riverTextR = svgEl('text', {
      x: bx(6), y: riverMidY + 2,
      'font-family': "'Noto Serif SC', serif",
      'font-size': '22',
      'font-weight': '700',
      fill: '#7a5a2a',
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      opacity: '0.6'
    });
    riverTextR.textContent = '漢 界';
    svg.appendChild(riverTextR);

    // ── Soldier/cannon position markers (small cross marks) ──
    const markerGroup = svgEl('g', { stroke: '#5a3a1a', 'stroke-width': '1' });
    // Cannon positions: (1,2), (7,2), (1,7), (7,7)
    const cannonPos = [[1,2],[7,2],[1,7],[7,7]];
    // Soldier/pawn positions
    const soldierPos = [[0,3],[2,3],[4,3],[6,3],[8,3],[0,6],[2,6],[4,6],[6,6],[8,6]];

    [...cannonPos, ...soldierPos].forEach(([x, y]) => {
      drawCrossMark(markerGroup, bx(x), by(y));
    });
    svg.appendChild(markerGroup);

    // ── Coordinate labels ──
    const labelGroup = svgEl('g', {
      'font-family': "'Inter', sans-serif",
      'font-size': '10',
      fill: '#7a5a2a',
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      opacity: '0.7'
    });

    const colLabels = 'abcdefghi';
    for (let c = 0; c < COLS; c++) {
      // Bottom
      labelGroup.appendChild(svgEl('text', {
        x: bx(c), y: by(0) + 18, 'font-size': '11'
      })).textContent = colLabels[c];
      // Top
      labelGroup.appendChild(svgEl('text', {
        x: bx(c), y: by(9) - 18, 'font-size': '11'
      })).textContent = colLabels[c];
    }

    for (let r = 0; r < ROWS; r++) {
      // Left
      labelGroup.appendChild(svgEl('text', {
        x: bx(0) - 18, y: by(r), 'font-size': '11'
      })).textContent = String(r + 1);
    }
    svg.appendChild(labelGroup);

    // ── Highlight layer (below pieces) ──
    const highlightLayer = svgEl('g', { id: 'highlightLayer' });
    svg.appendChild(highlightLayer);

    // ── Piece layer ──
    const pieceLayer = svgEl('g', { id: 'pieceLayer' });
    svg.appendChild(pieceLayer);
  }

  function drawCrossMark(parent, cx, cy) {
    const d = 5, gap = 3;
    const segs = [
      // Top-left
      [-gap, -(gap + d), -gap, -gap],
      [-(gap + d), -gap, -gap, -gap],
      // Top-right
      [gap, -(gap + d), gap, -gap],
      [(gap + d), -gap, gap, -gap],
      // Bottom-left
      [-gap, (gap + d), -gap, gap],
      [-(gap + d), gap, -gap, gap],
      // Bottom-right
      [gap, (gap + d), gap, gap],
      [(gap + d), gap, gap, gap],
    ];
    segs.forEach(([x1, y1, x2, y2]) => {
      parent.appendChild(svgEl('line', {
        x1: cx + x1, y1: cy + y1, x2: cx + x2, y2: cy + y2,
        'stroke-width': '0.8'
      }));
    });
  }

  // ══════════════════════════════════════════════════
  //  PIECE RENDERING
  // ══════════════════════════════════════════════════

  function renderPosition(grid, animateMove = null) {
    const pieceLayer = document.getElementById('pieceLayer');
    const highlightLayer = document.getElementById('highlightLayer');

    // Clear highlights
    highlightLayer.innerHTML = '';

    // Draw highlights for last move
    if (lastMoveFrom) {
      highlightLayer.appendChild(svgEl('rect', {
        x: bx(lastMoveFrom[0]) - CELL / 2 + 2,
        y: by(lastMoveFrom[1]) - CELL / 2 + 2,
        width: CELL - 4, height: CELL - 4,
        fill: 'rgba(255, 217, 0, 0.25)',
        rx: 4
      }));
    }
    if (lastMoveTo) {
      highlightLayer.appendChild(svgEl('rect', {
        x: bx(lastMoveTo[0]) - CELL / 2 + 2,
        y: by(lastMoveTo[1]) - CELL / 2 + 2,
        width: CELL - 4, height: CELL - 4,
        fill: 'rgba(255, 217, 0, 0.4)',
        rx: 4
      }));
    }

    // Re-draw pieces
    pieceLayer.innerHTML = '';

    for (let y = 0; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const ch = grid[y][x];
        if (ch === '.') continue;

        const cx = bx(x);
        const cy = by(y);
        const isChess = ch === ch.toUpperCase();

        const pieceGroup = svgEl('g', {
          transform: `translate(${cx}, ${cy})`,
          filter: 'url(#pieceShadow)',
          class: 'piece'
        });

        if (isChess) {
          drawChessPiece(pieceGroup, ch);
        } else {
          drawXiangqiPiece(pieceGroup, ch);
        }

        pieceLayer.appendChild(pieceGroup);
      }
    }
  }

  function drawChessPiece(group, ch) {
    const info = CHESS_PIECES[ch];
    if (!info) return;

    const r = 24;

    // Outer ring
    group.appendChild(svgEl('circle', {
      cx: 0, cy: 0, r: r,
      fill: 'url(#chessPieceGrad)',
      stroke: '#a09080',
      'stroke-width': '1.5'
    }));

    // Inner decorative ring
    group.appendChild(svgEl('circle', {
      cx: 0, cy: 0, r: r - 4,
      fill: 'none',
      stroke: '#c0b098',
      'stroke-width': '0.5'
    }));

    // Unicode chess symbol
    group.appendChild(svgEl('text', {
      x: 0, y: 2,
      'font-size': '30',
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      fill: '#3a3020',
      'font-weight': 'normal'
    })).textContent = info.unicode;
  }

  function drawXiangqiPiece(group, ch) {
    const info = XIANGQI_PIECES[ch];
    if (!info) return;

    const r = 24;

    // Outer ring
    group.appendChild(svgEl('circle', {
      cx: 0, cy: 0, r: r,
      fill: 'url(#xiangqiPieceGrad)',
      stroke: '#8a3030',
      'stroke-width': '1.5'
    }));

    // Inner ring
    group.appendChild(svgEl('circle', {
      cx: 0, cy: 0, r: r - 4,
      fill: 'none',
      stroke: '#aa5050',
      'stroke-width': '0.8'
    }));

    // Chinese character
    group.appendChild(svgEl('text', {
      x: 0, y: 2,
      'font-family': "'Noto Serif SC', serif",
      'font-size': '22',
      'font-weight': '700',
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      fill: '#f0c060'
    })).textContent = info.label;
  }

  // ══════════════════════════════════════════════════
  //  INITIAL BOARD STATE
  // ══════════════════════════════════════════════════

  function getInitialGrid() {
    // Build the default starting position as a 10x9 grid (indexed [y][x])
    // y=0 bottom (Chess), y=9 top (Xiangqi)
    const grid = Array.from({ length: ROWS }, () => Array(COLS).fill('.'));

    // Chess back rank (y=0): R N B Q K B N R .
    const chessBack = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R', '.'];
    for (let x = 0; x < COLS; x++) grid[0][x] = chessBack[x];

    // Chess pawns (y=1): 8 pawns + extra on i-file
    for (let x = 0; x < 9; x++) grid[1][x] = 'P';

    // Xiangqi back rank (y=9): c h e a g a e h c
    const xqBack = ['c', 'h', 'e', 'a', 'g', 'a', 'e', 'h', 'c'];
    for (let x = 0; x < COLS; x++) grid[9][x] = xqBack[x];

    // Xiangqi cannons (y=7)
    grid[7][1] = 'n';
    grid[7][7] = 'n';

    // Xiangqi soldiers (y=6)
    for (const x of [0, 2, 4, 6, 8]) grid[6][x] = 's';

    return grid;
  }

  // ══════════════════════════════════════════════════
  //  PARSE ASCII BOARD STATE
  // ══════════════════════════════════════════════════

  function parseAsciiState(asciiStr) {
    // The ASCII format from render.py:
    // "10 c h e a g a e h c"
    // " 9 . . . . . . . . ."
    // ...
    // " 1 R N B Q K B N R ."
    // "    a b c d e f g h i"
    const lines = asciiStr.trim().split('\n');
    const grid = Array.from({ length: ROWS }, () => Array(COLS).fill('.'));

    for (const line of lines) {
      const trimmed = line.trim();
      // Skip column labels
      if (trimmed.startsWith('a') || trimmed.startsWith('b')) continue;

      const match = trimmed.match(/^(\d+)\s+(.+)$/);
      if (!match) continue;

      const rowNum = parseInt(match[1]);   // 1-based, y = rowNum - 1
      const y = rowNum - 1;
      if (y < 0 || y >= ROWS) continue;

      const cells = match[2].trim().split(/\s+/);
      for (let x = 0; x < Math.min(cells.length, COLS); x++) {
        grid[y][x] = cells[x];
      }
    }

    return grid;
  }

  // ══════════════════════════════════════════════════
  //  GAME LOADING & REPLAY
  // ══════════════════════════════════════════════════

  function loadGames(data) {
    if (Array.isArray(data)) {
      games = data;
    } else {
      games = [data];
    }

    currentGameIdx = 0;
    currentStep = 0;

    // Setup game selector
    const sel = document.getElementById('gameSelect');
    if (games.length > 1) {
      sel.style.display = 'block';
      sel.innerHTML = '';
      games.forEach((g, i) => {
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `Game ${i + 1} — ${g.result || '?'}`;
        sel.appendChild(opt);
      });
      sel.onchange = () => {
        currentGameIdx = parseInt(sel.value);
        currentStep = 0;
        renderCurrentStep();
      };
    } else {
      sel.style.display = 'none';
    }

    renderCurrentStep();
    buildMoveList();
  }

  function getCurrentGame() {
    return games[currentGameIdx] || null;
  }

  function renderCurrentStep() {
    const game = getCurrentGame();
    if (!game) {
      renderPosition(getInitialGrid());
      updateInfo(null);
      return;
    }

    const states = game.states_ascii || [];
    const moves  = game.moves || [];

    // Clamp
    if (currentStep < 0) currentStep = 0;
    if (currentStep >= states.length) currentStep = states.length - 1;

    // Parse move for highlight
    lastMoveFrom = null;
    lastMoveTo = null;
    if (currentStep > 0 && currentStep <= moves.length) {
      const mv = moves[currentStep - 1];
      const parsed = parseMoveString(mv);
      if (parsed) {
        lastMoveFrom = parsed.from;
        lastMoveTo = parsed.to;
      }
    }

    // Render board
    const asciiState = states[currentStep];
    if (asciiState) {
      const grid = parseAsciiState(asciiState);
      renderPosition(grid);
    }

    // Update UI
    updateControlStates(states.length);
    updateStepLabel(moves);
    updateInfo(game);
    highlightMoveInList(currentStep);
  }

  function parseMoveString(mvStr) {
    // Format: "a1-b2" or "a1b2" or "(fx,fy)→(tx,ty)" etc.
    // Try common patterns
    const colMap = { a: 0, b: 1, c: 2, d: 3, e: 4, f: 5, g: 6, h: 7, i: 8 };

    // Try "a1-b2" or "a1b2"
    const m = mvStr.match(/([a-i])(\d+)\s*[-→]?\s*([a-i])(\d+)/i);
    if (m) {
      return {
        from: [colMap[m[1].toLowerCase()], parseInt(m[2]) - 1],
        to:   [colMap[m[3].toLowerCase()], parseInt(m[4]) - 1]
      };
    }

    // Try "(fx,fy)-(tx,ty)"
    const m2 = mvStr.match(/\(?(\d+)\s*,\s*(\d+)\)?.*\(?(\d+)\s*,\s*(\d+)\)?/);
    if (m2) {
      return {
        from: [parseInt(m2[1]), parseInt(m2[2])],
        to:   [parseInt(m2[3]), parseInt(m2[4])]
      };
    }

    return null;
  }

  function updateControlStates(totalStates) {
    const slider = document.getElementById('slider');
    slider.max = Math.max(0, totalStates - 1);
    slider.value = currentStep;

    document.getElementById('btnFirst').disabled = currentStep <= 0;
    document.getElementById('btnPrev').disabled  = currentStep <= 0;
    document.getElementById('btnNext').disabled  = currentStep >= totalStates - 1;
    document.getElementById('btnLast').disabled  = currentStep >= totalStates - 1;
  }

  function updateStepLabel(moves) {
    const label = document.getElementById('stepLabel');
    const count = document.getElementById('stepCount');
    const game = getCurrentGame();
    const total = game ? (game.states_ascii || []).length - 1 : 0;

    if (currentStep === 0) {
      label.textContent = 'Initial Position';
    } else if (currentStep <= moves.length) {
      const side = (currentStep % 2 === 1) ? 'Chess' : 'Xiangqi';
      label.textContent = `${side}: ${moves[currentStep - 1]}`;
    } else {
      label.textContent = `State ${currentStep}`;
    }

    count.textContent = `${currentStep} / ${total}`;
  }

  function updateInfo(game) {
    const status = document.getElementById('infoStatus');
    const result = document.getElementById('infoResult');
    const plies  = document.getElementById('infoPlies');
    const reason = document.getElementById('infoReason');

    if (!game) {
      status.textContent = 'Ready';
      result.textContent = '—';
      plies.textContent  = '—';
      reason.textContent = '—';
      return;
    }

    const meta = game.meta || {};
    const totalStates = (game.states_ascii || []).length;

    status.textContent = currentStep >= totalStates - 1 ? 'Finished' : 'In Progress';
    result.textContent = game.result || '—';
    plies.textContent  = meta.plies || totalStates - 1 || '—';
    reason.textContent = meta.reason || '—';
  }

  // ══════════════════════════════════════════════════
  //  MOVE LIST
  // ══════════════════════════════════════════════════

  function buildMoveList() {
    const container = document.getElementById('moveList');
    const game = getCurrentGame();

    if (!game || !game.moves || game.moves.length === 0) {
      container.innerHTML = '<p class="placeholder">No moves recorded</p>';
      return;
    }

    container.innerHTML = '';
    game.moves.forEach((mv, i) => {
      const entry = document.createElement('div');
      entry.className = 'move-entry';
      entry.dataset.step = i + 1;

      const isChess = (i % 2 === 0);  // Chess moves first (ply 1, 3, 5...)
      const sideClass = isChess ? 'move-side-chess' : 'move-side-xiangqi';

      entry.innerHTML = `
        <span class="move-number">${i + 1}.</span>
        <span class="move-text ${sideClass}">${escapeHtml(mv)}</span>
      `;

      entry.addEventListener('click', () => {
        currentStep = i + 1;
        renderCurrentStep();
      });

      container.appendChild(entry);
    });
  }

  function highlightMoveInList(step) {
    const entries = document.querySelectorAll('.move-entry');
    entries.forEach(e => {
      e.classList.toggle('active', parseInt(e.dataset.step) === step);
    });

    // Scroll active into view
    const active = document.querySelector('.move-entry.active');
    if (active) {
      active.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  // ══════════════════════════════════════════════════
  //  CONTROLS
  // ══════════════════════════════════════════════════

  function bindControls() {
    document.getElementById('btnFirst').addEventListener('click', goFirst);
    document.getElementById('btnPrev').addEventListener('click',  goPrev);
    document.getElementById('btnNext').addEventListener('click',  goNext);
    document.getElementById('btnLast').addEventListener('click',  goLast);
    document.getElementById('btnAutoPlay').addEventListener('click', toggleAutoPlay);

    document.getElementById('slider').addEventListener('input', (e) => {
      currentStep = parseInt(e.target.value);
      renderCurrentStep();
    });

    // File input
    document.getElementById('fileInput').addEventListener('change', handleFileInput);

    // Keyboard
    document.addEventListener('keydown', handleKeydown);
  }

  function goFirst() { currentStep = 0; renderCurrentStep(); }
  function goLast()  {
    const game = getCurrentGame();
    if (game) {
      currentStep = (game.states_ascii || []).length - 1;
    }
    renderCurrentStep();
  }
  function goPrev()  { if (currentStep > 0) { currentStep--; renderCurrentStep(); } }
  function goNext()  {
    const game = getCurrentGame();
    if (game && currentStep < (game.states_ascii || []).length - 1) {
      currentStep++;
      renderCurrentStep();
    }
  }

  function toggleAutoPlay() {
    const btn = document.getElementById('btnAutoPlay');
    const icon = document.getElementById('playIcon');

    if (autoPlayInterval) {
      clearInterval(autoPlayInterval);
      autoPlayInterval = null;
      btn.classList.remove('playing');
      icon.innerHTML = '<path fill="currentColor" d="M8 5v14l11-7z"/>';
    } else {
      btn.classList.add('playing');
      icon.innerHTML = '<path fill="currentColor" d="M6 6h4v12H6zm8 0h4v12h-4z"/>';
      autoPlayInterval = setInterval(() => {
        const game = getCurrentGame();
        if (!game || currentStep >= (game.states_ascii || []).length - 1) {
          toggleAutoPlay();  // Stop
          return;
        }
        currentStep++;
        renderCurrentStep();
      }, 600);
    }
  }

  function handleKeydown(e) {
    switch (e.key) {
      case 'ArrowLeft':  e.preventDefault(); goPrev();  break;
      case 'ArrowRight': e.preventDefault(); goNext();  break;
      case 'Home':       e.preventDefault(); goFirst(); break;
      case 'End':        e.preventDefault(); goLast();  break;
      case ' ':          e.preventDefault(); toggleAutoPlay(); break;
    }
  }

  function handleFileInput(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const text = evt.target.result;

        if (file.name.endsWith('.jsonl')) {
          const gamesList = text.trim().split('\n')
            .filter(l => l.trim())
            .map(l => JSON.parse(l));
          loadGames(gamesList);
        } else {
          const data = JSON.parse(text);
          loadGames(data);
        }
      } catch (err) {
        console.error('Failed to parse game file:', err);
        alert('Failed to parse game file: ' + err.message);
      }
    };
    reader.readAsText(file);
  }

  // ══════════════════════════════════════════════════
  //  PIECE LEGEND
  // ══════════════════════════════════════════════════

  function buildLegend() {
    const chessContainer = document.getElementById('chessLegend');
    const xqContainer = document.getElementById('xiangqiLegend');

    // Chess pieces
    for (const [ch, info] of Object.entries(CHESS_PIECES)) {
      const item = document.createElement('div');
      item.className = 'legend-item';

      // Mini SVG
      const miniSvg = document.createElementNS(NS, 'svg');
      miniSvg.setAttribute('viewBox', '-28 -28 56 56');
      miniSvg.setAttribute('width', '28');
      miniSvg.setAttribute('height', '28');
      miniSvg.classList.add('legend-piece');

      const g = svgEl('g');
      drawChessPiece(g, ch);
      miniSvg.appendChild(g);

      const textDiv = document.createElement('div');
      textDiv.innerHTML = `<div class="legend-name">${info.name}</div><div class="legend-desc">${info.desc}</div>`;

      item.appendChild(miniSvg);
      item.appendChild(textDiv);
      chessContainer.appendChild(item);
    }

    // Xiangqi pieces
    for (const [ch, info] of Object.entries(XIANGQI_PIECES)) {
      const item = document.createElement('div');
      item.className = 'legend-item';

      const miniSvg = document.createElementNS(NS, 'svg');
      miniSvg.setAttribute('viewBox', '-28 -28 56 56');
      miniSvg.setAttribute('width', '28');
      miniSvg.setAttribute('height', '28');
      miniSvg.classList.add('legend-piece');

      const g = svgEl('g');
      drawXiangqiPiece(g, ch);
      miniSvg.appendChild(g);

      const textDiv = document.createElement('div');
      textDiv.innerHTML = `<div class="legend-name">${info.label} ${info.name}</div><div class="legend-desc">${info.desc}</div>`;

      item.appendChild(miniSvg);
      item.appendChild(textDiv);
      xqContainer.appendChild(item);
    }
  }

  // ── Utility ──
  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

})();
