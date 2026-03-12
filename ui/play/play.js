/**
 * Hybrid Chess — Play Page Logic
 *
 * Interactive Human vs AI gameplay.
 * Communicates with hybrid.server via REST API.
 */

const BR = window.BoardRenderer;
const svg = document.getElementById('boardSvg');
const API = '';  // relative to server root

/* ── game state ── */
let gameState = {
  active: false,
  humanSide: 'chess',       // 'chess' or 'xiangqi'
  aiAgent: 'ab_d1',
  variant: 'none',
  flipped: false,
  sideToMove: 'chess',
  ply: 0,
  boardAscii: '',
  legalMoves: [],           // [{fx,fy,tx,ty,promotion}]
  selectedPiece: null,      // {x, y}
  moveHistory: [],
  gameOver: false,
  result: '',
};

let serverConnected = false;

/* ── render options ── */
function getRenderOpts() {
  return { ...BR.defaultOpts(), flipped: gameState.flipped };
}

/* ═══════════════════════════════════════════════
   Initialization
   ═══════════════════════════════════════════════ */
function init() {
  BR.drawBoard(svg, getRenderOpts());
  BR.drawCoordLabels(svg, getRenderOpts());
  drawInitialBoard();
  bindControls();
  checkServer();
}

function drawInitialBoard() {
  const initialAscii = [
    '10 c h e a g a e h c',
    ' 9 . . . . . . . . .',
    ' 8 . n . . . . . n .',
    ' 7 s . s . s . s . s',
    ' 6 . . . . . . . . .',
    ' 5 . . . . . . . . .',
    ' 4 P . P . P . P . P',
    ' 3 . . . . . . . . .',
    ' 2 . . . . . . . . .',
    ' 1 R N B Q K B N R .',
    '    a b c d e f g h i',
  ].join('\n');
  BR.drawPieces(svg, initialAscii, getRenderOpts());
}

/* ═══════════════════════════════════════════════
   Server Communication
   ═══════════════════════════════════════════════ */
async function checkServer() {
  try {
    const r = await fetch('/api/agents');
    if (r.ok) {
      serverConnected = true;
      document.getElementById('serverNotice').classList.add('connected');
      const data = await r.json();
      populateAgentSelect(data.agents || []);
    }
  } catch {
    serverConnected = false;
  }
}

function populateAgentSelect(agents) {
  const sel = document.getElementById('aiSelect');
  if (!agents.length) return;
  sel.innerHTML = agents.map(a =>
    `<option value="${a.id}" ${a.id === 'ab_d1' ? 'selected' : ''}>${a.label}</option>`
  ).join('');
}

async function apiCall(endpoint, body) {
  const r = await fetch(`/api/${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`API error: ${r.status}`);
  return r.json();
}

/* ═══════════════════════════════════════════════
   Game Flow
   ═══════════════════════════════════════════════ */
async function startNewGame() {
  if (!serverConnected) {
    alert('Server not connected.\nRun: python -m hybrid.server');
    return;
  }

  const humanSide = gameState.humanSide;
  const aiAgent = document.getElementById('aiSelect').value;
  const variant = document.getElementById('variantSelect').value;

  try {
    const data = await apiCall('new', {
      human_side: humanSide,
      ai_agent: aiAgent,
      variant: variant,
    });

    gameState.active = true;
    gameState.aiAgent = aiAgent;
    gameState.variant = variant;
    gameState.gameOver = false;
    gameState.moveHistory = [];
    gameState.selectedPiece = null;
    gameState.result = '';

    updateFromServerState(data);
    updateUI();

    document.getElementById('btnUndo').disabled = false;
    document.getElementById('btnResign').disabled = false;
    hideMessage();

    /* if AI moves first, let it go */
    if (gameState.sideToMove !== humanSide) {
      await aiMove();
    }
  } catch (err) {
    console.error('Failed to start game:', err);
    alert('Failed to start game: ' + err.message);
  }
}

async function submitHumanMove(fx, fy, tx, ty, promotion) {
  if (!gameState.active || gameState.gameOver) return;

  try {
    const data = await apiCall('move', {
      fx, fy, tx, ty,
      promotion: promotion || null,
    });

    /* record human move */
    const cols = 'abcdefghi';
    const notation = `${cols[fx]}${fy+1}-${cols[tx]}${ty+1}`;
    gameState.moveHistory.push({
      side: gameState.humanSide,
      notation,
    });

    updateFromServerState(data);
    updateUI();

    if (data.game_over) {
      handleGameOver(data);
      return;
    }

    /* AI responds */
    await aiMove();
  } catch (err) {
    console.error('Move failed:', err);
  }
}

async function aiMove() {
  if (!gameState.active || gameState.gameOver) return;

  updateStatus('AI thinking...', true);

  try {
    const data = await apiCall('ai_move', {});

    if (data.move) {
      const m = data.move;
      const cols = 'abcdefghi';
      gameState.moveHistory.push({
        side: gameState.humanSide === 'chess' ? 'xiangqi' : 'chess',
        notation: `${cols[m.fx]}${m.fy+1}-${cols[m.tx]}${m.ty+1}`,
      });
    }

    updateFromServerState(data);
    updateUI();

    if (data.game_over) {
      handleGameOver(data);
    }
  } catch (err) {
    console.error('AI move failed:', err);
    updateStatus('AI error', false);
  }
}

function updateFromServerState(data) {
  gameState.boardAscii = data.board_ascii || '';
  gameState.sideToMove = data.side_to_move || 'chess';
  gameState.ply = data.ply || 0;
  gameState.legalMoves = data.legal_moves || [];
  gameState.gameOver = data.game_over || false;
  gameState.result = data.result || '';
}

function handleGameOver(data) {
  gameState.active = false;
  gameState.gameOver = true;
  document.getElementById('btnUndo').disabled = true;
  document.getElementById('btnResign').disabled = true;

  let msg = data.result || 'Game Over';
  if (data.reason) msg += `\n${data.reason}`;
  showMessage(msg);
  updateStatus(data.result, false);
}

/* ═══════════════════════════════════════════════
   Board Interaction (Click-to-Move)
   ═══════════════════════════════════════════════ */
function setupBoardClick() {
  svg.addEventListener('click', (e) => {
    if (!gameState.active || gameState.gameOver) return;
    if (gameState.sideToMove !== gameState.humanSide) return;

    const rect = svg.getBoundingClientRect();
    const svgW = svg.viewBox.baseVal.width;
    const svgH = svg.viewBox.baseVal.height;
    const scaleX = svgW / rect.width;
    const scaleY = svgH / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;

    const coord = BR.pixelToBoard(px, py, getRenderOpts());
    if (!coord) return;

    handleSquareClick(coord.x, coord.y);
  });
}

function handleSquareClick(bx, by) {
  const sel = gameState.selectedPiece;

  if (sel) {
    /* check if clicking a legal destination */
    const isLegal = gameState.legalMoves.find(m =>
      m.fx === sel.x && m.fy === sel.y && m.tx === bx && m.ty === by
    );

    if (isLegal) {
      /* handle promotion */
      let promo = isLegal.promotion || null;
      /* if there are multiple promotions for same to-square, show picker */
      const promos = gameState.legalMoves.filter(m =>
        m.fx === sel.x && m.fy === sel.y && m.tx === bx && m.ty === by
      );
      if (promos.length > 1) {
        /* simple promotion picker — default to Queen */
        promo = 'QUEEN';
      }

      gameState.selectedPiece = null;
      BR.clearHighlights(svg);
      submitHumanMove(sel.x, sel.y, bx, by, promo);
      return;
    }

    /* clicking another own piece — re-select */
    if (isOwnPiece(bx, by)) {
      selectPiece(bx, by);
      return;
    }

    /* clicking elsewhere — deselect */
    gameState.selectedPiece = null;
    BR.clearHighlights(svg);
    return;
  }

  /* no piece selected — try to select */
  if (isOwnPiece(bx, by)) {
    selectPiece(bx, by);
  }
}

function isOwnPiece(bx, by) {
  const grid = BR.parseAsciiBoard(gameState.boardAscii);
  const ch = grid[by]?.[bx];
  if (!ch) return false;

  if (gameState.humanSide === 'chess') {
    return ch === ch.toUpperCase();
  } else {
    return ch === ch.toLowerCase();
  }
}

function selectPiece(bx, by) {
  gameState.selectedPiece = { x: bx, y: by };
  BR.clearHighlights(svg);

  /* find legal moves for this piece */
  const moves = gameState.legalMoves.filter(m => m.fx === bx && m.fy === by);
  const grid = BR.parseAsciiBoard(gameState.boardAscii);

  /* already-visited to-squares (for dedup with promotions) */
  const seen = new Set();
  const squares = [];

  for (const m of moves) {
    const key = `${m.tx},${m.ty}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const targetCh = grid[m.ty]?.[m.tx];
    const isCapture = !!targetCh;
    squares.push({
      x: m.tx, y: m.ty,
      fill: isCapture ? 'rgba(248,81,73,0.35)' : 'rgba(76,175,80,0.45)',
      stroke: isCapture ? 'rgba(248,81,73,0.7)' : 'rgba(76,175,80,0.7)',
    });
  }

  BR.highlightSquares(svg, squares, 'legal-move', getRenderOpts());

  /* highlight selected piece */
  const selHL = [{ x: bx, y: by, fill: 'rgba(212,168,67,0.3)', stroke: 'rgba(212,168,67,0.7)' }];
  BR.highlightSquares(svg, selHL, 'selected', getRenderOpts());
}

/* ═══════════════════════════════════════════════
   UI Updates
   ═══════════════════════════════════════════════ */
function updateUI() {
  const o = getRenderOpts();

  /* redraw board */
  BR.drawBoard(svg, o);
  BR.drawCoordLabels(svg, o);
  if (gameState.boardAscii) {
    BR.drawPieces(svg, gameState.boardAscii, o);
  }

  /* highlight last move */
  if (gameState.moveHistory.length > 0) {
    const last = gameState.moveHistory[gameState.moveHistory.length - 1];
    const mv = parseMoveNotation(last.notation);
    if (mv) BR.highlightMove(svg, mv.from, mv.to, o);
  }

  /* status */
  document.getElementById('statusTurn').textContent =
    gameState.sideToMove === 'chess' ? '♔ Chess' : '將 Xiangqi';
  document.getElementById('statusPly').textContent = gameState.ply;
  document.getElementById('statusYourSide').textContent =
    gameState.humanSide === 'chess' ? '♔ Chess (Bottom)' : '將 Xiangqi (Top)';

  if (!gameState.gameOver) {
    const yourTurn = gameState.sideToMove === gameState.humanSide;
    updateStatus(yourTurn ? 'Your turn' : 'AI turn', yourTurn);
  }

  /* player labels */
  const isFlipped = gameState.flipped;
  const topSide = isFlipped ? 'chess' : 'xiangqi';
  const botSide = isFlipped ? 'xiangqi' : 'chess';
  const topIsHuman = topSide === gameState.humanSide;
  const botIsHuman = botSide === gameState.humanSide;

  document.getElementById('playerTopName').textContent =
    `${topSide === 'chess' ? '♔ Chess' : '將 Xiangqi'} (${topIsHuman ? 'You' : 'AI'})`;
  document.getElementById('playerBottomName').textContent =
    `${botSide === 'chess' ? '♔ Chess' : '將 Xiangqi'} (${botIsHuman ? 'You' : 'AI'})`;

  document.getElementById('playerTop').classList.toggle('active-turn', gameState.sideToMove === topSide);
  document.getElementById('playerBottom').classList.toggle('active-turn', gameState.sideToMove === botSide);

  /* move history */
  updateMoveHistory();
}

function updateMoveHistory() {
  const el = document.getElementById('moveHistory');
  if (!gameState.moveHistory.length) {
    el.innerHTML = '<p class="text-dim">No moves yet</p>';
    return;
  }
  el.innerHTML = gameState.moveHistory.map((m, i) => `
    <div class="move-row">
      <span class="ply">${i + 1}.</span>
      <span class="side-indicator side-${m.side}"></span>
      <span class="notation">${m.notation}</span>
    </div>
  `).join('');

  el.scrollTop = el.scrollHeight;
}

function updateStatus(text, isYourTurn) {
  document.getElementById('statusState').textContent = text;
  document.getElementById('statusState').style.color =
    isYourTurn ? 'var(--accent-green)' : 'var(--text-secondary)';
}

function showMessage(text) {
  document.getElementById('msgText').textContent = text;
  document.getElementById('gameMessage').style.display = 'block';
}

function hideMessage() {
  document.getElementById('gameMessage').style.display = 'none';
}

function parseMoveNotation(notation) {
  const m = notation.match(/^([a-i])(\d+)-([a-i])(\d+)/);
  if (!m) return null;
  return {
    from: { x: m[1].charCodeAt(0) - 97, y: parseInt(m[2]) - 1 },
    to:   { x: m[3].charCodeAt(0) - 97, y: parseInt(m[4]) - 1 },
  };
}

/* ═══════════════════════════════════════════════
   Control Bindings
   ═══════════════════════════════════════════════ */
function bindControls() {
  /* side picker */
  document.querySelectorAll('.side-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.side-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      gameState.humanSide = btn.dataset.side;
      gameState.flipped = (btn.dataset.side === 'xiangqi');
      if (!gameState.active) {
        const o = getRenderOpts();
        BR.drawBoard(svg, o);
        BR.drawCoordLabels(svg, o);
        drawInitialBoard();
      }
    });
  });

  /* new game */
  document.getElementById('btnNewGame').addEventListener('click', startNewGame);

  /* flip */
  document.getElementById('btnFlip').addEventListener('click', () => {
    gameState.flipped = !gameState.flipped;
    updateUI();
  });

  /* resign */
  document.getElementById('btnResign').addEventListener('click', async () => {
    if (!gameState.active) return;
    if (confirm('Are you sure you want to resign?')) {
      try {
        const data = await apiCall('resign', {});
        handleGameOver(data);
      } catch (err) {
        console.error('Resign failed:', err);
      }
    }
  });

  /* undo */
  document.getElementById('btnUndo').addEventListener('click', async () => {
    if (!gameState.active) return;
    try {
      const data = await apiCall('undo', {});
      gameState.moveHistory.splice(-2); /* remove human + AI move */
      updateFromServerState(data);
      updateUI();
    } catch (err) {
      console.error('Undo failed:', err);
    }
  });

  /* board click */
  setupBoardClick();
}

/* ── boot ── */
init();
