/**
 * Hybrid Chess — Replay Page Logic
 *
 * Loads JSON/JSONL game files, provides step-through controls,
 * move list navigation, and auto-play functionality.
 */

const BR = window.BoardRenderer;
const svg = document.getElementById('boardSvg');
const opts = BR.defaultOpts();

/* ── state ── */
let games = [];
let currentGameIdx = 0;
let currentStep = 0;
let autoTimer = null;

/* ── init ── */
function init() {
  BR.drawBoard(svg, opts);
  BR.drawCoordLabels(svg, opts);
  BR.drawPieces(svg, initialBoard(), opts);
  buildLegends();
  bindControls();
}

/* ── default initial board ASCII ── */
function initialBoard() {
  return [
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
}

/* ── legends ── */
function buildLegends() {
  const chessEl = document.getElementById('chessLegend');
  const xiangqiEl = document.getElementById('xiangqiLegend');

  const chessData = [
    { icon: '♔', name: 'King', desc: 'All 8 dirs, 1 step' },
    { icon: '♕', name: 'Queen', desc: 'Orth + diag slide' },
    { icon: '♖', name: 'Rook', desc: 'Orth slide' },
    { icon: '♗', name: 'Bishop', desc: 'Diag slide' },
    { icon: '♘', name: 'Knight', desc: 'L-shape, no block' },
    { icon: '♙', name: 'Pawn', desc: 'Forward, promotes at y≥9' },
  ];
  const xiangqiData = [
    { ch: '將', name: 'General', desc: 'Orth 1-step, palace' },
    { ch: '士', name: 'Advisor', desc: 'Diag 1-step, palace' },
    { ch: '象', name: 'Elephant', desc: 'Diag 2-step, eye block' },
    { ch: '馬', name: 'Horse', desc: 'L-shape, leg block' },
    { ch: '車', name: 'Chariot', desc: 'Orth slide' },
    { ch: '砲', name: 'Cannon', desc: 'Screen jump capture' },
    { ch: '卒', name: 'Soldier', desc: 'Forward, +sideways after river' },
  ];

  chessEl.innerHTML = chessData.map(p => `
    <div class="legend-item">
      <div class="legend-piece chess">${p.icon}</div>
      <div class="legend-info">
        <div class="name">${p.name}</div>
        <div class="desc">${p.desc}</div>
      </div>
    </div>
  `).join('');

  xiangqiEl.innerHTML = xiangqiData.map(p => `
    <div class="legend-item">
      <div class="legend-piece xiangqi">${p.ch}</div>
      <div class="legend-info">
        <div class="name">${p.name}</div>
        <div class="desc">${p.desc}</div>
      </div>
    </div>
  `).join('');
}

/* ── load game data ── */
function loadGameData(data) {
  if (Array.isArray(data)) {
    games = data;
  } else {
    games = [data];
  }
  currentGameIdx = 0;
  showGame(0);
  updateGameSelect();
}

function showGame(idx) {
  currentGameIdx = idx;
  currentStep = 0;
  const g = games[idx];
  if (!g || !g.states_ascii) return;

  /* update info */
  document.getElementById('infoStatus').textContent = 'Loaded';
  document.getElementById('infoResult').textContent = g.result || '—';
  document.getElementById('infoPlies').textContent = g.meta?.plies ?? g.moves?.length ?? '—';
  document.getElementById('infoReason').textContent = g.meta?.reason || '—';

  /* slider */
  const slider = document.getElementById('stepSlider');
  slider.max = g.states_ascii.length - 1;
  slider.value = 0;

  /* move list */
  buildMoveList(g.moves || []);
  renderStep(0);
}

function renderStep(step) {
  const g = games[currentGameIdx];
  if (!g || !g.states_ascii) return;
  step = Math.max(0, Math.min(step, g.states_ascii.length - 1));
  currentStep = step;

  BR.drawPieces(svg, g.states_ascii[step], opts);
  BR.clearHighlights(svg);

  /* highlight last move */
  if (step > 0 && g.moves && g.moves[step - 1]) {
    const mv = parseMove(g.moves[step - 1]);
    if (mv) BR.highlightMove(svg, mv.from, mv.to, opts);
  }

  document.getElementById('stepLabel').textContent = `${step} / ${g.states_ascii.length - 1}`;
  document.getElementById('stepSlider').value = step;

  /* update move list active */
  document.querySelectorAll('.move-item').forEach((el, i) => {
    el.classList.toggle('active', i === step - 1);
  });
}

function parseMove(notation) {
  const m = notation.match(/^([a-i])(\d+)-([a-i])(\d+)/);
  if (!m) return null;
  return {
    from: { x: m[1].charCodeAt(0) - 97, y: parseInt(m[2]) - 1 },
    to:   { x: m[3].charCodeAt(0) - 97, y: parseInt(m[4]) - 1 },
  };
}

/* ── move list ── */
function buildMoveList(moves) {
  const el = document.getElementById('moveList');
  if (!moves.length) {
    el.innerHTML = '<p class="text-dim">No moves</p>';
    return;
  }
  el.innerHTML = moves.map((mv, i) => `
    <div class="move-item" data-step="${i + 1}">
      <span class="ply">${i + 1}.</span>
      <span class="notation">${escHtml(mv)}</span>
    </div>
  `).join('');

  el.querySelectorAll('.move-item').forEach(item => {
    item.addEventListener('click', () => renderStep(parseInt(item.dataset.step)));
  });
}

/* ── controls ── */
function bindControls() {
  document.getElementById('btnFirst').addEventListener('click', () => renderStep(0));
  document.getElementById('btnPrev').addEventListener('click', () => renderStep(currentStep - 1));
  document.getElementById('btnNext').addEventListener('click', () => renderStep(currentStep + 1));
  document.getElementById('btnLast').addEventListener('click', () => {
    const g = games[currentGameIdx];
    if (g) renderStep(g.states_ascii.length - 1);
  });

  document.getElementById('btnAuto').addEventListener('click', toggleAutoPlay);

  document.getElementById('stepSlider').addEventListener('input', e => {
    renderStep(parseInt(e.target.value));
  });

  document.getElementById('fileInput').addEventListener('change', handleFileLoad);

  document.getElementById('gameSelect').addEventListener('change', e => {
    showGame(parseInt(e.target.value));
  });

  /* keyboard */
  document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft')  { renderStep(currentStep - 1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { renderStep(currentStep + 1); e.preventDefault(); }
    if (e.key === 'Home')       { renderStep(0); e.preventDefault(); }
    if (e.key === 'End')        {
      const g = games[currentGameIdx];
      if (g) renderStep(g.states_ascii.length - 1);
      e.preventDefault();
    }
    if (e.key === ' ') { toggleAutoPlay(); e.preventDefault(); }
  });
}

function toggleAutoPlay() {
  const btn = document.getElementById('btnAuto');
  if (autoTimer) {
    clearInterval(autoTimer);
    autoTimer = null;
    btn.textContent = '▶';
  } else {
    btn.textContent = '⏸';
    autoTimer = setInterval(() => {
      const g = games[currentGameIdx];
      if (!g) return;
      if (currentStep >= g.states_ascii.length - 1) {
        clearInterval(autoTimer);
        autoTimer = null;
        btn.textContent = '▶';
        return;
      }
      renderStep(currentStep + 1);
    }, 600);
  }
}

/* ── file loading ── */
function handleFileLoad(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const text = reader.result.trim();
      let data;
      if (text.startsWith('[')) {
        data = JSON.parse(text);
      } else if (text.startsWith('{')) {
        /* could be single JSON or JSONL */
        if (text.includes('\n{')) {
          data = text.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
        } else {
          data = JSON.parse(text);
        }
      } else {
        data = text.split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
      }
      loadGameData(data);
    } catch (err) {
      alert('Failed to parse file: ' + err.message);
    }
  };
  reader.readAsText(file);
}

function updateGameSelect() {
  const sel = document.getElementById('gameSelect');
  if (games.length <= 1) {
    sel.style.display = 'none';
    return;
  }
  sel.style.display = 'block';
  sel.innerHTML = games.map((g, i) =>
    `<option value="${i}">Game ${i + 1} — ${g.result || '?'}</option>`
  ).join('');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ── boot ── */
init();
