"""Real-time training progress dashboard.

Usage:
    python scripts/training_dashboard.py --run-dir runs/rq4_default --port 8050

Open http://localhost:8050 in your browser.
"""

import argparse
import csv
import json
import os
import re
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ── HTML Dashboard (embedded) ──

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    min-height: 100vh;
    padding: 24px;
  }
  .header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 28px;
  }
  .header h1 {
    font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .header .status {
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem;
    padding: 6px 14px; border-radius: 20px;
    background: #1a1d2e; border: 1px solid #2d3348;
  }
  .status.running { border-color: #34d399; color: #34d399; }
  .status.done { border-color: #60a5fa; color: #60a5fa; }
  .status.waiting { border-color: #fbbf24; color: #fbbf24; }

  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .card {
    background: #1a1d2e; border-radius: 12px; padding: 20px;
    border: 1px solid #2d3348; transition: border-color 0.3s;
  }
  .card:hover { border-color: #4a5173; }
  .card .label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .card .value { font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
  .card .sub { font-size: 0.75rem; color: #666; margin-top: 4px; }

  .progress-section { margin-bottom: 20px; }
  .progress-bar-outer {
    background: #1a1d2e; border-radius: 10px; height: 28px; overflow: hidden;
    border: 1px solid #2d3348; position: relative;
  }
  .progress-bar-inner {
    height: 100%; border-radius: 10px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    transition: width 0.8s ease;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 600; color: white;
    font-family: 'JetBrains Mono', monospace;
    min-width: 40px;
  }
  .progress-label {
    display: flex; justify-content: space-between;
    font-size: 0.8rem; color: #888; margin-top: 6px;
  }

  .phase-indicator {
    display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap;
  }
  .phase {
    padding: 6px 16px; border-radius: 8px; font-size: 0.75rem;
    font-weight: 600; background: #1a1d2e; border: 1px solid #2d3348;
    color: #666; transition: all 0.3s;
  }
  .phase.active {
    background: linear-gradient(135deg, #1e3a5f, #2d1b69);
    border-color: #60a5fa; color: #60a5fa;
    box-shadow: 0 0 12px rgba(96, 165, 250, 0.15);
  }
  .phase.done { color: #34d399; border-color: #065f46; }

  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .full-width { grid-column: 1 / -1; }

  .log-box {
    background: #0d0f15; border-radius: 12px; padding: 16px;
    border: 1px solid #2d3348; max-height: 320px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    line-height: 1.6; color: #9ca3af;
  }
  .log-box .highlight { color: #60a5fa; }
  .log-box .success { color: #34d399; }
  .log-box .warn { color: #fbbf24; }
  .log-box .error { color: #ef4444; }

  .metrics-table {
    width: 100%; border-collapse: collapse; font-size: 0.78rem;
  }
  .metrics-table th {
    text-align: left; padding: 8px 12px; color: #888;
    border-bottom: 1px solid #2d3348; font-weight: 600;
    text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px;
  }
  .metrics-table td {
    padding: 8px 12px; border-bottom: 1px solid #1e2130;
    font-family: 'JetBrains Mono', monospace;
  }
  .metrics-table tr:hover { background: #1e2130; }

  .mini-chart { height: 120px; display: flex; align-items: flex-end; gap: 3px; padding-top: 10px; }
  .mini-bar {
    flex: 1; background: linear-gradient(to top, #3b82f6, #8b5cf6);
    border-radius: 3px 3px 0 0; min-width: 8px;
    transition: height 0.5s ease;
  }

  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
  .pulse { animation: pulse 2s ease-in-out infinite; }
</style>
</head>
<body>

<div class="header">
  <h1>🏁 AlphaZero Training Dashboard</h1>
  <div class="status waiting" id="status">⏳ Connecting...</div>
</div>

<div class="progress-section">
  <div class="progress-bar-outer">
    <div class="progress-bar-inner" id="mainProgress" style="width: 0%">0%</div>
  </div>
  <div class="progress-label">
    <span id="progressLeft">Iteration 0 / 20</span>
    <span id="progressRight">ETA: calculating...</span>
  </div>
</div>

<div class="phase-indicator" id="phases">
  <div class="phase" data-phase="selfplay">🎮 Self-Play</div>
  <div class="phase" data-phase="train">🧠 Training</div>
  <div class="phase" data-phase="gate">🚪 Gating</div>
  <div class="phase" data-phase="eval">📊 Evaluation</div>
</div>

<div class="grid">
  <div class="card">
    <div class="label">Current Iteration</div>
    <div class="value" id="currentIter">—</div>
    <div class="sub" id="iterSub">waiting for data</div>
  </div>
  <div class="card">
    <div class="label">Total Loss</div>
    <div class="value" id="totalLoss">—</div>
    <div class="sub" id="lossSub"></div>
  </div>
  <div class="card">
    <div class="label">Elapsed / ETA</div>
    <div class="value" id="elapsed">—</div>
    <div class="sub" id="etaSub"></div>
  </div>
</div>

<div class="two-col">
  <div class="card">
    <div class="label">Loss Trend</div>
    <div class="mini-chart" id="lossChart"></div>
  </div>
  <div class="card">
    <div class="label">Decisive Rate Trend</div>
    <div class="mini-chart" id="decisiveChart"></div>
  </div>
</div>

<div class="two-col">
  <div class="card">
    <div class="label">Completed Iterations</div>
    <div style="max-height: 250px; overflow-y: auto;">
      <table class="metrics-table">
        <thead><tr>
          <th>Iter</th><th>P.Loss</th><th>V.Loss</th><th>Total</th>
          <th>vs Rand</th><th>vs AB</th><th>Gate</th><th>Time</th>
        </tr></thead>
        <tbody id="metricsBody"></tbody>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="label">Live Log (last 30 lines)</div>
    <div class="log-box" id="logBox">Waiting for log data...</div>
  </div>
</div>

<div class="card full-width" style="margin-top: 4px;">
  <div class="label">Self-Play Side Balance (Chess vs Xiangqi)</div>
  <div id="sideBalance" style="font-family:'JetBrains Mono',monospace; font-size:0.82rem; margin-top:10px; color:#888;">
    Waiting for self-play data...
  </div>
</div>

<script>
const POLL_MS = 3000;
const TOTAL_ITERS = 20;

function formatTime(s) {
  if (!s || s < 0) return '—';
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${Math.floor(s % 60)}s`;
}

function colorLine(line) {
  if (/PASS|accept|Updated|OK/.test(line)) return 'success';
  if (/FAIL|REJECT|WARNING/.test(line)) return 'warn';
  if (/Error|error|Traceback/.test(line)) return 'error';
  if (/Iteration|Self-play|Train|Gating|Eval/.test(line)) return 'highlight';
  return '';
}

function renderChart(containerId, values, maxVal) {
  const el = document.getElementById(containerId);
  if (!values.length) { el.innerHTML = '<span style="color:#555;font-size:0.7rem;">no data</span>'; return; }
  const mx = maxVal || Math.max(...values);
  el.innerHTML = values.map(v =>
    `<div class="mini-bar" style="height:${Math.max(5, (v/mx)*100)}%" title="${v.toFixed(3)}"></div>`
  ).join('');
}

async function poll() {
  try {
    const resp = await fetch('/api/status');
    const data = await resp.json();

    // Status
    const statusEl = document.getElementById('status');
    if (data.completed >= TOTAL_ITERS) {
      statusEl.textContent = '✅ Complete'; statusEl.className = 'status done';
    } else if (data.completed >= 0) {
      statusEl.textContent = '🔄 Training'; statusEl.className = 'status running';
    } else {
      statusEl.textContent = '⏳ Waiting'; statusEl.className = 'status waiting';
    }

    // Progress
    const microPct = data.micro_progress || 0;
    const totalPct = Math.min(100, ((data.completed + microPct) / TOTAL_ITERS * 100));
    const bar = document.getElementById('mainProgress');
    bar.style.width = totalPct.toFixed(1) + '%';
    bar.textContent = totalPct.toFixed(1) + '%';
    document.getElementById('progressLeft').textContent =
      `Iteration ${data.completed} / ${TOTAL_ITERS}` + (data.current_phase ? ` — ${data.current_phase}` : '');

    // ETA
    if (data.avg_iter_seconds && data.completed < TOTAL_ITERS) {
      const remaining = (TOTAL_ITERS - data.completed - microPct) * data.avg_iter_seconds;
      document.getElementById('progressRight').textContent = `ETA: ${formatTime(remaining)}`;
    }

    // Cards
    document.getElementById('currentIter').textContent = data.completed;
    document.getElementById('iterSub').textContent = data.current_phase || '';
    if (data.rows && data.rows.length) {
      const last = data.rows[data.rows.length - 1];
      document.getElementById('totalLoss').textContent = parseFloat(last.total_loss || 0).toFixed(3);
      document.getElementById('lossSub').textContent =
        `P: ${parseFloat(last.policy_loss||0).toFixed(3)}  V: ${parseFloat(last.value_loss||0).toFixed(3)}`;
    }
    document.getElementById('elapsed').textContent = formatTime(data.elapsed_seconds);
    document.getElementById('etaSub').textContent =
      data.avg_iter_seconds ? `~${formatTime(data.avg_iter_seconds)}/iter` : '';

    // Charts
    if (data.rows) {
      renderChart('lossChart', data.rows.map(r => parseFloat(r.total_loss || 0)));
      renderChart('decisiveChart', data.rows.map(r => {
        const g = parseInt(r.sp_games || 0);
        const d = parseInt(r.sp_decisive || 0);
        return g > 0 ? d/g : 0;
      }), 1.0);
    }

    // Table
    const tbody = document.getElementById('metricsBody');
    if (data.rows) {
      tbody.innerHTML = data.rows.map(r => `<tr>
        <td>${r.iter}</td>
        <td>${parseFloat(r.policy_loss||0).toFixed(3)}</td>
        <td>${parseFloat(r.value_loss||0).toFixed(3)}</td>
        <td>${parseFloat(r.total_loss||0).toFixed(3)}</td>
        <td>${r.eval_random_w||'-'}/${r.eval_random_d||'-'}/${r.eval_random_l||'-'}</td>
        <td>${r.eval_ab_w||'-'}/${r.eval_ab_d||'-'}/${r.eval_ab_l||'-'}</td>
        <td style="color:${r.gate==='True'||r.gate==='1'?'#34d399':'#ef4444'}">${r.gate||'-'}</td>
        <td>${r.selfplay_seconds ? formatTime(parseFloat(r.selfplay_seconds)) : '-'}</td>
      </tr>`).join('');
    }

    // Log
    const logBox = document.getElementById('logBox');
    if (data.log_lines && data.log_lines.length) {
      logBox.innerHTML = data.log_lines.map(l =>
        `<div class="${colorLine(l)}">${l.replace(/</g,'&lt;')}</div>`
      ).join('');
      logBox.scrollTop = logBox.scrollHeight;
    }

    // Side balance
    if (data.rows && data.rows.length) {
      const sb = document.getElementById('sideBalance');
      const lines = data.rows.filter(r => r.sp_chess_wins).map(r =>
        `Iter ${r.iter}: Chess ${r.sp_chess_wins}W  Xiangqi ${r.sp_xiangqi_wins}W  Draw ${r.sp_draws}`
      );
      sb.innerHTML = lines.length ? lines.join('<br>') : 'Side data will appear after first iteration completes';
    }

    // Phases
    const phaseMap = { 'Self-play': 'selfplay', 'Train': 'train', 'Gating': 'gate', 'Eval': 'eval' };
    document.querySelectorAll('.phase').forEach(el => el.classList.remove('active'));
    if (data.current_phase) {
      for (const [key, val] of Object.entries(phaseMap)) {
        if (data.current_phase.includes(key)) {
          const el = document.querySelector(`.phase[data-phase="${val}"]`);
          if (el) el.classList.add('active');
        }
      }
    }

  } catch (e) {
    document.getElementById('status').textContent = '❌ Disconnected';
    document.getElementById('status').className = 'status';
  }
}

setInterval(poll, POLL_MS);
poll();
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the training dashboard."""

    run_dir = "."
    log_file = "training.log"

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/api/status":
            self._serve_status()
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

    def _serve_status(self):
        run_dir = Path(self.run_dir)
        data = {
            "completed": 0,
            "rows": [],
            "log_lines": [],
            "current_phase": "",
            "micro_progress": 0.0,
            "elapsed_seconds": 0,
            "avg_iter_seconds": None,
        }

        # Read metrics.csv
        csv_path = run_dir / "metrics.csv"
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    data["rows"] = list(reader)
                data["completed"] = len(data["rows"])
            except Exception:
                pass

        # Read log file for live progress
        log_path = run_dir / self.log_file
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                data["log_lines"] = [l.rstrip() for l in lines[-40:]]

                # Parse micro-progress from log
                full_text = "".join(lines[-80:])

                # Detect current phase
                for phase in ["[Eval]", "[Gating]", "[Train]", "[Self-play]"]:
                    # Find last occurrence
                    idx = full_text.rfind(phase)
                    if idx >= 0:
                        data["current_phase"] = phase.strip("[]")
                        break

                # Parse self-play game progress: "game X/Y"
                game_matches = re.findall(r"game\s+(\d+)/(\d+)", full_text)
                if game_matches:
                    last = game_matches[-1]
                    done, total = int(last[0]), int(last[1])
                    # Self-play is ~60% of iteration time
                    data["micro_progress"] = (done / max(total, 1)) * 0.6

                # Parse elapsed time
                iter_times = re.findall(r"Iteration.*?(\d+\.\d+)s total", full_text)
                if not iter_times:
                    # Try to find elapsed from self-play
                    elapsed_matches = re.findall(r"elapsed=(\d+\.?\d*)s", full_text)
                    if elapsed_matches:
                        pass  # just for progress

            except Exception:
                pass

        # Calculate timing
        if data["rows"]:
            sp_times = [float(r.get("selfplay_seconds", 0) or 0) for r in data["rows"]]
            if sp_times:
                # Rough estimate: self-play is ~70% of total iter time
                avg_sp = sum(sp_times) / len(sp_times)
                data["avg_iter_seconds"] = avg_sp / 0.7

            # Total elapsed from first to last
            total_sp = sum(sp_times)
            data["elapsed_seconds"] = total_sp / 0.7  # rough

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Training progress dashboard")
    parser.add_argument("--run-dir", required=True, help="Path to training run directory")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--log-file", default="training.log")
    args = parser.parse_args()

    DashboardHandler.run_dir = os.path.abspath(args.run_dir)
    DashboardHandler.log_file = args.log_file

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"\n  [Training Dashboard]")
    print(f"  ======================")
    print(f"  Run dir:  {args.run_dir}")
    print(f"  Log file: {args.log_file}")
    print(f"  URL:      http://localhost:{args.port}")
    print(f"\n  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
