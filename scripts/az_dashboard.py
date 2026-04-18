"""Dashboard v2: Enhanced with per-piece survival charts.

Usage: python scripts/az_dashboard.py runs/rq4_az_nq_allrules
"""
import csv, json, time, sys, os, webbrowser
from pathlib import Path
from datetime import datetime

REFRESH_INTERVAL = 15


def read_metrics(p):
    if not p.exists(): return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f): rows.append(row)
    return rows


def read_log_tail(d, n=30):
    logs = list(d.glob("*.log")) + list(d.glob("log.txt"))
    if not logs: return ["(Waiting for training to start...)"]
    lf = max(logs, key=lambda f: f.stat().st_mtime)
    try: return lf.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except: return ["(Error reading log)"]


def safe_float(row, key, default=0.0):
    v = row.get(key, "")
    try: return float(v) if v else default
    except: return default


def generate_html(run_dir, metrics, log_lines, variant_name="no_queen+ALL_RULES"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    iterations = [int(r.get("iter",0)) for r in metrics]
    chess_wr, xq_wr, draw_pct = [], [], []
    policy_loss, value_loss = [], []
    chess_end_mat, xq_end_mat = [], []
    
    # Per-piece survival
    chess_pieces = ["QUEEN","ROOK","BISHOP","KNIGHT","PAWN"]
    xq_pieces = ["CHARIOT","CANNON","HORSE","ELEPHANT","ADVISOR","SOLDIER"]
    surv_chess = {p: [] for p in chess_pieces}
    surv_xq = {p: [] for p in xq_pieces}
    
    for row in metrics:
        cw = safe_float(row, "sp_chess_wins")
        xw = safe_float(row, "sp_xiangqi_wins")
        dr = safe_float(row, "sp_draws")
        total = cw + xw + dr or 1
        chess_wr.append(round(cw/total*100, 1))
        xq_wr.append(round(xw/total*100, 1))
        draw_pct.append(round(dr/total*100, 1))
        
        policy_loss.append(round(safe_float(row, "policy_loss"), 4))
        value_loss.append(round(safe_float(row, "value_loss"), 4))
        
        chess_end_mat.append(safe_float(row, "sp_chess_end_mat"))
        xq_end_mat.append(safe_float(row, "sp_xq_end_mat"))
        
        for p in chess_pieces:
            surv_chess[p].append(round(safe_float(row, f"surv_chess_{p}") * 100, 1))
        for p in xq_pieces:
            surv_xq[p].append(round(safe_float(row, f"surv_xiangqi_{p}") * 100, 1))
    
    total_iters = len(iterations)
    target_iters = 20
    progress_pct = min(100, round(total_iters / target_iters * 100))
    
    latest = metrics[-1] if metrics else {}
    log_html = "\n".join(f"<div class='log-line'>{l}</div>" for l in log_lines[-25:])
    
    # Piece survival for latest iter bar chart
    latest_surv_labels = []
    latest_surv_values = []
    latest_surv_colors = []
    for p in chess_pieces:
        v = safe_float(latest, f"surv_chess_{p}") * 100
        latest_surv_labels.append(f"C:{p[:4]}")
        latest_surv_values.append(round(v, 1))
        latest_surv_colors.append("'rgba(79,195,247,0.8)'")
    for p in xq_pieces:
        v = safe_float(latest, f"surv_xiangqi_{p}") * 100
        latest_surv_labels.append(f"X:{p[:4]}")
        latest_surv_values.append(round(v, 1))
        latest_surv_colors.append("'rgba(239,83,80,0.8)'")
    
    # Eval data
    eval_iters = [int(r["iter"]) for r in metrics if r.get("eval_random_w")]
    eval_random = [round(safe_float(r,"eval_random_w")/(safe_float(r,"eval_random_w")+safe_float(r,"eval_random_d")+safe_float(r,"eval_random_l") or 1)*100,1) for r in metrics if r.get("eval_random_w")]
    eval_ab = [round(safe_float(r,"eval_ab_w")/(safe_float(r,"eval_ab_w")+safe_float(r,"eval_ab_d")+safe_float(r,"eval_ab_l") or 1)*100,1) for r in metrics if r.get("eval_random_w")]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="{REFRESH_INTERVAL}">
    <title>AZ Training: {variant_name}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%);
            color: #e0e0e0; min-height: 100vh; padding: 16px;
        }}
        .header {{ text-align: center; padding: 14px 0; }}
        .header h1 {{
            font-size: 24px;
            background: linear-gradient(90deg, #ff6b35, #f7c948, #00d2ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .header .sub {{ color: #888; font-size: 13px; margin-top: 4px; }}
        .progress-bar {{
            width: 100%; max-width: 500px; margin: 10px auto; height: 20px;
            background: rgba(255,255,255,0.08); border-radius: 10px; overflow: hidden; position: relative;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff6b35, #f7c948);
            border-radius: 10px; width: {progress_pct}%;
        }}
        .progress-text {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%);
            font-size: 11px; font-weight: bold; color: white; text-shadow: 0 1px 3px rgba(0,0,0,0.5);
        }}
        .grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 12px; max-width: 1400px; margin: 12px auto;
        }}
        .card {{
            background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px;
        }}
        .card h2 {{ font-size: 13px; color: #7b9fff; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
        .stat-row {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 13px; }}
        .stat-label {{ color: #999; }} .stat-value {{ font-weight: bold; }}
        .chess-c {{ color: #4fc3f7; }} .xq-c {{ color: #ef5350; }} .draw-c {{ color: #aaa; }}
        canvas {{ width: 100% !important; height: 180px !important; margin-top: 6px; }}
        .wide {{ grid-column: 1 / -1; }}
        .log-box {{
            background: rgba(0,0,0,0.35); border-radius: 6px; padding: 10px;
            font-family: Consolas, monospace; font-size: 10px; max-height: 220px; overflow-y: auto; line-height: 1.5;
        }}
        .log-line {{ white-space: pre-wrap; word-break: break-all; }}
        .ts {{ color: #555; font-size: 11px; text-align: center; margin-top: 10px; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }}
        .badge-run {{ background: #1b5e20; color: #66bb6a; }}
        .badge-done {{ background: #0d47a1; color: #42a5f5; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="header">
    <h1>⚗️ AZ Training: {variant_name}</h1>
    <div class="sub">20 iters × 100 games × 50 sims | C++ accelerated
        <span class="badge {'badge-run' if progress_pct<100 else 'badge-done'}">{'RUNNING' if progress_pct<100 else 'COMPLETE'}</span>
    </div>
</div>
<div class="progress-bar"><div class="progress-fill"></div>
    <div class="progress-text">{total_iters}/{target_iters} ({progress_pct}%)</div></div>

<div class="grid">
    <!-- Stats -->
    <div class="card">
        <h2>📊 Iter {total_iters} Stats</h2>
        <div class="stat-row"><span class="stat-label">Chess Wins</span><span class="stat-value chess-c">{latest.get('sp_chess_wins','?')}</span></div>
        <div class="stat-row"><span class="stat-label">XQ Wins</span><span class="stat-value xq-c">{latest.get('sp_xiangqi_wins','?')}</span></div>
        <div class="stat-row"><span class="stat-label">Draws</span><span class="stat-value draw-c">{latest.get('sp_draws','?')}</span></div>
        <div class="stat-row"><span class="stat-label">Avg Ply</span><span class="stat-value">{latest.get('sp_avg_ply','?')}</span></div>
        <div class="stat-row"><span class="stat-label">Mat Diff</span><span class="stat-value">{latest.get('sp_avg_mat_diff','?')}</span></div>
        <div class="stat-row"><span class="stat-label">Chess End Mat</span><span class="stat-value chess-c">{latest.get('sp_chess_end_mat','?')}</span></div>
        <div class="stat-row"><span class="stat-label">XQ End Mat</span><span class="stat-value xq-c">{latest.get('sp_xq_end_mat','?')}</span></div>
        <div class="stat-row"><span class="stat-label">Policy Loss</span><span class="stat-value">{policy_loss[-1] if policy_loss else '?'}</span></div>
        <div class="stat-row"><span class="stat-label">Value Loss</span><span class="stat-value">{value_loss[-1] if value_loss else '?'}</span></div>
    </div>
    
    <!-- Balance -->
    <div class="card"><h2>⚖️ Side Balance</h2><canvas id="balChart"></canvas></div>
    <div class="card"><h2>📉 Training Loss</h2><canvas id="lossChart"></canvas></div>
    <div class="card"><h2>💰 End-game Material</h2><canvas id="matChart"></canvas></div>
    
    <!-- Piece Survival Chart (latest) -->
    <div class="card wide"><h2>🛡️ Piece Survival Rate (Latest Iter)</h2><canvas id="survBar" style="height:200px !important;"></canvas></div>
    
    <!-- Chess piece survival over time -->
    <div class="card"><h2>♟️ Chess Piece Survival Trend</h2><canvas id="chessSurvChart"></canvas></div>
    <div class="card"><h2>🐴 XQ Piece Survival Trend</h2><canvas id="xqSurvChart"></canvas></div>
    
    <!-- Eval -->
    <div class="card"><h2>🎯 Eval Win Rate</h2><canvas id="evalChart"></canvas></div>
    
    <!-- Log -->
    <div class="card wide"><h2>📜 Log</h2><div class="log-box">{log_html}</div></div>
</div>
<div class="ts">Updated: {now} · Refresh: {REFRESH_INTERVAL}s</div>

<script>
const I = {json.dumps(iterations)};
const cWR = {json.dumps(chess_wr)}, xWR = {json.dumps(xq_wr)}, dP = {json.dumps(draw_pct)};
const pL = {json.dumps(policy_loss)}, vL = {json.dumps(value_loss)};
const cMat = {json.dumps(chess_end_mat)}, xMat = {json.dumps(xq_end_mat)};
const eI = {json.dumps(eval_iters)}, eR = {json.dumps(eval_random)}, eA = {json.dumps(eval_ab)};
const survLabels = {json.dumps(latest_surv_labels)};
const survVals = {json.dumps(latest_surv_values)};
const survColors = [{','.join(latest_surv_colors)}];

const O = {{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{color:'#ccc',font:{{size:10}}}}}}}},
    scales:{{x:{{ticks:{{color:'#888'}},grid:{{color:'rgba(255,255,255,0.04)'}}}},y:{{ticks:{{color:'#888'}},grid:{{color:'rgba(255,255,255,0.04)'}}}}}}}};

if(I.length>0){{
    new Chart(document.getElementById('balChart'),{{type:'line',data:{{labels:I,datasets:[
        {{label:'Chess%',data:cWR,borderColor:'#4fc3f7',tension:.3,pointRadius:2}},
        {{label:'XQ%',data:xWR,borderColor:'#ef5350',tension:.3,pointRadius:2}},
        {{label:'Draw%',data:dP,borderColor:'#888',borderDash:[4,4],tension:.3,pointRadius:1}}
    ]}},options:{{...O,scales:{{...O.scales,y:{{...O.scales.y,min:0,max:100}}}}}}}});
    
    new Chart(document.getElementById('lossChart'),{{type:'line',data:{{labels:I,datasets:[
        {{label:'Policy',data:pL,borderColor:'#ff9800',tension:.3,pointRadius:2}},
        {{label:'Value',data:vL,borderColor:'#8bc34a',tension:.3,pointRadius:2}}
    ]}},options:O}});
    
    new Chart(document.getElementById('matChart'),{{type:'line',data:{{labels:I,datasets:[
        {{label:'Chess Mat',data:cMat,borderColor:'#4fc3f7',tension:.3,pointRadius:2,fill:false}},
        {{label:'XQ Mat',data:xMat,borderColor:'#ef5350',tension:.3,pointRadius:2,fill:false}}
    ]}},options:O}});
    
    new Chart(document.getElementById('survBar'),{{type:'bar',data:{{labels:survLabels,datasets:[
        {{data:survVals,backgroundColor:survColors,borderWidth:0}}
    ]}},options:{{...O,plugins:{{legend:{{display:false}}}},scales:{{...O.scales,y:{{...O.scales.y,min:0,max:100,title:{{display:true,text:'Survival %',color:'#888'}}}}}}}}}});
    
    // Chess pieces over time
    const chessPieces = {json.dumps(chess_pieces)};
    const chessSurvData = {json.dumps({p: surv_chess[p] for p in chess_pieces})};
    const chessColors = ['#ff9800','#4fc3f7','#ab47bc','#66bb6a','#ffeb3b'];
    new Chart(document.getElementById('chessSurvChart'),{{type:'line',data:{{labels:I,datasets:
        chessPieces.map((p,i)=>({{label:p,data:chessSurvData[p],borderColor:chessColors[i],tension:.3,pointRadius:2}}))
    }},options:{{...O,scales:{{...O.scales,y:{{...O.scales.y,min:0,max:100}}}}}}}});
    
    // XQ pieces over time
    const xqPieces = {json.dumps(xq_pieces)};
    const xqSurvData = {json.dumps({p: surv_xq[p] for p in xq_pieces})};
    const xqColors = ['#ef5350','#ff7043','#ffa726','#66bb6a','#ab47bc','#78909c'];
    new Chart(document.getElementById('xqSurvChart'),{{type:'line',data:{{labels:I,datasets:
        xqPieces.map((p,i)=>({{label:p,data:xqSurvData[p],borderColor:xqColors[i],tension:.3,pointRadius:2}}))
    }},options:{{...O,scales:{{...O.scales,y:{{...O.scales.y,min:0,max:100}}}}}}}});
}}

if(eI.length>0){{
    new Chart(document.getElementById('evalChart'),{{type:'line',data:{{labels:eI,datasets:[
        {{label:'vs Random',data:eR,borderColor:'#66bb6a',tension:.3,pointRadius:3}},
        {{label:'vs AB d1',data:eA,borderColor:'#ab47bc',tension:.3,pointRadius:3}}
    ]}},options:{{...O,scales:{{...O.scales,y:{{...O.scales.y,min:0,max:100}}}}}}}});
}}

document.querySelector('.log-box')?.scrollTo(0,99999);
</script>
</body></html>"""
    return html


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/az_dashboard.py <run_dir> [variant_name]")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    run_dir.mkdir(parents=True, exist_ok=True)
    variant_name = sys.argv[2] if len(sys.argv) > 2 else "no_queen+ALL_RULES"
    html_path = run_dir / "dashboard.html"
    metrics_path = run_dir / "metrics.csv"
    print(f"Dashboard: {html_path.resolve()}")
    opened = False
    while True:
        try:
            m = read_metrics(metrics_path)
            l = read_log_tail(run_dir)
            html_path.write_text(generate_html(run_dir, m, l, variant_name), encoding="utf-8")
            if not opened:
                webbrowser.open(str(html_path.resolve()))
                opened = True
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(m)} iters", end="\r")
            time.sleep(REFRESH_INTERVAL)
        except KeyboardInterrupt:
            print("\nStopped."); break
        except Exception as e:
            print(f"Error: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()
