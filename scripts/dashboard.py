"""Live progress dashboard for runs/.

Usage::

    # In a separate terminal while the orchestrator runs.
    python -m scripts.dashboard

This polls every 30 seconds and writes ``runs/progress.html``.
Open that file in a browser — the page contains a meta-refresh tag, so it
reloads itself.

The dashboard reads, per variant directory:
  * ``metrics.csv`` — iteration progress, losses, eval results, draws/wins
  * ``orchestrator_status.json`` — which variant is currently running

The script exits cleanly on Ctrl-C.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REFRESH_SECONDS = 30


def read_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def fmt_pct(val) -> str:
    try:
        return f"{float(val) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_num(val, places: int = 3) -> str:
    try:
        return f"{float(val):.{places}f}"
    except (TypeError, ValueError):
        return "—"


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def variant_block(name: str, ablation: str, outdir: Path,
                  state: str, target_iters: int) -> str:
    rows = read_csv_rows(outdir / "metrics.csv")
    n_iters = len(rows)
    last = rows[-1] if rows else {}

    pct = min(100.0, 100.0 * n_iters / max(target_iters, 1))
    state_color = {
        "running": "#FFC857",
        "done":    "#7BC47F",
        "failed":  "#E15554",
        "pending": "#CCCCCC",
    }.get(state, "#CCCCCC")

    last_loss = (
        f"p={fmt_num(last.get('policy_loss'), 4)} "
        f"v={fmt_num(last.get('value_loss'), 4)}"
        if last else "—"
    )
    eval_random = (
        f"W={last.get('eval_random_w','')} "
        f"D={last.get('eval_random_d','')} "
        f"L={last.get('eval_random_l','')}"
        if last and str(last.get('eval_random_w', '')).strip() != "" else "—"
    )
    eval_ab = (
        f"W={last.get('eval_ab_w','')} "
        f"D={last.get('eval_ab_d','')} "
        f"L={last.get('eval_ab_l','')}"
        if last and str(last.get('eval_ab_w', '')).strip() != "" else "—"
    )
    draws = "—"
    decisive = "—"
    if last:
        try:
            decisive = (
                f"chess={last.get('sp_chess_wins','?')} "
                f"xq={last.get('sp_xiangqi_wins','?')} "
                f"draws={last.get('sp_draws','?')}"
            )
            draws = (
                f"limit={last.get('sp_draw_move_limit','?')} "
                f"3fold={last.get('sp_draw_threefold','?')} "
                f"adj={last.get('sp_draw_adjudicated','?')}"
            )
        except Exception:
            pass

    return f"""
    <div class="variant {state}">
      <div class="row1">
        <span class="badge" style="background:{state_color}">{html.escape(state)}</span>
        <span class="name">{html.escape(name)}</span>
        <span class="abl">{html.escape(ablation)}</span>
        <span class="pct">{n_iters}/{target_iters} iters · {pct:.0f}%</span>
      </div>
      <div class="bar"><div class="fill" style="width:{pct:.1f}%;background:{state_color}"></div></div>
      <div class="grid">
        <div><b>last loss</b><br>{last_loss}</div>
        <div><b>vs Random</b><br>{eval_random}</div>
        <div><b>vs AB(d=1)</b><br>{eval_ab}</div>
        <div><b>self-play outcomes</b><br>{decisive}</div>
        <div><b>draws (last iter)</b><br>{draws}</div>
      </div>
    </div>
    """


def render_html(root: Path) -> str:
    status_p = root / "orchestrator_status.json"
    status = {}
    if status_p.exists():
        try:
            with open(status_p, "r", encoding="utf-8") as f:
                status = json.load(f)
        except Exception:
            status = {}

    target_iters = (status.get("config") or {}).get("iterations", 50)
    variants = status.get("variants") or []
    if not variants:
        variants = []

    started_at = parse_iso(status.get("started_at"))
    finished_at = parse_iso(status.get("finished_at"))
    now = datetime.now(timezone.utc)

    elapsed = "—"
    if started_at:
        end = finished_at or now
        secs = int((end - started_at).total_seconds())
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        elapsed = f"{h}h {m:02d}m {s:02d}s"

    n_done = sum(1 for v in variants if v.get("state") == "done")
    n_running = sum(1 for v in variants if v.get("state") == "running")
    n_failed = sum(1 for v in variants if v.get("state") == "failed")
    n_pending = sum(1 for v in variants if v.get("state") == "pending")

    # ETA based on completed-variant pace.
    eta_str = "—"
    if started_at and n_done > 0 and n_pending + n_running > 0:
        secs = (now - started_at).total_seconds()
        per_variant = secs / max(n_done, 1)
        remaining = (n_pending + n_running) * per_variant
        eh, rem = divmod(int(remaining), 3600)
        em, _ = divmod(rem, 60)
        eta_str = f"~{eh}h {em:02d}m"

    blocks = []
    for v in variants:
        blocks.append(variant_block(
            name=v.get("name", "?"),
            ablation=v.get("ablation", "?"),
            outdir=Path(v.get("outdir", root / v.get("name", "?"))),
            state=v.get("state", "pending"),
            target_iters=target_iters,
        ))

    current_name = "—"
    if status.get("current"):
        current_name = status["current"].get("name", "—")

    body = f"""
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
<title>training progress · hybrid chess</title>
<style>
  body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
       margin:24px;background:#f7f7f7;color:#222;}}
  h1{{margin:0 0 8px 0;}}
  .meta{{color:#666;margin-bottom:18px;font-size:14px;}}
  .summary{{display:flex;gap:16px;margin-bottom:24px;}}
  .summary div{{padding:10px 14px;background:#fff;border-radius:6px;
               box-shadow:0 1px 2px rgba(0,0,0,.08);}}
  .summary b{{font-size:20px;}}
  .variant{{background:#fff;padding:14px 18px;border-radius:8px;
            margin-bottom:12px;box-shadow:0 1px 2px rgba(0,0,0,.08);}}
  .row1{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;}}
  .badge{{color:#222;padding:2px 8px;border-radius:4px;
         font-size:11px;font-weight:bold;text-transform:uppercase;}}
  .name{{font-weight:bold;font-size:15px;}}
  .abl{{color:#666;font-size:12px;font-family:monospace;}}
  .pct{{margin-left:auto;color:#666;font-size:13px;}}
  .bar{{height:8px;background:#eee;border-radius:4px;margin:8px 0;
        overflow:hidden;}}
  .fill{{height:100%;}}
  .grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;
         margin-top:8px;font-size:12px;color:#444;}}
  .grid b{{color:#222;font-size:11px;text-transform:uppercase;
          letter-spacing:.4px;}}
</style>
</head><body>
<h1>9-variant AlphaZero training</h1>
<div class="meta">
  Auto-refresh every {REFRESH_SECONDS}s · last regenerated
  {now.astimezone().strftime('%Y-%m-%d %H:%M:%S')} ·
  root: <code>{html.escape(str(root))}</code>
</div>

<div class="summary">
  <div>done<br><b>{n_done}/{len(variants)}</b></div>
  <div>running<br><b>{n_running}</b></div>
  <div>pending<br><b>{n_pending}</b></div>
  <div>failed<br><b style="color:#E15554">{n_failed}</b></div>
  <div>elapsed<br><b>{elapsed}</b></div>
  <div>ETA remaining<br><b>{eta_str}</b></div>
  <div>current<br><b>{html.escape(current_name)}</b></div>
</div>

{''.join(blocks)}

</body></html>
"""
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="runs")
    ap.add_argument("--out", type=str, default="",
                    help="output html path (default: <root>/progress.html)")
    ap.add_argument("--once", action="store_true",
                    help="render once and exit (default: loop)")
    ap.add_argument("--interval", type=int, default=REFRESH_SECONDS,
                    help=f"poll interval seconds (default: {REFRESH_SECONDS})")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else root / "progress.html"

    print(f"[dashboard] writing {out} every {args.interval}s "
          f"(Ctrl-C to stop) ...")
    try:
        while True:
            try:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(render_html(root))
            except Exception as e:
                print(f"[dashboard] render failed: {e}")
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[dashboard] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
