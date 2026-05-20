"""Post-hoc tournament analysis for cross_variant_tournament.

Reads one or more game_records.json files, merges them, and writes:

  * per_side.csv        — for each agent, win/draw/loss as Chess side vs
                           as Xiangqi side, with Wilson 95% CIs
  * decisive_rate.csv   — for each ordered pair, fraction of decisive
                           games (non-draw) + breakdown by termination
                           reason
  * game_length.csv     — for each ordered pair, mean / median / std of
                           game length in plies, split by outcome
  * pairwise_ci.csv     — symmetric pair score with Wilson CI; flag
                           "significant" if CI excludes 0.5
  * elo.csv             — maximum-likelihood Elo (Bradley-Terry on
                           score), with bootstrap 95% CIs
  * report.md           — markdown summary tying it all together

Usage (single corpus)::

    python -m scripts.tournament_analysis \\
        --records runs/cross_variant_tournament/game_records.json \\
        --outdir runs/cross_variant_tournament/analysis

Usage (merged 100 + 400 follow-up)::

    python -m scripts.tournament_analysis \\
        --records runs/cross_variant_tournament/game_records.json \\
                  runs/cross_variant_tournament_ext/game_records.json \\
        --outdir runs/cross_variant_tournament/analysis_500
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass


# Agent display order = order in the tournament script.
AGENT_ORDER = [
    "Default", "noQ", "xqQueen", "PK", "PK_noPromo",
    "PK_xqQueen", "noQ_noPromo", "noQ_PK", "noQ_ALL",
]


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI. Accepts fractional successes (for score with
    draws counted as 0.5) — this is a known approximation that is fine for
    visualisation but should not be over-interpreted."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_records(paths: list[Path]) -> list[dict]:
    """Read + concatenate. Deduplicates by (name_a, name_b, a_is_chess, seed)
    so accidental double-loading of the same file is a no-op."""
    seen: set[tuple] = set()
    merged: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            recs = json.load(f)
        kept = 0
        for r in recs:
            key = (r["name_a"], r["name_b"], r["a_is_chess"], r["seed"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(r)
            kept += 1
        print(f"  {p}: {len(recs)} records, {kept} new", file=sys.stderr)
    print(f"Total unique games: {len(merged)}", file=sys.stderr)
    return merged


# ── 1. Per-side breakdown ─────────────────────────────────────

def per_side_stats(records: list[dict], outdir: Path) -> dict:
    """For each agent, split performance by which side they played.

    A record contributes:
      * to (name_a, side=Chess if a_is_chess else Xiangqi) with outcome win_a
      * to (name_b, side=Xiangqi if a_is_chess else Chess) with outcome win_b
    """
    # stats[(agent, side)] = [W, D, L]
    stats: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])

    for r in records:
        a, b = r["name_a"], r["name_b"]
        a_side = "Chess" if r["a_is_chess"] else "Xiangqi"
        b_side = "Xiangqi" if r["a_is_chess"] else "Chess"
        out = r["outcome"]
        if out == "win_a":
            stats[(a, a_side)][0] += 1
            stats[(b, b_side)][2] += 1
        elif out == "win_b":
            stats[(a, a_side)][2] += 1
            stats[(b, b_side)][0] += 1
        else:
            stats[(a, a_side)][1] += 1
            stats[(b, b_side)][1] += 1

    rows = []
    by_agent: dict[str, dict] = {}
    for agent in AGENT_ORDER:
        entry: dict = {}
        for side in ("Chess", "Xiangqi"):
            W, D, L = stats[(agent, side)]
            N = W + D + L
            score = (W + 0.5 * D) / N if N > 0 else 0.0
            lo, hi = wilson_ci(W + 0.5 * D, N)
            rows.append({
                "agent": agent, "side": side,
                "wins": W, "draws": D, "losses": L, "n": N,
                "score": score, "ci_low": lo, "ci_high": hi,
            })
            entry[side] = {"score": score, "ci": (lo, hi),
                           "W": W, "D": D, "L": L, "N": N}
        entry["asymmetry"] = entry["Chess"]["score"] - entry["Xiangqi"]["score"]
        by_agent[agent] = entry

    with open(outdir / "per_side.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "agent", "side", "wins", "draws", "losses", "n",
            "score", "ci_low", "ci_high",
        ])
        w.writeheader()
        for row in rows:
            row = {**row,
                   "score": f"{row['score']:.4f}",
                   "ci_low": f"{row['ci_low']:.4f}",
                   "ci_high": f"{row['ci_high']:.4f}"}
            w.writerow(row)

    return by_agent


# ── 2. Decisive rate per ordered pair ─────────────────────────

def decisive_stats(records: list[dict], outdir: Path) -> dict:
    """For each unordered pair, fraction of games that ended with a winner
    (no draws) + breakdown by termination reason."""
    pair_stats: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "decisive": 0, "draws": 0, "reasons": defaultdict(int)})

    for r in records:
        key = tuple(sorted((r["name_a"], r["name_b"])))
        st = pair_stats[key]
        st["n"] += 1
        if r["outcome"] == "draw":
            st["draws"] += 1
        else:
            st["decisive"] += 1
        st["reasons"][r["reason"] or "Unknown"] += 1

    rows = []
    for (a, b), st in pair_stats.items():
        rate = st["decisive"] / st["n"] if st["n"] else 0.0
        rows.append({
            "agent_a": a, "agent_b": b, "n": st["n"],
            "decisive": st["decisive"], "draws": st["draws"],
            "decisive_rate": rate,
            "reasons": dict(st["reasons"]),
        })
    rows.sort(key=lambda r: (-r["decisive_rate"], r["agent_a"], r["agent_b"]))

    with open(outdir / "decisive_rate.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_a", "agent_b", "n", "decisive", "draws",
                    "decisive_rate", "top_reason", "top_reason_count"])
        for r in rows:
            top = max(r["reasons"].items(), key=lambda x: x[1])
            w.writerow([r["agent_a"], r["agent_b"], r["n"], r["decisive"],
                        r["draws"], f"{r['decisive_rate']:.4f}", top[0], top[1]])

    return {"rows": rows}


# ── 3. Game-length analysis ───────────────────────────────────

def game_length_stats(records: list[dict], outdir: Path) -> dict:
    """For each unordered pair, ply-length distribution (mean, median,
    std), split by overall outcome (decisive vs draw)."""
    pair_lens: dict[tuple, dict] = defaultdict(
        lambda: {"all": [], "decisive": [], "draw": []})

    for r in records:
        key = tuple(sorted((r["name_a"], r["name_b"])))
        L = r["plies"]
        pair_lens[key]["all"].append(L)
        if r["outcome"] == "draw":
            pair_lens[key]["draw"].append(L)
        else:
            pair_lens[key]["decisive"].append(L)

    def _stats(xs):
        if not xs:
            return None, None, None
        m = sum(xs) / len(xs)
        s = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
        med = sorted(xs)[len(xs) // 2]
        return m, med, s

    rows = []
    for (a, b), d in pair_lens.items():
        mA, medA, sA = _stats(d["all"])
        mD, _, _ = _stats(d["decisive"])
        mDr, _, _ = _stats(d["draw"])
        rows.append({
            "agent_a": a, "agent_b": b,
            "n": len(d["all"]),
            "mean_plies": mA, "median_plies": medA, "std_plies": sA,
            "mean_plies_decisive": mD,
            "mean_plies_draw": mDr,
        })
    rows.sort(key=lambda r: -r["mean_plies"])

    with open(outdir / "game_length.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_a", "agent_b", "n", "mean_plies", "median_plies",
                    "std_plies", "mean_plies_decisive", "mean_plies_draw"])
        for r in rows:
            w.writerow([
                r["agent_a"], r["agent_b"], r["n"],
                f"{r['mean_plies']:.1f}" if r["mean_plies"] is not None else "",
                r["median_plies"] if r["median_plies"] is not None else "",
                f"{r['std_plies']:.2f}" if r["std_plies"] is not None else "",
                f"{r['mean_plies_decisive']:.1f}" if r["mean_plies_decisive"] is not None else "",
                f"{r['mean_plies_draw']:.1f}" if r["mean_plies_draw"] is not None else "",
            ])
    return {"rows": rows}


# ── 4. Symmetric pairwise score + CI ──────────────────────────

def pairwise_significance(records: list[dict], outdir: Path) -> dict:
    """Score each unordered pair (A vs B) by averaging A's win-rate over
    both colour halves. Flag pairs whose 95% CI excludes 0.5."""
    pair_stats: dict[tuple, dict] = defaultdict(
        lambda: {"a_wins": 0, "b_wins": 0, "draws": 0, "n": 0})

    for r in records:
        a, b = sorted((r["name_a"], r["name_b"]))
        # canonicalise: "a" = lexicographically first agent
        if r["name_a"] == a:
            a_wins = (r["outcome"] == "win_a")
            b_wins = (r["outcome"] == "win_b")
        else:
            a_wins = (r["outcome"] == "win_b")
            b_wins = (r["outcome"] == "win_a")
        st = pair_stats[(a, b)]
        st["n"] += 1
        if a_wins:
            st["a_wins"] += 1
        elif b_wins:
            st["b_wins"] += 1
        else:
            st["draws"] += 1

    rows = []
    for (a, b), st in pair_stats.items():
        N = st["n"]
        score_a = (st["a_wins"] + 0.5 * st["draws"]) / N if N else 0.0
        lo, hi = wilson_ci(st["a_wins"] + 0.5 * st["draws"], N)
        sig = (lo > 0.5) or (hi < 0.5)
        rows.append({
            "agent_a": a, "agent_b": b, "n": N,
            "a_wins": st["a_wins"], "draws": st["draws"], "b_wins": st["b_wins"],
            "score_a": score_a, "ci_low": lo, "ci_high": hi,
            "significant": sig,
        })
    rows.sort(key=lambda r: (not r["significant"], r["agent_a"], r["agent_b"]))

    with open(outdir / "pairwise_significance.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["agent_a", "agent_b", "n", "a_wins", "draws", "b_wins",
                    "score_a", "ci_low", "ci_high", "significant"])
        for r in rows:
            w.writerow([
                r["agent_a"], r["agent_b"], r["n"],
                r["a_wins"], r["draws"], r["b_wins"],
                f"{r['score_a']:.4f}", f"{r['ci_low']:.4f}",
                f"{r['ci_high']:.4f}", "yes" if r["significant"] else "no",
            ])
    return {"rows": rows}


# ── 5. Elo (Bradley-Terry MLE + bootstrap CI) ─────────────────

def _fit_elo(records: list[dict], names: list[str],
             max_iter: int = 200, lr: float = 16.0) -> dict[str, float]:
    """Minorise-Maximise iteration for Bradley-Terry on score (draw = 0.5).
    Anchors the mean rating at 1500.

    For each game between i and j, expected_i = 1 / (1 + 10^((R_j-R_i)/400)).
    Gradient: actual_i - expected_i. Sum over games, scale by lr, iterate.
    """
    idx = {n: i for i, n in enumerate(names)}
    R = [1500.0] * len(names)

    # Pre-aggregate to speed up iteration: pair_score[i][j] = (sum_score_i, n)
    score_sum: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0])
    for r in records:
        i, j = idx[r["name_a"]], idx[r["name_b"]]
        if r["outcome"] == "win_a":
            s = 1.0
        elif r["outcome"] == "win_b":
            s = 0.0
        else:
            s = 0.5
        score_sum[(i, j)][0] += s
        score_sum[(i, j)][1] += 1

    for it in range(max_iter):
        grad = [0.0] * len(names)
        for (i, j), (s_sum, n) in score_sum.items():
            # expected_i over all n games (constant within an iteration)
            E = n / (1 + 10 ** ((R[j] - R[i]) / 400))
            grad[i] += (s_sum - E)
            grad[j] += ((n - s_sum) - (n - E))
        # MM-style step
        max_g = max(abs(g) for g in grad)
        for k in range(len(names)):
            R[k] += lr * grad[k] / max(max_g, 1.0)
        # Anchor mean at 1500
        mean_R = sum(R) / len(R)
        R = [r - mean_R + 1500.0 for r in R]

    return {n: R[idx[n]] for n in names}


def elo_with_bootstrap(records: list[dict], outdir: Path,
                       n_boot: int = 200, rng_seed: int = 42) -> dict:
    names = AGENT_ORDER
    base = _fit_elo(records, names)

    rng = random.Random(rng_seed)
    boot_ratings: dict[str, list[float]] = {n: [] for n in names}
    n_records = len(records)
    for b in range(n_boot):
        sample = [records[rng.randrange(n_records)] for _ in range(n_records)]
        r_b = _fit_elo(sample, names, max_iter=80)
        for n in names:
            boot_ratings[n].append(r_b[n])

    rows = []
    for n in names:
        rs = sorted(boot_ratings[n])
        lo = rs[int(0.025 * len(rs))]
        hi = rs[int(0.975 * len(rs))]
        rows.append({"agent": n, "elo": base[n],
                     "ci_low": lo, "ci_high": hi})
    rows.sort(key=lambda r: -r["elo"])

    with open(outdir / "elo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "agent", "elo", "ci_low", "ci_high"])
        for rank, r in enumerate(rows, 1):
            w.writerow([rank, r["agent"], f"{r['elo']:.1f}",
                        f"{r['ci_low']:.1f}", f"{r['ci_high']:.1f}"])
    return {"rows": rows, "n_boot": n_boot}


# ── Report ────────────────────────────────────────────────────

def write_report(records, per_side, decisive, length, pairwise, elo,
                 outdir: Path) -> None:
    lines = [
        f"# Tournament Analysis",
        "",
        f"Total games analysed: **{len(records)}**",
        "",
        "## 1. Per-side breakdown",
        "",
        "Each agent's score split by which side they played. "
        "Positive `asymmetry` = stronger as Chess side.",
        "",
        "| Agent | Chess score (CI) | Xiangqi score (CI) | Asymmetry |",
        "|---|---|---|---|",
    ]
    for agent in AGENT_ORDER:
        e = per_side[agent]
        c, x = e["Chess"], e["Xiangqi"]
        lines.append(
            f"| {agent} | {c['score']:.3f} "
            f"[{c['ci'][0]:.3f}, {c['ci'][1]:.3f}] (N={c['N']}) | "
            f"{x['score']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}] (N={x['N']}) | "
            f"{e['asymmetry']:+.3f} |"
        )

    lines += [
        "",
        "## 2. Elo (Bradley-Terry, anchored at mean=1500)",
        "",
        f"Bootstrap 95% CI from {elo['n_boot']} resamples.",
        "",
        "| Rank | Agent | Elo | 95% CI |",
        "|---|---|---|---|",
    ]
    for rank, r in enumerate(elo["rows"], 1):
        lines.append(f"| {rank} | {r['agent']} | {r['elo']:.1f} | "
                     f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}] |")

    sig = [r for r in pairwise["rows"] if r["significant"]]
    lines += [
        "",
        "## 3. Pairwise significance",
        "",
        f"**{len(sig)} / {len(pairwise['rows'])}** pairs have a 95% CI on "
        f"the symmetric score that excludes 0.5.",
        "",
    ]
    if sig:
        lines += [
            "| Agent A | Agent B | Score A | 95% CI | N |",
            "|---|---|---|---|---|",
        ]
        for r in sig:
            lines.append(
                f"| {r['agent_a']} | {r['agent_b']} | {r['score_a']:.3f} | "
                f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['n']} |"
            )

    lines += [
        "",
        "## 4. Decisive rate (top 10 most decisive pairs)",
        "",
        "| Agent A | Agent B | Decisive rate | N | Top reason |",
        "|---|---|---|---|---|",
    ]
    for r in decisive["rows"][:10]:
        top = max(r["reasons"].items(), key=lambda x: x[1])
        lines.append(
            f"| {r['agent_a']} | {r['agent_b']} | {r['decisive_rate']:.3f} | "
            f"{r['n']} | {top[0]} ({top[1]}) |"
        )

    lines += [
        "",
        "## 5. Game length (top 5 longest / shortest mean)",
        "",
        "Longest:",
        "",
        "| Agent A | Agent B | Mean plies | Median | Std |",
        "|---|---|---|---|---|",
    ]
    for r in length["rows"][:5]:
        lines.append(
            f"| {r['agent_a']} | {r['agent_b']} | "
            f"{r['mean_plies']:.1f} | {r['median_plies']} | "
            f"{r['std_plies']:.1f} |"
        )
    lines += ["", "Shortest:", "",
              "| Agent A | Agent B | Mean plies | Median | Std |",
              "|---|---|---|---|---|"]
    for r in length["rows"][-5:]:
        lines.append(
            f"| {r['agent_a']} | {r['agent_b']} | "
            f"{r['mean_plies']:.1f} | {r['median_plies']} | "
            f"{r['std_plies']:.1f} |"
        )

    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ── 6. Payoff matrix (symmetric, score from row's perspective) ─

def payoff_matrix(records: list[dict], outdir: Path) -> dict:
    """Write a 9x9 payoff_matrix.csv with cells = row's avg score vs col
    (draws = 0.5). Mirrors the format the tournament script writes per-run,
    but computed across the merged record set."""
    names = AGENT_ORDER
    idx = {n: i for i, n in enumerate(names)}
    n = len(names)
    wins = [[0] * n for _ in range(n)]
    draws = [[0] * n for _ in range(n)]
    total = [[0] * n for _ in range(n)]
    for r in records:
        i, j = idx.get(r["name_a"]), idx.get(r["name_b"])
        if i is None or j is None:
            continue
        total[i][j] += 1
        total[j][i] += 1
        if r["outcome"] == "win_a":
            wins[i][j] += 1
        elif r["outcome"] == "win_b":
            wins[j][i] += 1
        else:
            draws[i][j] += 1
            draws[j][i] += 1

    score = [[0.5 if i == j else
              ((wins[i][j] + 0.5 * draws[i][j]) / total[i][j]
               if total[i][j] > 0 else 0.0)
              for j in range(n)] for i in range(n)]

    with open(outdir / "payoff_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, name in enumerate(names):
            w.writerow([name] + [f"{score[i][j]:.3f}" for j in range(n)])

    return {"score": score, "names": names}


# ── Entry point ───────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", nargs="+", required=True,
                   help="one or more game_records.json files to merge")
    p.add_argument("--outdir", required=True,
                   help="output directory for CSVs and report.md")
    p.add_argument("--n-boot", type=int, default=200,
                   help="bootstrap resamples for Elo CI")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_records([Path(p) for p in args.records])

    print("\n[1/6] Per-side breakdown...", file=sys.stderr)
    per_side = per_side_stats(records, outdir)
    print("[2/6] Decisive rate per pair...", file=sys.stderr)
    decisive = decisive_stats(records, outdir)
    print("[3/6] Game-length stats...", file=sys.stderr)
    length = game_length_stats(records, outdir)
    print("[4/6] Pairwise significance...", file=sys.stderr)
    pairwise = pairwise_significance(records, outdir)
    print(f"[5/6] Elo (bootstrap n={args.n_boot})...", file=sys.stderr)
    elo = elo_with_bootstrap(records, outdir, n_boot=args.n_boot)
    print("[6/6] Payoff matrix...", file=sys.stderr)
    payoff_matrix(records, outdir)

    write_report(records, per_side, decisive, length, pairwise, elo, outdir)

    print(f"\nWrote analysis to: {outdir}/", file=sys.stderr)
    print("  per_side.csv  decisive_rate.csv  game_length.csv  "
          "pairwise_significance.csv  elo.csv  payoff_matrix.csv  "
          "report.md", file=sys.stderr)


if __name__ == "__main__":
    main()
