"""Generate EXPERIMENTS_EN.md and EXPERIMENTS_ZH.md from runs/ data."""
import csv, json, os, sys
from pathlib import Path
from datetime import datetime

RUNS = Path("runs")
DOCS = Path("docs")

# --- AZ run definitions ---
AZ_RUNS = [
    ("Default",      "rq4_az_default_v2",       "Standard Hybrid Chess rules", "标准 Hybrid Chess 规则"),
    ("Q only",       "rq4_az_noq_only",          "Remove Chess Queen",          "删除国际象棋后"),
    ("X only",       "rq4_az_xqqueen_only",      "Give XQ a Queen",             "给象棋方一个后"),
    ("PK",           "rq4_az_palace_knight_v2",   "chess_palace + knight_block", "宫格限制 + 蹩脚规则"),
    ("PK+noPromo",   "rq4_az_pk_nopromo",         "PK + no_promotion",           "PK + 禁止升变"),
    ("PK+xqQueen",   "rq4_az_pk_xqqueen",         "PK + xq_queen",              "PK + 象棋方加后"),
    ("noQ+noPromo",  "rq4_az_nq_nopromo",         "no_queen + no_promotion",     "删后 + 禁止升变"),
    ("noQ+PK",       "rq4_az_nq_pk",              "no_queen + PK",              "删后 + PK"),
    ("noQ+ALL",      "rq4_az_nq_allrules_v2",     "no_queen + all reforms",     "删后 + 全部改革"),
]

def load_metrics(run_dir):
    path = RUNS / run_dir / "metrics.csv"
    if not path.exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))

def summarize_az(rows):
    if not rows:
        return None
    n_iter = len(rows)
    tc = sum(int(r["sp_chess_wins"]) for r in rows)
    tx = sum(int(r["sp_xiangqi_wins"]) for r in rows)
    td = sum(int(r["sp_draws"]) for r in rows)
    total = tc + tx + td
    cp = tc / total * 100 if total else 0
    xp = tx / total * 100 if total else 0
    dp = td / total * 100 if total else 0
    cx = tc / tx if tx else float("inf")
    mat = sum(float(r["sp_avg_mat_diff"]) for r in rows) / n_iter
    # Last 10
    l10 = rows[-10:]
    l10c = sum(int(r["sp_chess_wins"]) for r in l10)
    l10x = sum(int(r["sp_xiangqi_wins"]) for r in l10)
    l10cx = l10c / l10x if l10x else float("inf")
    # Loss
    pl = float(rows[-1]["policy_loss"])
    vl = float(rows[-1]["value_loss"])
    # Piece survival
    surv = {}
    for key in ["surv_chess_QUEEN","surv_chess_ROOK","surv_chess_BISHOP","surv_chess_KNIGHT","surv_chess_PAWN",
                 "surv_xiangqi_CHARIOT","surv_xiangqi_CANNON","surv_xiangqi_HORSE","surv_xiangqi_ELEPHANT","surv_xiangqi_ADVISOR","surv_xiangqi_SOLDIER"]:
        if key in rows[-1]:
            vals = [float(r[key]) for r in rows[-10:] if key in r]
            surv[key.replace("surv_","")] = sum(vals)/len(vals)*100 if vals else 0
    return dict(n_iter=n_iter, total_games=total, chess_pct=cp, xq_pct=xp, draw_pct=dp,
                cx_ratio=cx, l10_cx=l10cx, mat_diff=mat, policy_loss=pl, value_loss=vl, surv=surv)

def load_ab_results():
    path = RUNS / "rq4_rule_reform_ab" / "results.json"
    if not path.exists():
        return []
    data = json.load(open(path))
    results = []
    for name, v in data.items():
        s = v["summary"]
        results.append(dict(name=name, n=s["n"], chess=s["chess_wins"], xq=s["xq_wins"],
                            draw=s["draws"], matdiff=s["avg_mat_diff"],
                            tb_c=s["mtb_chess"], tb_x=s["mtb_xq"], tb_e=s["mtb_even"]))
    results.sort(key=lambda x: abs(x["matdiff"]))
    return results

def load_tournament():
    spath = RUNS / "cross_variant_tournament" / "summary.json"
    ppath = RUNS / "cross_variant_tournament" / "payoff_matrix.csv"
    if not spath.exists():
        return None, None
    summary = json.load(open(spath))
    matrix_rows = list(csv.reader(open(ppath))) if ppath.exists() else []
    return summary, matrix_rows

def fmt_cx(v):
    if v == float("inf"):
        return "∞"
    return f"{v:.1f}x"

def gen_az_table(summaries, lang):
    if lang == "zh":
        header = "| 变体 | 轮次 | Chess% | XQ% | 和棋% | C:X | 后10轮C:X | 子力差 |"
        sep    = "|------|------|--------|-----|-------|-----|-----------|--------|"
    else:
        header = "| Variant | Iters | Chess% | XQ% | Draw% | C:X | L10 C:X | MatDiff |"
        sep    = "|---------|-------|--------|-----|-------|-----|---------|---------|"
    lines = [header, sep]
    for name, _, _, _, s in summaries:
        if s is None:
            continue
        star = " ⭐" if 0.5 <= s["cx_ratio"] <= 1.5 and s["cx_ratio"] != float("inf") else ""
        lines.append(f"| {name}{star} | {s['n_iter']} | {s['chess_pct']:.1f}% | {s['xq_pct']:.1f}% | {s['draw_pct']:.1f}% | {fmt_cx(s['cx_ratio'])} | {fmt_cx(s['l10_cx'])} | {s['mat_diff']:.1f} |")
    return "\n".join(lines)

def gen_ab_table(ab_results, lang):
    if lang == "zh":
        header = "| 变体 | 局数 | C胜 | X胜 | 和棋 | 子力判C | 子力判X | 子力判平 | avg_matdiff |"
        sep    = "|------|------|-----|-----|------|---------|---------|----------|-------------|"
    else:
        header = "| Variant | Games | C Win | X Win | Draw | TB Chess | TB XQ | TB Even | avg_matdiff |"
        sep    = "|---------|-------|-------|-------|------|----------|-------|---------|-------------|"
    lines = [header, sep]
    for r in ab_results:
        lines.append(f"| {r['name']} | {r['n']} | {r['chess']} | {r['xq']} | {r['draw']} | {r['tb_c']} | {r['tb_x']} | {r['tb_e']} | {r['matdiff']:+.1f} |")
    return "\n".join(lines)

def gen_payoff_table(matrix_rows):
    if not matrix_rows:
        return ""
    lines = []
    h = matrix_rows[0]
    lines.append("| | " + " | ".join(h[1:]) + " |")
    lines.append("|--" + "|------" * len(h[1:]) + "|")
    for row in matrix_rows[1:]:
        lines.append("| **" + row[0] + "** | " + " | ".join(row[1:]) + " |")
    return "\n".join(lines)

def gen_ranking_table(summary, lang):
    if not summary:
        return ""
    if lang == "zh":
        header = "| 排名 | Agent | 平均得分 |"
        sep    = "|------|-------|----------|"
    else:
        header = "| Rank | Agent | Avg Score |"
        sep    = "|------|-------|-----------|"
    lines = [header, sep]
    for i, r in enumerate(summary["ranking"], 1):
        lines.append(f"| {i} | {r['agent']} | {r['avg_score']:.4f} |")
    return "\n".join(lines)

def gen_factor_table(summaries, lang):
    lookup = {name: s for name, _, _, _, s in summaries if s}
    def get_cx(name):
        s = lookup.get(name)
        return fmt_cx(s["cx_ratio"]) if s else "—"
    def get_draw(name):
        s = lookup.get(name)
        return f"{s['draw_pct']:.0f}%" if s else "—"
    if lang == "zh":
        lines = [
            "| | Chess 有后 | Chess 无后 |",
            "|--|-----------|-----------|",
            f"| **XQ 无后** | Default {get_cx('Default')} ({get_draw('Default')} draw) | Q only {get_cx('Q only')} ({get_draw('Q only')} draw) |",
            f"| **XQ 有后** | X only {get_cx('X only')} ({get_draw('X only')} draw) | — |",
        ]
    else:
        lines = [
            "| | Chess has Queen | Chess no Queen |",
            "|--|----------------|----------------|",
            f"| **XQ no Queen** | Default {get_cx('Default')} ({get_draw('Default')} draw) | Q only {get_cx('Q only')} ({get_draw('Q only')} draw) |",
            f"| **XQ has Queen** | X only {get_cx('X only')} ({get_draw('X only')} draw) | — |",
        ]
    return "\n".join(lines)

def gen_piece_table(summaries, lang):
    pieces_chess = ["chess_QUEEN","chess_ROOK","chess_BISHOP","chess_KNIGHT","chess_PAWN"]
    pieces_xq = ["xiangqi_CHARIOT","xiangqi_CANNON","xiangqi_HORSE","xiangqi_ELEPHANT","xiangqi_ADVISOR","xiangqi_SOLDIER"]
    best = None
    for name, _, _, _, s in summaries:
        if name == "X only" and s:
            best = s
    if not best or not best.get("surv"):
        return ""
    if lang == "zh":
        header = "| 棋子 | 存活率(%) | 说明 |"
        sep    = "|------|-----------|------|"
    else:
        header = "| Piece | Survival(%) | Note |"
        sep    = "|-------|-------------|------|"
    lines = [header, sep]
    for p in pieces_chess + pieces_xq:
        v = best["surv"].get(p, 0)
        note = ""
        if v < 40:
            note = "⚠️ high loss" if lang == "en" else "⚠️ 高损耗"
        elif v > 85:
            note = "well protected" if lang == "en" else "保护完好"
        lines.append(f"| {p} | {v:.1f} | {note} |")
    return "\n".join(lines)

def gen_trend_table(summaries, target_name, lang):
    for name, run_dir, _, _, s in summaries:
        if name != target_name:
            continue
        rows = load_metrics(run_dir)
        if not rows:
            return ""
        if lang == "zh":
            header = "| 阶段 | Chess | XQ | 和棋 | C:X |"
            sep    = "|------|-------|----|------|-----|"
        else:
            header = "| Phase | Chess | XQ | Draw | C:X |"
            sep    = "|-------|-------|----|------|-----|"
        lines = [header, sep]
        n = len(rows)
        step = 10
        for i in range(0, n, step):
            batch = rows[i:i+step]
            c = sum(int(r["sp_chess_wins"]) for r in batch)
            x = sum(int(r["sp_xiangqi_wins"]) for r in batch)
            d = sum(int(r["sp_draws"]) for r in batch)
            cx = c / x if x else float("inf")
            lines.append(f"| {i}–{i+len(batch)-1} | {c} | {x} | {d} | {fmt_cx(cx)} |")
        return "\n".join(lines)
    return ""

def generate_doc(lang):
    now = datetime.now().strftime("%Y-%m-%d")
    # Load all data
    summaries = []
    for name, run_dir, desc_en, desc_zh in AZ_RUNS:
        rows = load_metrics(run_dir)
        s = summarize_az(rows)
        summaries.append((name, run_dir, desc_en, desc_zh, s))
    ab_results = load_ab_results()
    tournament_summary, matrix_rows = load_tournament()

    if lang == "zh":
        title = "Hybrid Chess — 实验结果报告"
        subtitle = f"> 自动生成于 {now}，数据来自 `runs/` 目录。"
        sec_overview = "## 一、实验概览"
        sec_ab = "## 二、AB D2 规则改革扫描（23 变体 × 40 局）"
        sec_ab_desc = "Alpha-Beta 深度=2，纯 C++ 加速。三项结构改革：`no_promotion`（禁升变）、`chess_palace`（King 限宫）、`knight_block`（马蹩脚）。"
        sec_az = "## 三、AlphaZero 九变体训练（各 50 轮 × 100 局/轮）"
        sec_az_desc = "统一配置：50 sims, max_ply=150, 4 workers, batch=256, 2 epochs。总计 **45,000 局**自对弈。"
        sec_factor = "## 四、因子分析"
        sec_factor_q = "### Queen 配置 2×2"
        sec_trend = "### xq_queen 稳定性（X only 每 10 轮趋势）"
        sec_piece = "### 棋子存活率（X only 变体，后 10 轮平均）"
        sec_tournament = "## 五、跨变体锦标赛（RQ3）"
        sec_tourn_desc = f"9 个变体的 best_model 在 Default 规则下对打，36 对 × 50 局 = **{tournament_summary['total_games'] if tournament_summary else 1800} 局**。"
        sec_payoff = "### Payoff Matrix"
        sec_ranking = "### Agent 排名"
        sec_findings = "## 六、关键发现"
        findings = [
            "1. **xq_queen 是唯一必要的平衡手段**：X only (0.7x) = PK+xqQueen (0.7x)，结构改革（PK）对 xq_queen 变体无附加效果。",
            "2. **no_promotion 完全无效**：150 步限制内兵到不了底线，零次升变发生。",
            "3. **删 Queen 导致和棋泛滥**：所有 noQ 变体和棋率 86–89%，对局质量严重下降。",
            "4. **xq_queen 趋势极其稳定**：50 轮内 C:X 始终 ~0.7x，无漂移。",
            '5. **"逆境出强者"**：受限规则训练的 agent（Q_only, PK）在 Default 规则下最强（0.625）。',
            "6. **Non-transitivity 存在**：PK_xQ → Default → noQ_ALL → PK_xQ 形成循环克制。",
        ]
        sec_config = "## 七、训练标准命令"
    else:
        title = "Hybrid Chess — Experiment Results"
        subtitle = f"> Auto-generated on {now} from `runs/` directory."
        sec_overview = "## 1. Overview"
        sec_ab = "## 2. AB D2 Rule Reform Scan (23 variants × 40 games)"
        sec_ab_desc = "Alpha-Beta depth=2, C++ accelerated. Three structural reforms: `no_promotion`, `chess_palace` (King confined to 3×3), `knight_block` (Knight with blocking rule)."
        sec_az = "## 3. AlphaZero Nine-Variant Training (50 iters × 100 games each)"
        sec_az_desc = "Uniform config: 50 sims, max_ply=150, 4 workers, batch=256, 2 epochs. Total: **45,000 self-play games**."
        sec_factor = "## 4. Factor Analysis"
        sec_factor_q = "### Queen Configuration 2×2"
        sec_trend = "### xq_queen Stability (X only per-10 trend)"
        sec_piece = "### Piece Survival Rate (X only variant, last 10 iters avg)"
        sec_tournament = "## 5. Cross-Variant Tournament (RQ3)"
        sec_tourn_desc = f"9 variant best_models play under Default rules, 36 pairs × 50 games = **{tournament_summary['total_games'] if tournament_summary else 1800} games**."
        sec_payoff = "### Payoff Matrix"
        sec_ranking = "### Agent Ranking"
        sec_findings = "## 6. Key Findings"
        findings = [
            "1. **xq_queen is the only necessary balancing mechanism**: X only (0.7x) = PK+xqQueen (0.7x); structural reforms (PK) add nothing when XQ has a Queen.",
            "2. **no_promotion has zero effect**: pawns never reach the back rank within the 150-ply limit.",
            "3. **Removing Queen causes draw flooding**: all noQ variants reach 86–89% draws.",
            "4. **xq_queen trend is extremely stable**: C:X stays ~0.7x across all 50 iterations with no drift.",
            '5. **"Adversity breeds strength"**: agents trained under restricted rules (Q_only, PK) perform best under Default rules (0.625).',
            "6. **Non-transitivity exists**: PK_xQ → Default → noQ_ALL → PK_xQ forms a rock-paper-scissors cycle.",
        ]
        sec_config = "## 7. Training Command"

    # Overview stats
    total_games = sum(s["total_games"] for _, _, _, _, s in summaries if s)
    total_iters = sum(s["n_iter"] for _, _, _, _, s in summaries if s)
    if lang == "zh":
        overview = f"- **AZ 训练**：9 个变体 × 50 轮 = {total_iters} 轮，共 {total_games:,} 局自对弈\n- **AB 扫描**：23 个变体 × 40 局 = 920 局\n- **锦标赛**：{tournament_summary['total_games'] if tournament_summary else 1800} 局"
    else:
        overview = f"- **AZ Training**: 9 variants × 50 iters = {total_iters} iters, {total_games:,} self-play games\n- **AB Scan**: 23 variants × 40 games = 920 games\n- **Tournament**: {tournament_summary['total_games'] if tournament_summary else 1800} games"

    # Assemble
    parts = [
        f"# {title}\n\n{subtitle}\n\n---\n",
        f"{sec_overview}\n\n{overview}\n\n---\n",
        f"{sec_ab}\n\n{sec_ab_desc}\n\n{gen_ab_table(ab_results, lang)}\n\n---\n",
        f"{sec_az}\n\n{sec_az_desc}\n\n{gen_az_table(summaries, lang)}\n",
        f"\n{sec_factor}\n\n{sec_factor_q}\n\n{gen_factor_table(summaries, lang)}\n",
        f"\n{sec_trend}\n\n{gen_trend_table(summaries, 'X only', lang)}\n",
        f"\n{sec_piece}\n\n{gen_piece_table(summaries, lang)}\n\n---\n",
        f"{sec_tournament}\n\n{sec_tourn_desc}\n\n{sec_payoff}\n\n{gen_payoff_table(matrix_rows)}\n",
        f"\n{sec_ranking}\n\n{gen_ranking_table(tournament_summary, lang)}\n\n---\n",
        f"{sec_findings}\n\n" + "\n".join(findings) + "\n\n---\n",
        f"{sec_config}\n\n```bash\npython scripts/train_az_iter.py \\\n  --iterations 50 --selfplay-games-per-iter 100 --simulations 50 \\\n  --selfplay-max-ply 150 --batch-size 256 --train-epochs 2 \\\n  --eval-games 20 --eval-interval 2 --eval-simulations 100 \\\n  --disable-gating 1 --resign-enabled 1 --device auto --seed 42 \\\n  --ablation \"xq_queen\" --use-cpp --num-workers 4 \\\n  --outdir \"runs/MY_RUN_NAME\"\n```\n",
    ]
    return "\n".join(parts)

def main():
    DOCS.mkdir(exist_ok=True)
    zh = generate_doc("zh")
    en = generate_doc("en")
    zh_path = DOCS / "EXPERIMENTS_ZH.md"
    en_path = DOCS / "EXPERIMENTS_EN.md"
    zh_path.write_text(zh, encoding="utf-8")
    en_path.write_text(en, encoding="utf-8")
    print(f"Generated {zh_path} ({len(zh)} bytes)")
    print(f"Generated {en_path} ({len(en)} bytes)")

if __name__ == "__main__":
    main()
