"""Chinese evidence report; observational differences are not causal proof."""
from collections import defaultdict,Counter
from pathlib import Path
import json
import numpy as np
from hybrid.rl.run_store import write_json
from .common import read,paired_interval,stage_path


def score_stats(scores):
    return dict(games=len(scores),wins=sum(s==1 for s in scores),draws=sum(s==.5 for s in scores),
                losses=sum(s==0 for s in scores),score=float(np.mean(scores)) if scores else None)


def arena_summary(folder):
    groups = defaultdict(list)
    records = [read(p) for p in sorted((folder/"games").glob("*.json"))]
    for row in records:
        assert len(row["states_ascii"])==len(row["moves"])+1==row["plies"]+1
        expected = .5 if row["winner"] is None else float(row["winner"]=="chess")
        assert expected==row["chess_score"]
        for key in (row["group"],row["preset"]+"/"+row["group"]):
            groups[key].append(row)
    result = {}
    for key,rows in groups.items():
        pairs = defaultdict(list)
        by_army = defaultdict(list)
        for row in rows:
            score = row["chess_score"] if row["assignment"]!="xiangqi" else 1-row["chess_score"]
            pairs[(row["preset"],row["seed"])].append(score)
            by_army[row["assignment"]].append(score)
        pair_values = [float(np.mean(v)) for v in pairs.values() if len(v)==2]
        selfplay = rows[0]["assignment"]=="self"
        scores = [r["chess_score"] for r in rows] if selfplay else sum(by_army.values(),[])
        times = [s["elapsed_seconds"] for row in rows for s in row["searches"]]
        depths = Counter(str(s["depth"]) for row in rows for s in row["searches"] if "depth" in s)
        result[key] = dict(**score_stats(scores),by_army={k:score_stats(v) for k,v in by_army.items()},
            score_perspective="chess" if selfplay else "tested_agent",complete_pairs=len(pair_values),
            score_95_ci=paired_interval(scores if selfplay else pair_values),
            reasons=Counter(r["reason"] for r in rows),depth_counts=depths,
            mean_plies=float(np.mean([r["plies"] for r in rows])),
            seconds_p50=float(np.median(times)) if times else None,
            seconds_p95=float(np.quantile(times,.95)) if times else None)
    return result,records


def differences(records,control,treatment):
    groups = defaultdict(dict)
    for row in records:
        if row["group"] in (control,treatment):
            groups[(row["preset"],row["seed"],row["assignment"])][row["group"]] = (
                row["chess_score"] if row["assignment"]!="xiangqi" else 1-row["chess_score"])
    pairs = defaultdict(list)
    for (preset,seed,army),values in groups.items():
        if set(values)=={control,treatment}:
            pairs[(preset,seed)].append(values[treatment]-values[control])
    values = [float(np.mean(rows)) for rows in pairs.values() if len(rows)==2]
    return dict(control=control,treatment=treatment,complete_opening_pairs=len(values),
        mean_difference=float(np.mean(values)) if values else None,interval_95=paired_interval(values,difference=True),
        method="Conservative Hoeffding bound over paired-opening score differences")


def report(args,cfg):
    root = Path(args.output)
    census = read(stage_path(args,"audit")/"summary.json")
    probes = [read(p) for p in (stage_path(args,"probe")/"cases").glob("*.json")]
    probe_summary = read(stage_path(args,"probe")/"summary.json")
    openings = read(stage_path(args,"probe")/"opening-evaluations.json")
    arena,arena_rows = arena_summary(stage_path(args,"arena"))
    after,after_rows = arena_summary(stage_path(args,"after"))
    wins = defaultdict(list)
    real_values = defaultdict(list)
    for row in probes:
        if row.get("mode") in ("policy","network","uniform_policy","zero_value","uniform_zero") and row.get("seconds") is None and row.get("simulations")==128 and not row.get("cpp"):
            if row.get("contract")=="win":
                wins[row["side"]+"/"+row["mode"]].append(int(row["immediate_win"]))
            if row["preset"]!="tactic":
                real_values[row["side"]+"/"+row["mode"]].append(row)
    tactic_rates = {key:dict(cases=len(values),immediate_wins=sum(values)) for key,values in wins.items()}
    real_metrics = {key:dict(cases=len(rows),mean_network_value=float(np.mean([r["network_value"] for r in rows])),
        mean_root_value=float(np.mean([r["root_value"] for r in rows if r["root_value"] is not None])) if rows[0]["mode"]!="policy" else None,
        unvisited_fraction=float(np.mean([r["unvisited_fraction"] for r in rows])),
        policy_entropy=float(np.mean([r["policy_entropy"] for r in rows])),
        observed_value_mse=float(np.mean([(r["network_value"]-r["observed_value"])**2 for r in rows])))
        for key,rows in real_values.items()}
    ablation = [differences(after_rows,f"uniform-{seed}",f"balanced-{seed}") for seed in cfg["training_seeds"]]
    evidence = dict(model_sha256=cfg["model_sha256"],source_version=args.source_version,
        stage_origins={stage:str(stage_path(args,stage).resolve()) for stage in ("audit","probe","arena","after")},
        census=census,opening_evaluations=openings,probe_summary=probe_summary,
        tactical_immediate_wins=tactic_rates,real_position_metrics=real_metrics,arena=arena,after=after,
        baseline_intervention=differences(arena_rows,"cross-ab_original","cross-ab_basic"),
        sampling_intervention=ablation)
    write_json(root/"report.json",evidence)
    lines = ["# 象棋阵营诊断", "",f"冻结模型：`{cfg['model_sha256']}`。", "",
        "本报告区分实现事实、代理水平下的阵营差异和采样干预结果；不把旧棋局标签当作最优棋力真值。", "",
        "## 评价函数线索", "", "| 规则 | 象棋子力差 | 原评价两视角之和 | 对称化后之和 |", "| --- | ---: | ---: | ---: |"]
    for row in openings:
        lines.append(f"| {row['preset']} | {row['material']['XIANGQI']:.1f} | {sum(row['values']['original'].values()):.3f} | {sum(row['values']['symmetric'].values()):.3f} |")
    lines += ["", "原评价的残局开关仅取决于子力差大于 5。开局触发及视角不对称属于可复现行为；它们对模型偏科的贡献需要结合下面的独立干预结果。", "",
        "## 教师与训练分布", "",f"已统计 {census['games']} / {census['expected_games']} 局。教师重复循环长度分布：`{json.dumps(census['teacher_cycle_lengths'])}`。", "",
        "| 数据 | 棋局 | 局面 | 国际象棋走棋样本 | 象棋走棋样本 | 最长 10% 棋局占局面比例 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("teacher","selfplay"):
        row = census["groups"].get(name)
        if row:
            lines.append(f"| {name} | {row['games']} | {row['samples']} | {row['chess_samples']} | {row['xiangqi_samples']} | {row['longest_10_percent_position_share']:.1%} |")
    lines += ["", "## 战术与搜索", "", "规则安全案例单独验证，不计为棋力成功。下表仅统计存在立即获胜走法的案例。", "",
        "| 阵营 / 模式 | 立即获胜 | 案例数 |", "| --- | ---: | ---: |"]
    for name,row in sorted(tactic_rates.items()):
        lines.append(f"| {name} | {row['immediate_wins']} | {row['cases']} |")
    lines += ["",f"搜索任务完成 {probe_summary['completed_cases']} / {probe_summary['expected_cases']}。单叶 Python/C++ 对照详见 report.json。", "",
              "## 新开局对弈", "", "自对弈得分以国际象棋方为视角，交叉对弈以被测学习代理为视角。", "",
              "| 组别 | 胜 / 和 / 负 | 得分 |", "| --- | --- | ---: |"]
    for name,row in arena.items():
        if "/" not in name:
            lines.append(f"| {name} | {row['wins']} / {row['draws']} / {row['losses']} | {row['score']:.1%} |")
    lines += ["", "## 采样对照", "", "每个种子单独比较平衡采样减去原采样的配对得分。正值有利于平衡采样。", ""]
    for row in ablation:
        lines.append(f"- {row['treatment']}：完整开局配对 {row['complete_opening_pairs']}，差值 {row['mean_difference']}，95% 区间 {row['interval_95']}。")
    lines += ["", "## 解释边界", "", "- 区间覆盖零的对照不能确认为稳定棋力改进；两个训练种子也不代表充分的训练随机性覆盖。",
        "- 禁用网络价值或替换先验产生变化，只能定位敏感环节，需结合可判定战术与搜索记录判断机制。",
        "- 微训练只改变采样，不足以证明早期教师造成了全部问题。",
        "- 未完成棋局不计为和棋，未完成配对不进入配对区间。最终解释和最小修复建议见随交付附上的人工核对结论。", ""]
    (root/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
