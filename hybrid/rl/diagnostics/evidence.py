"""Derive auditable tables and replay checks without rerunning the experiments.

Run on a compute node after the relevant stages finish. Original records remain
immutable; summaries and selected replay exports go into a separate directory.
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import os

import numpy as np

from hybrid.core.env import HybridChessEnv
from hybrid.core.render import render_board
from hybrid.core.types import Side
from hybrid.rl.general_train import require_compute
from hybrid.rl.run_store import write_json, sha256
from hybrid.web_variants import parse_variant
from .common import read, parse_move
from .data import metadata
from .report import arena_summary, differences


def average(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def quantiles(values):
    values = list(values)
    return dict(zip(("min", "p50", "p95", "max"), map(float, np.quantile(values, [0,.5,.95,1])))) if values else None


def probe_tables(folder):
    rows = [read(p) for p in sorted((folder/"cases").glob("*.json"))]
    summary = read(folder/"summary.json")
    assert summary["complete"] and len(rows)==summary["expected_cases"]
    groups = defaultdict(list)
    for r in rows:
        if r.get("simulations")==128 and r.get("seconds") is None and not r.get("cpp"):
            for key in (f"{r['side']}/{r['mode']}", f"{r['preset']}/{r['side']}/{r['mode']}"):
                groups[key].append(r)
    metrics = {}
    for key, group in groups.items():
        real = [r for r in group if r["preset"]!="tactic"]
        wins = [r for r in group if r.get("contract")=="win"]
        metrics[key] = dict(cases=len(group), winning_cases=len(wins),
            immediate_wins=sum(r["immediate_win"] for r in wins),
            real_cases=len(real), network_value=average(r["network_value"] for r in real),
            observed_label=average(r["observed_value"] for r in real),
            observed_mse=average((r["network_value"]-r["observed_value"])**2 for r in real),
            saturated_fraction=average(abs(r["network_value"])>=.95 for r in real),
            legal_count=average(r["legal_count"] for r in real),
            positive_prior_coverage=average(sum(a["prior"]>0 for a in r["actions"])/r["legal_count"] for r in real),
            normalized_entropy=average(r["policy_entropy"]/np.log(r["legal_count"]) for r in real if r["legal_count"]>1),
            unvisited_fraction=average(r["unvisited_fraction"] for r in real),
            root_value=average(r["root_value"] for r in real if r["root_value"] is not None))
    index = {r["id"]:r for r in rows}
    comparisons = defaultdict(list)
    batch = []
    failures = []
    for r in rows:
        prefix = r["position_id"].replace(":","-")+"--"
        base = index[prefix+"network"]
        if r.get("contract")=="win" and not r["immediate_win"]:
            failures.append({k:r.get(k) for k in ("id","side","move","mode","evaluator","depth","simulations","seconds","cpp","batch")})
        if r.get("evaluator"):
            original = index[prefix+"original"+str(r["depth"])]
            comparisons[f"{r['evaluator']}/depth{r['depth']}/{r['side']}"].append(dict(
                changed=r["move"]!=original["move"], seconds=r["elapsed_seconds"], original_seconds=original["elapsed_seconds"]))
        if r.get("cpp") and r["batch"]==8:
            batch.append(dict(position_id=r["position_id"],side=r["side"],same_move=r["move"]==base["move"],
                completed_simulations=r["completed_simulations"],
                root_value_gap=r["root_value"]-base["root_value"],
                immediate_win=r["immediate_win"],single_immediate_win=base["immediate_win"],contract=r["contract"],
                seconds=r["elapsed_seconds"],single_seconds=base["elapsed_seconds"]))
    budget_groups = defaultdict(list)
    for r in rows:
        if r.get("mode")=="network" and not r.get("cpp") and (
                r.get("seconds") or r.get("simulations") in (64,512)):
            label = f"seconds{r['seconds']}" if r.get("seconds") else f"simulations{r['simulations']}"
            budget_groups[label+"/"+r["side"]].append(r)
    return dict(metrics=metrics, failures=failures, batch=batch, parity=summary["parity"],
        evaluator_controls={k:dict(cases=len(v),changed_moves=sum(r["changed"] for r in v),
            seconds=quantiles(r["seconds"] for r in v),original_seconds=quantiles(r["original_seconds"] for r in v)) for k,v in comparisons.items()},
        budgets={k:dict(cases=len(v),simulations=quantiles(r["completed_simulations"] for r in v),
            elapsed=quantiles(r["elapsed_seconds"] for r in v),winning_cases=sum(r["contract"]=="win" for r in v),
            immediate_wins=sum(r["immediate_win"] for r in v if r["contract"]=="win")) for k,v in budget_groups.items()})


def census_tables(folder, output):
    rows = [read(p) for p in sorted((folder/"games").glob("*.json"))]
    assert len(rows)==read(folder/"summary.json")["expected_games"]
    strata = {}
    cycles = [r for r in rows if "cycle" in r]
    for r in rows:
        key = "/".join(str(r[k]) for k in ("mode","family","iteration","winner","reason"))
        cell = strata.setdefault(key,dict(games=0,positions=0,chess_positions=0,xiangqi_positions=0))
        cell["games"]+=1
        for target,source in (("positions","samples"),("chess_positions","chess_samples"),("xiangqi_positions","xiangqi_samples")):
            cell[target]+=r[source]
    representatives = [next(r for r in cycles if r["cycle"]["last_length"]==length) for length in (4,6,8)]
    exports = []
    for r in representatives:
        meta = metadata(r["path"])
        env = HybridChessEnv(variant=parse_variant(meta["variant"]),use_cpp=True)
        state = env.reset()
        boards = [render_board(state.board)]
        for text in meta["moves"]:
            state,_,done,info = env.step(parse_move(text))
            boards.append(render_board(state.board))
        assert done and info.reason==meta["reason"]
        replay = dict(id=r["id"],variant=meta["variant"],moves=meta["moves"],states_ascii=boards,
            winner=None,reason=info.reason,cycle=r["cycle"],source_sha256=sha256(r["path"]),
            chess="teacher-ab",xiangqi="teacher-ab")
        write_json(output/"replays"/(r["id"]+".json"),replay)
        exports.append(dict(id=r["id"],cycle=r["cycle"],plies=len(meta["moves"])))
    return dict(strata=strata,teacher_repetition=dict(games=len(cycles),
        onset=quantiles(r["cycle"]["start"] for r in cycles),
        onset_at_or_after_ply12=sum(r["cycle"]["start"]>=12 for r in cycles),
        cycle_lengths=Counter(str(r["cycle"]["last_length"]) for r in cycles),representatives=exports))


def verify_replays(folder, output):
    schedule = read(folder/"schedule.json")
    rows = [read(p) for p in sorted((folder/"games").glob("*.json"))]
    assert len(rows)==len(schedule) and {r["id"] for r in rows}=={r["id"] for r in schedule}
    schedule = {r["id"]:r for r in schedule}
    moves = 0
    for row in rows:
        assert all(row[k]==v for k,v in schedule[row["id"]].items())
        env = HybridChessEnv(variant=parse_variant(row["variant"]),use_cpp=True)
        state = env.reset()
        assert render_board(state.board)==row["states_ascii"][0]
        for i,text in enumerate(row["moves"]):
            move = parse_move(text)
            assert move in env.legal_moves()
            state,_,done,info = env.step(move)
            assert render_board(state.board)==row["states_ascii"][i+1]
            assert done==(i==len(row["moves"])-1)
        assert (info.winner.name.lower() if info.winner else None)==row["winner"]
        assert info.reason==row["reason"] and state.ply==row["plies"]
        assert len(row["searches"])==max(0,state.ply-4)
        moves+=state.ply
    return dict(games=len(rows),plies=moves,all_legal=True,all_boards_match=True,all_results_match=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",required=True)
    parser.add_argument("--corrected",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--final",action="store_true")
    args = parser.parse_args()
    require_compute()
    original,corrected,output = map(Path,(args.original,args.corrected,args.output))
    assert output.resolve() not in (original.resolve(),corrected.resolve())
    cfg = read(original/"identity.json")["config"]
    evidence = dict(job_id=os.environ["SLURM_JOB_ID"],final=args.final,
        source=read("SOURCE.json")["source_version"],
        inputs={str(p.resolve()):sha256(p/"identity.json") for p in (original,corrected)},
        probes=probe_tables(corrected/"probe"),census=census_tables(original/"audit",output))
    evidence["baseline"],rows = arena_summary(original/"arena")
    evidence["baseline_difference"] = differences(rows,"cross-ab_original","cross-ab_basic")
    evidence["fitting"] = {p.parent.name:read(p) for p in (original/"ablate").glob("*/metrics.json")}
    evidence["composition"] = read(original/"ablate/composition.json")
    if args.final:
        evidence["after"],rows = arena_summary(original/"after")
        evidence["sampling_differences"] = [differences(rows,f"uniform-{s}",f"balanced-{s}") for s in cfg["training_seeds"]]
        evidence["replay_checks"] = {stage:verify_replays(original/stage,output) for stage in ("arena","after")}
    write_json(output/"evidence.json",evidence)
    # Small console summary for the human interpretation; full tables stay JSON.
    compact = {k:v for k,v in evidence.items() if k in ("baseline_difference","sampling_differences","replay_checks","composition")}
    compact["baseline"] = {k:v for k,v in evidence["baseline"].items() if "/" not in k}
    compact["probes"] = {k:v for k,v in evidence["probes"]["metrics"].items() if k.count("/")==1}
    compact["teacher_repetition"] = evidence["census"]["teacher_repetition"]
    print(json.dumps(compact,indent=2))


if __name__=="__main__":
    main()
