"""Reachable mate-in-one confirmation, separate from synthetic rule contracts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

import numpy as np
from hybrid.core.env import HybridChessEnv
from hybrid.core.render import render_board
from hybrid.core.rules import is_in_check
from hybrid.core.types import Side
from hybrid.rl.general_train import require_compute
from hybrid.rl.run_store import write_json, sha256
from hybrid.web_variants import parse_variant
from .common import read, parse_move, position, restore, search_record
from .fixtures import verify_contract
from .probe import init_worker


def work(job):
    from .probe import _model
    env,state = restore(job["position"])
    move,stats = search_record(_model,state,env.legal_moves(),**job["settings"])
    initial = render_board(state.board)
    child,_,done,info = env.step(move)
    stats.update(position_id=job["position"]["id"],side=state.side_to_move.name,
        immediate_win=bool(done and info.winner==state.side_to_move),
        expected_wins=job["position"]["winning_moves"],model_sha256=job["model_sha256"])
    write_json(job["replay"],dict(id=Path(job["output"]).stem,variant=state.variant.to_dict(),
        initial_ply=state.ply,initial_side=state.side_to_move.name.lower(),
        moves=[stats["move"]],states_ascii=[initial,render_board(child.board)],
        winner=info.winner.name.lower() if done and info.winner else None,
        reason=info.reason if done else "Nonterminal after selected move",
        origin_game=job["position"]["origin_game"],complete_game=False))
    write_json(job["output"],stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    require_compute()
    import multiprocessing as mp
    mp.set_start_method("spawn",force=True)
    root,out = Path(args.original),Path(args.output)
    cfg = read(root/"identity.json")["config"]
    assert sha256(args.model)==cfg["model_sha256"]
    identity = dict(source_version=read("SOURCE.json")["source_version"],model_sha256=cfg["model_sha256"],
        seed=9970000,origin_sha256=sha256(root/"arena/schedule.json"),cases_per_army=12)
    from hybrid.rl.run_store import RunStore,run_lock
    with run_lock(out):
        RunStore(out,identity)
        records = [read(p) for p in sorted((root/"arena/games").glob("*.json"))]
        buckets = {s:[] for s in Side}
        rng = np.random.default_rng(identity["seed"])
        for index in rng.permutation(len(records)):
            row = records[int(index)]
            if row["reason"]!="Checkmate":
                continue
            side = Side[row["winner"].upper()]
            if len(buckets[side])>=12:
                continue
            env = HybridChessEnv(variant=parse_variant(row["variant"]),use_cpp=True)
            state = env.reset()
            for move in row["moves"][:-1]:
                state,_,done,_ = env.step(parse_move(move))
                assert not done
            assert state.side_to_move==side and not is_in_check(state.board,side.opponent())
            record = position(state,id=row["id"]+"-mate",origin_game=row["id"],
                moves_before=row["moves"][:-1],preset=row["preset"],contract="win")
            record["winning_moves"] = verify_contract(record)["winning_moves"]
            assert row["moves"][-1] in record["winning_moves"]
            buckets[side].append(record)
        assert all(len(rows)==12 for rows in buckets.values())
        positions = sum(buckets.values(),[])
        write_json(out/"positions.json",positions)
        jobs = []
        for record in positions:
            settings = [(m,dict(mode=m,simulations=128)) for m in
                        ("policy","network","uniform_policy","zero_value","uniform_zero")]
            settings += [("sim512",dict(simulations=512)),("time1",dict(simulations=100000,seconds=1.))]
            for label,config in settings:
                name = record["id"]+"--"+label+".json"
                jobs.append(dict(position=record,settings=config,model_sha256=cfg["model_sha256"],
                    output=str(out/"cases"/name),replay=str(out/"replays"/name)))
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(args.model,)) as pool:
            futures = [pool.submit(work,j) for j in jobs if not Path(j["output"]).exists()]
            for future in as_completed(futures):
                future.result()
        summary = {}
        for job in jobs:
            result = read(job["output"])
            key = result["side"]+"/"+Path(job["output"]).stem.split("--")[-1]
            group = summary.setdefault(key,dict(cases=0,immediate_wins=0))
            group["cases"]+=1
            group["immediate_wins"]+=int(result["immediate_win"])
        write_json(out/"summary.json",dict(complete=True,cases=len(jobs),positions=len(positions),
            job_id=os.environ["SLURM_JOB_ID"],groups=summary))
        print(summary,flush=True)


if __name__=="__main__":
    main()
