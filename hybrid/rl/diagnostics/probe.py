"""Matched-position interventions with per-case atomic completion."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import torch
from hybrid.agents.alphazero_stub import TorchPolicyValueModel
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.agents.eval import material_score
from hybrid.core.types import Side
from hybrid.rl.general_model import load_general_model
from hybrid.rl.run_store import write_json
from .common import read, restore, search_record, EVALUATORS, notation, fingerprint, stage_path
from .fixtures import tactical_positions,verify_contract

_model = None


def init_worker(path):
    global _model
    torch.set_num_threads(1)
    _model = TorchPolicyValueModel(load_general_model(path),"cpu")


def work(job):
    if time.time()>=job["deadline"]:
        return None
    record,settings = job["position"],job["settings"]
    env,state = restore(record,settings.get("cpp",False))
    legal = env.legal_moves()
    if settings.get("evaluator"):
        before = fingerprint(state)
        agent = AlphaBetaAgent(SearchConfig(depth=settings["depth"]),evaluator=EVALUATORS[settings["evaluator"]])
        started = time.perf_counter()
        move = agent.select_move(state,legal)
        result = dict(settings,move=notation(move),elapsed_seconds=time.perf_counter()-started,
                      completed_depth=agent.last_completed_depth)
        assert fingerprint(state)==before
    else:
        move,result = search_record(_model,state,legal,**settings)
    _,_,done,info = env.step(move)
    result.update(id=job["id"],position_id=record["id"],side=record["side"],preset=record["preset"],
                  immediate_win=bool(done and info.winner==state.side_to_move),
                  immediate_draw=bool(done and info.winner is None),contract=record.get("contract"),
                  observed_value=record.get("observed_value"))
    write_json(job["path"],result)
    return job["id"]


def probe(args,cfg,stop):
    out = Path(args.output)/"probe"
    real = read(stage_path(args,"audit")/"positions.json")
    tactics = tactical_positions()
    contracts = {p["id"]:verify_contract(p) for p in tactics}
    write_json(out/"tactics.json",tactics)
    write_json(out/"contracts.json",contracts)
    positions = tactics+real
    evaluations = []
    from hybrid.web_variants import PRESETS,parse_variant
    from hybrid.core.env import HybridChessEnv
    for preset in PRESETS:
        env = HybridChessEnv(variant=parse_variant(preset["variant"]))
        state = env.reset()
        evaluations.append(dict(preset=preset["id"],material={s.name:material_score(state,s) for s in Side},
            values={name:{s.name:fn(state,s) for s in Side} for name,fn in EVALUATORS.items()}))
    write_json(out/"opening-evaluations.json",evaluations)
    jobs = []
    def add(p,label,settings):
        identifier = p["id"].replace(":","-")+"--"+label
        jobs.append(dict(id=identifier,position=p,settings=settings,deadline=stop.deadline,
                         path=str(out/"cases"/(identifier+".json"))))
    for p in positions:
        for mode in ("policy","network","uniform_policy","zero_value","uniform_zero"):
            add(p,mode,dict(mode=mode,simulations=cfg["simulations"]))
    # One real position per rule/army + all hand-checkable fixtures.
    representatives = tactics+real[::16]
    for p in representatives:
        for sims in (64,512):
            add(p,"sim"+str(sims),dict(simulations=sims))
        for seconds in (1.,3.,6.):
            add(p,"time"+str(int(seconds)),dict(simulations=100000,seconds=seconds))
        for batch in (1,8):
            add(p,"cpp"+str(batch),dict(simulations=128,cpp=True,batch=batch))
        for name in EVALUATORS:
            for depth in (1,2):
                add(p,name+str(depth),dict(evaluator=name,depth=depth))
    pending = [j for j in jobs if not Path(j["path"]).exists()]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(args.model,)) as pool:
        futures = [pool.submit(work,j) for j in pending]
        for i,future in enumerate(as_completed(futures)):
            future.result()
            if (i+1)%25==0:
                print(json.dumps({"probe_completed_new":i+1,"pending_at_start":len(pending)}),flush=True)
            if stop():
                for f in futures:
                    f.cancel()
                break
    completed = [read(j["path"]) for j in jobs if Path(j["path"]).exists()]
    parity = []
    by_id = {r["id"]:r for r in completed}
    for p in representatives:
        prefix = p["id"].replace(":","-")+"--"
        a,b = by_id.get(prefix+"network"),by_id.get(prefix+"cpp1")
        if a and b:
            parity.append(dict(position_id=p["id"],same_move=a["move"]==b["move"],
                same_visits={r["move"]:r["visits"] for r in a["actions"]}=={r["move"]:r["visits"] for r in b["actions"]},
                value_difference=abs(a["root_value"]-b["root_value"])))
    write_json(out/"summary.json",dict(complete=len(completed)==len(jobs),completed_cases=len(completed),
        expected_cases=len(jobs),positions=len(positions),tactics=len(tactics),real_positions=len(real),parity=parity))
