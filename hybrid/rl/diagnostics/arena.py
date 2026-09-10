"""Fresh openings, explicit army assignments and complete-game persistence."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import random
import time

import torch
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.agents.alphazero_stub import TorchPolicyValueModel
from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.core.render import render_board
from hybrid.rl.general_model import load_general_model
from hybrid.rl.run_store import write_json,sha256
from hybrid.web_variants import PRESETS,parse_variant
from .common import read,notation,neural_agent,EVALUATORS

_models = {}


def init_worker(paths):
    global _models
    torch.set_num_threads(1)
    _models = {key:TorchPolicyValueModel(load_general_model(path),"cpu") for key,path in paths.items()}


def agent_for(name,seconds,seed):
    if name.startswith("ab_"):
        return AlphaBetaAgent(SearchConfig(depth=4,time_limit_seconds=seconds),evaluator=EVALUATORS[name[3:]])
    return neural_agent(_models[name],simulations=100000,seconds=seconds,seed=seed)


def play(job):
    if time.time()>=job["deadline"]:
        return None
    env = HybridChessEnv(variant=parse_variant(job["variant"]))
    state = env.reset()
    agents = {Side.CHESS:agent_for(job["chess"],job["seconds"],job["seed"]),
              Side.XIANGQI:agent_for(job["xiangqi"],job["seconds"],job["seed"])}
    rng = random.Random(job["seed"])
    moves,boards,searches = [],[render_board(state.board)],[]
    while True:
        if time.time()>=job["deadline"]:
            return None
        legal = env.legal_moves()
        if state.ply<4:
            move = rng.choice(legal)
        else:
            current = agents[state.side_to_move]
            started = time.perf_counter()
            if isinstance(current,AlphaBetaAgent):
                move = current.select_move(state,legal)
                detail = dict(depth=current.last_completed_depth)
            else:
                root = current._run_mcts_search(state,legal,add_noise=False)
                move = max(root.children,key=lambda m:root.children[m].N)
                detail = dict(simulations=root.N,root_value=root.Q,unvisited_fraction=
                              sum(ch.N==0 for ch in root.children.values())/len(legal))
            searches.append(dict(ply=state.ply,side=state.side_to_move.name.lower(),
                                 elapsed_seconds=time.perf_counter()-started,legal_count=len(legal),**detail))
        moves.append(notation(move))
        state,_,done,info = env.step(move)
        boards.append(render_board(state.board))
        if done:
            break
    row = {k:v for k,v in job.items() if k not in ("deadline","path")}
    row.update(winner=info.winner.name.lower() if info.winner else None,
        result=info.winner.name.lower()+"_win" if info.winner else "draw",reason=info.reason,
        plies=state.ply,moves=moves,states_ascii=boards,searches=searches,
        chess_score=.5 if info.winner is None else float(info.winner==Side.CHESS))
    write_json(job["path"],row)
    return job["id"]


def make_jobs(cfg,after=False,candidates=()):
    jobs = []
    def append(preset,group,chess,xiangqi,seed,assignment):
        jobs.append(dict(id=f"{preset['id']}-{group}-{seed}-{assignment}",preset=preset["id"],
            variant=parse_variant(preset["variant"]).to_dict(),group=group,chess=chess,xiangqi=xiangqi,
            seed=seed,assignment=assignment,seconds=cfg["seconds"]))
    for index,preset in enumerate(PRESETS):
        if after:
            if preset["id"] not in cfg["after_presets"]:
                continue
            for candidate in candidates:
                for opening in range(cfg["after_openings"]):
                    seed = cfg["after_seed"]+index*100+opening
                    append(preset,candidate,candidate,"frozen",seed,"chess")
                    append(preset,candidate,"frozen",candidate,seed,"xiangqi")
        else:
            for agent in ("ab_original","ab_basic","frozen"):
                for opening in range(cfg["self_openings"]):
                    append(preset,"self-"+agent,agent,agent,cfg["arena_seed"]+index*100+opening,"self")
            for opponent in ("ab_original","ab_basic"):
                for opening in range(cfg["cross_openings"]):
                    seed = cfg["arena_seed"]+10000+index*100+opening
                    append(preset,"cross-"+opponent,"frozen",opponent,seed,"chess")
                    append(preset,"cross-"+opponent,opponent,"frozen",seed,"xiangqi")
    return jobs


def arena(args,cfg,stop):
    paths = {"frozen":args.model}
    if args.after:
        candidates = read(Path(args.output)/"ablate/candidates.json")
        if len(candidates)!=4:
            raise ValueError("Need all four completed ablation candidates")
        for row in candidates:
            if sha256(row["path"])!=row["sha256"]:
                raise ValueError("Candidate changed")
            paths[row["name"]] = row["path"]
    folder = Path(args.output)/("after" if args.after else "arena")
    hashes = {name:sha256(path) for name,path in paths.items()}
    jobs = make_jobs(cfg,args.after,[name for name in paths if name!="frozen"])
    # Distribute opening pairs across rules/agents before scheduling another opening.
    jobs.sort(key=lambda j:(j["seed"]%100,j["preset"],j["group"],j["assignment"]))
    for job in jobs:
        job.update(model_hashes=hashes,deadline=stop.deadline,path=str(folder/"games"/(job["id"]+".json")))
    write_json(folder/"schedule.json",[{k:v for k,v in j.items() if k not in ("deadline","path")} for j in jobs])
    pending = [j for j in jobs if not Path(j["path"]).exists()]
    for job in jobs:
        if Path(job["path"]).exists():
            row = read(job["path"])
            if any(row.get(k)!=job[k] for k in ("id","model_hashes","variant","seed","chess","xiangqi")):
                raise ValueError("Arena resume identity mismatch")
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,initargs=(paths,)) as pool:
        futures = [pool.submit(play,j) for j in pending]
        for future in as_completed(futures):
            result = future.result()
            if result:
                print(json.dumps({"arena_game":result}),flush=True)
            if stop():
                for f in futures:
                    f.cancel()
                break
    completed = sum(Path(j["path"]).exists() for j in jobs)
    write_json(folder/"summary.json",dict(complete=completed==len(jobs),completed_games=completed,
        expected_games=len(jobs),model_hashes=hashes,protocol="one CPU thread, nominal 1 second, AB max depth 4"))
