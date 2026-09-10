"""Whole-game census and game-disjoint replay-pool reconstruction."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import zipfile

import numpy as np
import torch
from hybrid.core.env import HybridChessEnv
from hybrid.core.rules import board_hash
from hybrid.core.types import Side
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.rl.general_games import load_game
from hybrid.rl.general_model import encode_general
from hybrid.rl.run_store import write_json, sha256
from hybrid.web_variants import PRESETS, parse_variant
from .common import read, parse_move, position


def metadata(path):
    # Read only the metadata member; do not inflate every feature tensor for a census.
    with zipfile.ZipFile(path) as archive, archive.open("metadata.npy") as stream:
        return json.loads(str(np.load(stream,allow_pickle=False)))


def sampling_key(ex):
    return (ex.side_to_move.name, int(ex.z))


class BalancedBuffer(ReplayBuffer):
    def __init__(self, examples):
        super().__init__(max(1,len(examples)))
        self.examples = examples
        groups = defaultdict(list)
        for i,ex in enumerate(examples):
            groups[sampling_key(ex)].append(i)
        self.groups = [np.asarray(groups[k]) for k in sorted(groups)]
        if len(self.groups)!=6:
            raise ValueError("The balanced intervention requires all six army/result strata")

    def sample_batch(self,batch_size,rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        # Each draw first chooses a stratum uniformly, then a position uniformly.
        strata = rng.integers(len(self.groups),size=batch_size)
        ids = [int(rng.choice(self.groups[g])) for g in strata]
        examples = [self.examples[i] for i in ids]
        return (np.stack([e.state for e in examples]).astype(np.float32),
                [e.pi_indices for e in examples],[e.pi_probs for e in examples],
                np.asarray([e.z for e in examples],dtype=np.float32))


def prepare_pool(args,cfg):
    path = Path(args.training)/cfg["training_checkpoint"]
    if sha256(path)!=cfg["training_checkpoint_sha256"]:
        raise ValueError("Frozen training checkpoint changed")
    checkpoint = torch.load(path,map_location="cpu",weights_only=False)
    if checkpoint["iteration"]!=104:
        raise ValueError("Expected the round 104 replay pool")
    refs = checkpoint["replay"]
    order = np.random.default_rng(cfg["data_seed"]).permutation(len(refs))
    valid_ids = set(int(i) for i in order[:max(1,round(len(refs)*.1))])
    games, total = [],0
    for i,ref in enumerate(refs):
        if sha256(ref["path"])!=ref["sha256"]:
            raise ValueError("Replay shard changed")
        meta = metadata(ref["path"])
        games.append(dict(ref,index=i,samples=meta["samples"],game_id=meta["game_id"],
                          validation=i in valid_ids,start=total))
        total += meta["samples"]
    # Match the exact last 100,000 positions the frozen model trained on.
    cutoff = max(0,total-100000)
    for game in games:
        game["skip"] = min(game["samples"],max(0,cutoff-game["start"]))
    manifest = dict(games=games,cutoff=cutoff,pool_samples=min(total,100000),
        checkpoint_sha256=cfg["training_checkpoint_sha256"],split_seed=cfg["data_seed"],
        note="Game-disjoint within this experiment; the initial model has seen this pool before.")
    write_json(Path(args.output)/"pool.json",manifest)
    return manifest


def load_pool(manifest):
    train,valid = ReplayBuffer(),ReplayBuffer()
    for game in manifest["games"]:
        if game["skip"]==game["samples"]:
            continue
        if sha256(game["path"])!=game["sha256"]:
            raise ValueError("Replay shard changed")
        examples,_ = load_game(game["path"])
        (valid if game["validation"] else train).append(examples[game["skip"]:])
    return train,valid


def audit(args,cfg,stop):
    out = Path(args.output)/"audit"
    cache = out/"games"
    teacher_manifest = read(Path(args.teacher)/"dataset.json")
    teacher = [Path(r["path"]) for part in ("train","validation") for r in teacher_manifest[part]]
    selfplay = sorted((Path(args.training)/"selfplay").glob("iter-*/game-*.npz"))
    jobs = [("teacher",p) for p in teacher]+[("selfplay",p) for p in selfplay]
    records = []
    for mode,path in jobs:
        key = ("teacher-"+path.stem if mode=="teacher" else path.parent.name+"-"+path.stem)
        target = cache/(key+".json")
        if target.exists():
            row = read(target)
        else:
            if stop():
                break
            meta = metadata(path)
            n = meta["samples"]
            winner = meta["winner"] or "draw"
            row = dict(id=key,path=str(path),mode=mode,family=meta["family"],variant=meta["variant"],
                iteration=meta["game_id"]//128 if mode=="selfplay" else None,
                samples=n,plies=meta["plies"],winner=winner,reason=meta["reason"],
                chess_samples=(n+1)//2,xiangqi_samples=n//2)
            if mode=="teacher" and meta["reason"]=="Threefold repetition":
                env = HybridChessEnv(variant=parse_variant(meta["variant"]),use_cpp=True)
                state = env.reset()
                seen = defaultdict(list)
                seen[board_hash(state.board,state.side_to_move)].append(0)
                for text in meta["moves"]:
                    state,_,done,info = env.step(parse_move(text))
                    seen[board_hash(state.board,state.side_to_move)].append(state.ply)
                occurrences = seen[board_hash(state.board,state.side_to_move)]
                assert done and info.reason==meta["reason"] and len(occurrences)>=3
                row["cycle"] = dict(occurrences=occurrences,last_length=occurrences[-1]-occurrences[-2],
                    moves=meta["moves"][occurrences[-2]:occurrences[-1]],
                    start=occurrences[-3],verified=True)
            write_json(target,row)
        records.append(row)
        if len(records)%500==0:
            print(json.dumps({"audit_games":len(records),"expected":len(jobs)}),flush=True)
    groups = {}
    for row in records:
        for key in (row["mode"],row["mode"]+"/"+row["family"],row["mode"]+"/iteration/"+str(row["iteration"])):
            group = groups.setdefault(key,dict(games=0,samples=0,chess_samples=0,xiangqi_samples=0,
                                               winners=Counter(),reasons=Counter(),lengths=[]))
            for field in ("samples","chess_samples","xiangqi_samples"):
                group[field] += row[field]
            group["games"] += 1
            group["winners"][row["winner"]] += 1
            group["reasons"][row["reason"]] += 1
            group["lengths"].append(row["samples"])
    for group in groups.values():
        lengths = sorted(group.pop("lengths"),reverse=True)
        group["longest_10_percent_position_share"] = sum(lengths[:max(1,int(np.ceil(len(lengths)*.1)))] )/sum(lengths)
    cycles = Counter(str(r["cycle"]["last_length"]) for r in records if "cycle" in r)
    result = dict(complete=len(records)==len(jobs),games=len(records),expected_games=len(jobs),
                  groups=groups,teacher_cycle_lengths=cycles,
                  partial_last_collection_included_in_census=True)
    write_json(out/"summary.json",result)
    if not result["complete"]:
        return
    prepare_pool(args,cfg)
    positions_path = out/"positions.json"
    if positions_path.exists():
        return
    selected = []
    rng = np.random.default_rng(cfg["data_seed"])
    for preset in PRESETS:
        rows = [read(p) for p in sorted((Path(args.evaluation)/"games").glob("*.json"))
                if p.name.startswith(preset["id"]+"-")]
        rows = [r for r in rows if r["preset"]==preset["id"]]
        candidates = {side:[] for side in Side}
        # Replay a fixed random subset of complete games, not just saved board snapshots.
        for index in rng.permutation(len(rows))[:4]:
            row = rows[int(index)]
            env = HybridChessEnv(variant=parse_variant(row["variant"]),use_cpp=True)
            state = env.reset()
            for text in row["moves"]:
                if state.ply>=4:
                    candidates[state.side_to_move].append(position(state,id=f"{row['id']}:{state.ply}",
                        preset=preset["id"],origin_game=row["id"],
                        observed_value=0. if row["winner"] is None else (1. if row["winner"]==state.side_to_move.name.lower() else -1.)))
                state,_,_,_ = env.step(parse_move(text))
        for side in Side:
            bucket = candidates[side]
            if len(bucket)<cfg["real_positions_per_army_preset"]:
                raise ValueError("Insufficient real positions")
            selected.extend(bucket[int(i)] for i in rng.choice(len(bucket),cfg["real_positions_per_army_preset"],replace=False))
    write_json(positions_path,selected)
