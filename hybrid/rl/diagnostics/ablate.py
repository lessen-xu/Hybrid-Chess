"""Two sampling-only interventions, with complete optimizer/RNG recovery."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from hybrid.rl.az_train import train_one_epoch
from hybrid.rl.general_model import load_general_model,model_payload
from hybrid.rl.run_store import RunStore,rng_state,restore_rng,write_json,atomic_write,sha256
from .common import read,json_hash
from .data import load_pool,BalancedBuffer,sampling_key


def metrics(net,buffer,device):
    net.eval()
    groups = defaultdict(list)
    with torch.no_grad():
        for start in range(0,len(buffer),256):
            examples = buffer.examples[start:start+256]
            x = torch.from_numpy(np.stack([e.state for e in examples]).astype(np.float32)).to(device)
            logits,values = net(x)
            for i,ex in enumerate(examples):
                indices = torch.as_tensor(ex.pi_indices.astype(np.int64),device=device)
                logp = torch.log_softmax(logits[i].flatten()[indices],0)
                target = torch.as_tensor(ex.pi_probs,device=device)
                v = float(values[i,0])
                row = dict(value=v,target=ex.z,squared_error=(v-ex.z)**2,
                    saturated=float(abs(v)>=.95),policy_loss=float(-(target*logp).sum()),
                    policy_entropy=float(-(logp.exp()*logp).sum()),
                    legal_count=len(indices),zero_target_fraction=float(np.mean(ex.pi_probs==0)))
                groups[ex.side_to_move.name].append(row)
                groups[str(sampling_key(ex))].append(row)
    return {key:dict(samples=len(rows),**{metric:float(np.mean([r[metric] for r in rows]))
                for metric in rows[0]}) for key,rows in groups.items()}


def fit(args,cfg,arm,seed,train,valid,stop,device):
    root = Path(args.output)/"ablate"/f"{arm}-{seed}"
    identity = dict(source_version=args.source_version,config=cfg,arm=arm,seed=seed,
                    pool_sha256=sha256(Path(args.output)/"pool.json"),model_sha256=cfg["model_sha256"])
    store = RunStore(root,identity)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = load_general_model(args.model,device)
    optimizer = torch.optim.AdamW(net.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    state = store.load()
    if state:
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"],rng)
    else:
        state = dict(stage="ablate",global_step=0,metrics=[],before=metrics(net,valid,device))
    last = time.monotonic()
    def save():
        state.update(model=net.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(rng))
        store.save(state)
    data = train if arm=="uniform" else BalancedBuffer(train.examples)
    while state["global_step"]<cfg["updates"]:
        if stop():
            save()
            return None
        result = train_one_epoch(net,data,optimizer,torch.device(device),cfg["batch_size"],max_steps=1,rng=rng)
        if not all(np.isfinite(value) for value in result.values()):
            raise RuntimeError("Non-finite training statistics")
        state["global_step"] += 1
        state["metrics"].append(result)
        if time.monotonic()-last>=cfg["checkpoint_seconds"]:
            save()
            last = time.monotonic()
    if "after" not in state:
        state["after"] = metrics(net,valid,device)
    save()
    model_path = root/"candidate.pt"
    if not model_path.exists():
        atomic_write(model_path,lambda stream:torch.save(model_payload(net,source_version=args.source_version,
            experiment="sampling-ablation",arm=arm,seed=seed,global_step=cfg["updates"],
            parent_sha256=cfg["model_sha256"]),stream))
    write_json(root/"metrics.json",dict(arm=arm,seed=seed,updates=state["global_step"],before=state["before"],
        after=state["after"],training=state["metrics"],train_samples=len(train),validation_samples=len(valid)))
    return dict(name=f"{arm}-{seed}",arm=arm,seed=seed,path=str(model_path),sha256=sha256(model_path))


def ablate(args,cfg,stop):
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Ablation fitting requires the allocated GPU")
    train,valid = load_pool(read(Path(args.output)/"pool.json"))
    if not len(train) or not len(valid):
        raise ValueError("Need nonempty game-disjoint splits")
    write_json(Path(args.output)/"ablate/composition.json",dict(train=Counter(str(sampling_key(e)) for e in train.examples),
        validation=Counter(str(sampling_key(e)) for e in valid.examples)))
    candidates = []
    for seed in cfg["training_seeds"]:
        for arm in ("uniform","balanced"):
            result = fit(args,cfg,arm,seed,train,valid,stop,"cuda")
            if result is None:
                write_json(Path(args.output)/"ablate/candidates.json",candidates)
                return
            candidates.append(result)
            write_json(Path(args.output)/"ablate/candidates.json",candidates)
            print(json.dumps(result),flush=True)
