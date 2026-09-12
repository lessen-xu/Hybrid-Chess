"""Finite, resumable teacher -> supervised -> multivariant self-play pipeline.

Run on compute nodes: python -m hybrid.rl.general_train --help
Full checkpoints are trusted local data; exported models are weights-only.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import socket
import time

import numpy as np
import torch
import torch.nn.functional as F

from hybrid.rl.az_replay import ReplayBuffer, BalancedBuffer
from hybrid.rl.az_train import train_one_epoch
from hybrid.rl.general_games import collect_games, load_game
from hybrid.rl.general_model import new_model, model_payload
from hybrid.rl.run_store import RunStore, atomic_write, write_json, sha256, rng_state, restore_rng, run_lock


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0].startswith("submit"):
        raise RuntimeError("Run training and benchmarks inside a Slurm compute allocation")


class StopControl:
    def __init__(self, wall_seconds):
        self.deadline = time.time() + wall_seconds
        self.requested = False
        for name in ("SIGUSR1", "SIGTERM", "SIGINT"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), self._signal)

    def _signal(self, *_):
        self.requested = True

    def __call__(self):
        return self.requested or time.time() >= self.deadline


def game_reference(path):
    return {"path": str(path), "sha256": sha256(path)}


def load_buffer(references, capacity, balanced: bool = False):
    buffer = BalancedBuffer(capacity) if balanced else ReplayBuffer(capacity)
    for reference in references:
        if sha256(reference["path"]) != reference["sha256"]:
            raise RuntimeError(f"Changed replay shard: {reference['path']}")
        examples, _ = load_game(reference["path"])
        buffer.append(examples)
    return buffer



def validation_loss(net, buffer, device, batch_size):
    """Full validation set; legal zero-target moves remain in the denominator."""
    net.eval()
    total, count = 0., 0
    with torch.no_grad():
        for offset in range(0, len(buffer), batch_size):
            examples = buffer.examples[offset:offset+batch_size]
            states = torch.from_numpy(np.stack([ex.state for ex in examples]).astype(np.float32)).to(device)
            policy, value = net(states)
            flat = policy.flatten(1)
            losses = []
            for i, ex in enumerate(examples):
                indices = torch.as_tensor(ex.pi_indices.astype(np.int64), device=device)
                probs = torch.as_tensor(ex.pi_probs, device=device)
                losses.append(-(probs * F.log_softmax(flat[i, indices], dim=0)).sum() + (value[i, 0]-ex.z)**2)
            total += float(torch.stack(losses).sum())
            count += len(examples)
    if not count:
        raise RuntimeError("Empty validation split")
    return total / count


class Trainer:
    def __init__(self, config, root, code_version, device, stop):
        self.config, self.root = config, Path(root)
        self.device, self.stop = device, stop
        self.store = RunStore(root, {"config": config, "source_version": code_version, "format": 2})
        torch.set_num_threads(1)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        self.rng = np.random.default_rng(config["seed"])
        self.net = new_model(config["channels"], config["res_blocks"]).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config["supervised_lr"],
                                          weight_decay=config["weight_decay"])
        self.state = self.store.load()
        if self.state is None:
            self.state = {"stage": "supervised", "epoch": 0, "iteration": 0, "global_step": 0,
                "batch_step": 0, "epoch_steps": 0, "epoch_loss": 0., "best_validation": float("inf"),
                "bad_epochs": 0, "completed_games": [], "replay": [], "metrics": [],
                "source_version": code_version}
            self.save()
        else:
            self.net.load_state_dict(self.state["model"])
            self.optimizer.load_state_dict(self.state["optimizer"])
            restore_rng(self.state["rng"], self.rng)
        self.last_save = time.monotonic()

    def save(self):
        self.state["model"] = self.net.state_dict()
        self.state["optimizer"] = self.optimizer.state_dict()
        self.state["rng"] = rng_state(self.rng)
        self.store.save(self.state)
        write_json(self.root / "metrics.json", self.state["metrics"])
        self.last_save = time.monotonic()

    def export(self, name):
        path = self.root / name
        payload = model_payload(self.net, source_version=self.state["source_version"],
            iteration=self.state["iteration"], global_step=self.state["global_step"],
            training_config=self.config)
        atomic_write(path, lambda f: torch.save(payload, f))
        write_json(path.with_suffix(".json"), {"sha256": sha256(path),
                   "source_version": self.state["source_version"], "encoding_version": 2,
                   "iteration": self.state["iteration"], "global_step": self.state["global_step"]})
        return path

    def train_epoch(self, buffer):
        state, cfg = self.state, self.config
        if not state["epoch_steps"]:
            state["epoch_steps"] = max(1, math.ceil(len(buffer)/cfg["batch_size"]))
        while state["batch_step"] < state["epoch_steps"]:
            if self.stop():
                self.save()
                return False
            stats = train_one_epoch(self.net, buffer, self.optimizer, torch.device(self.device),
                batch_size=cfg["batch_size"], max_steps=1, rng=self.rng)
            if not math.isfinite(stats["total_loss"]):
                raise RuntimeError("Non-finite training loss")
            state["epoch_loss"] += stats["total_loss"]
            state["batch_step"] += 1
            state["global_step"] += 1
            if time.monotonic()-self.last_save >= cfg["checkpoint_seconds"]:
                self.save()
        return True

    def finish_epoch(self):
        loss = self.state["epoch_loss"] / self.state["epoch_steps"]
        self.state.update(batch_step=0, epoch_steps=0, epoch_loss=0.)
        return loss

    def run(self, teacher_manifest, max_iterations=None):
        cfg, state = self.config, self.state
        if state["stage"] == "supervised":
            manifest = json.loads(Path(teacher_manifest).read_text())
            if "teacher_manifest_sha256" in state and state["teacher_manifest_sha256"] != sha256(teacher_manifest):
                raise ValueError("Teacher dataset changed while resuming")
            state["teacher_manifest_sha256"] = sha256(teacher_manifest)
            balanced = cfg.get("balanced_sampling", True)
            train = load_buffer(manifest["train"], cfg["teacher_samples"], balanced=balanced)
            valid = load_buffer(manifest["validation"], cfg["teacher_samples"], balanced=False)
            if not len(train) or not len(valid):
                raise RuntimeError("Need nonempty game-disjoint teacher train and validation sets")
            while state["epoch"] < cfg["supervised_epochs"] and state["bad_epochs"] < cfg["patience"]:
                if not self.train_epoch(train):
                    return
                val = validation_loss(self.net, valid, self.device, cfg["batch_size"])
                metric = {"stage": "supervised", "epoch": state["epoch"],
                          "train_loss": self.finish_epoch(), "validation_loss": val}
                state["metrics"].append(metric)
                print(json.dumps(metric), flush=True)
                state["epoch"] += 1
                if val < state["best_validation"]:
                    state["best_validation"] = val
                    state["bad_epochs"] = 0
                    state["supervised_best"] = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    self.export("supervised.pt")
                else:
                    state["bad_epochs"] += 1
                self.save()
            self.net.load_state_dict(state["supervised_best"])
            self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=cfg["selfplay_lr"],
                                               weight_decay=cfg["weight_decay"])
            state["stage"] = "collect"
            self.export("candidate.pt")
            self.save()
        while not self.stop() and (max_iterations is None or state["iteration"] < max_iterations):
            if state["stage"] == "collect":
                checkpoint = self.export("current.pt")
                model_hash = sha256(checkpoint)
                # torch.save container metadata need not be byte-stable on resume.
                # Persist the source checkpoint used by this collection exactly once.
                frozen = self.root / "selfplay" / f"iter-{state['iteration']:05d}" / "model.pt"
                if not frozen.exists():
                    atomic_write(frozen, lambda f: f.write(checkpoint.read_bytes()))
                else:
                    from hybrid.rl.general_model import load_general_model
                    frozen_net = load_general_model(frozen)
                    if any(not torch.equal(v.cpu(), frozen_net.state_dict()[k]) for k, v in self.net.state_dict().items()):
                        raise RuntimeError("Collection model differs from resumed checkpoint")
                model_hash = sha256(frozen)
                sims = cfg["initial_simulations"] if state["iteration"] < cfg["initial_iterations"] else cfg["simulations"]
                jobs = [{"mode": "selfplay", "game_id": state["iteration"]*cfg["games_per_iteration"]+i,
                    "seed": cfg["seed"]+100000, "simulations": sims, "use_cpp": cfg["use_cpp"],
                    "max_plies": cfg["max_plies"], "deadline": self.stop.deadline,
                    "model_sha256": model_hash, "record": i < 2,
                    "path": str(frozen.parent / f"game-{i:04d}.npz")} for i in range(cfg["games_per_iteration"])]
                def on_game(meta):
                    if meta["game_id"] not in state["completed_games"]:
                        state["completed_games"].append(meta["game_id"])
                    print(json.dumps({"stage": "selfplay", "iteration": state["iteration"],
                        **{k: meta[k] for k in ("game_id", "family", "plies", "winner", "reason", "seconds")}}), flush=True)
                    if time.monotonic()-self.last_save >= cfg["checkpoint_seconds"]:
                        self.save()
                collect_games(jobs, cfg["workers"], frozen, self.device, on_game, self.stop)
                if not all(Path(job["path"]).exists() for job in jobs):
                    self.save()
                    return
                references = [game_reference(job["path"]) for job in jobs]
                state["replay"].extend(references)
                # Drop whole oldest shards only when the remaining shards still fill the buffer.
                sizes = [load_game(r["path"])[1]["samples"] for r in state["replay"]]
                total = sum(sizes)
                while sizes and total-sizes[0] >= cfg["replay_capacity"]:
                    total -= sizes.pop(0)
                    state["replay"].pop(0)
                records = [load_game(job["path"])[1] for job in jobs]
                state["metrics"].append({"stage": "collection", "iteration": state["iteration"],
                    "games": len(records), "samples": sum(r["samples"] for r in records),
                    "chess_wins": sum(r["winner"] == "chess" for r in records),
                    "xiangqi_wins": sum(r["winner"] == "xiangqi" for r in records),
                    "draws": sum(r["winner"] is None for r in records),
                    "seconds_sum": sum(r["seconds"] for r in records), "simulations": sims})
                state["stage"] = "train"
                self.save()
            if state["stage"] == "train":
                balanced = cfg.get("balanced_sampling", True)
                buffer = load_buffer(state["replay"], cfg["replay_capacity"], balanced=balanced)
                if not self.train_epoch(buffer):
                    return
                metric = {"stage": "selfplay_train", "iteration": state["iteration"],
                    "loss": self.finish_epoch(), "buffer_samples": len(buffer)}
                state["metrics"].append(metric)
                print(json.dumps(metric), flush=True)
                state["iteration"] += 1
                state["stage"] = "collect"
                self.export(f"candidate-{state['iteration']:04d}.pt")
                self.export("candidate.pt")
                self.save()
        self.save()


def generate_teacher(cfg, root, source_version, stop):
    root = Path(root)
    store = RunStore(root, {"config": cfg, "source_version": source_version, "format": 2, "stage": "teacher"})
    previous = store.load() or {"stage": "teacher", "completed_games": [], "iteration": 0, "global_step": 0,
                                "elapsed": 0.}
    if previous.get("finalized"):
        if not (root / "dataset.json").exists():
            raise RuntimeError("Finalized teacher run is missing its manifest")
        return
    started = time.monotonic()
    remaining = max(0., cfg["teacher_wall_seconds"]-previous["elapsed"])
    stop.deadline = min(stop.deadline, time.time()+remaining)
    records = {}
    for path in sorted((root / "games").glob("game-*.npz")):
        _, meta = load_game(path)
        if meta["seed"] != cfg["seed"] or meta["mode"] != "teacher":
            raise ValueError("Teacher game identity mismatch")
        records[meta["game_id"]] = meta
    last_save = time.monotonic()
    def save():
        previous.update(completed_games=sorted(records), global_step=sum(r["samples"] for r in records.values()),
                        elapsed=previous.get("elapsed", 0.)+time.monotonic()-save.last)
        save.last = time.monotonic()
        store.save(previous)
    save.last = started
    def finished():
        return stop() or sum(r["samples"] for r in records.values()) >= cfg["teacher_samples"]
    def on_game(meta):
        nonlocal last_save
        records[meta["game_id"]] = meta
        print(json.dumps({"stage": "teacher", "game": meta["game_id"], "plies": meta["plies"],
                          "samples": sum(r["samples"] for r in records.values()), "winner": meta["winner"]}), flush=True)
        if time.monotonic()-last_save >= cfg["checkpoint_seconds"]:
            save()
            last_save = time.monotonic()
    wave_size = max(30, cfg["workers"]*2)
    game_start = 0
    while not finished():
        jobs = [{"mode": "teacher", "game_id": i, "seed": cfg["seed"], "teacher_seconds": cfg["teacher_seconds"],
            "max_plies": cfg["max_plies"], "deadline": stop.deadline, "record": i < 2,
            "use_cpp": False, "path": str(root / "games" / f"game-{i:05d}.npz")}
            for i in range(game_start, game_start+wave_size)]
        collect_games(jobs, cfg["workers"], on_game=on_game, should_stop=finished)
        game_start += wave_size
    save()
    references, total = [], 0
    for path in sorted((root / "games").glob("game-*.npz")):
        _, meta = load_game(path)
        if total+meta["samples"] > cfg["teacher_samples"]:
            continue
        total += meta["samples"]
        references.append(game_reference(path))
    if len(references) < 2:
        raise RuntimeError("Teacher run produced insufficient game-disjoint data")
    order = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 918])).permutation(len(references))
    validation_count = max(1, round(len(references)*.1))
    valid = [references[i] for i in order[:validation_count]]
    train = [references[i] for i in order[validation_count:]]
    write_json(root / "dataset.json", {"train": train, "validation": valid, "samples": total,
        "seed": cfg["seed"], "source_version": source_version})
    write_json(root / "summary.json", {"samples": total, "games": len(train)+len(valid),
        "train_games": len(train), "validation_games": len(valid), "elapsed_seconds": previous["elapsed"]})
    previous["finalized"] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher", "train"))
    parser.add_argument("--config", default="configs/general-ai.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--teacher-manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--wall-seconds", type=float, default=13500)
    parser.add_argument("--max-iterations", type=int)
    args = parser.parse_args()
    require_compute()
    mp.set_start_method("spawn", force=True)
    config = json.loads(Path(args.config).read_text())
    stop = StopControl(args.wall_seconds)
    with run_lock(args.output):
        if args.stage == "teacher":
            generate_teacher(config, args.output, args.source_version, stop)
        else:
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA allocation unavailable")
            Trainer(config, args.output, args.source_version, args.device, stop).run(args.teacher_manifest, args.max_iterations)


if __name__ == "__main__":
    main()
