"""Dedicated CPU allocation probe for batch USR1 and cross-allocation recovery."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Side
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.rl.az_selfplay import Example
from hybrid.rl.az_train import train_one_epoch
from hybrid.rl.general_model import encode_general
from hybrid.rl.general_train import require_compute, Trainer, StopControl
from hybrid.rl.run_store import run_lock, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require_compute()
    cfg = json.loads(Path("configs/general-ai.json").read_text())
    cfg.update(channels=8, res_blocks=1, batch_size=4)
    stop = StopControl(150)
    buffer = ReplayBuffer()
    state = HybridChessEnv().reset()
    for i in range(32):
        buffer.append([Example(encode_general(state).numpy(), np.array([0, 90], dtype=np.uint16),
            np.array([float(i % 2), float(1-i % 2)], dtype=np.float32), Side.CHESS, float(i % 3-1))])
    root = Path(args.output)
    with run_lock(root):
        if args.mode == "prepare":
            trainer = Trainer(cfg, root / "interrupted", "recovery-probe-v1", "cpu", stop)
            if trainer.state["global_step"]:
                raise RuntimeError("Probe already has progress")
            trainer.state["epoch_steps"] = 8
            for _ in range(3):
                stats = train_one_epoch(trainer.net, buffer, trainer.optimizer, torch.device("cpu"),
                                        batch_size=4, max_steps=1, rng=trainer.rng)
                trainer.state["batch_step"] += 1
                trainer.state["global_step"] += 1
                trainer.state["epoch_loss"] += stats["total_loss"]
            print("READY_FOR_BATCH_USR1", flush=True)
            while not stop():
                time.sleep(.1)
            if not stop.requested:
                raise RuntimeError("No batch signal reached Python")
            trainer.save()
            write_json(root / "signal.json", {"signal_received": True, "saved_step": trainer.state["global_step"]})
        else:
            if not (root / "signal.json").exists():
                raise RuntimeError("Signal checkpoint has not been accepted")
            continuous = Trainer(cfg, root / "continuous", "recovery-probe-v1", "cpu", stop)
            assert continuous.train_epoch(buffer)
            resumed = Trainer(cfg, root / "interrupted", "recovery-probe-v1", "cpu", stop)
            assert resumed.state["global_step"] == 3
            assert resumed.train_epoch(buffer)
            assert resumed.state["global_step"] == continuous.state["global_step"] == 8
            assert resumed.rng.bit_generator.state == continuous.rng.bit_generator.state
            for key, value in continuous.net.state_dict().items():
                assert torch.equal(value, resumed.net.state_dict()[key]), key
            for parameter, fields in continuous.optimizer.state_dict()["state"].items():
                for key, value in fields.items():
                    assert torch.equal(value, resumed.optimizer.state_dict()["state"][parameter][key])
            write_json(root / "verified.json", {"cross_allocation_resume": True,
                "model_optimizer_rng_exact": True, "steps": 8})
            print("CROSS_ALLOCATION_RESUME_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
