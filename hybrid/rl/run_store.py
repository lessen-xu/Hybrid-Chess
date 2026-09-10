"""Atomic run storage; exported models use the restricted weights-only loader."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, data):
    atomic_write(path, lambda f: f.write((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")))


@contextmanager
def run_lock(root):
    """OS releases the advisory lock even after SIGKILL; file is not a PID lock."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "run.lock", "a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def rng_state(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "generator": generator.bit_generator.state,
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.bit_generator.state = state["generator"]
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class RunStore:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        manifest = self.root / "identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Run identity/config changed; use a new output directory")
        else:
            write_json(manifest, identity)

    def load(self):
        index_path = self.root / "checkpoints.json"
        if not index_path.exists():
            if (self.root / "status.json").exists():
                raise RuntimeError("Run has progress but no checkpoint index")
            return None
        index = json.loads(index_path.read_text())
        for offset, item in enumerate(index):
            path = self.root / item["file"]
            if path.exists() and sha256(path) == item["sha256"]:
                # Full training state is trusted local run data, not a downloaded model.
                data = torch.load(path, map_location="cpu", weights_only=False)
                if data["identity"] != self.identity:
                    raise ValueError("Checkpoint identity mismatch")
                if offset:
                    write_json(self.root / "recovery.json", {"fallback": item["file"],
                        "discarded_checkpoint": index[0], "restored_steps": data.get("global_step", 0)})
                return data
        raise RuntimeError("No intact checkpoint available; refusing to restart from zero")

    def save(self, state):
        index_path = self.root / "checkpoints.json"
        previous = json.loads(index_path.read_text()) if index_path.exists() else []
        sequence = max([p["sequence"] for p in previous] + [-1]) + 1
        path = self.root / f"state-{sequence:06d}.pt"
        atomic_write(path, lambda f: torch.save({**state, "identity": self.identity}, f))
        valid_previous = [p for p in previous if (self.root / p["file"]).exists()
                          and sha256(self.root / p["file"]) == p["sha256"]]
        index = [{"file": path.name, "sha256": sha256(path), "sequence": sequence}] + valid_previous[:1]
        write_json(index_path, index)
        write_json(self.root / "status.json", {
            k: state[k] for k in ("stage", "iteration", "global_step", "epoch", "completed_games") if k in state
        })
        keep = {p["file"] for p in index}
        for item in previous:
            if item["file"] not in keep:
                (self.root / item["file"]).unlink(missing_ok=True)
