"""Opt-in, finite root-cause experiments for army imbalance.

All experiment execution requires a Slurm compute node. Existing game, engine,
model, and general evaluation defaults are not changed by this command.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import socket
import subprocess

from hybrid.rl.general_train import require_compute,StopControl
from hybrid.rl.run_store import RunStore,run_lock,sha256,write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("audit","probe","arena","ablate","report"))
    parser.add_argument("--config",default="configs/diagnose-ai.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--teacher",required=True)
    parser.add_argument("--training",required=True)
    parser.add_argument("--evaluation",required=True)
    parser.add_argument("--source-version",required=True)
    parser.add_argument("--wall-seconds",type=float,default=3300)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--after",action="store_true",help="Evaluate all four sampling-ablation models")
    parser.add_argument("--reference-run",help="Explicit immutable input run for corrected probes/report aggregation")
    args = parser.parse_args()
    require_compute()
    if args.workers<1 or args.workers>16 or args.wall_seconds<=0:
        raise ValueError("Invalid bounded experiment configuration")
    if args.after and args.stage!="arena":
        raise ValueError("--after only applies to arena")
    cfg = json.loads(Path(args.config).read_text())
    if cfg["protocol"]!=1 or sha256(args.model)!=cfg["model_sha256"]:
        raise ValueError("Frozen model or protocol mismatch")
    source_manifest = Path("SOURCE.json")
    if not source_manifest.exists() or json.loads(source_manifest.read_text())["source_version"]!=args.source_version:
        raise ValueError("Run from the immutable source release matching --source-version")
    mp.set_start_method("spawn",force=True)
    root = Path(args.output)
    identity = dict(config=cfg,source_version=args.source_version,
        inputs={key:str(Path(getattr(args,key)).resolve()) for key in ("model","teacher","training","evaluation")})
    if args.reference_run:
        if args.stage not in ("probe","report"):
            raise ValueError("A reference run is only supported for probe and report")
        reference = Path(args.reference_run).resolve()
        old = json.loads((reference/"identity.json").read_text())
        if old["config"]!=cfg or old["inputs"]!=identity["inputs"] or reference==root.resolve():
            raise ValueError("Reference run inputs or configuration differ")
        identity["reference_run"] = dict(path=str(reference),identity_sha256=sha256(reference/"identity.json"),
                                         source_version=old["source_version"])
    # Lock each stage independently so training and CPU probes can run concurrently.
    stage = "after" if args.after else args.stage
    with run_lock(root/(stage+"-lock")):
        with run_lock(root/"identity-lock"):
            RunStore(root,identity)
        write_json(root/stage/"hardware.json",dict(hostname=socket.gethostname(),
            cpu=json.loads(subprocess.check_output(["lscpu","--json"],text=True))))
        stop = StopControl(args.wall_seconds)
        if args.stage=="audit":
            from hybrid.rl.diagnostics.data import audit
            audit(args,cfg,stop)
        elif args.stage=="probe":
            from hybrid.rl.diagnostics.probe import probe
            probe(args,cfg,stop)
        elif args.stage=="arena":
            from hybrid.rl.diagnostics.arena import arena
            arena(args,cfg,stop)
        elif args.stage=="ablate":
            from hybrid.rl.diagnostics.ablate import ablate
            ablate(args,cfg,stop)
        else:
            from hybrid.rl.diagnostics.report import report
            report(args,cfg)


if __name__=="__main__":
    main()
