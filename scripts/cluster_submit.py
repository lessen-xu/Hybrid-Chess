"""Submit a finite UBELIX job with a persistent budget ledger and duplicate guard.

Run on the login node. An ambiguous submission is queried, never blindly retried.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--kind", choices=("cpu", "gpu", "debug", "preempt"), required=True)
    parser.add_argument("--minutes", type=int, required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--memory", default="64G")
    parser.add_argument("--budget-config", help="JSON containing a budget object for a new bounded run")
    parser.add_argument("--cpu-group", help="Required when the ledger has CPU stage limits")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ledger = Path(args.ledger).resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    gpu = args.kind != "cpu"
    if args.minutes <= 0 or args.minutes > (20 if args.kind == "debug" else 240):
        raise ValueError("Job wall time outside this round's limits")
    if not 1 <= args.cpus <= 16:
        raise ValueError("CPU count outside this round's limits")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Missing command")
    logdir = ledger.parent / "slurm"
    logdir.mkdir(exist_ok=True)
    with ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        limits = (json.loads(Path(args.budget_config).read_text())["budget"] if args.budget_config else
                  {"gpu_seconds_limit": 28800, "cpu_core_seconds_limit": 230400})
        if any(not isinstance(limits[k],int) or limits[k]<0 for k in ("gpu_seconds_limit","cpu_core_seconds_limit")):
            raise ValueError("Budget limits must be nonnegative integer seconds")
        data = json.loads(ledger.read_text()) if ledger.exists() else {**limits,"jobs":[]}
        if args.budget_config and any(data.get(k)!=v for k,v in limits.items()):
            raise ValueError("Existing ledger budget differs; refusing to modify limits")
        if not gpu and data.get("cpu_groups") and args.cpu_group not in data["cpu_groups"]:
            raise ValueError("Choose one of the ledger's CPU budget groups")
        same = [j for j in data["jobs"] if j["name"] == args.name]
        if same:
            print(json.dumps(same[0]))
            if same[0].get("job_id"):
                return
            query = subprocess.check_output(["squeue", "--noheader", "--me", "--name", args.name,
                                               "--format=%i"], text=True).strip().splitlines()
            if len(query) == 1:
                same[0]["job_id"] = query[0]
                atomic_json(ledger, data)
                print(json.dumps(same[0]))
                return
            raise RuntimeError("Unresolved prior submission. Check sacct before changing the ledger.")
        # Reserve queued/running wall limits; charge completed allocations by actual elapsed time.
        for job in data["jobs"]:
            if job.get("job_id") and "elapsed_seconds" not in job:
                result = subprocess.check_output(["sacct", "-X", "-nP", "-j", str(job["job_id"]),
                    "--format=JobIDRaw,State,ElapsedRaw"], text=True).strip().splitlines()
                for line in result:
                    jid, status, elapsed, *_ = line.split("|")
                    if jid == str(job["job_id"]) and status.split()[0] not in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"):
                        job.update(elapsed_seconds=int(elapsed), slurm_state=status)
        gpu_used = sum(j.get("elapsed_seconds", j["wall_seconds"])*int(j["gpu"]) for j in data["jobs"])
        cpu_used = sum(j.get("elapsed_seconds", j["wall_seconds"])*j["cpus"] for j in data["jobs"] if not j["gpu"])
        wall = args.minutes*60
        if not gpu and data.get("cpu_groups"):
            group_used = sum(j.get("elapsed_seconds",j["wall_seconds"])*j["cpus"]
                for j in data["jobs"] if not j["gpu"] and j.get("cpu_group")==args.cpu_group)
            if group_used+wall*args.cpus>data["cpu_groups"][args.cpu_group]:
                raise RuntimeError(f"CPU stage budget exceeded: {args.cpu_group}")
        if gpu_used+wall*int(gpu) > data["gpu_seconds_limit"] or cpu_used+wall*args.cpus*int(not gpu) > data["cpu_core_seconds_limit"]:
            raise RuntimeError(f"Budget exceeded: reserved/used GPU={gpu_used}s CPU={cpu_used} core-s")
        script = logdir / f"{args.name}.sh"
        # USR1 reaches the application through srun; the Python handler only sets a flag.
        text = "#!/usr/bin/env bash\nset -euo pipefail\nmodule load Python/3.11.3-GCCcore-12.3.0\n"
        text += f"export PATH={shlex.quote(str(Path(args.python).parent))}:\"$PATH\"\n"
        text += "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1\n"
        text += f"cd {shlex.quote(str(Path(args.release).resolve()))}\n"
        text += "step_pid=\ntrap 'if [[ -n \"$step_pid\" ]]; then kill -USR1 \"$step_pid\" 2>/dev/null || true; fi' USR1 TERM\n"
        text += "srun --cpu-bind=cores " + shlex.join(command) + " &\nstep_pid=$!\n"
        text += "set +e\nwhile true; do wait \"$step_pid\"; result=$?; kill -0 \"$step_pid\" 2>/dev/null || break; done\nexit \"$result\"\n"
        script.write_text(text)
        entry = {"name": args.name, "gpu": gpu, "cpus": args.cpus, "wall_seconds": wall,
            "cpu_group": args.cpu_group,
            "release": str(Path(args.release).resolve()), "command": command,
            "script_sha256": hashlib.sha256(text.encode()).hexdigest(), "intent_time": time.time()}
        data["jobs"].append(entry)
        atomic_json(ledger, data)
        partition = "gpu-invest" if args.kind=="preempt" else "gpu" if gpu else "epyc2"
        qos = "job_gpu_preemptable" if args.kind=="preempt" else "job_debug" if args.kind=="debug" else "job_gratis"
        submit = ["sbatch", "--parsable", "--account=gratis", "--partition="+partition,
            "--qos="+qos, "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={args.cpus}", f"--mem={args.memory}", f"--time={args.minutes}",
            f"--job-name={args.name}", f"--comment=hybrid-chess:{args.name}", "--signal=B:USR1@120",
            f"--output={logdir}/%x-%j.out", f"--error={logdir}/%x-%j.err"]
        if gpu:
            submit.append("--gres=gpu:h100:1")
        if args.kind=="preempt":
            # Every allocation must have its own final accounting record. Resume
            # explicitly with a new job ID instead of hiding time in requeues.
            submit.append("--no-requeue")
        env = {k:v for k,v in os.environ.items() if not k.startswith("SBATCH_")}
        result = subprocess.run(submit+[str(script)], env=env, text=True, capture_output=True)
        if result.returncode:
            entry.update(rejected=result.stderr, elapsed_seconds=0)
        else:
            entry["job_id"] = result.stdout.strip().split(";")[0]
        atomic_json(ledger, data)
        print(json.dumps(entry), flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr)


if __name__ == "__main__":
    main()
