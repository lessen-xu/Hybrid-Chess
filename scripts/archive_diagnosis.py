"""Archive completed diagnostic runs, checking every payload before publication.

Original training datasets are external inputs, identified by the frozen pool
and original backup hashes. This archive includes all new recovery checkpoints.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Archive compression runs in a compute allocation")
    root,output = Path(args.root).resolve(),Path(args.output).resolve()
    assert root.is_dir() and not output.exists()
    for stage,count in (("run-r01/arena",120),("run-r01/after",48),("run-r02/probe",1620)):
        summary = read(root/stage/"summary.json")
        assert summary["complete"]
        assert summary.get("completed_games",summary.get("completed_cases"))==count
    assert read(root/"confirm-r05/summary.json")["cases"]==168
    evidence = read(root/"evidence-final/evidence.json")
    assert evidence["final"] and all(v["all_results_match"] for v in evidence["replay_checks"].values())
    assert (root/"run-r02/report.json").exists()
    candidates = read(root/"run-r01/ablate/candidates.json")
    assert len(candidates)==4
    for row in candidates:
        assert digest(row["path"])==row["sha256"]
        assert read(Path(row["path"]).parent/"metrics.json")["updates"]==128
    files = {}
    assert read(root/"replay-import-check.json")["complete"]
    for name in ("run-r01","run-r02","evidence-baseline","evidence-final","confirm-r05","ui-replays","slurm"):
        for path in sorted((root/name).rglob("*")):
            assert not path.is_symlink()
            if path.is_file() and path.name!="run.lock" and not path.name.endswith(".tmp"):
                if name=="slurm" and os.environ["SLURM_JOB_ID"] in path.name:
                    continue  # This allocation's log is still open.
                files[path.relative_to(root).as_posix()] = path
    files["budget-at-archive.json"] = root/"budget.json"
    files["replay-import-check.json"] = root/"replay-import-check.json"
    files["helpers/archive_diagnosis.py"] = Path(__file__).resolve()
    files["helpers/check_diagnostic_replays.cjs"] = root/"check_diagnostic_replays.cjs"
    for name in ("launch.py","monitor.py","accounting.py"):
        files["helpers/"+name] = root/name
    project = root.parent.parent
    for index in range(1,6):
        path = project/"releases"/f"diagnose-r{index:02d}.tar.gz"
        assert path.is_file()
        files["sources/"+path.name] = path
    frozen = project/"outputs/round01/eval-inputs/candidate-0104-6d71c98a.pt"
    assert digest(frozen)==read(root/"run-r01/identity.json")["config"]["model_sha256"]
    files["models/frozen-0104.pt"] = frozen
    manifest = dict(job_id=os.environ["SLURM_JOB_ID"],
        primary_probe="run-r02/probe",excluded_partial_probe="run-r01/probe",
        note="Original teacher/selfplay shards remain external SHA-identified inputs; final Slurm accounting is issued after this archive allocation ends.",
        files=[dict(file=name,bytes=path.stat().st_size,sha256=digest(path)) for name,path in sorted(files.items())])
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_name(output.name+".tmp")
    assert not temporary.exists()
    with temporary.open("xb") as stream:
        with tarfile.open(fileobj=stream,mode="w:gz") as archive:
            for row in manifest["files"]:
                data = files[row["file"]].read_bytes()
                assert len(data)==row["bytes"] and hashlib.sha256(data).hexdigest()==row["sha256"]
                info = tarfile.TarInfo("diagnosis01/"+row["file"])
                info.size = len(data)
                archive.addfile(info,io.BytesIO(data))
            data = json.dumps(manifest,indent=2).encode()
            info = tarfile.TarInfo("diagnosis01/ARCHIVE_MANIFEST.json")
            info.size = len(data)
            archive.addfile(info,io.BytesIO(data))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,output)
    receipt = dict(path=str(output),sha256=digest(output),bytes=output.stat().st_size,files=len(files))
    output.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
