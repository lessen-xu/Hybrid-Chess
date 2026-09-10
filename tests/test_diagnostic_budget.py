"""Budget guards and preemption flags tested with a mocked scheduler."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import pytest

pytest.importorskip("fcntl",reason="Slurm submission is supported on Linux")


def invoke(tmp_path,monkeypatch,kind="preempt",minutes=15,cpus=16,existing=None):
    ledger = tmp_path/"budget.json"
    limits = dict(gpu_seconds_limit=7200,cpu_core_seconds_limit=86400,cpu_groups={"checks":18000})
    config = tmp_path/"config.json"
    config.write_text(json.dumps({"budget":limits}))
    if existing:
        ledger.write_text(json.dumps(existing))
    calls = []
    def submit(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout="12345\n",stderr="")
    monkeypatch.setattr(subprocess,"run",submit)
    monkeypatch.setattr(sys,"argv",["cluster_submit","--ledger",str(ledger),"--budget-config",str(config),
        "--name","test","--release",str(tmp_path),"--python",sys.executable,"--kind",kind,
        "--minutes",str(minutes),"--cpus",str(cpus),"--cpu-group","checks","--","python","work.py"])
    runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts/cluster_submit.py"),run_name="__main__")
    return calls,json.loads(ledger.read_text())


def test_preemptible_gpu_is_free_and_never_requeues_silently(tmp_path,monkeypatch):
    calls,ledger = invoke(tmp_path,monkeypatch)
    command = calls[0]
    assert "--account=gratis" in command and "--partition=gpu-invest" in command
    assert "--qos=job_gpu_preemptable" in command and "--no-requeue" in command
    assert "--gres=gpu:h100:1" in command
    assert ledger["gpu_seconds_limit"]==7200 and ledger["jobs"][0]["gpu"]


@pytest.mark.parametrize("kind,minutes,cpus",[("gpu",121,1),("cpu",240,2)])
def test_total_and_stage_caps_reject_before_submission(tmp_path,monkeypatch,kind,minutes,cpus):
    with pytest.raises(RuntimeError,match="budget|Budget"):
        invoke(tmp_path,monkeypatch,kind,minutes,cpus)
    ledger = tmp_path/"budget.json"
    assert not ledger.exists()


def test_budget_configuration_cannot_enlarge_existing_ledger(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="budget differs"):
        invoke(tmp_path,monkeypatch,existing=dict(gpu_seconds_limit=3600,cpu_core_seconds_limit=86400,
            cpu_groups={"checks":18000},jobs=[]))
