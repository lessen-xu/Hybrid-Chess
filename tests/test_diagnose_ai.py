"""Diagnostic controls must be reproducible without changing production play."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from hybrid.agents.alphabeta_agent import AlphaBetaAgent,SearchConfig
from hybrid.agents.alphazero_stub import TorchPolicyValueModel
from hybrid.agents.eval import evaluate
from hybrid.core.board import Board
from hybrid.core.env import HybridChessEnv
from hybrid.core.types import Piece,PieceKind as K,Side
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.rl.az_selfplay import Example
from hybrid.rl.general_model import new_model,model_payload,encode_general
from hybrid.rl.run_store import write_json
from hybrid.rl.diagnostics.common import (position,restore,fingerprint,basic_evaluate,symmetric_evaluate,
    no_stalemate_evaluate,neural_agent,search_record,notation,parse_move,paired_interval)
from hybrid.rl.diagnostics.fixtures import tactical_positions,verify_contract
from hybrid.rl.diagnostics.data import BalancedBuffer
from hybrid.rl.diagnostics.arena import make_jobs


@pytest.mark.parametrize("record",tactical_positions(),ids=lambda r:r["id"])
def test_fixture_contract(record):
    assert verify_contract(record)["rule_contract_verified"]


def test_legacy_evaluator_and_opt_in_controls():
    env = HybridChessEnv()
    state = env.reset()
    assert abs(evaluate(state,Side.CHESS)+evaluate(state,Side.XIANGQI))>1
    for function in (basic_evaluate,symmetric_evaluate):
        assert function(state,Side.CHESS)==pytest.approx(-function(state,Side.XIANGQI))
    original = AlphaBetaAgent(SearchConfig(depth=1)).select_move(state,env.legal_moves())
    explicit = AlphaBetaAgent(SearchConfig(depth=1),evaluator=evaluate).select_move(state,env.legal_moves())
    assert original==explicit


def test_stalemate_penalty_can_be_isolated():
    board = Board.empty()
    for x,y,kind,side in [(0,0,K.KING,Side.CHESS),(3,0,K.ROOK,Side.CHESS),
                          (5,0,K.ROOK,Side.CHESS),(8,0,K.QUEEN,Side.CHESS),(4,9,K.GENERAL,Side.XIANGQI)]:
        board.set(x,y,Piece(kind,side))
    env = HybridChessEnv()
    state = env.reset_from_board(board,Side.XIANGQI)
    assert len(env.legal_moves())==1
    assert no_stalemate_evaluate(state,Side.CHESS)-evaluate(state,Side.CHESS)==pytest.approx(8.)


@pytest.mark.parametrize("cpp",[False,True])
def test_history_restore_and_recording_do_not_change_search(cpp):
    if cpp:
        pytest.importorskip("hybrid.cpp_engine.hybrid_cpp_engine")
    torch.set_num_threads(1)
    torch.manual_seed(18)
    record = tactical_positions()[-1]
    env,state = restore(record,cpp)
    assert position(state)=={k:v for k,v in record.items() if k in position(state)}
    model = TorchPolicyValueModel(new_model(8,1).eval())
    before = fingerprint(state)
    chosen = neural_agent(model,simulations=8,cpp=cpp,batch=1).select_move(state,env.legal_moves())
    move,stats = search_record(model,state,env.legal_moves(),simulations=8,cpp=cpp,batch=1)
    assert move==chosen and stats["state_preserved"] and fingerprint(state)==before
    assert stats["root_value"]==0
    assert parse_move(notation(move))==move
    _,_,done,info = env.step(move)
    assert done and info.winner is None


@pytest.mark.parametrize("mode",["network","uniform_policy","zero_value","uniform_zero"])
def test_intervention_accepts_the_engine_batch_interface(mode):
    from hybrid.rl.diagnostics.common import InterventionModel
    torch.set_num_threads(1)
    model = TorchPolicyValueModel(new_model(8,1).eval())
    env = HybridChessEnv()
    state = env.reset()
    adapter = InterventionModel(model,mode)
    assert adapter.predict_batch([])==[]
    results = adapter.predict_batch([(state,env.legal_moves())]*2)
    assert len(results)==2
    single = adapter.predict(state,env.legal_moves())
    for policy,value in results:
        assert value==pytest.approx(single[1],abs=1e-6)
        assert policy==pytest.approx(single[0],abs=1e-6)


def test_cpp_batch_eight_calls_adapter_and_clears_virtual_loss():
    pytest.importorskip("hybrid.cpp_engine.hybrid_cpp_engine")
    torch.set_num_threads(1)
    model = TorchPolicyValueModel(new_model(8,1).eval())
    env = HybridChessEnv(use_cpp=True)
    state = env.reset()
    before = fingerprint(state)
    agent = neural_agent(model,simulations=16,cpp=True,batch=8)
    root = agent._run_mcts_search(state,env.legal_moves(),add_noise=False)
    agent._assert_no_vl_leak(root)
    assert root.N==16 and fingerprint(state)==before


def examples():
    state = HybridChessEnv().reset()
    return [Example(encode_general(state).numpy(),np.array([0,90],dtype=np.uint16),
        np.array([.25,.75],dtype=np.float32),side,float(z))
        for side in Side for z in (-1,0,1) for _ in range(2 if z else 20)]


def test_balanced_sampling_is_uniform_over_six_strata_and_seeded():
    data = BalancedBuffer(examples())
    a,b = np.random.default_rng(18),np.random.default_rng(18)
    for _ in range(3):
        batch1,batch2 = data.sample_batch(600,a),data.sample_batch(600,b)
        assert np.array_equal(batch1[3],batch2[3])
        for z in (-1,0,1):
            assert 140<sum(batch1[3]==z)<260
    with pytest.raises(ValueError):
        BalancedBuffer(examples()[:2])


def test_schedule_pairs_both_armies_without_duplicate_games():
    cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/diagnose-ai.json").read_text())
    initial = make_jobs(cfg)
    assert len(initial)==len({j["id"] for j in initial})==120
    assert sum(j["assignment"]=="self" for j in initial)==72
    after = make_jobs(cfg,True,[f"{arm}-{seed}" for seed in cfg["training_seeds"] for arm in ("uniform","balanced")])
    assert len(after)==48
    for row in [j for j in initial+after if j["assignment"]=="chess"]:
        others = [j for j in initial+after if j["group"]==row["group"] and j["seed"]==row["seed"] and j["assignment"]=="xiangqi"]
        assert len(others)==1
        assert others[0]["chess"]==row["xiangqi"] and others[0]["xiangqi"]==row["chess"]


def test_positive_differences_still_use_difference_range():
    assert paired_interval([.5]*20,difference=True)[0]<paired_interval([.5]*20)[0]


@pytest.mark.parametrize("arm",["uniform","balanced"])
def test_ablation_resume_matches_uninterrupted(tmp_path,arm):
    from hybrid.rl.diagnostics.ablate import fit
    torch.set_num_threads(1)
    model = tmp_path/"initial.pt"
    torch.manual_seed(27)
    torch.save(model_payload(new_model(8,1)),model)
    train,valid = ReplayBuffer(),ReplayBuffer()
    train.append(examples())
    valid.append(examples())
    cfg = dict(learning_rate=.0001,weight_decay=.0001,batch_size=8,updates=4,
               checkpoint_seconds=600,model_sha256="test")
    def args(name):
        root = tmp_path/name
        write_json(root/"pool.json",{})
        return SimpleNamespace(output=str(root),model=str(model),source_version="test")
    continuous,resumed = args("continuous"),args("resumed")
    fit(continuous,cfg,arm,8,train,valid,lambda:False,"cpu")
    class Interrupt:
        calls = 0
        def __call__(self):
            self.calls+=1
            return self.calls>2
    assert fit(resumed,cfg,arm,8,train,valid,Interrupt(),"cpu") is None
    fit(resumed,cfg,arm,8,train,valid,lambda:False,"cpu")
    def checkpoint(path):
        root = Path(path.output)/"ablate"/f"{arm}-8"
        index = json.loads((root/"checkpoints.json").read_text())
        return torch.load(root/index[0]["file"],weights_only=False)
    a,b = checkpoint(continuous),checkpoint(resumed)
    assert a["global_step"]==b["global_step"]==4
    assert a["rng"]["generator"]==b["rng"]["generator"]
    assert torch.equal(a["rng"]["torch"],b["rng"]["torch"])
    for key,value in a["model"].items():
        assert torch.equal(value,b["model"][key])
    for parameter,values in a["optimizer"]["state"].items():
        for key,value in values.items():
            assert torch.equal(value,b["optimizer"]["state"][parameter][key])
