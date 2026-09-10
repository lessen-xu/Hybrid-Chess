"""Rule-aware learning regressions. Run the entire file on a compute node."""
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig, Node, TorchPolicyValueModel
from hybrid.core.env import HybridChessEnv
from hybrid.core.rules import board_hash
from hybrid.core.types import Side, Move
from hybrid.rl.az_replay import ReplayBuffer
from hybrid.rl.az_selfplay import Example
from hybrid.rl.az_train import train_one_epoch
from hybrid.rl.general_model import (encode_general, context_features, sample_variant, new_model,
    model_payload, load_general_model, RULE_FIELDS)
from hybrid.rl.run_store import RunStore, rng_state, restore_rng
from hybrid.web_variants import parse_variant, PRESETS


@pytest.fixture(autouse=True)
def one_torch_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("key", RULE_FIELDS)
def test_rule_is_observable_with_identical_board(key):
    first = HybridChessEnv().reset()
    second = first.clone()
    flags = first.variant.to_dict()
    flags[key] = not flags[key]
    second.variant = parse_variant(flags)
    a, b = encode_general(first), encode_general(second)
    assert torch.equal(a[:15], b[:15])
    assert not torch.equal(a[15:], b[15:])
    assert a.shape == (29, 10, 9)


def test_fractional_features_and_game_storage(tmp_path):
    from hybrid.rl.general_games import save_game, load_game
    state = HybridChessEnv().reset()
    state.ply = 123
    state.repetition[board_hash(state.board, state.side_to_move)] = 2
    encoded = encode_general(state).numpy().astype(np.float16)
    ex = Example(encoded, np.array([0, 1], dtype=np.uint16), np.array([1., 0.], dtype=np.float32), Side.CHESS, 1.)
    path = tmp_path / "game.npz"
    save_game(path, [ex], {"complete": True})
    examples, meta = load_game(path)
    assert np.allclose(examples[0].state[-2:, 0, 0], [(400-123)/400, 2/3], atol=0.0005)
    assert examples[0].pi_probs.tolist() == [1., 0.]


def test_full_legal_actions_receive_policy_gradients():
    net = new_model(channels=8, res_blocks=1)
    state = HybridChessEnv().reset()
    ex = Example(encode_general(state).numpy(), np.array([0, 90], dtype=np.uint16),
                 np.array([1., 0.], dtype=np.float32), Side.CHESS, 0.)
    buffer = ReplayBuffer()
    buffer.append([ex, ex])
    optimizer = torch.optim.SGD(net.parameters(), lr=.01)
    stats = train_one_epoch(net, buffer, optimizer, torch.device("cpu"), batch_size=2, max_steps=1)
    assert stats["policy_loss"] > 0
    assert net.policy_conv.bias.grad[0] < 0
    assert net.policy_conv.bias.grad[1] > 0


@pytest.mark.parametrize("cpp", [False, True])
def test_search_child_records_repetition_without_parent_mutation(cpp):
    if cpp:
        pytest.importorskip("hybrid.cpp_engine.hybrid_cpp_engine")
    env = HybridChessEnv(use_cpp=cpp)
    state = env.reset()
    net = TorchPolicyValueModel(new_model(8, 1))
    agent = AlphaZeroMiniAgent(net, MCTSConfig(simulations=1, leaf_batch_size=1), use_cpp=cpp)
    root = agent._run_mcts_search(state, env.legal_moves(), add_noise=False)
    child = next(iter(root.children.values()))
    if cpp:
        key = child.cpp_board.board_hash(child.cpp_side)
    else:
        key = board_hash(child.state.board, child.state.side_to_move)
    assert child.state.repetition[key] == 1
    assert key not in state.repetition
    assert child.state.repetition is not state.repetition


def test_virtual_loss_discourages_a_reserved_branch():
    state = HybridChessEnv().reset()
    root = Node(state)
    a, b = Move(0, 1, 0, 2), Move(1, 1, 1, 2)
    root.children = {a: Node(state, prior=.5, virtual_loss=1), b: Node(state, prior=.5)}
    agent = AlphaZeroMiniAgent(None)
    assert agent._select_child(root)[0] == b


@pytest.mark.parametrize("side", [Side.CHESS, Side.XIANGQI])
def test_backup_and_selection_use_opponent_view(side):
    state = HybridChessEnv().reset()
    state.side_to_move = side
    root = Node(state)
    a, b = Move(0, 1, 0, 2), Move(1, 1, 1, 2)
    losing_child = Node(state, prior=.5)
    winning_child = Node(state, prior=.5)
    root.children = {a: losing_child, b: winning_child}
    agent = AlphaZeroMiniAgent(None, MCTSConfig(discount_factor=1.))
    agent._backup([root, losing_child], 1.)
    agent._backup([root, winning_child], -1.)
    assert agent._select_child(root)[0] == b
    assert root.Q == 0.


def test_deadline_returns_legal_move():
    env = HybridChessEnv()
    state = env.reset()
    agent = AlphaZeroMiniAgent(TorchPolicyValueModel(new_model(8, 1)),
        MCTSConfig(simulations=100000, time_limit_seconds=0., dirichlet_eps=0.))
    assert agent.select_move(state, env.legal_moves()) in env.legal_moves()


@pytest.mark.parametrize("variant", [p["variant"] for p in PRESETS] +
    [{k: k != "flying_general"} for k in RULE_FIELDS])
def test_multivariant_python_cpp_agree(variant):
    pytest.importorskip("hybrid.cpp_engine.hybrid_cpp_engine")
    py = HybridChessEnv(variant=parse_variant(variant))
    cpp = HybridChessEnv(variant=parse_variant(variant), use_cpp=True)
    py.reset()
    cpp.reset()
    rng = np.random.default_rng(72)
    for _ in range(20):
        py_moves, cpp_moves = py.legal_moves(), cpp.legal_moves()
        assert set(py_moves) == set(cpp_moves)
        if not py_moves:
            break
        move = py_moves[int(rng.integers(len(py_moves)))]
        a, b = py.step(move), cpp.step(move)
        assert a[1:] == b[1:]
        assert a[0].repetition == b[0].repetition
        assert np.array_equal(context_features(a[0]), context_features(b[0]))
        if a[2]:
            break


def test_run_store_fallback_and_identity(tmp_path):
    store = RunStore(tmp_path, {"config": 1})
    store.save({"stage": "train", "global_step": 1})
    store.save({"stage": "train", "global_step": 2})
    index = json.loads((tmp_path / "checkpoints.json").read_text())
    (tmp_path / index[0]["file"]).write_bytes(b"broken")
    assert store.load()["global_step"] == 1
    assert (tmp_path / "recovery.json").exists()
    with pytest.raises(ValueError):
        RunStore(tmp_path, {"config": 2})
    (tmp_path / index[1]["file"]).write_bytes(b"broken too")
    with pytest.raises(RuntimeError):
        store.load()


def test_rng_restores_independent_generator():
    rng = np.random.default_rng(15)
    saved = rng_state(rng)
    expected = (rng.random(), torch.rand(4))
    restore_rng(saved, rng)
    assert rng.random() == expected[0]
    assert torch.equal(torch.rand(4), expected[1])


def test_mid_epoch_resume_matches_continuous_training(tmp_path):
    from hybrid.rl.general_train import Trainer, StopControl
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/general-ai.json").read_text())
    cfg.update(channels=8, res_blocks=1, batch_size=4)
    buffer = ReplayBuffer()
    state = HybridChessEnv().reset()
    for i in range(32):
        buffer.append([Example(encode_general(state).numpy(), np.array([0, 90], dtype=np.uint16),
            np.array([float(i % 2), float(1-i % 2)], dtype=np.float32), Side.CHESS, float(i % 3-1))])
    continuous = Trainer(cfg, tmp_path / "continuous", "same", "cpu", StopControl(60))
    assert continuous.train_epoch(buffer)
    class Interrupt:
        deadline = time.time()+60
        calls = 0
        def __call__(self):
            self.calls += 1
            return self.calls > 3
    interrupted = Trainer(cfg, tmp_path / "resumed", "same", "cpu", Interrupt())
    assert not interrupted.train_epoch(buffer)
    assert interrupted.state["batch_step"] == 3
    resumed = Trainer(cfg, tmp_path / "resumed", "same", "cpu", StopControl(60))
    assert resumed.train_epoch(buffer)
    assert resumed.state["global_step"] == continuous.state["global_step"]
    assert resumed.rng.bit_generator.state == continuous.rng.bit_generator.state
    for key, value in continuous.net.state_dict().items():
        assert torch.equal(value, resumed.net.state_dict()[key]), key
    a, b = continuous.optimizer.state_dict(), resumed.optimizer.state_dict()
    assert a["param_groups"] == b["param_groups"]
    for parameter, fields in a["state"].items():
        for key, value in fields.items():
            assert torch.equal(value, b["state"][parameter][key])


def test_weights_version_and_web_loading(tmp_path):
    import hybrid.server as server
    path = tmp_path / "model.pt"
    torch.save(model_payload(new_model(8, 1)), path)
    loaded = load_general_model(path)
    assert loaded.initial_conv.in_channels == 29
    try:
        server.configure_model(path)
        game = server.GameSession("chess", "az_fast", "pk")
        assert game.ai_agent.cfg.time_limit_seconds == 1
        assert game.env.state.variant.chess_palace
    finally:
        server.configure_model(None)
    torch.save({"model": loaded.state_dict()}, path)
    with pytest.raises(ValueError, match="version 2"):
        load_general_model(path)


def test_sampling_balances_presets_and_is_reproducible():
    families = [sample_variant(i, 1)[1] for i in range(30)]
    assert families.count("custom") == 6
    assert all(families.count(p["id"]) == 4 for p in PRESETS)
    assert sample_variant(25, 1) == sample_variant(25, 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_teacher_and_training_resume_smoke(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA compute allocation required")
    from hybrid.rl.general_train import generate_teacher, Trainer, StopControl
    mp.set_start_method("spawn", force=True)
    cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/general-ai.json").read_text())
    cfg.update(channels=8, res_blocks=1, max_plies=4, teacher_samples=120, teacher_seconds=.001,
        teacher_wall_seconds=90, supervised_epochs=1, batch_size=8, games_per_iteration=2,
        initial_simulations=2, simulations=2, workers=2)
    teacher = tmp_path / "teacher"
    generate_teacher(cfg, teacher, "test", StopControl(120))
    manifest = json.loads((teacher / "dataset.json").read_text())
    assert set(r["path"] for r in manifest["train"]).isdisjoint(r["path"] for r in manifest["validation"])
    root = tmp_path / "train"
    trainer = Trainer(cfg, root, "test", device, StopControl(120))
    trainer.run(teacher / "dataset.json", max_iterations=1)
    steps = trainer.state["global_step"]
    assert trainer.state["iteration"] == 1
    assert (root / "candidate.pt").exists()
    resumed = Trainer(cfg, root, "test", device, StopControl(120))
    resumed.run(teacher / "dataset.json", max_iterations=1)
    assert resumed.state["global_step"] == steps
    for key, value in trainer.net.state_dict().items():
        assert torch.equal(value, resumed.net.state_dict()[key])
