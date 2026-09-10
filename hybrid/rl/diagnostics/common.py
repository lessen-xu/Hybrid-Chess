"""Stable experiment identities, positions and explicit diagnostic agents."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from hybrid.agents.alphabeta_agent import AlphaBetaAgent, SearchConfig
from hybrid.agents.alphazero_stub import AlphaZeroMiniAgent, MCTSConfig
from hybrid.agents.eval import evaluate, material_score, mobility_score, EvalWeights
from hybrid.core.board import Board
from hybrid.core.env import HybridChessEnv
from hybrid.core.rules import board_hash, generate_legal_moves, is_in_check
from hybrid.core.types import Move, Piece, PieceKind, Side
from hybrid.web_variants import parse_variant


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def notation(move):
    text = f"{'abcdefghi'[move.fx]}{move.fy+1}-{'abcdefghi'[move.tx]}{move.ty+1}"
    if move.promotion:
        text += "=" + {PieceKind.QUEEN: "Q", PieceKind.ROOK: "R", PieceKind.BISHOP: "B", PieceKind.KNIGHT: "N"}[move.promotion]
    return text


def parse_move(text):
    body, _, promotion = text.partition("=")
    origin, target = body.split("-")
    kind = {"Q": PieceKind.QUEEN, "R": PieceKind.ROOK, "B": PieceKind.BISHOP, "N": PieceKind.KNIGHT}.get(promotion)
    return Move(ord(origin[0])-97, int(origin[1:])-1, ord(target[0])-97, int(target[1:])-1, kind)


def position(state, **metadata):
    return dict(pieces=[[x,y,p.kind.name,p.side.name] for x,y,p in state.board.iter_pieces()],
                side=state.side_to_move.name, ply=state.ply, repetition=dict(state.repetition),
                variant=state.variant.to_dict(), max_plies=state.max_plies, **metadata)


def restore(record, cpp=False):
    board = Board.empty()
    for x,y,kind,side in record["pieces"]:
        board.set(x,y,Piece(PieceKind[kind], Side[side]))
    env = HybridChessEnv(variant=parse_variant(record["variant"]), use_cpp=cpp,
                         max_plies=record.get("max_plies",400))
    state = env.reset_from_board(board, Side[record["side"]])
    state.ply = record["ply"]
    state.repetition = dict(record["repetition"])
    env.state = state
    return env, state


def fingerprint(state):
    return json_hash(position(state))


def basic_evaluate(state, side, weights=EvalWeights()):
    check = int(is_in_check(state.board,side.opponent()))-int(is_in_check(state.board,side))
    return material_score(state,side)+weights.mobility*mobility_score(state,side)+weights.check_bonus*check


def symmetric_evaluate(state, side, weights=EvalWeights()):
    return (evaluate(state,side,weights)-evaluate(state,side.opponent(),weights))/2


def no_stalemate_evaluate(state, side, weights=EvalWeights()):
    value = evaluate(state,side,weights)
    opponent = side.opponent()
    if (material_score(state,side)>5 and sum(p.side==opponent for _,_,p in state.board.iter_pieces())<=6
            and len(generate_legal_moves(state.board,opponent))<=1 and not is_in_check(state.board,opponent)):
        value += 8.
    return value


EVALUATORS = {"original": evaluate, "basic": basic_evaluate,
              "symmetric": symmetric_evaluate, "no_stalemate": no_stalemate_evaluate}


class InterventionModel:
    def __init__(self, model, mode="network"):
        self.model, self.mode = model, mode

    def predict(self,state,moves):
        policy,value = self.model.predict(state,moves)
        return self._change(policy,value,moves)

    def _change(self,policy,value,moves):
        if self.mode in ("uniform_policy","uniform_zero"):
            policy = {m:1/len(moves) for m in moves}
        if self.mode in ("zero_value","uniform_zero"):
            value = 0.
        return policy,value

    def predict_batch(self, inputs):
        results = (self.model.predict_batch(inputs) if hasattr(self.model,"predict_batch") else
                   [self.model.predict(state,moves) for state,moves in inputs])
        return [self._change(policy,value,moves) for (_,moves),(policy,value) in zip(inputs,results)]


def stage_path(args,stage):
    local = Path(args.output)/stage
    reference = getattr(args,"reference_run",None)
    return local if local.exists() or not reference else Path(reference)/stage


def neural_agent(model, *, simulations=128, seconds=None, cpp=False, batch=1, mode="network", seed=0):
    return AlphaZeroMiniAgent(InterventionModel(model,mode),MCTSConfig(simulations=simulations,
        time_limit_seconds=seconds,discount_factor=1.,dirichlet_eps=0.,leaf_batch_size=batch,max_plies=400),
        seed=seed,use_cpp=cpp)


def search_record(model,state,legal,*,simulations=128,seconds=None,cpp=False,batch=1,mode="network"):
    before = fingerprint(state)
    adapter = InterventionModel(model,mode)
    started = time.perf_counter()
    priors,value = adapter.predict(state,legal)
    if mode == "policy":
        chosen = max(legal,key=lambda m:priors[m])
        root = None
    else:
        agent = neural_agent(model,simulations=simulations,seconds=seconds,cpp=cpp,batch=batch,mode=mode)
        root = agent._run_mcts_search(state,legal,add_noise=False)
        chosen = max(root.children,key=lambda m:root.children[m].N)
    elapsed = time.perf_counter()-started
    assert fingerprint(state)==before and chosen in legal
    rows = [dict(move=notation(m),prior=priors[m],visits=root.children[m].N if root else 0,
                 parent_value=-root.children[m].Q if root else None) for m in legal]
    return chosen,dict(mode=mode,simulations=simulations,seconds=seconds,cpp=cpp,batch=batch,
        move=notation(chosen),elapsed_seconds=elapsed,legal_count=len(legal),network_value=value,
        root_value=root.Q if root else None,completed_simulations=root.N if root else 0,
        unvisited_fraction=sum(r["visits"]==0 for r in rows)/len(rows),
        policy_entropy=-sum(p*np.log(p) for p in priors.values() if p>0),actions=rows,state_preserved=True)


def paired_interval(values, *, difference=False):
    values = np.asarray(values,dtype=float)
    if not len(values):
        return None
    # Differences can span [-1,1]; this bound intentionally stays conservative.
    radius = (2 if difference else 1)*np.sqrt(np.log(40)/(2*len(values)))
    return [max(-1. if difference else 0.,float(values.mean()-radius)),min(1.,float(values.mean()+radius))]
