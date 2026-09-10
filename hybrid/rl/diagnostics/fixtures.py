"""Twenty-four explicit rule/tactical contracts, twelve for each army.

Rule-safety cases are reported separately from winning-move tests. They are not
counted as evidence of strategic strength merely because a move was legal.
"""
from hybrid.core.board import Board
from hybrid.core.env import HybridChessEnv
from hybrid.core.rules import apply_move, board_hash, is_in_check, terminal_info, TerminalStatus
from hybrid.core.types import Piece, PieceKind as K, Side, Move
from hybrid.web_variants import parse_variant
from .common import position, restore, notation


def tactical_positions():
    records = []
    C,X = Side.CHESS,Side.XIANGQI
    def add(name,side,entries,flags=None,contract="win",forbidden=None):
        board = Board.empty()
        for x,y,kind,army in entries:
            board.set(x,y,Piece(kind,army))
        env = HybridChessEnv(variant=parse_variant(flags or {}))
        state = env.reset_from_board(board,side)
        records.append(position(state,id=f"{side.name.lower()}-{name}",preset="tactic",
            contract=contract,forbidden=notation(forbidden) if forbidden else None))
    for side in (C,X):
        royals = [(0,0,K.KING,C),(4,9,K.GENERAL,X)]
        wins = ([(K.QUEEN,4,7),(K.ROOK,4,7),(K.BISHOP,2,7)] if side==C
                else [(K.XQ_QUEEN,0,2),(K.CHARIOT,0,2),(K.HORSE,1,2)])
        for kind,x,y in wins:
            add("capture-"+kind.name.lower(),side,royals+[(x,y,kind,side)])
        if side==C:
            add("escape-check",side,royals+[(0,4,K.CHARIOT,X)],contract="escape")
            knight = [(3,7,K.KNIGHT,C)]
            add("blocked-knight",side,royals+knight+[(3,8,K.PAWN,C)],
                {"knight_block":True},"legal",Move(3,7,4,9))
            add("free-knight",side,royals+knight,{"knight_block":True})
            add("cannon-screen",side,royals+[(0,3,K.CANNON,X),(0,2,K.SOLDIER,X)],contract="escape")
            add("palace",side,[(3,0,K.KING,C),(4,9,K.GENERAL,X)],
                {"chess_palace":True},"legal",Move(3,0,2,0))
        else:
            add("escape-check",side,royals+[(4,5,K.ROOK,C)],contract="escape")
            horse = [(1,2,K.HORSE,X)]
            add("blocked-knight",side,royals+horse+[(1,1,K.SOLDIER,X)],None,"legal",Move(1,2,0,0))
            add("free-knight",side,royals+horse)
            add("cannon-screen",side,royals+[(0,3,K.CANNON,X),(0,2,K.SOLDIER,X)])
            add("palace",side,[(0,0,K.KING,C),(3,9,K.GENERAL,X)],None,"legal",Move(3,9,2,9))
        for label,flags in [("all",{}),("no-queen",{"no_queen_promotion":True}),("none",{"no_promotion":True})]:
            entries = [(8,0,K.KING,C),(4,9,K.GENERAL,X),(0,8,K.PAWN,C)]
            if side==X:
                entries.append((1,8,K.CHARIOT,X))
            add("promotion-"+label,side,entries,flags,"promotion")
        env = HybridChessEnv()
        state = env.reset_from_board(env.reset().board,side)
        for move in env.legal_moves():
            state.repetition[board_hash(apply_move(state.board,move),side.opponent())] = 2
        records.append(position(state,id=side.name.lower()+"-repetition",preset="tactic",contract="draw",forbidden=None))
    assert len(records)==24
    return records


def verify_contract(record):
    env,state = restore(record)
    legal = env.legal_moves()
    assert legal and terminal_info(state.board,state.side_to_move,state.repetition,state.ply,400).status==TerminalStatus.ONGOING
    if record.get("forbidden"):
        assert record["forbidden"] not in {notation(m) for m in legal}
    if record["contract"]=="escape":
        assert is_in_check(state.board,state.side_to_move)
    if record["contract"]=="promotion":
        # Opponent-promotion cases independently verify the same rule contract.
        chess = HybridChessEnv(variant=state.variant)
        chess.reset_from_board(state.board,Side.CHESS)
        actual = {m.promotion for m in chess.legal_moves() if (m.fx,m.fy,m.tx,m.ty)==(0,8,0,9)}
        expected = ({None} if state.variant.no_promotion else
                    {K.ROOK,K.BISHOP,K.KNIGHT} if state.variant.no_queen_promotion else
                    {K.QUEEN,K.ROOK,K.BISHOP,K.KNIGHT})
        assert actual==expected
    env._set_active_variant()
    wins,draws = [],[]
    for move in legal:
        child_env,_ = restore(record)
        child,_,done,info = child_env.step(move)
        if done and info.winner==state.side_to_move:
            wins.append(notation(move))
        if done and info.winner is None:
            draws.append(notation(move))
        assert not is_in_check(child.board,state.side_to_move)
    if record["contract"]=="win":
        assert wins, record["id"]
    if record["contract"]=="draw":
        assert len(draws)==len(legal)
    return dict(winning_moves=wins,drawing_moves=draws,rule_contract_verified=True)
