# hybrid/cpp_engine/__init__.py
"""C++ game engine for hybrid chess, exposed via pybind11."""

from .hybrid_cpp_engine import (   # noqa: F401
    Side,
    PieceKind,
    Piece,
    Move,
    Board,
    GameInfo,
    SearchResult,
    RuleFlags,
    opponent,
    generate_pseudo_legal_moves,
    generate_legal_moves,
    apply_move,
    is_square_attacked,
    is_in_check,
    terminal_info,
    best_move,
    perft_nodes,
    set_rule_flags,
    BOARD_W,
    BOARD_H,
    MAX_PLIES,
)
