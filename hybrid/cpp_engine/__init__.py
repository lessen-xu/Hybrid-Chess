# hybrid/cpp_engine/__init__.py
"""Python facade for the compiled C++ engine.

The actual extension is ``hybrid.cpp_engine.hybrid_cpp_engine`` (built from
``cpp/src/`` via pybind11). We re-export only the symbols that Python code
actually imports through this package; the rest of the binding surface is
reachable by importing the raw extension directly (the tests do this for
the perft helper and the slow attack-detection variant).
"""

import sys
import os

if sys.platform == "win32":
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if os.path.isdir(p) and (os.path.exists(os.path.join(p, "g++.exe")) or os.path.exists(os.path.join(p, "libwinpthread-1.dll"))):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass

from .hybrid_cpp_engine import (   # noqa: F401
    Side,
    PieceKind,
    Piece,
    Move,
    Board,
    GameInfo,
    RuleFlags,
    generate_legal_moves,
    apply_move,
    is_square_attacked,
    is_in_check,
    terminal_info,
    best_move,
    set_rule_flags,
    BOARD_W,
    BOARD_H,
    MAX_PLIES,
)
