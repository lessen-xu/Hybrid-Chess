#pragma once
// types.h — Core data types for the hybrid chess engine.
// Mirrors hybrid/core/types.py exactly.

#include <optional>
#include <tuple>

// Board dimensions (9 columns x 10 rows, Xiangqi standard)
constexpr int BOARD_W = 9;
constexpr int BOARD_H = 10;
constexpr int MAX_PLIES = 400;

// ---------------------------------------------------------------------------
// Side
// ---------------------------------------------------------------------------
enum class Side { CHESS, XIANGQI };

inline Side opponent(Side s) {
    return s == Side::CHESS ? Side::XIANGQI : Side::CHESS;
}

// ---------------------------------------------------------------------------
// PieceKind  (same order as Python PieceKind enum)
// ---------------------------------------------------------------------------
enum class PieceKind {
    // Chess side
    KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN,
    // Xiangqi side
    GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER,
    // Sentinel for "no promotion"
    NONE
};

// ---------------------------------------------------------------------------
// Piece
// ---------------------------------------------------------------------------
struct Piece {
    PieceKind kind;
    Side side;

    bool operator==(const Piece& o) const {
        return kind == o.kind && side == o.side;
    }
    bool operator!=(const Piece& o) const { return !(*this == o); }
};

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------
struct Move {
    int fx, fy, tx, ty;
    PieceKind promotion = PieceKind::NONE;

    std::tuple<int,int> from_sq() const { return {fx, fy}; }
    std::tuple<int,int> to_sq()   const { return {tx, ty}; }

    bool operator==(const Move& o) const {
        return fx == o.fx && fy == o.fy &&
               tx == o.tx && ty == o.ty &&
               promotion == o.promotion;
    }
};

// ---------------------------------------------------------------------------
// RuleFlags — variant config passed from Python to C++ engine
// ---------------------------------------------------------------------------
struct RuleFlags {
    bool no_queen_promotion = false;  // Pawn promotes to R/B/N only (no Queen)
    bool no_promotion = false;        // Completely disable pawn promotion
    bool chess_palace = false;        // Confine Chess King to 3x3 palace (x=3-5, y=0-2)
    bool knight_block = false;        // Chess Knight uses XQ Horse leg-blocking rules
};

// Global rule flags — set once per game by Python wrapper.
// Thread-local to avoid issues with multiprocessing.
inline thread_local RuleFlags g_rule_flags;

inline void set_rule_flags(const RuleFlags& f) { g_rule_flags = f; }
