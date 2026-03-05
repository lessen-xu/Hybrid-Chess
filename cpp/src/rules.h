#pragma once
// rules.h — Move generation, check detection, terminal state.
// Mirrors hybrid/core/rules.py.

#include "board.h"
#include <vector>
#include <string>
#include <unordered_map>

// Generate all pseudo-legal moves (ignoring self-check).
std::vector<Move> generate_pseudo_legal_moves(const Board& board, Side side);

// Generate legal moves (filters out moves leaving own royal in check).
std::vector<Move> generate_legal_moves(const Board& board, Side side);

// Apply a move on a cloned board and return the new board.
Board apply_move(const Board& board, const Move& mv);

// ── In-place move execution / reversal (zero-clone) ──

struct UndoInfo {
    Piece moved;                    // piece that was on from-square
    std::optional<Piece> captured;  // piece that was on to-square (if any)
    bool did_promotion;             // was this a pawn promotion?
};

// Execute move in-place (mutates board). Fills undo for reversal.
void make_move(Board& board, const Move& mv, UndoInfo& undo);

// Reverse a make_move (restores board to pre-move state).
void unmake_move(Board& board, const Move& mv, const UndoInfo& undo);

// Like generate_legal_moves but takes a mutable board ref and
// does do/undo filtering in-place (zero clone).
// Board is guaranteed unchanged on return.
void generate_legal_moves_inplace(Board& board, Side side, std::vector<Move>& out);

// Full scratch version: caller provides pseudo_scratch buffer to avoid
// internal pseudo-legal vector allocation (zero hidden heap alloc).
void generate_legal_moves_inplace(Board& board, Side side,
                                   std::vector<Move>& out,
                                   std::vector<Move>& pseudo_scratch);

// In-place pseudo-legal move generation (appends to out, caller clears).
void generate_pseudo_legal_moves_inplace(const Board& board, Side side,
                                          std::vector<Move>& out);

// ── Attack detection ──
//
// SEMANTICS: "attacked" here means "reachable by a pseudo-legal move of
// by_side on the current board", NOT the textbook notion of "control"
// (which ignores occupancy).  Consequences:
//
//   • Most pieces: if (x,y) holds a friendly piece (same side as by_side),
//     the square is considered NOT attacked because no pseudo-legal move
//     can land on a same-side piece.
//
//   • EXCEPTION — Chess Pawn: pawns "attack" diagonals (x±1, y+1) even
//     when those squares hold friendly pieces.  This matches the slow
//     implementation's special-case (which does not go through piece_moves
//     for pawn attack detection).
//
//   • EXCEPTION — Xiangqi Cannon: can reach (x,y) either by non-capture
//     slide (target must be empty) or by screen-jump capture (target must
//     hold an enemy piece).  It cannot reach a same-side piece.
//
//   • EXCEPTION — Xiangqi Flying General: only applicable when (x,y)
//     holds the Chess King (one-directional rule: General→King only).
//
// The _fast variant uses reverse-ray/offset probes from (x,y) instead of
// scanning the whole board.  It is semantically equivalent to _slow; the
// equivalence is enforced by TestAttackEquivalence in test_ab_cpp.py.

bool is_square_attacked(const Board& board, int x, int y, Side by_side);
bool is_square_attacked_slow(const Board& board, int x, int y, Side by_side);
bool is_square_attacked_fast(const Board& board, int x, int y, Side by_side);

// Check if side's royal piece is in check.
bool is_in_check(const Board& board, Side side);

// Terminal status strings.
namespace TerminalStatus {
    constexpr const char* ONGOING     = "ongoing";
    constexpr const char* CHESS_WIN   = "chess_win";
    constexpr const char* XIANGQI_WIN = "xiangqi_win";
    constexpr const char* DRAW        = "draw";
}

// Game termination info.
struct GameInfo {
    std::string status;
    int winner;   // 0=none, 1=CHESS, 2=XIANGQI
    std::string reason;
};

// Determine if the game is over.
GameInfo terminal_info(const Board& board, Side side_to_move,
                       const std::unordered_map<std::string,int>& repetition_table,
                       int ply, int max_plies);
