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

// Check if square (x,y) is attacked by any piece of by_side.
bool is_square_attacked(const Board& board, int x, int y, Side by_side);

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
