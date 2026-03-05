#pragma once
// ab_search.h — Alpha-Beta (Negamax) search engine.
// Single-call entry: best_move() runs the entire search in C++.

#include "board.h"
#include "types.h"
#include <cstdint>
#include <string>
#include <unordered_map>

// Result returned by best_move().
struct SearchResult {
    Move best_move;
    double score;
    int64_t nodes;   // total nodes visited
};

// Run a full alpha-beta search and return the best move.
// All tree expansion, evaluation, and pruning happens in C++.
SearchResult best_move(
    const Board& board,
    Side side_to_move,
    int depth,
    const std::unordered_map<std::string,int>& repetition_table,
    int ply,
    int max_plies
);
