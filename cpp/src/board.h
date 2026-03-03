#pragma once
// board.h — Board class: 9x10 grid stored as grid[y][x].
// Mirrors hybrid/core/board.py.

#include "types.h"
#include <optional>
#include <vector>
#include <string>
#include <tuple>

class Board {
public:
    std::optional<Piece> grid[BOARD_H][BOARD_W];  // grid[y][x]

    // Create an empty board (all cells nullopt).
    static Board empty();

    // Deep copy.
    Board clone() const;

    // Bounds check.
    bool in_bounds(int x, int y) const;

    // Get piece at (x,y). Returns nullopt if out of bounds or empty.
    std::optional<Piece> get(int x, int y) const;

    // Set piece at (x,y). Asserts in_bounds.
    void set(int x, int y, std::optional<Piece> piece);

    // Move piece from (fx,fy) to (tx,ty). Returns captured piece if any.
    std::optional<Piece> move_piece(int fx, int fy, int tx, int ty);

    // Iterate over all pieces: returns vector of (x, y, piece).
    std::vector<std::tuple<int,int,Piece>> iter_pieces() const;

    // Stable hash matching Python's board_hash(board, side_to_move).
    // Uses SHA1 of the same string representation.
    std::string board_hash(Side side_to_move) const;
};
