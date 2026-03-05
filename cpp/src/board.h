#pragma once
// board.h — Board class: 9x10 grid stored as grid[y][x].
// Mirrors hybrid/core/board.py.

#include "types.h"
#include "zobrist.h"
#include <optional>
#include <vector>
#include <string>
#include <tuple>
#include <utility>

// Side index for royal cache: CHESS=0, XIANGQI=1
inline int side_index(Side s) { return s == Side::CHESS ? 0 : 1; }

// Encode/decode grid square index (0..89), -1 = absent
inline int sq_encode(int x, int y) { return y * BOARD_W + x; }
inline int sq_x(int sq) { return sq % BOARD_W; }
inline int sq_y(int sq) { return sq / BOARD_W; }

// Is this piece kind a royal?
inline bool is_royal_kind(Side side, PieceKind kind) {
    return (side == Side::CHESS && kind == PieceKind::KING) ||
           (side == Side::XIANGQI && kind == PieceKind::GENERAL);
}

class Board {
public:
    std::optional<Piece> grid[BOARD_H][BOARD_W];  // grid[y][x]
    ZKey128 zkey{0, 0};  // Incremental Zobrist hash (piece configuration only, no side-to-move)
    int16_t royal_sq[2] = {-1, -1};  // CHESS King sq, XIANGQI General sq (-1 = absent)

    // Create an empty board (all cells nullopt).
    static Board empty();

    // Deep copy.
    Board clone() const;

    // Bounds check.
    bool in_bounds(int x, int y) const;

    // Get piece at (x,y). Returns nullopt if out of bounds or empty.
    std::optional<Piece> get(int x, int y) const;

    // Set piece at (x,y). Asserts in_bounds. Maintains zkey and royal cache.
    void set(int x, int y, std::optional<Piece> piece);

    // Move piece from (fx,fy) to (tx,ty). Returns captured piece if any.
    // Maintains zkey and royal cache.
    std::optional<Piece> move_piece(int fx, int fy, int tx, int ty);

    // Iterate over all pieces: returns vector of (x, y, piece).
    std::vector<std::tuple<int,int,Piece>> iter_pieces() const;

    // Stable hash matching Python's board_hash(board, side_to_move).
    // Uses SHA1 of the same string representation.
    std::string board_hash(Side side_to_move) const;

    // ── Royal cache API ──

    // O(1) cached: royal square index (0..89), or -1 if absent.
    int royal_square(Side s) const { return royal_sq[side_index(s)]; }

    // O(1) cached: does this side still have its royal piece?
    bool has_royal(Side s) const { return royal_sq[side_index(s)] >= 0; }

    // O(1) cached: royal (x,y). Returns (-1,-1) if absent.
    std::pair<int,int> royal_xy(Side s) const {
        int sq = royal_sq[side_index(s)];
        return sq >= 0 ? std::make_pair(sq_x(sq), sq_y(sq)) : std::make_pair(-1, -1);
    }

    // Full grid scan to rebuild royal_sq. Call after bulk grid writes.
    void recompute_royal_cache();

    // Full grid scan — returns square index without modifying cache (for testing).
    int royal_square_recompute(Side s) const;

    // ── Zobrist 128-bit API ──

    // Raw Zobrist key (piece configuration only, no side-to-move).
    ZKey128 zobrist_key_raw() const;

    // Zobrist key XOR'd with side-to-move toggle.
    ZKey128 zobrist_key(Side stm) const;

    // 32-char hex string of zobrist_key(stm).
    std::string zobrist_key_hex(Side stm) const;

    // Recompute Zobrist key from scratch (full grid scan) — for testing.
    std::string zobrist_key_hex_recompute(Side stm) const;

    // Recompute zkey from grid (call after bulk grid setup).
    void recompute_zkey();
};
