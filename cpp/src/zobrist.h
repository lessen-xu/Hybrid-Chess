#pragma once
// zobrist.h — Incremental Zobrist 128-bit hashing for Board.
// Deterministic: uses splitmix64 with a fixed seed.
// Header-only: no .cpp needed.

#include "types.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <functional>
// ZKey128 — 128-bit Zobrist key

struct ZKey128 {
    uint64_t lo = 0;
    uint64_t hi = 0;

    bool operator==(const ZKey128& o) const { return lo == o.lo && hi == o.hi; }
    bool operator!=(const ZKey128& o) const { return !(*this == o); }

    ZKey128 operator^(const ZKey128& o) const { return {lo ^ o.lo, hi ^ o.hi}; }
    ZKey128& operator^=(const ZKey128& o) { lo ^= o.lo; hi ^= o.hi; return *this; }

    // 32-char hex string: hi first (16 hex), then lo (16 hex), lowercase
    std::string to_hex() const {
        char buf[33];
        std::snprintf(buf, sizeof(buf), "%016llx%016llx",
                      (unsigned long long)hi, (unsigned long long)lo);
        return std::string(buf, 32);
    }
};

// std::hash specialization for ZKey128 (for unordered_map)
namespace std {
    template<> struct hash<ZKey128> {
        size_t operator()(const ZKey128& k) const noexcept {
            // Combine hi and lo with a good mixing constant
            return static_cast<size_t>(k.lo ^ (k.hi * 0x9E3779B97F4A7C15ULL));
        }
    };
}
// PieceKind → index mapping (explicit, future-proof)

static constexpr int ZOBRIST_NUM_KINDS = 13;  // KING..SOLDIER (excl. NONE)

inline int piece_kind_index(PieceKind k) {
    switch (k) {
        case PieceKind::KING:     return 0;
        case PieceKind::QUEEN:    return 1;
        case PieceKind::ROOK:     return 2;
        case PieceKind::BISHOP:   return 3;
        case PieceKind::KNIGHT:   return 4;
        case PieceKind::PAWN:     return 5;
        case PieceKind::GENERAL:  return 6;
        case PieceKind::ADVISOR:  return 7;
        case PieceKind::ELEPHANT: return 8;
        case PieceKind::HORSE:    return 9;
        case PieceKind::CHARIOT:  return 10;
        case PieceKind::CANNON:   return 11;
        case PieceKind::SOLDIER:  return 12;
        default:                  return -1;  // NONE — should never happen
    }
}
// Deterministic random table (splitmix64, fixed seed)

namespace {

inline uint64_t splitmix64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct ZobristTable {
    // piece_key[side][kind_index][square] where square = y*BOARD_W + x
    ZKey128 piece_key[2][ZOBRIST_NUM_KINDS][BOARD_W * BOARD_H];
    ZKey128 side_to_move_key;  // XOR toggle for side-to-move

    ZobristTable() {
        uint64_t seed = 0xC0FFEE123456789ULL;
        // Generate piece keys
        for (int s = 0; s < 2; ++s)
            for (int k = 0; k < ZOBRIST_NUM_KINDS; ++k)
                for (int sq = 0; sq < BOARD_W * BOARD_H; ++sq) {
                    piece_key[s][k][sq].lo = splitmix64(seed);
                    piece_key[s][k][sq].hi = splitmix64(seed);
                }
        // Side-to-move key
        side_to_move_key.lo = splitmix64(seed);
        side_to_move_key.hi = splitmix64(seed);
    }

    int side_index(Side s) const {
        return s == Side::CHESS ? 0 : 1;
    }

    ZKey128 get_piece_key(Side side, PieceKind kind, int x, int y) const {
        int si = side_index(side);
        int ki = piece_kind_index(kind);
        int sq = y * BOARD_W + x;
        return piece_key[si][ki][sq];
    }
};

// Singleton — constructed once at program start, deterministic
inline const ZobristTable& zobrist_table() {
    static const ZobristTable instance;
    return instance;
}

}  // anonymous namespace
// Compute full Zobrist key from grid (for init / recompute)

#include <optional>

inline ZKey128 compute_zkey_from_grid(
        const std::optional<Piece> grid[BOARD_H][BOARD_W]) {
    const auto& zt = zobrist_table();
    ZKey128 key{0, 0};
    for (int y = 0; y < BOARD_H; ++y)
        for (int x = 0; x < BOARD_W; ++x) {
            const auto& cell = grid[y][x];
            if (cell.has_value())
                key ^= zt.get_piece_key(cell->side, cell->kind, x, y);
        }
    return key;
}
