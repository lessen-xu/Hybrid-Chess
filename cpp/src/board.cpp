// board.cpp — Board implementation.

#include "board.h"
#include <cassert>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cstdio>

// ═══════════════════════════════════════════════════════════════
// Minimal SHA1 implementation (matches Python's hashlib.sha1).
// Only used for board_hash() — no external dependency needed.
// ═══════════════════════════════════════════════════════════════

namespace {

static inline uint32_t sha1_rol(uint32_t v, int bits) {
    return (v << bits) | (v >> (32 - bits));
}

struct SHA1Context {
    uint32_t h[5];
    uint64_t total_len;
    uint8_t  buf[64];
    size_t   buf_len;

    void init() {
        h[0] = 0x67452301u;
        h[1] = 0xEFCDAB89u;
        h[2] = 0x98BADCFEu;
        h[3] = 0x10325476u;
        h[4] = 0xC3D2E1F0u;
        total_len = 0;
        buf_len = 0;
    }

    void process_block(const uint8_t block[64]) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) {
            w[i] = (uint32_t(block[i*4])<<24) | (uint32_t(block[i*4+1])<<16) |
                    (uint32_t(block[i*4+2])<<8) | uint32_t(block[i*4+3]);
        }
        for (int i = 16; i < 80; i++) {
            w[i] = sha1_rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];

        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | ((~b) & d); k = 0x5A827999u; }
            else if (i < 40) { f = b ^ c ^ d;             k = 0x6ED9EBA1u; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
            else              { f = b ^ c ^ d;             k = 0xCA62C1D6u; }
            uint32_t temp = sha1_rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = sha1_rol(b, 30); b = a; a = temp;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }

    void update(const uint8_t* data, size_t len) {
        total_len += len;
        for (size_t i = 0; i < len; i++) {
            buf[buf_len++] = data[i];
            if (buf_len == 64) {
                process_block(buf);
                buf_len = 0;
            }
        }
    }

    void update_str(const std::string& s) {
        update(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

    std::string hexdigest() {
        // Padding
        uint64_t bits = total_len * 8;
        uint8_t one = 0x80;
        update(&one, 1);
        uint8_t zero = 0;
        while (buf_len != 56) {
            update(&zero, 1);
        }
        uint8_t len_be[8];
        for (int i = 7; i >= 0; i--) {
            len_be[i] = static_cast<uint8_t>(bits & 0xff);
            bits >>= 8;
        }
        update(len_be, 8);

        char hex[41];
        for (int i = 0; i < 5; i++) {
            std::snprintf(hex + i * 8, 9, "%08x", h[i]);
        }
        hex[40] = '\0';
        return std::string(hex);
    }
};

// Side name first char (matching Python's Side.name[0])
char side_char(Side s) {
    return s == Side::CHESS ? 'C' : 'X';
}

// PieceKind name first char (matching Python's PieceKind.name[0])
char kind_char(PieceKind k) {
    switch (k) {
        case PieceKind::KING:     return 'K';
        case PieceKind::QUEEN:    return 'Q';
        case PieceKind::ROOK:     return 'R';
        case PieceKind::BISHOP:   return 'B';
        case PieceKind::KNIGHT:   return 'K';  // KNIGHT.name[0] = 'K'
        case PieceKind::PAWN:     return 'P';
        case PieceKind::GENERAL:  return 'G';
        case PieceKind::ADVISOR:  return 'A';
        case PieceKind::ELEPHANT: return 'E';
        case PieceKind::HORSE:    return 'H';
        case PieceKind::CHARIOT:  return 'C';
        case PieceKind::CANNON:   return 'C';  // CANNON.name[0] = 'C'
        case PieceKind::SOLDIER:  return 'S';
        default:                  return '?';
    }
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════
// Board methods
// ═══════════════════════════════════════════════════════════════

Board Board::empty() {
    Board b;
    for (int y = 0; y < BOARD_H; y++)
        for (int x = 0; x < BOARD_W; x++)
            b.grid[y][x] = std::nullopt;
    return b;
}

Board Board::clone() const {
    Board b;
    for (int y = 0; y < BOARD_H; y++)
        for (int x = 0; x < BOARD_W; x++)
            b.grid[y][x] = grid[y][x];
    return b;
}

bool Board::in_bounds(int x, int y) const {
    return x >= 0 && x < BOARD_W && y >= 0 && y < BOARD_H;
}

std::optional<Piece> Board::get(int x, int y) const {
    if (!in_bounds(x, y)) return std::nullopt;
    return grid[y][x];
}

void Board::set(int x, int y, std::optional<Piece> piece) {
    assert(in_bounds(x, y));
    grid[y][x] = piece;
}

std::optional<Piece> Board::move_piece(int fx, int fy, int tx, int ty) {
    auto p = get(fx, fy);
    assert(p.has_value());
    auto captured = get(tx, ty);
    set(tx, ty, p);
    set(fx, fy, std::nullopt);
    return captured;
}

std::vector<std::tuple<int,int,Piece>> Board::iter_pieces() const {
    std::vector<std::tuple<int,int,Piece>> out;
    for (int y = 0; y < BOARD_H; y++)
        for (int x = 0; x < BOARD_W; x++)
            if (grid[y][x].has_value())
                out.emplace_back(x, y, grid[y][x].value());
    return out;
}

std::string Board::board_hash(Side side_to_move) const {
    // Must match Python's board_hash() exactly:
    //   row tokens joined by '|', then SHA1 hex.
    std::string joined;
    bool first = true;
    for (int y = 0; y < BOARD_H; y++) {
        for (int x = 0; x < BOARD_W; x++) {
            if (!first) joined += '|';
            first = false;
            auto& p = grid[y][x];
            if (!p.has_value()) {
                joined += '.';
            } else {
                joined += side_char(p->side);
                joined += kind_char(p->kind);
            }
        }
    }
    joined += '|';
    joined += 'T';
    joined += side_char(side_to_move);

    SHA1Context sha;
    sha.init();
    sha.update_str(joined);
    return sha.hexdigest();
}
