// rules.cpp — Move generation, legality checks, check/checkmate/draw detection.
// Mirrors hybrid/core/rules.py exactly.
//
// KEY RULE: Flying general is ONE-DIRECTIONAL (General -> King only).
// is_square_attacked by CHESS does NOT include flying general.

#include "rules.h"
#include <cassert>
#include <algorithm>
#include <array>
#include <tuple>
#include <optional>

// ── Geometry helpers ──

static constexpr std::array<std::pair<int,int>, 4> ORTH_DIRS = {{{1,0},{-1,0},{0,1},{0,-1}}};
static constexpr std::array<std::pair<int,int>, 4> DIAG_DIRS = {{{1,1},{1,-1},{-1,1},{-1,-1}}};

static constexpr std::array<std::pair<int,int>, 8> KNIGHT_DELTAS = {{
    {1,2},{2,1},{-1,2},{-2,1},
    {1,-2},{2,-1},{-1,-2},{-2,-1}
}};

// ── Helper: find royal piece (direct grid scan, no vector alloc) ──

static std::optional<std::pair<int,int>> find_royal(const Board& board, Side side) {
    PieceKind target = (side == Side::CHESS) ? PieceKind::KING : PieceKind::GENERAL;
    for (int y = 0; y < BOARD_H; ++y) {
        for (int x = 0; x < BOARD_W; ++x) {
            auto& cell = board.grid[y][x];
            if (cell.has_value() && cell->side == side && cell->kind == target)
                return std::make_pair(x, y);
        }
    }
    return std::nullopt;
}

// ── Palace check ──

static bool palace_contains(Side side, int x, int y) {
    if (side != Side::XIANGQI) return true;   // Chess has no palace
    return (x >= 3 && x <= 5) && (y >= 7 && y <= 9);
}

// ── Sliding moves (Rook/Chariot/Bishop/Queen) ──

template<size_t N>
static void slide_moves(const Board& board, int x, int y, Side side,
                        const std::array<std::pair<int,int>, N>& dirs,
                        std::vector<Move>& out) {
    for (auto [dx, dy] : dirs) {
        int cx = x + dx, cy = y + dy;
        while (board.in_bounds(cx, cy)) {
            auto p = board.get(cx, cy);
            if (!p.has_value()) {
                out.push_back({x, y, cx, cy});
            } else {
                if (p->side != side)
                    out.push_back({x, y, cx, cy});
                break;
            }
            cx += dx;
            cy += dy;
        }
    }
}

// ── Chess Pawn promotion helper ──

static void maybe_promotions(int fx, int fy, int tx, int ty,
                             bool no_queen_promotion,
                             std::vector<Move>& out) {
    if (ty != 9) {
        out.push_back({fx, fy, tx, ty});
        return;
    }
    // Promotion at y=9
    if (!no_queen_promotion)
        out.push_back({fx, fy, tx, ty, PieceKind::QUEEN});
    out.push_back({fx, fy, tx, ty, PieceKind::ROOK});
    out.push_back({fx, fy, tx, ty, PieceKind::BISHOP});
    out.push_back({fx, fy, tx, ty, PieceKind::KNIGHT});
}

// ── Chess Pawn moves ──

static void chess_pawn_moves(const Board& board, int x, int y, Side side,
                             bool no_queen_promotion,
                             std::vector<Move>& out) {
    // Forward 1
    int nx = x, ny = y + 1;
    if (board.in_bounds(nx, ny) && !board.get(nx, ny).has_value()) {
        maybe_promotions(x, y, nx, ny, no_queen_promotion, out);
        // Double step from starting rank
        if (y == 1) {
            int nx2 = x, ny2 = y + 2;
            if (board.in_bounds(nx2, ny2) && !board.get(nx2, ny2).has_value()) {
                out.push_back({x, y, nx2, ny2});
            }
        }
    }
    // Diagonal capture
    for (int dx : {-1, 1}) {
        int cx = x + dx, cy = y + 1;
        if (!board.in_bounds(cx, cy)) continue;
        auto t = board.get(cx, cy);
        if (t.has_value() && t->side != side) {
            maybe_promotions(x, y, cx, cy, no_queen_promotion, out);
        }
    }
}

// ── Xiangqi Cannon moves ──

static void xiangqi_cannon_moves(const Board& board, int x, int y, Side side,
                                 std::vector<Move>& out) {
    // Non-capture: slide until blocked
    for (auto [dx, dy] : ORTH_DIRS) {
        int cx = x + dx, cy = y + dy;
        while (board.in_bounds(cx, cy) && !board.get(cx, cy).has_value()) {
            out.push_back({x, y, cx, cy});
            cx += dx;
            cy += dy;
        }
    }
    // Capture: find screen, then find enemy target behind it
    for (auto [dx, dy] : ORTH_DIRS) {
        int cx = x + dx, cy = y + dy;
        // Find screen
        while (board.in_bounds(cx, cy) && !board.get(cx, cy).has_value()) {
            cx += dx;
            cy += dy;
        }
        if (!board.in_bounds(cx, cy)) continue;
        // Skip screen, find target
        cx += dx;
        cy += dy;
        while (board.in_bounds(cx, cy) && !board.get(cx, cy).has_value()) {
            cx += dx;
            cy += dy;
        }
        if (!board.in_bounds(cx, cy)) continue;
        auto target = board.get(cx, cy);
        if (target.has_value() && target->side != side) {
            out.push_back({x, y, cx, cy});
        }
    }
}

// ── Xiangqi Horse moves (with leg block) ──

static void xiangqi_horse_moves(const Board& board, int x, int y, Side side,
                                std::vector<Move>& out) {
    // (leg_dx, leg_dy, dst_dx, dst_dy)
    static constexpr int candidates[][4] = {
        {1,0,2,1}, {1,0,2,-1},
        {-1,0,-2,1}, {-1,0,-2,-1},
        {0,1,1,2}, {0,1,-1,2},
        {0,-1,1,-2}, {0,-1,-1,-2},
    };
    for (auto& c : candidates) {
        int lx = x + c[0], ly = y + c[1];
        if (!board.in_bounds(lx, ly)) continue;
        if (board.get(lx, ly).has_value()) continue; // leg blocked
        int nx = x + c[2], ny = y + c[3];
        if (!board.in_bounds(nx, ny)) continue;
        auto t = board.get(nx, ny);
        if (!t.has_value() || t->side != side)
            out.push_back({x, y, nx, ny});
    }
}

// ── Xiangqi Elephant moves ──

static void xiangqi_elephant_moves(const Board& board, int x, int y, Side side,
                                   std::vector<Move>& out) {
    static constexpr int deltas[][2] = {{2,2},{2,-2},{-2,2},{-2,-2}};
    for (auto& d : deltas) {
        int nx = x + d[0], ny = y + d[1];
        if (!board.in_bounds(nx, ny)) continue;
        if (ny < 5) continue; // cannot cross river
        int ex = x + d[0]/2, ey = y + d[1]/2; // eye square
        if (board.get(ex, ey).has_value()) continue; // eye blocked
        auto t = board.get(nx, ny);
        if (!t.has_value() || t->side != side)
            out.push_back({x, y, nx, ny});
    }
}

// ── Xiangqi Advisor moves ──

static void xiangqi_advisor_moves(const Board& board, int x, int y, Side side,
                                  std::vector<Move>& out) {
    for (auto [dx, dy] : DIAG_DIRS) {
        int nx = x + dx, ny = y + dy;
        if (!board.in_bounds(nx, ny)) continue;
        if (!palace_contains(side, nx, ny)) continue;
        auto t = board.get(nx, ny);
        if (!t.has_value() || t->side != side)
            out.push_back({x, y, nx, ny});
    }
}

// ── Xiangqi General moves (including flying general) ──

static void xiangqi_general_moves(const Board& board, int x, int y, Side side,
                                  std::vector<Move>& out) {
    // Normal orthogonal moves within palace
    for (auto [dx, dy] : ORTH_DIRS) {
        int nx = x + dx, ny = y + dy;
        if (!board.in_bounds(nx, ny)) continue;
        if (!palace_contains(side, nx, ny)) continue;
        auto t = board.get(nx, ny);
        if (!t.has_value() || t->side != side)
            out.push_back({x, y, nx, ny});
    }

    // Flying general: if King is on the same file with no pieces between
    auto [kx, ky] = board.royal_xy(Side::CHESS);
    if (kx >= 0) {
        if (kx == x) {
            int step = (ky > y) ? 1 : -1;
            int cy = y + step;
            bool blocked = false;
            while (cy != ky) {
                if (board.get(x, cy).has_value()) {
                    blocked = true;
                    break;
                }
                cy += step;
            }
            if (!blocked) {
                out.push_back({x, y, kx, ky});
            }
        }
    }
}

// ── Xiangqi Soldier moves ──

static void xiangqi_soldier_moves(const Board& board, int x, int y, Side side,
                                  std::vector<Move>& out) {
    // Forward for Xiangqi is y-1 (towards Chess side)
    // After crossing river (y <= 4), can also move sideways
    static constexpr int deltas[][2] = {{0,-1},{1,0},{-1,0}};
    int n = (y <= 4) ? 3 : 1;
    for (int i = 0; i < n; ++i) {
        int nx = x + deltas[i][0], ny = y + deltas[i][1];
        if (!board.in_bounds(nx, ny)) continue;
        auto t = board.get(nx, ny);
        if (!t.has_value() || t->side != side)
            out.push_back({x, y, nx, ny});
    }
}

// ── Piece move dispatch ──

static void piece_moves(const Board& board, int x, int y, const Piece& p,
                        bool no_queen_promotion,
                        std::vector<Move>& out) {
    auto k = p.kind;
    auto s = p.side;

    // Chess pieces
    if (k == PieceKind::ROOK)   { slide_moves(board, x, y, s, ORTH_DIRS, out); return; }
    if (k == PieceKind::BISHOP) { slide_moves(board, x, y, s, DIAG_DIRS, out); return; }
    if (k == PieceKind::QUEEN) {
        slide_moves(board, x, y, s, ORTH_DIRS, out);
        slide_moves(board, x, y, s, DIAG_DIRS, out);
        return;
    }
    if (k == PieceKind::KNIGHT) {
        for (auto [dx, dy] : KNIGHT_DELTAS) {
            int nx = x + dx, ny = y + dy;
            if (!board.in_bounds(nx, ny)) continue;
            auto t = board.get(nx, ny);
            if (!t.has_value() || t->side != s)
                out.push_back({x, y, nx, ny});
        }
        return;
    }
    if (k == PieceKind::KING) {
        // All 8 directions, 1 step
        for (auto [dx, dy] : ORTH_DIRS) {
            int nx = x + dx, ny = y + dy;
            if (!board.in_bounds(nx, ny)) continue;
            auto t = board.get(nx, ny);
            if (!t.has_value() || t->side != s)
                out.push_back({x, y, nx, ny});
        }
        for (auto [dx, dy] : DIAG_DIRS) {
            int nx = x + dx, ny = y + dy;
            if (!board.in_bounds(nx, ny)) continue;
            auto t = board.get(nx, ny);
            if (!t.has_value() || t->side != s)
                out.push_back({x, y, nx, ny});
        }
        return;
    }
    if (k == PieceKind::PAWN) {
        chess_pawn_moves(board, x, y, s, no_queen_promotion, out);
        return;
    }

    // Xiangqi pieces
    if (k == PieceKind::CHARIOT) { slide_moves(board, x, y, s, ORTH_DIRS, out); return; }
    if (k == PieceKind::CANNON)  { xiangqi_cannon_moves(board, x, y, s, out); return; }
    if (k == PieceKind::HORSE)   { xiangqi_horse_moves(board, x, y, s, out); return; }
    if (k == PieceKind::ELEPHANT){ xiangqi_elephant_moves(board, x, y, s, out); return; }
    if (k == PieceKind::ADVISOR) { xiangqi_advisor_moves(board, x, y, s, out); return; }
    if (k == PieceKind::GENERAL) { xiangqi_general_moves(board, x, y, s, out); return; }
    if (k == PieceKind::SOLDIER) { xiangqi_soldier_moves(board, x, y, s, out); return; }
}

// ── Pseudo-legal move generation ──

void generate_pseudo_legal_moves_inplace(const Board& board, Side side,
                                         std::vector<Move>& out) {
    // out is assumed already cleared by the caller.
    for (int y = 0; y < BOARD_H; ++y) {
        for (int x = 0; x < BOARD_W; ++x) {
            auto& cell = board.grid[y][x];
            if (!cell.has_value()) continue;
            if (cell->side != side) continue;
            piece_moves(board, x, y, *cell, false, out);
        }
    }
}

std::vector<Move> generate_pseudo_legal_moves(const Board& board, Side side) {
    std::vector<Move> moves;
    moves.reserve(128);
    generate_pseudo_legal_moves_inplace(board, side, moves);
    return moves;
}

// ═══════════════════════════════════════════════════════════════
// In-place move execution / reversal
// ═══════════════════════════════════════════════════════════════

void make_move(Board& board, const Move& mv, UndoInfo& undo) {
    auto from_piece = board.get(mv.fx, mv.fy);
    assert(from_piece.has_value());
    undo.moved = *from_piece;
    undo.captured = board.get(mv.tx, mv.ty);

    board.move_piece(mv.fx, mv.fy, mv.tx, mv.ty);

    // Handle Chess pawn promotion
    if (from_piece->kind == PieceKind::PAWN && from_piece->side == Side::CHESS
        && mv.promotion != PieceKind::NONE) {
        board.set(mv.tx, mv.ty, Piece{mv.promotion, Side::CHESS});
        undo.did_promotion = true;
    } else {
        undo.did_promotion = false;
    }
}

void unmake_move(Board& board, const Move& mv, const UndoInfo& undo) {
    // Restore source square: if promotion happened, put back the pawn
    if (undo.did_promotion) {
        board.set(mv.fx, mv.fy, Piece{PieceKind::PAWN, Side::CHESS});
    } else {
        board.set(mv.fx, mv.fy, undo.moved);
    }
    // Restore target square: captured piece or empty
    board.set(mv.tx, mv.ty, undo.captured);

#ifndef NDEBUG
    // Debug-only: verify royal cache consistency after undo.
    // Catches future grid mutations that bypass set/move_piece.
    for (int si = 0; si < 2; ++si) {
        Side s = (si == 0) ? Side::CHESS : Side::XIANGQI;
        assert(board.royal_square(s) == board.royal_square_recompute(s));
    }
#endif
}

// ── Apply move (clone-based, for external/pybind API) ──

Board apply_move(const Board& board, const Move& mv) {
    Board nb = board.clone();
    UndoInfo u;
    make_move(nb, mv, u);
    return nb;
}

// ── Attack detection (SLOW — preserved for equivalence testing) ──

bool is_square_attacked_slow(const Board& board, int x, int y, Side by_side) {
    for (int py = 0; py < BOARD_H; ++py) {
        for (int px = 0; px < BOARD_W; ++px) {
            auto& cell = board.grid[py][px];
            if (!cell.has_value()) continue;
            if (cell->side != by_side) continue;

            // Chess pawn attacks diagonally (not same as its movement)
            if (cell->kind == PieceKind::PAWN && cell->side == Side::CHESS) {
                for (int dx : {-1, 1}) {
                    int ax = px + dx, ay = py + 1;
                    if (ax == x && ay == y && board.in_bounds(ax, ay))
                        return true;
                }
                continue;
            }

            // Other pieces: any pseudo-legal move landing on (x,y) counts
            std::vector<Move> pmoves;
            piece_moves(board, px, py, *cell, false, pmoves);
            for (auto& mv : pmoves) {
                if (mv.tx == x && mv.ty == y)
                    return true;
            }
        }
    }
    return false;
}

// ── Attack detection (FAST — reverse ray/offset from target square) ──
//
// Semantics: pseudo-legal reachability (not textbook "control").
// See rules.h for full docstring and the three exception cases
// (Chess Pawn, Xiangqi Cannon, Flying General).

static inline bool ib(int x, int y) {
    return x >= 0 && x < BOARD_W && y >= 0 && y < BOARD_H;
}

static inline bool has_piece(const Board& b, int x, int y, Side s, PieceKind k) {
    auto& c = b.grid[y][x];
    return c.has_value() && c->side == s && c->kind == k;
}

bool is_square_attacked_fast(const Board& board, int x, int y, Side by_side) {

    auto& target_cell = board.grid[y][x];
    bool target_friendly = target_cell.has_value() && target_cell->side == by_side;

    // ── CHESS pieces ──
    if (by_side == Side::CHESS) {
        // Pawn: chess pawn attacks diagonally forward (y+1) REGARDLESS of target
        // (standard chess convention: pawns "control" squares diagonally)
        for (int dx : {-1, 1}) {
            int px = x + dx, py = y - 1;
            if (ib(px, py) && has_piece(board, px, py, Side::CHESS, PieceKind::PAWN))
                return true;
        }

        // For all other CHESS pieces, if target has a friendly piece, no move can land there
        if (target_friendly)
            return false;

        // King: 8 neighbors
        for (auto [dx, dy] : ORTH_DIRS) {
            int nx = x + dx, ny = y + dy;
            if (ib(nx, ny) && has_piece(board, nx, ny, Side::CHESS, PieceKind::KING))
                return true;
        }
        for (auto [dx, dy] : DIAG_DIRS) {
            int nx = x + dx, ny = y + dy;
            if (ib(nx, ny) && has_piece(board, nx, ny, Side::CHESS, PieceKind::KING))
                return true;
        }

        // Knight: 8 L-offsets (no leg block for chess knight)
        for (auto [dx, dy] : KNIGHT_DELTAS) {
            int nx = x + dx, ny = y + dy;
            if (ib(nx, ny) && has_piece(board, nx, ny, Side::CHESS, PieceKind::KNIGHT))
                return true;
        }

        // Rook / Queen: orthogonal rays
        for (auto [dx, dy] : ORTH_DIRS) {
            int cx = x + dx, cy = y + dy;
            while (ib(cx, cy)) {
                auto& c = board.grid[cy][cx];
                if (c.has_value()) {
                    if (c->side == Side::CHESS &&
                        (c->kind == PieceKind::ROOK || c->kind == PieceKind::QUEEN))
                        return true;
                    break;
                }
                cx += dx; cy += dy;
            }
        }

        // Bishop / Queen: diagonal rays
        for (auto [dx, dy] : DIAG_DIRS) {
            int cx = x + dx, cy = y + dy;
            while (ib(cx, cy)) {
                auto& c = board.grid[cy][cx];
                if (c.has_value()) {
                    if (c->side == Side::CHESS &&
                        (c->kind == PieceKind::BISHOP || c->kind == PieceKind::QUEEN))
                        return true;
                    break;
                }
                cx += dx; cy += dy;
            }
        }

        return false;
    }

    // ── XIANGQI pieces ──
    // (by_side == Side::XIANGQI)

    // If target square has a friendly XIANGQI piece, no move can land there
    if (target_friendly)
        return false;

    // General: orthogonal 1-step within palace
    // Original checks: destination must be in palace. Here (x,y) is destination.
    if (palace_contains(Side::XIANGQI, x, y)) {
        for (auto [dx, dy] : ORTH_DIRS) {
            int nx = x + dx, ny = y + dy;
            if (ib(nx, ny) &&
                has_piece(board, nx, ny, Side::XIANGQI, PieceKind::GENERAL))
                return true;
        }
    }

    // Flying general: General can capture King on same column with no pieces between.
    // This is one-directional: General -> King. So if by_side is XIANGQI and target
    // is on same column as the General, check for clear path.
    {
        // Only applies if target (x,y) is the Chess King (one-directional rule)
        if (target_cell.has_value() && target_cell->kind == PieceKind::KING &&
            target_cell->side == Side::CHESS) {
            int gsq = board.royal_square(Side::XIANGQI);
            if (gsq >= 0) {
                int gx = sq_x(gsq), gy = sq_y(gsq);
                if (gx == x && gy != y) {
                    int step = (y > gy) ? 1 : -1;
                    int cy = gy + step;
                    bool blocked = false;
                    while (cy != y) {
                        if (board.grid[cy][x].has_value()) { blocked = true; break; }
                        cy += step;
                    }
                    if (!blocked) return true;
                }
            }
        }
    }

    // Advisor: diagonal 1-step within palace
    // Original checks: destination must be in palace. Here (x,y) is destination.
    if (palace_contains(Side::XIANGQI, x, y)) {
        for (auto [dx, dy] : DIAG_DIRS) {
            int nx = x + dx, ny = y + dy;
            if (ib(nx, ny) &&
                has_piece(board, nx, ny, Side::XIANGQI, PieceKind::ADVISOR))
                return true;
        }
    }

    // Elephant: diagonal 2-step, eye must be empty
    // Both destination (x,y) AND elephant position (nx,ny) must be y>=5 (own half)
    if (y >= 5) {
        static constexpr int deltas[][2] = {{2,2},{2,-2},{-2,2},{-2,-2}};
        for (auto& d : deltas) {
            int nx = x + d[0], ny = y + d[1];
            if (!ib(nx, ny)) continue;
            if (ny < 5) continue; // elephant must also stay on own half
            int ex = x + d[0]/2, ey = y + d[1]/2;
            if (board.grid[ey][ex].has_value()) continue; // eye blocked
            if (has_piece(board, nx, ny, Side::XIANGQI, PieceKind::ELEPHANT))
                return true;
        }
    }

    // Horse: reverse-L with leg block
    // A horse at (hx,hy) attacks (x,y) if:
    //   (hx,hy) = (x-dx, y-dy) for some horse delta (dx,dy)
    //   and the leg at (hx + leg_dx, hy + leg_dy) is empty
    // We need to reverse the horse movement table.
    // Original: horse at (hx,hy), leg at (hx+c[0], hy+c[1]), dest at (hx+c[2], hy+c[3])
    // So: hx = x - c[2], hy = y - c[3], leg at (hx+c[0], hy+c[1])
    {
        static constexpr int candidates[][4] = {
            {1,0,2,1}, {1,0,2,-1},
            {-1,0,-2,1}, {-1,0,-2,-1},
            {0,1,1,2}, {0,1,-1,2},
            {0,-1,1,-2}, {0,-1,-1,-2},
        };
        for (auto& c : candidates) {
            int hx = x - c[2], hy = y - c[3];
            if (!ib(hx, hy)) continue;
            if (!has_piece(board, hx, hy, Side::XIANGQI, PieceKind::HORSE)) continue;
            int lx = hx + c[0], ly = hy + c[1];
            if (!ib(lx, ly)) continue;
            if (board.grid[ly][lx].has_value()) continue; // leg blocked
            return true;
        }
    }

    // Chariot + Cannon slides: orthogonal rays
    // Chariot: first piece on ray = attack (same as Rook)
    // Cannon non-capture: can slide to empty (x,y) if no piece blocks the path
    //   (the target_cell check at top already returned false for friendly pieces,
    //    so if we reach here, target is either empty or has enemy)
    bool target_empty = !target_cell.has_value();
    for (auto [dx, dy] : ORTH_DIRS) {
        int cx = x + dx, cy = y + dy;
        while (ib(cx, cy)) {
            auto& c = board.grid[cy][cx];
            if (c.has_value()) {
                if (c->side == Side::XIANGQI) {
                    if (c->kind == PieceKind::CHARIOT) return true;
                    if (c->kind == PieceKind::CANNON && target_empty) return true;
                }
                break;
            }
            cx += dx; cy += dy;
        }
    }

    // Cannon: orthogonal rays with exactly one screen between
    // Cannon jump is capture-only, so target (x,y) must have an enemy piece.
    // Non-capture slides are already handled above in the Chariot+Cannon section.
    if (!target_empty) {
        for (auto [dx, dy] : ORTH_DIRS) {
            int cx = x + dx, cy = y + dy;
            // Find first piece (screen)
            while (ib(cx, cy) && !board.grid[cy][cx].has_value()) {
                cx += dx; cy += dy;
            }
            if (!ib(cx, cy)) continue;
            // Skip screen, find next piece
            cx += dx; cy += dy;
            while (ib(cx, cy) && !board.grid[cy][cx].has_value()) {
                cx += dx; cy += dy;
            }
            if (!ib(cx, cy)) continue;
            if (board.grid[cy][cx].has_value() &&
                board.grid[cy][cx]->side == Side::XIANGQI &&
                board.grid[cy][cx]->kind == PieceKind::CANNON)
                return true;
        }
    }

    // Soldier: can attack (x,y) from positions that move to (x,y)
    // Soldier moves: always forward (y-1), and sideways (±1,0) if past river (y<=4)
    // So a soldier at (sx,sy) attacks (x,y) if:
    //   (sx, sy) = (x, y+1) [forward]  — always valid
    //   (sx, sy) = (x±1, y) [sideways]  — only if sy <= 4 (soldier past river)
    {
        // Forward: soldier at (x, y+1)
        if (ib(x, y + 1) && has_piece(board, x, y + 1, Side::XIANGQI, PieceKind::SOLDIER))
            return true;
        // Sideways: soldier at (x-1, y) or (x+1, y), only if that soldier is past river
        for (int dx : {-1, 1}) {
            int sx = x + dx;
            if (ib(sx, y) && y <= 4 &&
                has_piece(board, sx, y, Side::XIANGQI, PieceKind::SOLDIER))
                return true;
        }
    }

    return false;
}

// ── Public API: uses fast path ──

bool is_square_attacked(const Board& board, int x, int y, Side by_side) {
    return is_square_attacked_fast(board, x, y, by_side);
}

// ── Check detection ──

bool is_in_check(const Board& board, Side side) {
    int sq = board.royal_square(side);
    if (sq < 0) return true;  // royal captured
    int rx = sq_x(sq), ry = sq_y(sq);
    return is_square_attacked(board, rx, ry, opponent(side));
}

// ── Legal move generation ──

// Full scratch version: zero hidden heap alloc when buffers are pre-allocated.
void generate_legal_moves_inplace(Board& board, Side side,
                                   std::vector<Move>& out,
                                   std::vector<Move>& pseudo_scratch) {
    pseudo_scratch.clear();
    generate_pseudo_legal_moves_inplace(board, side, pseudo_scratch);
    out.clear();
    out.reserve(pseudo_scratch.size());
    for (auto& mv : pseudo_scratch) {
        UndoInfo u;
        make_move(board, mv, u);
        if (!is_in_check(board, side))
            out.push_back(mv);
        unmake_move(board, mv, u);
    }
}

// Convenience: creates local pseudo_scratch (one alloc per call).
void generate_legal_moves_inplace(Board& board, Side side,
                                   std::vector<Move>& out) {
    std::vector<Move> pseudo_scratch;
    pseudo_scratch.reserve(128);
    generate_legal_moves_inplace(board, side, out, pseudo_scratch);
}

// Const-board version: clones board, returns vector.
std::vector<Move> generate_legal_moves(const Board& board, Side side) {
    Board tmp = board.clone();
    std::vector<Move> out;
    std::vector<Move> pseudo;
    pseudo.reserve(128);
    generate_legal_moves_inplace(tmp, side, out, pseudo);
    return out;
}

// ── Terminal state detection ──

GameInfo terminal_info(const Board& board, Side side_to_move,
                       const std::unordered_map<std::string,int>& repetition_table,
                       int ply, int max_plies) {
    // 1) Royal existence
    bool chess_has = board.has_royal(Side::CHESS);
    bool xiangqi_has = board.has_royal(Side::XIANGQI);
    if (!chess_has)
        return {TerminalStatus::XIANGQI_WIN, 2, "Chess king captured"};
    if (!xiangqi_has)
        return {TerminalStatus::CHESS_WIN, 1, "Xiangqi general captured"};

    // 2) Move limit
    if (ply >= max_plies)
        return {TerminalStatus::DRAW, 0, "Max plies reached"};

    // 3) Threefold repetition
    std::string key = board.board_hash(side_to_move);
    auto it = repetition_table.find(key);
    if (it != repetition_table.end() && it->second >= 3)
        return {TerminalStatus::DRAW, 0, "Threefold repetition"};

    // 4) Legal moves
    auto legal = generate_legal_moves(board, side_to_move);
    if (!legal.empty())
        return {TerminalStatus::ONGOING, 0, ""};

    // No legal moves
    if (is_in_check(board, side_to_move)) {
        Side winner = opponent(side_to_move);
        std::string status = (winner == Side::CHESS)
            ? TerminalStatus::CHESS_WIN : TerminalStatus::XIANGQI_WIN;
        int w = (winner == Side::CHESS) ? 1 : 2;
        return {status, w, "Checkmate"};
    } else {
        return {TerminalStatus::DRAW, 0, "Stalemate (draw by rule)"};
    }
}
