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

// ── Helper: find royal piece ──

static std::optional<std::pair<int,int>> find_royal(const Board& board, Side side) {
    PieceKind target = (side == Side::CHESS) ? PieceKind::KING : PieceKind::GENERAL;
    for (auto& [x, y, p] : board.iter_pieces()) {
        if (p.side == side && p.kind == target)
            return std::make_pair(x, y);
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
    auto king_pos = find_royal(board, Side::CHESS);
    if (king_pos.has_value()) {
        auto [kx, ky] = *king_pos;
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
    std::vector<std::pair<int,int>> candidates = {{0, -1}};
    // After crossing river (y <= 4), can also move sideways
    if (y <= 4) {
        candidates.push_back({1, 0});
        candidates.push_back({-1, 0});
    }
    for (auto [dx, dy] : candidates) {
        int nx = x + dx, ny = y + dy;
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

std::vector<Move> generate_pseudo_legal_moves(const Board& board, Side side) {
    // Note: no_queen_promotion defaults to false (matching Python default)
    std::vector<Move> moves;
    for (auto& [x, y, p] : board.iter_pieces()) {
        if (p.side != side) continue;
        piece_moves(board, x, y, p, false, moves);
    }
    return moves;
}

// ── Apply move ──

Board apply_move(const Board& board, const Move& mv) {
    Board nb = board.clone();
    auto piece = nb.get(mv.fx, mv.fy);
    assert(piece.has_value());
    nb.move_piece(mv.fx, mv.fy, mv.tx, mv.ty);
    // Handle pawn promotion
    if (piece->kind == PieceKind::PAWN && piece->side == Side::CHESS
        && mv.promotion != PieceKind::NONE) {
        nb.set(mv.tx, mv.ty, Piece{mv.promotion, Side::CHESS});
    }
    return nb;
}

// ── Attack detection ──

bool is_square_attacked(const Board& board, int x, int y, Side by_side) {
    for (auto& [px, py, p] : board.iter_pieces()) {
        if (p.side != by_side) continue;

        // Chess pawn attacks diagonally (not same as its movement)
        if (p.kind == PieceKind::PAWN && p.side == Side::CHESS) {
            for (int dx : {-1, 1}) {
                int ax = px + dx, ay = py + 1;
                if (ax == x && ay == y && board.in_bounds(ax, ay))
                    return true;
            }
            continue;
        }

        // Other pieces: any pseudo-legal move landing on (x,y) counts
        std::vector<Move> pmoves;
        piece_moves(board, px, py, p, false, pmoves);
        for (auto& mv : pmoves) {
            if (mv.tx == x && mv.ty == y)
                return true;
        }
    }
    return false;
}

// ── Check detection ──

bool is_in_check(const Board& board, Side side) {
    auto royal = find_royal(board, side);
    if (!royal.has_value()) return true;  // royal captured
    auto [rx, ry] = *royal;
    return is_square_attacked(board, rx, ry, opponent(side));
}

// ── Legal move generation ──

std::vector<Move> generate_legal_moves(const Board& board, Side side) {
    std::vector<Move> out;
    auto pseudo = generate_pseudo_legal_moves(board, side);
    for (auto& mv : pseudo) {
        Board nb = apply_move(board, mv);
        if (!is_in_check(nb, side))
            out.push_back(mv);
    }
    return out;
}

// ── Terminal state detection ──

GameInfo terminal_info(const Board& board, Side side_to_move,
                       const std::unordered_map<std::string,int>& repetition_table,
                       int ply, int max_plies) {
    // 1) Royal existence
    auto chess_royal = find_royal(board, Side::CHESS);
    auto xiangqi_royal = find_royal(board, Side::XIANGQI);
    if (!chess_royal.has_value())
        return {TerminalStatus::XIANGQI_WIN, 2, "Chess king captured"};
    if (!xiangqi_royal.has_value())
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
