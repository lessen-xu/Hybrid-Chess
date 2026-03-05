// ab_search.cpp — Full Alpha-Beta (Negamax) search in C++.
// Mirrors hybrid/agents/alphabeta_agent.py + hybrid/agents/eval.py.
//
// Performance: zero Board clones during search. Uses make_move/unmake_move
// for in-place board mutation. Only one clone at best_move() entry.
// Each node: 1× generate_legal_moves_inplace, 1× board_hash per child.
// No calls to terminal_info() — inline terminal detection.

#include "ab_search.h"
#include "rules.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

// ═══════════════════════════════════════════════════════════════
// Piece values — matches hybrid/agents/eval.py PIECE_VALUES
// ═══════════════════════════════════════════════════════════════

static double piece_value(PieceKind k) {
    switch (k) {
        // Chess
        case PieceKind::KING:     return 0.0;
        case PieceKind::QUEEN:    return 9.0;
        case PieceKind::ROOK:     return 5.0;
        case PieceKind::BISHOP:   return 3.0;
        case PieceKind::KNIGHT:   return 3.0;
        case PieceKind::PAWN:     return 1.0;
        // Xiangqi
        case PieceKind::GENERAL:  return 0.0;
        case PieceKind::ADVISOR:  return 2.0;
        case PieceKind::ELEPHANT: return 2.0;
        case PieceKind::HORSE:    return 4.0;
        case PieceKind::CHARIOT:  return 9.0;
        case PieceKind::CANNON:   return 5.0;
        case PieceKind::SOLDIER:  return 1.0;
        default:                  return 0.0;
    }
}

// ═══════════════════════════════════════════════════════════════
// Royal existence check (direct grid scan, local to AB search)
// ═══════════════════════════════════════════════════════════════

static bool has_royal(const Board& board, Side side) {
    PieceKind target = (side == Side::CHESS) ? PieceKind::KING : PieceKind::GENERAL;
    for (int y = 0; y < BOARD_H; ++y) {
        for (int x = 0; x < BOARD_W; ++x) {
            auto& cell = board.grid[y][x];
            if (cell.has_value() && cell->side == side && cell->kind == target)
                return true;
        }
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════
// Evaluation — material + mobility + check bonus
// ═══════════════════════════════════════════════════════════════

static constexpr double MOBILITY_WEIGHT = 0.05;
static constexpr double CHECK_BONUS     = 0.3;

static double evaluate(Board& board, Side perspective) {
    // Material (direct grid scan)
    double mat = 0.0;
    for (int y = 0; y < BOARD_H; ++y) {
        for (int x = 0; x < BOARD_W; ++x) {
            auto& cell = board.grid[y][x];
            if (cell.has_value()) {
                double v = piece_value(cell->kind);
                mat += (cell->side == perspective) ? v : -v;
            }
        }
    }

    // Mobility (uses generate_legal_moves_inplace — zero clone)
    Side opp = opponent(perspective);
    std::vector<Move> my_m, op_m;
    generate_legal_moves_inplace(board, perspective, my_m);
    generate_legal_moves_inplace(board, opp, op_m);
    double mob = MOBILITY_WEIGHT * static_cast<double>(
        static_cast<int>(my_m.size()) - static_cast<int>(op_m.size()));

    // Check bonus
    double chk = 0.0;
    if (is_in_check(board, opp))         chk += CHECK_BONUS;
    if (is_in_check(board, perspective)) chk -= CHECK_BONUS;

    return mat + mob + chk;
}

// ═══════════════════════════════════════════════════════════════
// Move ordering — captures (by victim value) > checks > stable
// Uses make/unmake for check detection (no clone).
// ═══════════════════════════════════════════════════════════════

struct MoveKey {
    double score;       // primary sort (descending)
    int fx, fy, tx, ty; // tie-break (ascending tuple order)
    int promo;
};

static MoveKey move_order_key(Board& board, Side stm, const Move& mv) {
    // Capture bonus: value of captured piece
    double cap = 0.0;
    auto target = board.get(mv.tx, mv.ty);
    if (target.has_value()) {
        cap = piece_value(target->kind);
    }

    // Check bonus: does this move give check? (make/unmake, no clone)
    UndoInfo u;
    make_move(board, mv, u);
    double chk = is_in_check(board, opponent(stm)) ? 2.0 : 0.0;
    unmake_move(board, mv, u);

    return MoveKey{
        cap + chk,
        mv.fx, mv.fy, mv.tx, mv.ty,
        static_cast<int>(mv.promotion)
    };
}

static bool move_key_cmp(const std::pair<MoveKey, Move>& a,
                          const std::pair<MoveKey, Move>& b) {
    // Primary: higher score first
    if (a.first.score != b.first.score)
        return a.first.score > b.first.score;
    // Tie-break: lexicographic by (fx,fy,tx,ty,promo) ascending
    auto ta = std::make_tuple(a.first.fx, a.first.fy, a.first.tx, a.first.ty, a.first.promo);
    auto tb = std::make_tuple(b.first.fx, b.first.fy, b.first.tx, b.first.ty, b.first.promo);
    return ta < tb;
}

static std::vector<Move> order_moves(Board& board, Side stm,
                                      const std::vector<Move>& moves) {
    std::vector<std::pair<MoveKey, Move>> keyed;
    keyed.reserve(moves.size());
    for (auto& mv : moves) {
        keyed.push_back({move_order_key(board, stm, mv), mv});
    }
    std::stable_sort(keyed.begin(), keyed.end(), move_key_cmp);
    std::vector<Move> result;
    result.reserve(moves.size());
    for (auto& [k, mv] : keyed) {
        result.push_back(mv);
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════
// Repetition guard — RAII push/pop for search-local repetition
// Key is moved in for efficiency, held by value.
// ═══════════════════════════════════════════════════════════════

struct RepGuard {
    std::unordered_map<std::string,int>& rep;
    std::string key;

    RepGuard(std::unordered_map<std::string,int>& r, std::string k)
        : rep(r), key(std::move(k)) {
        rep[key]++;
    }
    ~RepGuard() {
        if (--rep[key] <= 0) rep.erase(key);
    }
};

// ═══════════════════════════════════════════════════════════════
// Negamax with alpha-beta pruning (zero-clone, in-place board)
//
// Inline terminal detection — no call to terminal_info().
// stm_hash_key = board.board_hash(stm), computed by parent.
// Single generate_legal_moves_inplace per node.
// ═══════════════════════════════════════════════════════════════

static constexpr double INF = 1e18;
static constexpr double WIN_SCORE = 1e6;

static double negamax(Board& board, Side stm, int depth,
                      double alpha, double beta,
                      std::unordered_map<std::string,int>& rep,
                      const std::string& stm_hash_key,
                      int ply, int max_plies,
                      Side root_perspective, int64_t& nodes) {
    nodes++;

    // ── Inline terminal detection (matches terminal_info order) ──

    // 1) Royal existence
    if (!has_royal(board, Side::CHESS)) {
        // Xiangqi wins
        double score = WIN_SCORE - static_cast<double>(ply);
        return (Side::XIANGQI == root_perspective) ? score : -score;
    }
    if (!has_royal(board, Side::XIANGQI)) {
        // Chess wins
        double score = WIN_SCORE - static_cast<double>(ply);
        return (Side::CHESS == root_perspective) ? score : -score;
    }

    // 2) Move limit
    if (ply >= max_plies) return 0.0;

    // 3) Threefold repetition (use pre-computed stm_hash_key)
    {
        auto it = rep.find(stm_hash_key);
        if (it != rep.end() && it->second >= 3) return 0.0;
    }

    // 4) Legal moves (single generation, used for both terminal + search)
    std::vector<Move> moves;
    generate_legal_moves_inplace(board, stm, moves);

    if (moves.empty()) {
        // Checkmate or stalemate
        if (is_in_check(board, stm)) {
            // Checkmate: opponent wins
            Side winner = opponent(stm);
            double score = WIN_SCORE - static_cast<double>(ply);
            return (winner == root_perspective) ? score : -score;
        }
        return 0.0;  // Stalemate
    }

    // ── Leaf evaluation (after terminal check) ──
    if (depth <= 0) {
        return evaluate(board, root_perspective);
    }

    // ── Search ──
    auto ordered = order_moves(board, stm, moves);
    Side opp = opponent(stm);
    double best = -INF;

    for (auto& mv : ordered) {
        UndoInfo u;
        make_move(board, mv, u);

        // Compute child hash once — used for RepGuard AND passed to child
        RepGuard guard(rep, board.board_hash(opp));

        double v = -negamax(board, opp, depth - 1, -beta, -alpha,
                            rep, guard.key,
                            ply + 1, max_plies,
                            root_perspective, nodes);
        unmake_move(board, mv, u);

        best = std::max(best, v);
        alpha = std::max(alpha, v);
        if (alpha >= beta) break;
    }

    return best;
}

// ═══════════════════════════════════════════════════════════════
// Public API: best_move()
// ═══════════════════════════════════════════════════════════════

SearchResult best_move(
        const Board& board,
        Side side_to_move,
        int depth,
        const std::unordered_map<std::string,int>& repetition_table,
        int ply,
        int max_plies) {

    // Single clone at entry — protects Python's board from mutation
    Board b = board.clone();

    // Mutable copy of repetition table for search-local tracking
    auto rep = repetition_table;

    // Root hash (for root-level terminal checks, computed once)
    std::string root_key = b.board_hash(side_to_move);

    // ── Root terminal detection (same order as negamax) ──
    if (!has_royal(b, Side::CHESS) || !has_royal(b, Side::XIANGQI)) {
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};
    }
    if (ply >= max_plies) {
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};
    }

    std::vector<Move> moves;
    generate_legal_moves_inplace(b, side_to_move, moves);
    if (moves.empty()) {
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};
    }

    auto ordered = order_moves(b, side_to_move, moves);

    int64_t total_nodes = 0;
    double alpha = -INF, beta = INF;
    Move best_mv = ordered[0];
    double best_val = -INF;
    Side opp = opponent(side_to_move);

    for (auto& mv : ordered) {
        UndoInfo u;
        make_move(b, mv, u);

        // Child hash: computed once, used for rep + passed to child
        RepGuard guard(rep, b.board_hash(opp));

        double v = -negamax(b, opp, depth - 1, -beta, -alpha,
                            rep, guard.key,
                            ply + 1, max_plies,
                            side_to_move, total_nodes);
        unmake_move(b, mv, u);

        if (v > best_val) {
            best_val = v;
            best_mv = mv;
        }
        alpha = std::max(alpha, v);
    }

    return SearchResult{best_mv, best_val, total_nodes};
}
