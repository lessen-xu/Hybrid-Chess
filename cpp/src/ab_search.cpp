// ab_search.cpp — Full Alpha-Beta (Negamax) search in C++.
// Mirrors hybrid/agents/alphabeta_agent.py + hybrid/agents/eval.py.

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
// Evaluation — material + mobility + check bonus
// ═══════════════════════════════════════════════════════════════

static constexpr double MOBILITY_WEIGHT = 0.05;
static constexpr double CHECK_BONUS     = 0.3;

static double evaluate(const Board& board, Side perspective) {
    // Material
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

    // Mobility
    Side opp = opponent(perspective);
    int my_moves = static_cast<int>(generate_legal_moves(board, perspective).size());
    int op_moves = static_cast<int>(generate_legal_moves(board, opp).size());
    double mob = MOBILITY_WEIGHT * static_cast<double>(my_moves - op_moves);

    // Check bonus
    double chk = 0.0;
    if (is_in_check(board, opp))         chk += CHECK_BONUS;
    if (is_in_check(board, perspective)) chk -= CHECK_BONUS;

    return mat + mob + chk;
}

// ═══════════════════════════════════════════════════════════════
// Move ordering — captures (by victim value) > checks > stable
// ═══════════════════════════════════════════════════════════════

struct MoveKey {
    double score;       // primary sort (descending)
    int fx, fy, tx, ty; // tie-break (ascending tuple order)
    int promo;
};

static MoveKey move_order_key(const Board& board, Side stm, const Move& mv) {
    // Capture bonus: value of captured piece
    double cap = 0.0;
    auto target = board.get(mv.tx, mv.ty);
    if (target.has_value()) {
        cap = piece_value(target->kind);
    }

    // Check bonus: does this move give check?
    Board after = apply_move(board, mv);
    double chk = is_in_check(after, opponent(stm)) ? 2.0 : 0.0;

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

static std::vector<Move> order_moves(const Board& board, Side stm,
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
// ═══════════════════════════════════════════════════════════════

struct RepGuard {
    std::unordered_map<std::string,int>& rep;
    std::string key;

    RepGuard(std::unordered_map<std::string,int>& r, const std::string& k)
        : rep(r), key(k) {
        rep[key]++;
    }
    ~RepGuard() {
        if (--rep[key] <= 0) rep.erase(key);
    }
};

// ═══════════════════════════════════════════════════════════════
// Negamax with alpha-beta pruning
// ═══════════════════════════════════════════════════════════════

static constexpr double INF = 1e18;
static constexpr double WIN_SCORE = 1e6;

static double negamax(const Board& board, Side stm, int depth,
                      double alpha, double beta,
                      std::unordered_map<std::string,int>& rep,
                      int ply, int max_plies,
                      Side root_perspective, int64_t& nodes) {
    nodes++;

    // Terminal check
    GameInfo info = terminal_info(board, stm, rep, ply, max_plies);
    if (info.status != std::string(TerminalStatus::ONGOING)) {
        if (info.status == std::string(TerminalStatus::DRAW)) {
            return 0.0;
        }
        // Win/loss with ply correction (prefer faster wins)
        Side winner = (info.winner == 1) ? Side::CHESS : Side::XIANGQI;
        double score = WIN_SCORE - static_cast<double>(ply);
        return (winner == root_perspective) ? score : -score;
    }

    // Leaf: evaluate
    if (depth <= 0) {
        // Evaluate from side-to-move's perspective, then adjust to
        // root_perspective's perspective via negamax sign flipping.
        // Actually: negamax returns value from stm's perspective,
        // and we need it relative to root_perspective.
        // Simpler: evaluate always returns from root_perspective.
        return evaluate(board, root_perspective);
    }

    // Generate & order moves
    auto moves = generate_legal_moves(board, stm);
    if (moves.empty()) {
        return evaluate(board, root_perspective);
    }
    auto ordered = order_moves(board, stm, moves);

    Side opp = opponent(stm);
    double best = -INF;

    for (auto& mv : ordered) {
        Board child = apply_move(board, mv);
        std::string hash_key = child.board_hash(opp);
        RepGuard guard(rep, hash_key);

        double v = -negamax(child, opp, depth - 1, -beta, -alpha,
                            rep, ply + 1, max_plies,
                            root_perspective, nodes);
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

    // We need a mutable copy for search-local repetition tracking
    auto rep = repetition_table;

    auto moves = generate_legal_moves(board, side_to_move);
    if (moves.empty()) {
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};
    }

    auto ordered = order_moves(board, side_to_move, moves);

    int64_t total_nodes = 0;
    double alpha = -INF, beta = INF;
    Move best_mv = ordered[0];
    double best_val = -INF;
    Side opp = opponent(side_to_move);

    for (auto& mv : ordered) {
        Board child = apply_move(board, mv);
        std::string hash_key = child.board_hash(opp);
        RepGuard guard(rep, hash_key);

        double v = -negamax(child, opp, depth - 1, -beta, -alpha,
                            rep, ply + 1, max_plies,
                            side_to_move, total_nodes);
        if (v > best_val) {
            best_val = v;
            best_mv = mv;
        }
        alpha = std::max(alpha, v);
    }

    return SearchResult{best_mv, best_val, total_nodes};
}
