// ab_search.cpp — Full Alpha-Beta (Negamax) search in C++.
// Step 7: PVS (Negascout) + Root Aspiration Windows.
// Step 6: Per-ply buffer pre-allocation, leaf eval 1× movegen.
// Step 5: Zobrist 128-bit hash replaces SHA1 in hot path.
// Step 4: TT + Iterative Deepening + Killer/History heuristics.
//
// Performance: zero Board clones during search. Uses make_move/unmake_move.
// Zobrist mode: each node O(1) ZKey XOR — zero SHA1 calls in hot path.
// PVS: null-window scout for non-PV moves, re-search on fail-high.
// Root aspiration windows narrow search from d>=2 (exponential widening on fail).
// Per-ply pre-allocated buffers — zero heap allocation in recursive search.
// Leaf eval: 1× movegen (opp only), reuses stm move count from search.
// Transposition Table with generation-based isolation (deterministic).
// Iterative Deepening from depth 1 to requested depth.

#include "ab_search.h"
#include "rules.h"
#include "zobrist.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

// ═══════════════════════════════════════════════════════════════
// Piece values — matches hybrid/agents/eval.py PIECE_VALUES
// ═══════════════════════════════════════════════════════════════

static double piece_value(PieceKind k) {
    switch (k) {
        case PieceKind::KING:     return 0.0;
        case PieceKind::QUEEN:    return 9.0;
        case PieceKind::ROOK:     return 5.0;
        case PieceKind::BISHOP:   return 3.0;
        case PieceKind::KNIGHT:   return 3.0;
        case PieceKind::PAWN:     return 1.0;
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
// Royal existence check (direct grid scan)
// ═══════════════════════════════════════════════════════════════

static bool has_royal(const Board& board, Side side) {
    PieceKind target = (side == Side::CHESS) ? PieceKind::KING : PieceKind::GENERAL;
    for (int y = 0; y < BOARD_H; ++y)
        for (int x = 0; x < BOARD_W; ++x) {
            auto& cell = board.grid[y][x];
            if (cell.has_value() && cell->side == side && cell->kind == target)
                return true;
        }
    return false;
}

// ═══════════════════════════════════════════════════════════════
// Evaluation constants
// ═══════════════════════════════════════════════════════════════

static constexpr double MOBILITY_WEIGHT = 0.05;
static constexpr double CHECK_BONUS     = 0.3;

// ═══════════════════════════════════════════════════════════════
// Transposition Table
// ═══════════════════════════════════════════════════════════════

static constexpr double INF      = 1e18;
static constexpr double WIN_SCORE = 1e6;
static constexpr double MATE_TH  = WIN_SCORE * 0.5;

// PVS null-window epsilon (scores are double)
static constexpr double PVS_EPS = 1e-6;

// Root aspiration window parameters
static constexpr double INITIAL_ASP = 0.75;
static constexpr int    ASP_MAX_RETRIES = 5;
static constexpr double ASP_FALLBACK_TH = WIN_SCORE * 0.5;

enum TTFlag : uint8_t { TT_EXACT = 0, TT_LOWER = 1, TT_UPPER = 2, TT_EMPTY = 3 };

struct TTEntry {
    uint64_t key1 = 0;
    uint64_t key2 = 0;
    uint8_t  rep_bucket = 0;
    uint8_t  generation = 0;
    uint8_t  flag = TT_EMPTY;
    int      depth = -1;
    double   value = 0.0;
    Move     best_move{0, 0, 0, 0, PieceKind::NONE};
};

// Parse first 32 hex chars of SHA1 string into two uint64_t (legacy only)
static void parse_hash_key(const std::string& hex, uint64_t& k1, uint64_t& k2) {
    auto hex4 = [](char c) -> uint64_t {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    k1 = 0; k2 = 0;
    for (int i = 0; i < 16; ++i)
        k1 = (k1 << 4) | hex4(hex[i]);
    for (int i = 16; i < 32; ++i)
        k2 = (k2 << 4) | hex4(hex[i]);
}

// Mate-score TT pack/unpack
static inline double tt_pack(double s, int ply) {
    if (s >  MATE_TH)  return s + ply;
    if (s < -MATE_TH)  return s - ply;
    return s;
}
static inline double tt_unpack(double s, int ply) {
    if (s >  MATE_TH)  return s - ply;
    if (s < -MATE_TH)  return s + ply;
    return s;
}

static constexpr size_t TT_SIZE = 1 << 19;  // 512K entries
static constexpr size_t TT_MASK = TT_SIZE - 1;

struct TranspositionTable {
    std::vector<TTEntry> table;
    uint8_t generation = 0;

    TranspositionTable() : table(TT_SIZE) {}

    void new_search() {
        generation++;
        if (generation == 0) {
            for (auto& e : table) e.generation = 0;
            generation = 1;
        }
    }

    size_t index(uint64_t k1, uint64_t k2, uint8_t rb) const {
        uint64_t h = k1 ^ (k2 * 0x9E3779B97F4A7C15ULL) ^ (static_cast<uint64_t>(rb) << 1);
        return static_cast<size_t>(h & TT_MASK);
    }

    const TTEntry* probe(uint64_t k1, uint64_t k2, uint8_t rb) const {
        size_t idx = index(k1, k2, rb);
        const auto& e = table[idx];
        if (e.generation == generation &&
            e.flag != TT_EMPTY &&
            e.key1 == k1 && e.key2 == k2 &&
            e.rep_bucket == rb)
            return &e;
        return nullptr;
    }

    void store(uint64_t k1, uint64_t k2, uint8_t rb,
               int depth, double value, uint8_t flag, const Move& bm) {
        size_t idx = index(k1, k2, rb);
        auto& e = table[idx];
        if (e.generation != generation || depth >= e.depth) {
            e.key1 = k1;
            e.key2 = k2;
            e.rep_bucket = rb;
            e.generation = generation;
            e.depth = depth;
            e.value = value;
            e.flag = flag;
            e.best_move = bm;
        }
    }
};

static TranspositionTable g_tt;

// ═══════════════════════════════════════════════════════════════
// ScoredMove — for sorting without extra pair<MoveKey,Move> alloc
// ═══════════════════════════════════════════════════════════════

struct ScoredMove {
    Move mv;
    int    phase;       // 0=TT PV, 1=capture/check, 2=killer, 3=quiet
    double score;       // within-phase sort (descending)
    int    hist;        // history bonus for quiet moves
};

static bool scored_move_cmp(const ScoredMove& a, const ScoredMove& b) {
    if (a.phase != b.phase) return a.phase < b.phase;
    if (a.score != b.score) return a.score > b.score;
    if (a.hist  != b.hist)  return a.hist  > b.hist;
    // Tie-break by move coordinates (deterministic)
    auto ta = std::make_tuple(a.mv.fx, a.mv.fy, a.mv.tx, a.mv.ty, static_cast<int>(a.mv.promotion));
    auto tb = std::make_tuple(b.mv.fx, b.mv.fy, b.mv.tx, b.mv.ty, static_cast<int>(b.mv.promotion));
    return ta < tb;
}

// ═══════════════════════════════════════════════════════════════
// Per-ply buffers — pre-allocated, reused across recursive calls
// ═══════════════════════════════════════════════════════════════

struct PlyBuffers {
    std::vector<Move> moves;       // stm legal moves
    std::vector<Move> moves_opp;   // opp legal moves (leaf eval)
    std::vector<ScoredMove> scored; // for sorting

    PlyBuffers() {
        moves.reserve(192);
        moves_opp.reserve(192);
        scored.reserve(192);
    }
};

// ═══════════════════════════════════════════════════════════════
// Search context — per best_move() call
// ═══════════════════════════════════════════════════════════════

static constexpr int MAX_SEARCH_PLY = 256;

struct SearchContext {
    // Killer moves: 2 slots per search ply
    Move killer1[MAX_SEARCH_PLY];
    Move killer2[MAX_SEARCH_PLY];

    // History heuristic: from_sq * 90 + to_sq
    int history[90 * 90];

    // Per-ply pre-allocated buffers
    std::vector<PlyBuffers> bufs;

    // Zobrist-mode repetition tables
    std::unordered_map<ZKey128,int> base_rep_z;
    std::unordered_map<ZKey128,int> path_rep_z;

    SearchContext() {
        Move empty{0, 0, 0, 0, PieceKind::NONE};
        for (int i = 0; i < MAX_SEARCH_PLY; ++i) {
            killer1[i] = empty;
            killer2[i] = empty;
        }
        std::memset(history, 0, sizeof(history));
    }

    void init_buffers(int max_depth) {
        // sply ranges from 0 (root) up to max_depth + some margin
        bufs.resize(max_depth + 4);
    }

    void update_killers(int sply, const Move& mv) {
        if (sply >= MAX_SEARCH_PLY) return;
        if (!(mv.fx == killer1[sply].fx && mv.fy == killer1[sply].fy &&
              mv.tx == killer1[sply].tx && mv.ty == killer1[sply].ty &&
              mv.promotion == killer1[sply].promotion)) {
            killer2[sply] = killer1[sply];
            killer1[sply] = mv;
        }
    }

    void update_history(const Move& mv, int depth) {
        int from_sq = mv.fy * 9 + mv.fx;
        int to_sq   = mv.ty * 9 + mv.tx;
        history[from_sq * 90 + to_sq] += depth * depth;
    }

    int get_history(const Move& mv) const {
        int from_sq = mv.fy * 9 + mv.fx;
        int to_sq   = mv.ty * 9 + mv.tx;
        return history[from_sq * 90 + to_sq];
    }

    bool is_killer(int sply, const Move& mv) const {
        if (sply >= MAX_SEARCH_PLY) return false;
        auto match = [&](const Move& k) {
            return k.fx == mv.fx && k.fy == mv.fy &&
                   k.tx == mv.tx && k.ty == mv.ty &&
                   k.promotion == mv.promotion;
        };
        return match(killer1[sply]) || match(killer2[sply]);
    }

    int rep_count_z(const ZKey128& key) const {
        int c = 0;
        auto it1 = base_rep_z.find(key);
        if (it1 != base_rep_z.end()) c += it1->second;
        auto it2 = path_rep_z.find(key);
        if (it2 != path_rep_z.end()) c += it2->second;
        return c;
    }
};

// ═══════════════════════════════════════════════════════════════
// Move ordering — writes sorted result into moves buffer in-place
// ═══════════════════════════════════════════════════════════════

static bool moves_equal(const Move& a, const Move& b) {
    return a.fx == b.fx && a.fy == b.fy &&
           a.tx == b.tx && a.ty == b.ty &&
           a.promotion == b.promotion;
}

// Sorts moves in-place using scored buffer. Preserves exact same ordering
// as the old order_moves function for determinism.
static void order_moves_inplace(Board& board, Side stm,
                                std::vector<Move>& moves,
                                const Move& tt_move,
                                SearchContext& ctx, int sply) {
    bool has_tt = (tt_move.fx != 0 || tt_move.fy != 0 ||
                   tt_move.tx != 0 || tt_move.ty != 0 ||
                   tt_move.promotion != PieceKind::NONE);

    auto& scored = ctx.bufs[sply].scored;
    scored.clear();

    for (auto& mv : moves) {
        ScoredMove sm;
        sm.mv = mv;

        // Phase 0: TT PV move
        if (has_tt && moves_equal(mv, tt_move)) {
            sm.phase = 0; sm.score = 100.0; sm.hist = 0;
            scored.push_back(sm);
            continue;
        }

        // Capture bonus
        double cap = 0.0;
        auto target = board.get(mv.tx, mv.ty);
        if (target.has_value()) {
            cap = piece_value(target->kind);
        }

        // Check bonus (make/unmake)
        UndoInfo u;
        make_move(board, mv, u);
        double chk = is_in_check(board, opponent(stm)) ? 2.0 : 0.0;
        unmake_move(board, mv, u);

        double tactical_score = cap + chk;

        if (tactical_score > 0.0) {
            sm.phase = 1; sm.score = tactical_score; sm.hist = 0;
        } else if (ctx.is_killer(sply, mv)) {
            sm.phase = 2; sm.score = 0.0; sm.hist = 0;
        } else {
            sm.phase = 3; sm.score = 0.0; sm.hist = ctx.get_history(mv);
        }
        scored.push_back(sm);
    }

    std::stable_sort(scored.begin(), scored.end(), scored_move_cmp);

    // Write back sorted moves
    for (size_t i = 0; i < scored.size(); ++i) {
        moves[i] = scored[i].mv;
    }
}

// ═══════════════════════════════════════════════════════════════
// Leaf evaluation — reuses stm_moves_count, only generates opp
// ═══════════════════════════════════════════════════════════════

static inline double evaluate_leaf(
        Board& board,
        Side root_perspective,
        Side stm,
        int stm_moves_count,
        SearchContext& ctx,
        int sply) {
    // Material (grid scan)
    double mat = 0.0;
    for (int y = 0; y < BOARD_H; ++y)
        for (int x = 0; x < BOARD_W; ++x) {
            auto& cell = board.grid[y][x];
            if (cell.has_value()) {
                double v = piece_value(cell->kind);
                mat += (cell->side == root_perspective) ? v : -v;
            }
        }

    // Mobility: only generate opp moves (stm moves already known)
    Side opp = opponent(stm);
    auto& opp_moves = ctx.bufs[sply].moves_opp;
    opp_moves.clear();
    generate_legal_moves_inplace(board, opp, opp_moves);
    int opp_moves_count = static_cast<int>(opp_moves.size());

    // Compute perspective-relative mobility
    // Old code: generate_legal_moves(perspective) - generate_legal_moves(opp)
    // root_perspective's move count depends on whether root_perspective == stm
    int perspective_count, opp_persp_count;
    if (root_perspective == stm) {
        perspective_count = stm_moves_count;
        opp_persp_count = opp_moves_count;
    } else {
        perspective_count = opp_moves_count;
        opp_persp_count = stm_moves_count;
    }
    double mob = MOBILITY_WEIGHT * static_cast<double>(perspective_count - opp_persp_count);

    // Check bonus
    double chk = 0.0;
    Side rp_opp = opponent(root_perspective);
    if (is_in_check(board, rp_opp))            chk += CHECK_BONUS;
    if (is_in_check(board, root_perspective))   chk -= CHECK_BONUS;

    return mat + mob + chk;
}

// ═══════════════════════════════════════════════════════════════
// Repetition guards — RAII push/pop
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

struct RepGuardZ {
    std::unordered_map<ZKey128,int>& rep;
    ZKey128 key;

    RepGuardZ(std::unordered_map<ZKey128,int>& r, ZKey128 k)
        : rep(r), key(k) {
        rep[key]++;
    }
    ~RepGuardZ() {
        if (--rep[key] <= 0) rep.erase(key);
    }
};

// ═══════════════════════════════════════════════════════════════
// Negamax — ZOBRIST fast path (zero SHA1 in hot path)
// ═══════════════════════════════════════════════════════════════

static double negamax_z(Board& board, Side stm, int depth,
                        double alpha, double beta,
                        SearchContext& ctx,
                        ZKey128 stm_key,
                        int ply, int max_plies,
                        Side root_perspective, int64_t& nodes,
                        int sply) {
    nodes++;

    // ── Inline terminal detection ──
    if (!has_royal(board, Side::CHESS)) {
        double score = WIN_SCORE - static_cast<double>(ply);
        return (Side::XIANGQI == root_perspective) ? score : -score;
    }
    if (!has_royal(board, Side::XIANGQI)) {
        double score = WIN_SCORE - static_cast<double>(ply);
        return (Side::CHESS == root_perspective) ? score : -score;
    }
    if (ply >= max_plies) return 0.0;

    // Threefold repetition
    int rep_count = ctx.rep_count_z(stm_key);
    if (rep_count >= 3) return 0.0;

    // ── TT probe ──
    uint64_t k1 = stm_key.lo, k2 = stm_key.hi;
    uint8_t rb = static_cast<uint8_t>(std::min(rep_count, 3));

    Move tt_best{0, 0, 0, 0, PieceKind::NONE};
    double alpha0 = alpha;

    const TTEntry* tte = g_tt.probe(k1, k2, rb);
    if (tte) {
        tt_best = tte->best_move;
        if (tte->depth >= depth) {
            double v = tt_unpack(tte->value, ply);
            if (tte->flag == TT_EXACT) return v;
            if (tte->flag == TT_LOWER) alpha = std::max(alpha, v);
            if (tte->flag == TT_UPPER) beta  = std::min(beta, v);
            if (alpha >= beta) return v;
        }
    }

    // ── Generate legal moves (into pre-allocated buffer) ──
    auto& moves = ctx.bufs[sply].moves;
    moves.clear();
    generate_legal_moves_inplace(board, stm, moves);

    if (moves.empty()) {
        if (is_in_check(board, stm)) {
            Side winner = opponent(stm);
            double score = WIN_SCORE - static_cast<double>(ply);
            return (winner == root_perspective) ? score : -score;
        }
        return 0.0;
    }

    // ── Leaf evaluation (reuse stm move count) ──
    if (depth <= 0) {
        return evaluate_leaf(board, root_perspective, stm,
                             static_cast<int>(moves.size()), ctx, sply);
    }

    // ── Search (PVS / Negascout) ──
    order_moves_inplace(board, stm, moves, tt_best, ctx, sply);
    Side opp = opponent(stm);
    double best = -INF;
    Move best_mv = moves[0];
    bool first_move = true;

    for (auto& mv : moves) {
        UndoInfo u;
        make_move(board, mv, u);
        ZKey128 child_key = board.zobrist_key(opp);
        RepGuardZ guard(ctx.path_rep_z, child_key);

        double v;
        if (first_move) {
            // First move: full window
            v = -negamax_z(board, opp, depth - 1, -beta, -alpha,
                           ctx, child_key,
                           ply + 1, max_plies,
                           root_perspective, nodes, sply + 1);
            first_move = false;
        } else if (alpha + PVS_EPS < beta) {
            // Null-window scout
            v = -negamax_z(board, opp, depth - 1, -alpha - PVS_EPS, -alpha,
                           ctx, child_key,
                           ply + 1, max_plies,
                           root_perspective, nodes, sply + 1);
            // Re-search if scout found a better move within true window
            if (v > alpha && v < beta) {
                v = -negamax_z(board, opp, depth - 1, -beta, -alpha,
                               ctx, child_key,
                               ply + 1, max_plies,
                               root_perspective, nodes, sply + 1);
            }
        } else {
            // Window already narrow, full search
            v = -negamax_z(board, opp, depth - 1, -beta, -alpha,
                           ctx, child_key,
                           ply + 1, max_plies,
                           root_perspective, nodes, sply + 1);
        }
        unmake_move(board, mv, u);

        if (v > best) {
            best = v;
            best_mv = mv;
        }
        alpha = std::max(alpha, v);
        if (alpha >= beta) {
            bool is_capture = board.get(mv.tx, mv.ty).has_value();
            if (!is_capture) {
                ctx.update_killers(sply, mv);
                ctx.update_history(mv, depth);
            }
            break;
        }
    }

    // ── TT store ──
    uint8_t flag;
    if (best <= alpha0)     flag = TT_UPPER;
    else if (best >= beta)  flag = TT_LOWER;
    else                    flag = TT_EXACT;

    g_tt.store(k1, k2, rb, depth, tt_pack(best, ply), flag, best_mv);

    return best;
}

// ═══════════════════════════════════════════════════════════════
// Negamax — SHA1 legacy path (backward compatible)
// ═══════════════════════════════════════════════════════════════

static double negamax_sha1(Board& board, Side stm, int depth,
                           double alpha, double beta,
                           std::unordered_map<std::string,int>& rep,
                           const std::string& stm_hash_key,
                           int ply, int max_plies,
                           Side root_perspective, int64_t& nodes,
                           SearchContext& ctx, int sply) {
    nodes++;

    if (!has_royal(board, Side::CHESS)) {
        double score = WIN_SCORE - static_cast<double>(ply);
        return (Side::XIANGQI == root_perspective) ? score : -score;
    }
    if (!has_royal(board, Side::XIANGQI)) {
        double score = WIN_SCORE - static_cast<double>(ply);
        return (Side::CHESS == root_perspective) ? score : -score;
    }
    if (ply >= max_plies) return 0.0;

    int rep_count = 0;
    {
        auto it = rep.find(stm_hash_key);
        if (it != rep.end()) {
            rep_count = it->second;
            if (rep_count >= 3) return 0.0;
        }
    }

    // TT probe (Zobrist key)
    ZKey128 zk = board.zobrist_key(stm);
    uint64_t k1 = zk.lo, k2 = zk.hi;
    uint8_t rb = static_cast<uint8_t>(std::min(rep_count, 3));

    Move tt_best{0, 0, 0, 0, PieceKind::NONE};
    double alpha0 = alpha;

    const TTEntry* tte = g_tt.probe(k1, k2, rb);
    if (tte) {
        tt_best = tte->best_move;
        if (tte->depth >= depth) {
            double v = tt_unpack(tte->value, ply);
            if (tte->flag == TT_EXACT) return v;
            if (tte->flag == TT_LOWER) alpha = std::max(alpha, v);
            if (tte->flag == TT_UPPER) beta  = std::min(beta, v);
            if (alpha >= beta) return v;
        }
    }

    // Generate legal moves (into pre-allocated buffer)
    auto& moves = ctx.bufs[sply].moves;
    moves.clear();
    generate_legal_moves_inplace(board, stm, moves);

    if (moves.empty()) {
        if (is_in_check(board, stm)) {
            Side winner = opponent(stm);
            double score = WIN_SCORE - static_cast<double>(ply);
            return (winner == root_perspective) ? score : -score;
        }
        return 0.0;
    }

    // Leaf evaluation (reuse stm move count)
    if (depth <= 0) {
        return evaluate_leaf(board, root_perspective, stm,
                             static_cast<int>(moves.size()), ctx, sply);
    }

    // Search (PVS / Negascout)
    order_moves_inplace(board, stm, moves, tt_best, ctx, sply);
    Side opp = opponent(stm);
    double best = -INF;
    Move best_mv = moves[0];
    bool first_move = true;

    for (auto& mv : moves) {
        UndoInfo u;
        make_move(board, mv, u);
        RepGuard guard(rep, board.board_hash(opp));

        double v;
        if (first_move) {
            v = -negamax_sha1(board, opp, depth - 1, -beta, -alpha,
                              rep, guard.key,
                              ply + 1, max_plies,
                              root_perspective, nodes, ctx, sply + 1);
            first_move = false;
        } else if (alpha + PVS_EPS < beta) {
            v = -negamax_sha1(board, opp, depth - 1, -alpha - PVS_EPS, -alpha,
                              rep, guard.key,
                              ply + 1, max_plies,
                              root_perspective, nodes, ctx, sply + 1);
            if (v > alpha && v < beta) {
                v = -negamax_sha1(board, opp, depth - 1, -beta, -alpha,
                                  rep, guard.key,
                                  ply + 1, max_plies,
                                  root_perspective, nodes, ctx, sply + 1);
            }
        } else {
            v = -negamax_sha1(board, opp, depth - 1, -beta, -alpha,
                              rep, guard.key,
                              ply + 1, max_plies,
                              root_perspective, nodes, ctx, sply + 1);
        }
        unmake_move(board, mv, u);

        if (v > best) {
            best = v;
            best_mv = mv;
        }
        alpha = std::max(alpha, v);
        if (alpha >= beta) {
            bool is_capture = board.get(mv.tx, mv.ty).has_value();
            if (!is_capture) {
                ctx.update_killers(sply, mv);
                ctx.update_history(mv, depth);
            }
            break;
        }
    }

    uint8_t flag;
    if (best <= alpha0)     flag = TT_UPPER;
    else if (best >= beta)  flag = TT_LOWER;
    else                    flag = TT_EXACT;

    g_tt.store(k1, k2, rb, depth, tt_pack(best, ply), flag, best_mv);

    return best;
}

// ═══════════════════════════════════════════════════════════════
// Helper: detect repetition table mode
// ═══════════════════════════════════════════════════════════════

enum class RepMode { ZOBRIST, SHA1 };

static RepMode detect_rep_mode(const std::unordered_map<std::string,int>& rep) {
    if (rep.empty()) return RepMode::ZOBRIST;
    const auto& key = rep.begin()->first;
    auto is_hex = [](const std::string& s) {
        for (char c : s) {
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
                return false;
        }
        return true;
    };
    if (key.size() == 32 && is_hex(key)) return RepMode::ZOBRIST;
    if (key.size() == 40 && is_hex(key)) return RepMode::SHA1;
    return RepMode::SHA1;
}

static ZKey128 parse_zkey_hex(const std::string& hex) {
    auto hex4 = [](char c) -> uint64_t {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    ZKey128 k{0, 0};
    for (int i = 0; i < 16; ++i)
        k.hi = (k.hi << 4) | hex4(hex[i]);
    for (int i = 16; i < 32; ++i)
        k.lo = (k.lo << 4) | hex4(hex[i]);
    return k;
}

// ═══════════════════════════════════════════════════════════════
// Public API: best_move() — Iterative Deepening wrapper
// ═══════════════════════════════════════════════════════════════

SearchResult best_move(
        const Board& board,
        Side side_to_move,
        int depth,
        const std::unordered_map<std::string,int>& repetition_table,
        int ply,
        int max_plies) {

    Board b = board.clone();
    g_tt.new_search();

    if (!has_royal(b, Side::CHESS) || !has_royal(b, Side::XIANGQI))
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};
    if (ply >= max_plies)
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};

    // Generate root moves (not in PlyBuffers — root uses its own vector)
    std::vector<Move> root_moves;
    generate_legal_moves_inplace(b, side_to_move, root_moves);
    if (root_moves.empty())
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};

    SearchContext ctx;
    ctx.init_buffers(depth + 2);

    int64_t total_nodes = 0;
    Move best_mv = root_moves[0];
    double best_val = -INF;
    Side opp = opponent(side_to_move);

    RepMode mode = detect_rep_mode(repetition_table);

    if (mode == RepMode::ZOBRIST) {
        for (auto& [k, v] : repetition_table) {
            ctx.base_rep_z[parse_zkey_hex(k)] = v;
        }

        ZKey128 root_key = b.zobrist_key(side_to_move);

        for (int d = 1; d <= depth; ++d) {
            // Aspiration window
            double asp_alpha, asp_beta;
            if (d == 1) {
                asp_alpha = -INF;
                asp_beta  =  INF;
            } else {
                double w = INITIAL_ASP;
                asp_alpha = best_val - w;
                asp_beta  = best_val + w;
            }

            Move iter_best = root_moves[0];
            double iter_val = -INF;
            bool accepted = false;

            for (int retry = 0; !accepted; ++retry) {
                ctx.path_rep_z.clear();

                double alpha = asp_alpha, beta = asp_beta;
                iter_best = root_moves[0];
                iter_val = -INF;

                int root_rep = ctx.rep_count_z(root_key);
                uint8_t root_rb = static_cast<uint8_t>(std::min(root_rep, 3));
                Move root_tt_mv{0, 0, 0, 0, PieceKind::NONE};
                const TTEntry* root_tte = g_tt.probe(root_key.lo, root_key.hi, root_rb);
                if (root_tte) root_tt_mv = root_tte->best_move;

                ctx.bufs[0].moves = root_moves;
                order_moves_inplace(b, side_to_move, ctx.bufs[0].moves, root_tt_mv, ctx, 0);

                bool first_root_move = true;
                for (auto& mv : ctx.bufs[0].moves) {
                    UndoInfo u;
                    make_move(b, mv, u);
                    ZKey128 child_key = b.zobrist_key(opp);
                    RepGuardZ guard(ctx.path_rep_z, child_key);

                    double v;
                    if (first_root_move) {
                        v = -negamax_z(b, opp, d - 1, -beta, -alpha,
                                       ctx, child_key,
                                       ply + 1, max_plies,
                                       side_to_move, total_nodes, 1);
                        first_root_move = false;
                    } else if (alpha + PVS_EPS < beta) {
                        v = -negamax_z(b, opp, d - 1, -alpha - PVS_EPS, -alpha,
                                       ctx, child_key,
                                       ply + 1, max_plies,
                                       side_to_move, total_nodes, 1);
                        if (v > alpha && v < beta) {
                            v = -negamax_z(b, opp, d - 1, -beta, -alpha,
                                           ctx, child_key,
                                           ply + 1, max_plies,
                                           side_to_move, total_nodes, 1);
                        }
                    } else {
                        v = -negamax_z(b, opp, d - 1, -beta, -alpha,
                                       ctx, child_key,
                                       ply + 1, max_plies,
                                       side_to_move, total_nodes, 1);
                    }
                    unmake_move(b, mv, u);

                    if (v > iter_val) {
                        iter_val = v;
                        iter_best = mv;
                    }
                    alpha = std::max(alpha, v);
                }

                // Check aspiration result
                if (d == 1 || (iter_val > asp_alpha && iter_val < asp_beta)) {
                    accepted = true;
                } else if (retry >= ASP_MAX_RETRIES || (asp_beta - asp_alpha) > ASP_FALLBACK_TH) {
                    // Fallback to full window
                    asp_alpha = -INF;
                    asp_beta  =  INF;
                    // One more iteration with full window
                } else {
                    // Widen
                    double w = INITIAL_ASP;
                    for (int i = 0; i <= retry; ++i) w *= 2.0;
                    asp_alpha = best_val - w;
                    asp_beta  = best_val + w;
                }
            }

            best_mv = iter_best;
            best_val = iter_val;
        }
    } else {
        std::string root_key = b.board_hash(side_to_move);

        for (int d = 1; d <= depth; ++d) {
            double asp_alpha, asp_beta;
            if (d == 1) {
                asp_alpha = -INF;
                asp_beta  =  INF;
            } else {
                double w = INITIAL_ASP;
                asp_alpha = best_val - w;
                asp_beta  = best_val + w;
            }

            Move iter_best = root_moves[0];
            double iter_val = -INF;
            bool accepted = false;

            for (int retry = 0; !accepted; ++retry) {
                auto rep = repetition_table;

                double alpha = asp_alpha, beta = asp_beta;
                iter_best = root_moves[0];
                iter_val = -INF;

                ZKey128 root_zk = b.zobrist_key(side_to_move);
                int root_rep = 0;
                {
                    auto it = rep.find(root_key);
                    if (it != rep.end()) root_rep = it->second;
                }
                uint8_t root_rb = static_cast<uint8_t>(std::min(root_rep, 3));
                Move root_tt_mv{0, 0, 0, 0, PieceKind::NONE};
                const TTEntry* root_tte = g_tt.probe(root_zk.lo, root_zk.hi, root_rb);
                if (root_tte) root_tt_mv = root_tte->best_move;

                ctx.bufs[0].moves = root_moves;
                order_moves_inplace(b, side_to_move, ctx.bufs[0].moves, root_tt_mv, ctx, 0);

                bool first_root_move = true;
                for (auto& mv : ctx.bufs[0].moves) {
                    UndoInfo u;
                    make_move(b, mv, u);
                    RepGuard guard(rep, b.board_hash(opp));

                    double v;
                    if (first_root_move) {
                        v = -negamax_sha1(b, opp, d - 1, -beta, -alpha,
                                          rep, guard.key,
                                          ply + 1, max_plies,
                                          side_to_move, total_nodes, ctx, 1);
                        first_root_move = false;
                    } else if (alpha + PVS_EPS < beta) {
                        v = -negamax_sha1(b, opp, d - 1, -alpha - PVS_EPS, -alpha,
                                          rep, guard.key,
                                          ply + 1, max_plies,
                                          side_to_move, total_nodes, ctx, 1);
                        if (v > alpha && v < beta) {
                            v = -negamax_sha1(b, opp, d - 1, -beta, -alpha,
                                              rep, guard.key,
                                              ply + 1, max_plies,
                                              side_to_move, total_nodes, ctx, 1);
                        }
                    } else {
                        v = -negamax_sha1(b, opp, d - 1, -beta, -alpha,
                                          rep, guard.key,
                                          ply + 1, max_plies,
                                          side_to_move, total_nodes, ctx, 1);
                    }
                    unmake_move(b, mv, u);

                    if (v > iter_val) {
                        iter_val = v;
                        iter_best = mv;
                    }
                    alpha = std::max(alpha, v);
                }

                if (d == 1 || (iter_val > asp_alpha && iter_val < asp_beta)) {
                    accepted = true;
                } else if (retry >= ASP_MAX_RETRIES || (asp_beta - asp_alpha) > ASP_FALLBACK_TH) {
                    asp_alpha = -INF;
                    asp_beta  =  INF;
                } else {
                    double w = INITIAL_ASP;
                    for (int i = 0; i <= retry; ++i) w *= 2.0;
                    asp_alpha = best_val - w;
                    asp_beta  = best_val + w;
                }
            }

            best_mv = iter_best;
            best_val = iter_val;
        }
    }

    return SearchResult{best_mv, best_val, total_nodes};
}
