// ab_search.cpp — Full Alpha-Beta (Negamax) search in C++.
// Step 4: TT + Iterative Deepening + Killer/History heuristics.
//
// Performance: zero Board clones during search. Uses make_move/unmake_move.
// Each node: 1× generate_legal_moves_inplace, 1× board_hash per child.
// Inline terminal detection — no call to terminal_info().
// Transposition Table with generation-based isolation (deterministic).
// Iterative Deepening from depth 1 to requested depth.

#include "ab_search.h"
#include "rules.h"

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
// Evaluation — material + mobility + check bonus
// ═══════════════════════════════════════════════════════════════

static constexpr double MOBILITY_WEIGHT = 0.05;
static constexpr double CHECK_BONUS     = 0.3;

static double evaluate(Board& board, Side perspective) {
    double mat = 0.0;
    for (int y = 0; y < BOARD_H; ++y)
        for (int x = 0; x < BOARD_W; ++x) {
            auto& cell = board.grid[y][x];
            if (cell.has_value()) {
                double v = piece_value(cell->kind);
                mat += (cell->side == perspective) ? v : -v;
            }
        }

    Side opp = opponent(perspective);
    std::vector<Move> my_m, op_m;
    generate_legal_moves_inplace(board, perspective, my_m);
    generate_legal_moves_inplace(board, opp, op_m);
    double mob = MOBILITY_WEIGHT * static_cast<double>(
        static_cast<int>(my_m.size()) - static_cast<int>(op_m.size()));

    double chk = 0.0;
    if (is_in_check(board, opp))         chk += CHECK_BONUS;
    if (is_in_check(board, perspective)) chk -= CHECK_BONUS;

    return mat + mob + chk;
}

// ═══════════════════════════════════════════════════════════════
// Transposition Table
// ═══════════════════════════════════════════════════════════════

static constexpr double INF      = 1e18;
static constexpr double WIN_SCORE = 1e6;
static constexpr double MATE_TH  = WIN_SCORE * 0.5;

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

// Parse first 32 hex chars of SHA1 string into two uint64_t
static void parse_hash_key(const std::string& hex, uint64_t& k1, uint64_t& k2) {
    // hex is 40 chars; we use first 32 (128 bits)
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
            // Overflow: clear all generation fields, then set to 1
            for (auto& e : table) e.generation = 0;
            generation = 1;
        }
    }

    size_t index(uint64_t k1, uint64_t k2, uint8_t rb) const {
        uint64_t h = k1 ^ (k2 * 0x9E3779B97F4A7C15ULL) ^ (static_cast<uint64_t>(rb) << 1);
        return static_cast<size_t>(h & TT_MASK);
    }

    // Returns pointer to entry if hit, nullptr if miss
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
        // Deterministic replace: stale generation -> always replace; same gen -> only if deeper
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

// Global TT (persists across best_move calls, but generation isolates them)
static TranspositionTable g_tt;

// ═══════════════════════════════════════════════════════════════
// Search context — per best_move() call, holds killer/history
// ═══════════════════════════════════════════════════════════════

static constexpr int MAX_SEARCH_PLY = 256;

struct SearchContext {
    // Killer moves: 2 slots per search ply
    Move killer1[MAX_SEARCH_PLY];
    Move killer2[MAX_SEARCH_PLY];

    // History heuristic: from_sq * 90 + to_sq
    int history[90 * 90];

    SearchContext() {
        Move empty{0, 0, 0, 0, PieceKind::NONE};
        for (int i = 0; i < MAX_SEARCH_PLY; ++i) {
            killer1[i] = empty;
            killer2[i] = empty;
        }
        std::memset(history, 0, sizeof(history));
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
};

// ═══════════════════════════════════════════════════════════════
// Move ordering: TT PV > captures+checks > killers > history > tie-break
// ═══════════════════════════════════════════════════════════════

struct MoveKey {
    int    phase;       // 0=TT PV, 1=capture/check, 2=killer, 3=quiet
    double score;       // within-phase sort (descending)
    int    hist;        // history bonus for quiet moves
    int fx, fy, tx, ty; // tie-break
    int promo;
};

static bool move_key_cmp(const std::pair<MoveKey, Move>& a,
                          const std::pair<MoveKey, Move>& b) {
    if (a.first.phase != b.first.phase)
        return a.first.phase < b.first.phase;
    if (a.first.score != b.first.score)
        return a.first.score > b.first.score;
    if (a.first.hist != b.first.hist)
        return a.first.hist > b.first.hist;
    auto ta = std::make_tuple(a.first.fx, a.first.fy, a.first.tx, a.first.ty, a.first.promo);
    auto tb = std::make_tuple(b.first.fx, b.first.fy, b.first.tx, b.first.ty, b.first.promo);
    return ta < tb;
}

static bool moves_equal(const Move& a, const Move& b) {
    return a.fx == b.fx && a.fy == b.fy &&
           a.tx == b.tx && a.ty == b.ty &&
           a.promotion == b.promotion;
}

static std::vector<Move> order_moves(Board& board, Side stm,
                                      const std::vector<Move>& moves,
                                      const Move& tt_move,
                                      const SearchContext& ctx, int sply) {
    bool has_tt = (tt_move.fx != 0 || tt_move.fy != 0 ||
                   tt_move.tx != 0 || tt_move.ty != 0 ||
                   tt_move.promotion != PieceKind::NONE);

    std::vector<std::pair<MoveKey, Move>> keyed;
    keyed.reserve(moves.size());

    for (auto& mv : moves) {
        // Phase 0: TT PV move
        if (has_tt && moves_equal(mv, tt_move)) {
            keyed.push_back({MoveKey{0, 100.0, 0,
                mv.fx, mv.fy, mv.tx, mv.ty, static_cast<int>(mv.promotion)}, mv});
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
            // Phase 1: captures/checks
            keyed.push_back({MoveKey{1, tactical_score, 0,
                mv.fx, mv.fy, mv.tx, mv.ty, static_cast<int>(mv.promotion)}, mv});
        } else if (ctx.is_killer(sply, mv)) {
            // Phase 2: killer moves
            keyed.push_back({MoveKey{2, 0.0, 0,
                mv.fx, mv.fy, mv.tx, mv.ty, static_cast<int>(mv.promotion)}, mv});
        } else {
            // Phase 3: quiet moves, sorted by history
            int hist = ctx.get_history(mv);
            keyed.push_back({MoveKey{3, 0.0, hist,
                mv.fx, mv.fy, mv.tx, mv.ty, static_cast<int>(mv.promotion)}, mv});
        }
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
// Repetition guard — RAII push/pop
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
// Negamax with alpha-beta, TT, killer/history
// ═══════════════════════════════════════════════════════════════

static double negamax(Board& board, Side stm, int depth,
                      double alpha, double beta,
                      std::unordered_map<std::string,int>& rep,
                      const std::string& stm_hash_key,
                      int ply, int max_plies,
                      Side root_perspective, int64_t& nodes,
                      SearchContext& ctx, int sply) {
    nodes++;

    // ── Inline terminal detection ──

    // 1) Royal existence
    if (!has_royal(board, Side::CHESS)) {
        double score = WIN_SCORE - static_cast<double>(ply);
        return (Side::XIANGQI == root_perspective) ? score : -score;
    }
    if (!has_royal(board, Side::XIANGQI)) {
        double score = WIN_SCORE - static_cast<double>(ply);
        return (Side::CHESS == root_perspective) ? score : -score;
    }

    // 2) Move limit
    if (ply >= max_plies) return 0.0;

    // 3) Threefold repetition (MUST check before TT probe)
    int rep_count = 0;
    {
        auto it = rep.find(stm_hash_key);
        if (it != rep.end()) {
            rep_count = it->second;
            if (rep_count >= 3) return 0.0;
        }
    }

    // ── TT probe ──
    uint64_t k1, k2;
    parse_hash_key(stm_hash_key, k1, k2);
    uint8_t rb = static_cast<uint8_t>(std::min(rep_count, 3));

    Move tt_best{0, 0, 0, 0, PieceKind::NONE};
    double alpha0 = alpha;  // save original alpha for TT flag computation

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

    // ── Generate legal moves ──
    std::vector<Move> moves;
    generate_legal_moves_inplace(board, stm, moves);

    if (moves.empty()) {
        if (is_in_check(board, stm)) {
            Side winner = opponent(stm);
            double score = WIN_SCORE - static_cast<double>(ply);
            return (winner == root_perspective) ? score : -score;
        }
        return 0.0;  // stalemate
    }

    // ── Leaf evaluation ──
    if (depth <= 0) {
        return evaluate(board, root_perspective);
    }

    // ── Search ──
    auto ordered = order_moves(board, stm, moves, tt_best, ctx, sply);
    Side opp = opponent(stm);
    double best = -INF;
    Move best_mv = ordered[0];

    for (auto& mv : ordered) {
        UndoInfo u;
        make_move(board, mv, u);
        RepGuard guard(rep, board.board_hash(opp));

        double v = -negamax(board, opp, depth - 1, -beta, -alpha,
                            rep, guard.key,
                            ply + 1, max_plies,
                            root_perspective, nodes,
                            ctx, sply + 1);
        unmake_move(board, mv, u);

        if (v > best) {
            best = v;
            best_mv = mv;
        }
        alpha = std::max(alpha, v);
        if (alpha >= beta) {
            // Beta cutoff: update killer/history for quiet moves
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
// Public API: best_move() — Iterative Deepening wrapper
// ═══════════════════════════════════════════════════════════════

SearchResult best_move(
        const Board& board,
        Side side_to_move,
        int depth,
        const std::unordered_map<std::string,int>& repetition_table,
        int ply,
        int max_plies) {

    // Single clone at entry — protects Python's board
    Board b = board.clone();

    // New TT generation for this call (isolates from previous calls)
    g_tt.new_search();

    // Root hash
    std::string root_key = b.board_hash(side_to_move);

    // Root terminal checks
    if (!has_royal(b, Side::CHESS) || !has_royal(b, Side::XIANGQI))
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};
    if (ply >= max_plies)
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};

    std::vector<Move> moves;
    generate_legal_moves_inplace(b, side_to_move, moves);
    if (moves.empty())
        return SearchResult{Move{0,0,0,0,PieceKind::NONE}, 0.0, 0};

    // Fresh search context (killer/history reset each call for determinism)
    SearchContext ctx;

    int64_t total_nodes = 0;
    Move best_mv = moves[0];
    double best_val = -INF;
    Side opp = opponent(side_to_move);

    // ── Iterative deepening: depth 1..depth ──
    for (int d = 1; d <= depth; ++d) {
        // Mutable copy of repetition table (reset per ID iteration for consistency
        // with previous behavior — rep state should reflect game history only)
        auto rep = repetition_table;

        double alpha = -INF, beta = INF;
        Move iter_best = moves[0];
        double iter_val = -INF;

        // Use TT PV from previous iteration for root move ordering
        uint64_t rk1, rk2;
        parse_hash_key(root_key, rk1, rk2);
        int root_rep = 0;
        {
            auto it = rep.find(root_key);
            if (it != rep.end()) root_rep = it->second;
        }
        uint8_t root_rb = static_cast<uint8_t>(std::min(root_rep, 3));
        Move root_tt_mv{0, 0, 0, 0, PieceKind::NONE};
        const TTEntry* root_tte = g_tt.probe(rk1, rk2, root_rb);
        if (root_tte) root_tt_mv = root_tte->best_move;

        auto ordered = order_moves(b, side_to_move, moves, root_tt_mv, ctx, 0);

        for (auto& mv : ordered) {
            UndoInfo u;
            make_move(b, mv, u);
            RepGuard guard(rep, b.board_hash(opp));

            double v = -negamax(b, opp, d - 1, -beta, -alpha,
                                rep, guard.key,
                                ply + 1, max_plies,
                                side_to_move, total_nodes,
                                ctx, 1);
            unmake_move(b, mv, u);

            if (v > iter_val) {
                iter_val = v;
                iter_best = mv;
            }
            alpha = std::max(alpha, v);
        }

        best_mv = iter_best;
        best_val = iter_val;
    }

    return SearchResult{best_mv, best_val, total_nodes};
}
