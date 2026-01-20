#ifndef PUCT_MCTS_H
#define PUCT_MCTS_H

#include "chess_types.cuh"
#include "search_config.h"
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <random>
#include <unordered_map>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Forward declarations
struct PUCTNode;
class PUCTEngine;

// ============================================================================
// MOVE HEURISTICS (for policy priors)
// ============================================================================

class MoveHeuristics {
public:
    // Piece-square value estimation
    static int piece_square_value(int piece, int square, int color);

    // MVV-LVA (Most Valuable Victim - Least Valuable Attacker) score
    static int mvv_lva_score(Move move, const BoardState& state);

    // SEE (Static Exchange Evaluation) score
    static int see_score(Move move, const BoardState& state);

    // Check if move is a killer move
    static bool is_killer_move(Move move, int ply);

    // History heuristic score
    static int history_score(Move move, int color);

    // Check if move gives check
    static bool is_check(Move move, const BoardState& state);

    // Move type checks
    static bool is_capture(Move move);
    static bool is_promotion(Move move);
    static bool is_passed_pawn_push(Move move, const BoardState& state);

    // Compute heuristic policy prior for a move
    static float heuristic_policy_prior(Move move, const BoardState& state, int ply,
                                        float capture_weight = 3.0f, float check_weight = 5.0f);
};

// ============================================================================
// KILLER MOVE TABLE
// ============================================================================

class KillerMoveTable {
public:
    KillerMoveTable() {
        clear();
    }

    void clear() {
        for (int i = 0; i < MAX_PLY; i++) {
            killers[i][0] = 0;
            killers[i][1] = 0;
        }
    }

    void add(Move move, int ply) {
        if (ply >= MAX_PLY) return;
        if (move != killers[ply][0]) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = move;
        }
    }

    bool is_killer(Move move, int ply) const {
        if (ply >= MAX_PLY) return false;
        return move == killers[ply][0] || move == killers[ply][1];
    }

private:
    static const int MAX_PLY = 64;
    Move killers[MAX_PLY][2];
};

// ============================================================================
// HISTORY HEURISTIC TABLE
// ============================================================================

class HistoryTable {
public:
    HistoryTable() {
        clear();
    }

    void clear() {
        for (int c = 0; c < 2; c++) {
            for (int from = 0; from < 64; from++) {
                for (int to = 0; to < 64; to++) {
                    history[c][from][to] = 0;
                }
            }
        }
    }

    void update(Move move, int color, int depth) {
        int from = move & 0x3F;
        int to = (move >> 6) & 0x3F;
        history[color][from][to] += depth * depth;

        // Cap at maximum
        if (history[color][from][to] > 10000) {
            history[color][from][to] = 10000;
        }
    }

    int get(Move move, int color) const {
        int from = move & 0x3F;
        int to = (move >> 6) & 0x3F;
        return history[color][from][to];
    }

private:
    int history[2][64][64];
};

// ============================================================================
// PUCT NODE (AlphaZero-style MCTS node)
// ============================================================================

struct PUCTNode {
    BoardState state;
    Move move_from_parent;
    PUCTNode* parent;
    std::vector<std::unique_ptr<PUCTNode>> children;

    // PUCT statistics (thread-safe)
    std::atomic<int> visits;
    std::atomic<float> total_value;  // Sum of values (Q * N)
    float prior_prob;                // Policy prior probability

    // RAVE (Rapid Action Value Estimation) statistics
    std::atomic<int> amaf_visits;
    std::atomic<float> amaf_value;

    // Move generation
    std::vector<Move> legal_moves;
    std::vector<float> move_priors;
    bool moves_generated;

    // Node state
    bool is_terminal;
    bool evaluated;
    float value_estimate;  // From GPU evaluation
    int depth;

    // Thread safety
    std::mutex expansion_mutex;

    // Constructor
    PUCTNode(const BoardState& s, Move m = 0, PUCTNode* p = nullptr, float prior = 1.0f)
        : state(s), move_from_parent(m), parent(p), prior_prob(prior),
          visits(0), total_value(0.0f), amaf_visits(0), amaf_value(0.0f),
          moves_generated(false), is_terminal(false), evaluated(false),
          value_estimate(0.0f), depth(p ? p->depth + 1 : 0) {}

    // Q value (average value)
    float Q() const {
        int n = visits.load(std::memory_order_relaxed);
        if (n == 0) return 0.0f;
        return total_value.load(std::memory_order_relaxed) / n;
    }

    // AMAF Q value
    float Q_amaf() const {
        int n = amaf_visits.load(std::memory_order_relaxed);
        if (n == 0) return 0.0f;
        return amaf_value.load(std::memory_order_relaxed) / n;
    }

    // PUCT score for selection
    float puct_score(int parent_visits, float c_puct, float fpu_value = 0.0f) const {
        int n = visits.load(std::memory_order_relaxed);

        if (n == 0) {
            // First Play Urgency
            float u = c_puct * prior_prob * std::sqrt((float)parent_visits);
            return fpu_value + u;
        }

        float q = Q();
        float u = c_puct * prior_prob * std::sqrt((float)parent_visits) / (1.0f + n);
        return q + u;
    }

    // RAVE-enhanced PUCT score
    float rave_puct_score(int parent_visits, float c_puct, float fpu_value,
                          float rave_k, bool use_rave) const {
        if (!use_rave) {
            return puct_score(parent_visits, c_puct, fpu_value);
        }

        int n = visits.load(std::memory_order_relaxed);
        int n_amaf = amaf_visits.load(std::memory_order_relaxed);

        if (n == 0) {
            float u = c_puct * prior_prob * std::sqrt((float)parent_visits);
            return fpu_value + u;
        }

        // RAVE mixing: Q_mixed = beta * Q_amaf + (1 - beta) * Q
        // beta = n_amaf / (n + n_amaf + n * n_amaf * k)
        float beta = 0.0f;
        if (n_amaf > 0) {
            beta = (float)n_amaf / (n + n_amaf + n * n_amaf * rave_k);
        }

        float q = Q();
        float q_amaf = Q_amaf();
        float q_mixed = beta * q_amaf + (1.0f - beta) * q;

        float u = c_puct * prior_prob * std::sqrt((float)parent_visits) / (1.0f + n);
        return q_mixed + u;
    }

    // Update node statistics
    void update(float value) {
        visits.fetch_add(1, std::memory_order_relaxed);
        // C++17 compatible: use load/store loop for atomic float
        float old_val = total_value.load(std::memory_order_relaxed);
        while (!total_value.compare_exchange_weak(old_val, old_val + value, 
                                                   std::memory_order_relaxed)) {}
    }

    // Update AMAF statistics
    void update_amaf(float value) {
        amaf_visits.fetch_add(1, std::memory_order_relaxed);
        float old_val = amaf_value.load(std::memory_order_relaxed);
        while (!amaf_value.compare_exchange_weak(old_val, old_val + value,
                                                  std::memory_order_relaxed)) {}
    }

    // Virtual loss for parallel MCTS
    void add_virtual_loss(float loss) {
        visits.fetch_add(1, std::memory_order_relaxed);
        float old_val = total_value.load(std::memory_order_relaxed);
        while (!total_value.compare_exchange_weak(old_val, old_val - loss,
                                                   std::memory_order_relaxed)) {}
    }

    void remove_virtual_loss(float loss) {
        visits.fetch_sub(1, std::memory_order_relaxed);
        float old_val = total_value.load(std::memory_order_relaxed);
        while (!total_value.compare_exchange_weak(old_val, old_val + loss,
                                                   std::memory_order_relaxed)) {}
    }

    // Check if fully expanded
    bool is_fully_expanded() const {
        return moves_generated && (children.size() == legal_moves.size() || is_terminal);
    }

    // Check if leaf
    bool is_leaf() const {
        return children.empty();
    }
};

// ============================================================================
// PUCT ENGINE CONFIGURATION
// ============================================================================

struct PUCTConfig {
    // Search parameters
    int num_simulations = 10000;
    int batch_size = 512;
    float temperature = 0.0f;  // 0 = greedy, 1 = proportional to visit count

    // PUCT parameters
    float c_puct = 1.5f;               // Exploration constant
    bool use_dynamic_cpuct = false;    // AlphaGo Zero dynamic c_puct
    float c_puct_base = 19652.0f;      // Base for dynamic c_puct
    float c_puct_init = 1.25f;         // Init for dynamic c_puct

    // First Play Urgency (FPU)
    bool use_fpu = true;
    float fpu_reduction = 0.2f;        // FPU value = parent_Q - reduction

    // Dirichlet noise (for exploration at root)
    bool add_dirichlet_noise = false;
    float dirichlet_alpha = 0.3f;
    float dirichlet_epsilon = 0.25f;

    // Virtual loss (for parallel MCTS)
    bool use_virtual_loss = true;
    float virtual_loss = 3.0f;

    // RAVE (Rapid Action Value Estimation)
    bool use_rave = false;
    float rave_k = 0.00001f;           // RAVE mixing parameter
    int rave_update_depth = 10;        // Max depth for AMAF updates

    // Move ordering heuristics
    float capture_weight = 3.0f;
    float check_weight = 5.0f;
    float killer_weight = 1.5f;
    float history_weight = 1.0f;

    // Playout configuration
    PlayoutMode playout_mode = PlayoutMode::QUIESCENCE;
    int quiescence_depth = 3;

    // Verbose output
    bool verbose = false;
    int info_interval = 0;  // Print info every N simulations (0 = disabled)

    // Preset configurations
    static PUCTConfig Quick() {
        PUCTConfig cfg;
        cfg.num_simulations = 1000;
        cfg.batch_size = 256;
        return cfg;
    }

    static PUCTConfig Normal() {
        PUCTConfig cfg;
        cfg.num_simulations = 10000;
        cfg.batch_size = 512;
        return cfg;
    }

    static PUCTConfig Strong() {
        PUCTConfig cfg;
        cfg.num_simulations = 50000;
        cfg.batch_size = 1024;
        cfg.use_rave = true;
        return cfg;
    }
};

// ============================================================================
// PUCT ENGINE (Heuristic AlphaZero-style MCTS)
// ============================================================================

class PUCTEngine {
public:
    PUCTEngine(const PUCTConfig& cfg = PUCTConfig());
    ~PUCTEngine();

    // Initialize GPU resources
    void init();

    // Main search function
    Move search(const BoardState& root_state);

    // Get statistics
    int get_total_visits() const { return root ? root->visits.load() : 0; }
    float get_root_value() const { return root ? root->Q() : 0.0f; }

    // Get principal variation
    std::vector<Move> get_pv(int max_length = 10) const;

    // Get move probabilities (for training data generation)
    std::vector<float> get_move_probabilities(float temperature = 1.0f);

private:
    // Configuration
    PUCTConfig config;

    // Tree
    std::unique_ptr<PUCTNode> root;

    // GPU resources
    BoardState* d_boards;
    float* d_results;
    BoardState* h_boards;
    float* h_results;
    int max_batch_size;

    // Move ordering heuristics
    KillerMoveTable killer_moves;
    HistoryTable history_table;

    // Random number generator
    std::mt19937 rng;

    // Statistics
    int total_simulations;

    // MCTS phases
    PUCTNode* select(PUCTNode* node);
    PUCTNode* best_child_puct(PUCTNode* node);
    void expand_and_evaluate(PUCTNode* node);
    void expand_and_evaluate_batch(const std::vector<PUCTNode*>& nodes);

    // RAVE selection
    PUCTNode* select_and_track(PUCTNode* node, std::vector<Move>& tree_moves);
    void backpropagate_with_rave(PUCTNode* node, float value, const std::vector<Move>& tree_moves);
    void update_amaf_stats(PUCTNode* node, const std::vector<Move>& sim_moves, float value);

    // GPU evaluation
    void evaluate_positions_gpu(const std::vector<PUCTNode*>& nodes);

    // Move generation and priors
    void generate_legal_moves(PUCTNode* node);
    void compute_move_priors(PUCTNode* node);

    // Root node utilities
    void add_dirichlet_noise_to_root();
    Move select_move_by_temperature(float temperature);

    // Dynamic c_puct
    float get_dynamic_cpuct() const;

    // Memory management
    void ensure_batch_capacity(int size);
};

#endif
