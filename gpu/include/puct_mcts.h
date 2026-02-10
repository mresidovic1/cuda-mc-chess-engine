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
struct PUCTConfig;
struct PUCTNode;
class PUCTEngine;


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


// Killer Move Table - remembers moves that caused beta-cutoffs
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


// History Heuristic - tracks historically good moves across the search tree
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


// Continuation History - Stockfish-style multi-level history (move pairs) 
// https://www.chessprogramming.org/History_Heuristic
class ContinuationHistory {
public:
    ContinuationHistory() {
        clear();
    }

    void clear() {
        for (int c = 0; c < 2; c++) {
            for (int pt = 0; pt < 6; pt++) {
                for (int to = 0; to < 64; to++) {
                    history[c][pt][to] = 0;
                }
            }
        }
    }

    // Stockfish-style history update with gravity
    void update(Move move, int color, Piece piece_type, int bonus) {
        int to = (move >> 6) & 0x3F;
        int pt = piece_type % 6;

        // Apply bonus with gravity (values decay toward zero)
        // Formula: new = old + bonus - old * abs(bonus) / 512
        int old_value = history[color][pt][to];
        history[color][pt][to] = old_value + bonus - old_value * abs(bonus) / 512;

        // Clamp to reasonable range
        if (history[color][pt][to] > 16384) history[color][pt][to] = 16384;
        if (history[color][pt][to] < -16384) history[color][pt][to] = -16384;
    }

    int get(Move move, int color, Piece piece_type) const {
        int to = (move >> 6) & 0x3F;
        int pt = piece_type % 6;
        return history[color][pt][to];
    }

private:
    int history[2][6][64];  // [color][piece_type][to_square]
};


// Best Move Stability Tracker - root enhancement for early stopping
struct BestMoveTracker {
    Move current_best;
    int changes_count;
    int last_change_simulation;
    float stability_score;  // 1.0 = very stable, 0.0 = very unstable

    BestMoveTracker()
        : current_best(0), changes_count(0), last_change_simulation(0), stability_score(1.0f) {}

    void update(Move new_best, int current_simulation) {
        if (new_best != current_best && current_best != 0) {
            changes_count++;
            last_change_simulation = current_simulation;
            current_best = new_best;
        } else if (current_best == 0) {
            current_best = new_best;
        }

        // Decay stability (Stockfish uses 0.517 decay factor per iteration)
        stability_score *= 0.517f;

        // Boost stability if no recent changes
        int sims_since_change = current_simulation - last_change_simulation;
        if (sims_since_change > 1000) {
            stability_score = std::min(1.0f, stability_score + 0.1f);
        }
    }

    bool should_extend_search() const {
        // If best move is unstable, allocate more simulations
        return stability_score < 0.5f && changes_count > 3;
    }

    void reset() {
        current_best = 0;
        changes_count = 0;
        last_change_simulation = 0;
        stability_score = 1.0f;
    }
};


// Aspiration Window - dynamic search window for faster convergence
struct AspirationWindowState {
    float previous_best_q;
    float window_size;
    int fail_high_count;
    int fail_low_count;
    bool active;

    AspirationWindowState()
        : previous_best_q(0.5f), window_size(0.1f),
          fail_high_count(0), fail_low_count(0), active(false) {}

    void reset(float initial_window = 0.1f) {
        previous_best_q = 0.5f;
        window_size = initial_window;
        fail_high_count = 0;
        fail_low_count = 0;
        active = false;
    }

    void update(float best_q, float initial_window) {
        float q_lower = previous_best_q - window_size;
        float q_upper = previous_best_q + window_size;

        if (best_q < q_lower) {
            // Fail low: widen downward
            fail_low_count++;
            window_size *= 1.5f;
            previous_best_q = best_q;
        } else if (best_q > q_upper) {
            // Fail high: widen upward
            fail_high_count++;
            window_size *= 1.5f;
            previous_best_q = best_q;
        } else {
            // Success: narrow window
            previous_best_q = best_q;
            window_size = std::max(initial_window * 0.9f, initial_window * 0.5f);
            active = true;  // Window is stable now
        }
    }

    bool is_within_window(float q) const {
        float q_lower = previous_best_q - window_size;
        float q_upper = previous_best_q + window_size;
        return q >= q_lower && q <= q_upper;
    }
};


// Temperature Schedule - AlphaZero-inspired exploration control
// https://www.chessprogramming.org/Simulated_Annealing
struct TemperatureSchedule {
    float initial_temp;
    float final_temp;
    int warmup_sims;

    TemperatureSchedule()
        : initial_temp(1.0f), final_temp(0.01f), warmup_sims(10000) {}

    TemperatureSchedule(float initial, float final, int warmup)
        : initial_temp(initial), final_temp(final), warmup_sims(warmup) {}

    float get_temperature(int current_sim, int total_sims) const {
        if (current_sim < warmup_sims) {
            return initial_temp;
        }

        // Linear decay from initial to final
        if (total_sims <= warmup_sims) {
            return final_temp;
        }

        float progress = (float)(current_sim - warmup_sims) / (total_sims - warmup_sims);
        progress = std::min(1.0f, std::max(0.0f, progress));

        return initial_temp + progress * (final_temp - initial_temp);
    }
};


// Multi-PV Result - top-N best moves with PV lines
struct MultiPVResult {
    Move move;
    float q_value;
    int visits;
    std::vector<Move> pv;

    MultiPVResult() : move(0), q_value(0.0f), visits(0) {}

    MultiPVResult(Move m, float q, int v, const std::vector<Move>& p = {})
        : move(m), q_value(q), visits(v), pv(p) {}
};


// Node Dynamics Tracking - for adaptive exploration based on position stability
struct NodeDynamics {
    bool improving;                // Q-value increasing over recent visits
    float volatility;              // Stddev of recent Q-values
    float q_history[10];           // Last 10 Q-values for volatility calculation
    int q_history_idx;             // Circular buffer index
    int q_history_count;           // Number of entries in history
    int game_phase;                // Cached game phase (0-256)

    NodeDynamics()
        : improving(false), volatility(0.0f), q_history_idx(0), q_history_count(0), game_phase(128) {
        for (int i = 0; i < 10; i++) q_history[i] = 0.0f;
    }

    void update_q_history(float q) {
        q_history[q_history_idx] = q;
        q_history_idx = (q_history_idx + 1) % 10;
        if (q_history_count < 10) q_history_count++;

        // Update volatility (stddev of recent Q values)
        if (q_history_count >= 3) {
            float mean = 0.0f;
            for (int i = 0; i < q_history_count; i++) {
                mean += q_history[i];
            }
            mean /= q_history_count;

            float variance = 0.0f;
            for (int i = 0; i < q_history_count; i++) {
                float diff = q_history[i] - mean;
                variance += diff * diff;
            }
            variance /= q_history_count;
            volatility = std::sqrt(variance);
        }
    }

    void update_improving(float current_q, float parent_q) {
        improving = current_q > parent_q + 0.05f;
    }
};


struct PUCTNode {
    BoardState state;
    Move move_from_parent;
    PUCTNode* parent;
    std::vector<std::unique_ptr<PUCTNode>> children;

    // PUCT statistics (thread-safe)
    std::atomic<int> visits;
    std::atomic<float> total_value;  // Sum of values (Q * N)
    float prior_prob;                // Policy prior probability

    // RAVE (Rapid Action Value Estimation) - accelerates convergence via AMAF updates
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

    // Node dynamics for adaptive exploration
    NodeDynamics dynamics;

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

    // Adaptive PUCT score with dynamic c_puct (declaration only, defined after PUCTConfig)
    float adaptive_puct_score(int parent_visits, float base_c_puct, float fpu_value,
                             const PUCTConfig& config) const;

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

    // Virtual Loss - enables safe parallel tree exploration without thread collision
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


struct PUCTConfig {
    // Search parameters
    int num_simulations = 350000;  // Increased for better tactical strength
    int batch_size = 512;
    float temperature = 0.0f;  // 0 = greedy, 1 = proportional to visit count

    // Time-based search (optional, overrides num_simulations if set)
    bool use_time_limit = false;
    int time_limit_ms = 10000;  // 10 seconds default

    // PUCT parameters
    float c_puct = 1.5f;               // Exploration constant
    bool use_dynamic_cpuct = false;    // AlphaGo Zero dynamic c_puct
    float c_puct_base = 19652.0f;      // Base for dynamic c_puct
    float c_puct_init = 1.25f;         // Init for dynamic c_puct

    // First Play Urgency (FPU) - handles unvisited nodes more aggressively
    bool use_fpu = true;
    float fpu_reduction = 0.2f;        // FPU value = parent_Q - reduction

    // Dirichlet Noise - AlphaZero-style root exploration
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

    // Move Ordering Enhancements - MVV-LVA, killer moves, history heuristic
    float capture_weight = 3.0f;
    float check_weight = 5.0f;
    float killer_weight = 1.5f;
    float history_weight = 1.0f;
    bool use_continuation_history = true;
    float continuation_weight = 0.5f;

    // Adaptive Exploration - dynamic c_puct based on node visits
    bool use_adaptive_cpuct = false;       // Enable dynamic c_puct
    float depth_decay_factor = 0.05f;      // c_puct reduction per depth level
    float improving_bonus = 0.1f;          // Bonus when position improving
    float volatility_factor = 0.5f;        // Volatility multiplier
    float opening_phase_bonus = 0.2f;      // Extra exploration in opening
    float endgame_phase_reduction = 0.2f;  // Less exploration in endgame
    bool use_move_number_scaling = true;   // Scale exploration by move number

    // Root node enhancements
    bool use_aspiration_windows = false;       // Enable aspiration window search
    float aspiration_initial_window = 0.1f;    // Initial Q window size (+/-)
    float aspiration_widening_factor = 1.5f;   // How much to widen on fail
    int aspiration_min_sims = 5000;            // Min sims before using aspiration

    bool use_temperature_decay = false;        // Enable temperature decay
    float temperature_initial = 1.0f;          // Starting temperature
    float temperature_final = 0.01f;           // Final temperature
    int temperature_warmup_sims = 10000;       // Sims before decay starts

    bool use_multi_pv = false;                 // Enable multi-PV output
    int multi_pv_count = 3;                    // Number of PV lines to track

    // Playout configuration
    PlayoutMode playout_mode = PlayoutMode::QUIESCENCE;
    int quiescence_depth = 3;

    // Verbose output
    bool verbose = false;
    int info_interval = 0;  // Print info every N simulations (0 = disabled)

    // Preset configurations
    static PUCTConfig Quick() {
        PUCTConfig cfg;
        cfg.num_simulations = 50000;
        cfg.batch_size = 512;
        cfg.use_adaptive_cpuct = false;  // Keep simple for speed
        return cfg;
    }

    static PUCTConfig Normal() {
        PUCTConfig cfg;
        cfg.num_simulations = 350000;
        cfg.batch_size = 512;
        cfg.use_adaptive_cpuct = true;   // Enable adaptive features
        cfg.use_continuation_history = true;
        cfg.use_aspiration_windows = true;
        cfg.use_temperature_decay = true;
        return cfg;
    }

    static PUCTConfig Strong() {
        PUCTConfig cfg;
        cfg.num_simulations = 1000000;
        cfg.batch_size = 1024;
        cfg.use_rave = true;
        cfg.use_adaptive_cpuct = true;   // Enable all features
        cfg.use_continuation_history = true;
        cfg.use_aspiration_windows = true;
        cfg.use_temperature_decay = true;
        cfg.use_multi_pv = true;
        cfg.multi_pv_count = 5;
        return cfg;
    }

    // Optimized preset with adaptive exploration
    static PUCTConfig Adaptive() {
        PUCTConfig cfg;
        cfg.num_simulations = 350000;
        cfg.batch_size = 512;

        // Enable all adaptive features
        cfg.use_continuation_history = true;
        cfg.use_adaptive_cpuct = true;
        cfg.use_move_number_scaling = true;

        // Tuned parameters for adaptive exploration
        cfg.depth_decay_factor = 0.05f;
        cfg.improving_bonus = 0.15f;
        cfg.volatility_factor = 0.6f;
        cfg.opening_phase_bonus = 0.25f;
        cfg.endgame_phase_reduction = 0.2f;

        return cfg;
    }

    // Optimized preset with all features enabled
    static PUCTConfig Advanced() {
        PUCTConfig cfg;
        cfg.num_simulations = 350000;
        cfg.batch_size = 4096;

        // Move ordering features
        cfg.use_continuation_history = true;
        cfg.continuation_weight = 0.5f;

        // Adaptive exploration features
        cfg.use_adaptive_cpuct = true;
        cfg.use_move_number_scaling = true;
        cfg.depth_decay_factor = 0.05f;
        cfg.improving_bonus = 0.15f;
        cfg.volatility_factor = 0.6f;
        cfg.opening_phase_bonus = 0.25f;
        cfg.endgame_phase_reduction = 0.2f;

        // Root enhancements
        cfg.use_aspiration_windows = true;
        cfg.aspiration_initial_window = 0.1f;
        cfg.aspiration_widening_factor = 1.5f;
        cfg.aspiration_min_sims = 5000;

        cfg.use_temperature_decay = true;
        cfg.temperature_initial = 1.0f;
        cfg.temperature_final = 0.01f;
        cfg.temperature_warmup_sims = 10000;

        cfg.use_multi_pv = true;
        cfg.multi_pv_count = 3;

        return cfg;
    }
};


// Adaptive PUCT Score - dynamic exploration based on node visits
// Uses AlphaGo Zero formula: log((1 + N + c_base) / c_base) + c_init
inline float PUCTNode::adaptive_puct_score(int parent_visits, float base_c_puct,
                                           float fpu_value, const PUCTConfig& config) const {
    int n = visits.load(std::memory_order_relaxed);

    // Calculate adaptive c_puct
    float c_puct = base_c_puct;

    if (config.use_adaptive_cpuct && n > 0) {
        // AlphaGo Zero base formula: log((1 + N + c_base) / c_base)
        float c_base_term = std::log((1.0f + parent_visits + config.c_puct_base) / config.c_puct_base);

        // Depth-based reduction (deeper = less exploration)
        float depth_factor = 1.0f / (1.0f + depth * config.depth_decay_factor);

        // Improving bonus
        float improving_factor = dynamics.improving ? config.improving_bonus : -config.improving_bonus;

        // Volatility bonus (more volatile = more exploration)
        float volatility_bonus = dynamics.volatility * config.volatility_factor;

        // Phase-based adjustment
        float phase_factor = 1.0f;
        int phase = dynamics.game_phase;
        if (phase > 200) {
            // Opening: explore more
            phase_factor = 1.0f + config.opening_phase_bonus;
        } else if (phase < 50) {
            // Endgame: explore less
            phase_factor = 1.0f - config.endgame_phase_reduction;
        }

        c_puct = (c_base_term + config.c_puct_init + improving_factor + volatility_bonus)
                 * depth_factor * phase_factor;

        // Clamp to reasonable range
        c_puct = std::max(0.5f, std::min(c_puct, 5.0f));
    }

    if (n == 0) {
        // First Play Urgency
        float u = c_puct * prior_prob * std::sqrt((float)parent_visits);
        return fpu_value + u;
    }

    float q = Q();
    float u = c_puct * prior_prob * std::sqrt((float)parent_visits) / (1.0f + n);
    return q + u;
}


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

    // Multi-PV support
    std::vector<MultiPVResult> get_multi_pv(int num_pvs = 3) const;

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
    ContinuationHistory continuation_history;

    // Best move stability tracking
    BestMoveTracker stability_tracker;

    // Aspiration window state
    AspirationWindowState aspiration_window;

    // Temperature schedule
    TemperatureSchedule temp_schedule;

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

    // Helper methods
    Piece get_moved_piece(const BoardState& state, Move move) const;
    void update_history_tables(Move move, const BoardState& state, int depth);
    Move get_best_move() const;
    std::string move_to_string(Move move) const;

    // Adaptive exploration helpers
    void update_node_dynamics(PUCTNode* node);
    float compute_move_number_factor(int move_idx, int total_moves) const;

    // Root enhancement helpers
    void initialize_aspiration_window();
    void update_aspiration_window(float best_q);
    bool should_use_aspiration_window(int simulations_done) const;
    PUCTNode* select_child_in_window(PUCTNode* node, float q_lower, float q_upper);
    float get_current_temperature(int simulations_done) const;

    // Memory management
    void ensure_batch_capacity(int size);
};

#endif
