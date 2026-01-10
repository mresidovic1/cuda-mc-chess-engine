#ifndef PUCT_MCTS_H
#define PUCT_MCTS_H

#include "chess_types.cuh"
#include "search_config.h"
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <random>

// ============================================================================
// HEURISTIC-BASED PUCT MCTS (AlphaZero concepts WITHOUT neural networks)
// ============================================================================
//
// This engine combines:
// - PUCT selection (from AlphaZero)
// - Virtual loss for parallel search (from AlphaGo Zero)
// - Dirichlet noise for exploration (from AlphaZero)
// - Advanced tactical heuristics (history, killers, MVV-LVA, etc.)
// - GPU-batched playout evaluation
//
// PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
//
// Where P(s,a) comes from HEURISTIC move ordering, NOT neural networks!
//
// ============================================================================

// Move heuristic scoring for policy priors
struct MoveHeuristics {
    // Piece-square tables
    static int piece_square_value(int piece, int square, int color);
    
    // MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
    static int mvv_lva_score(Move move, const BoardState& state);
    
    // SEE (Static Exchange Evaluation)
    static int see_score(Move move, const BoardState& state);
    
    // Killer move detection
    static bool is_killer_move(Move move, int ply);
    
    // History heuristic score
    static int history_score(Move move, int color);
    
    // Tactical move detection
    static bool is_check(Move move, const BoardState& state);
    static bool is_capture(Move move);
    static bool is_promotion(Move move);
    static bool is_passed_pawn_push(Move move, const BoardState& state);
    
    // Combined heuristic score for move ordering
    static float heuristic_policy_prior(Move move, const BoardState& state, int ply);
};

// Killer moves table (indexed by ply)
struct KillerMoves {
    static constexpr int MAX_PLY = 128;
    static constexpr int KILLERS_PER_PLY = 2;
    
    Move killers[MAX_PLY][KILLERS_PER_PLY];
    
    void clear() {
        memset(killers, 0, sizeof(killers));
    }
    
    void add(Move move, int ply) {
        if (ply >= MAX_PLY) return;
        
        // Shift killers
        if (killers[ply][0] != move) {
            killers[ply][1] = killers[ply][0];
            killers[ply][0] = move;
        }
    }
    
    bool is_killer(Move move, int ply) const {
        if (ply >= MAX_PLY) return false;
        return killers[ply][0] == move || killers[ply][1] == move;
    }
};

// History heuristic table
struct HistoryTable {
    static constexpr int MAX_SQUARES = 64;
    
    int history[2][MAX_SQUARES][MAX_SQUARES];  // [color][from][to]
    
    void clear() {
        memset(history, 0, sizeof(history));
    }
    
    void update(Move move, int color, int depth) {
        int from = move & 0x3F;
        int to = (move >> 6) & 0x3F;
        
        // Depth-based bonus: deeper = more important
        history[color][from][to] += depth * depth;
        
        // Prevent overflow
        if (history[color][from][to] > 1000000) {
            age_history();
        }
    }
    
    int get(Move move, int color) const {
        int from = move & 0x3F;
        int to = (move >> 6) & 0x3F;
        return history[color][from][to];
    }
    
    void age_history() {
        for (int c = 0; c < 2; c++) {
            for (int f = 0; f < MAX_SQUARES; f++) {
                for (int t = 0; t < MAX_SQUARES; t++) {
                    history[c][f][t] /= 2;
                }
            }
        }
    }
};

// PUCT Configuration (AlphaZero-style but with heuristics)
struct PUCTConfig {
    // PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    float c_puct;                      // PUCT exploration constant (1.5-3.0 for heuristic)
    float c_puct_base;                 // Base for dynamic adjustment
    float c_puct_init;                 // Initial value
    bool use_dynamic_cpuct;            // Dynamic c_puct
    
    // Dirichlet noise for root exploration
    float dirichlet_alpha;             // Dirichlet concentration (0.3 for chess)
    float dirichlet_epsilon;           // Mixing weight (0.25)
    bool add_dirichlet_noise;          // Enable noise
    
    // Virtual loss for parallel search
    float virtual_loss;                // Virtual loss value (3.0)
    bool use_virtual_loss;             // Enable virtual loss
    
    // Temperature for move selection
    float temperature;                 // Move selection temperature
    
    // Search parameters
    int num_simulations;               // MCTS simulations per move
    int batch_size;                    // GPU batch size
    
    // FPU (First Play Urgency)
    float fpu_reduction;               // FPU reduction (0.25)
    bool use_fpu;                      // Use FPU
    
    // Playout configuration
    PlayoutMode playout_mode;          // GPU playout type
    int quiescence_depth;              // Quiescence search depth
    
    // Heuristic weights
    float capture_weight;              // Weight for captures in prior
    float check_weight;                // Weight for checks in prior
    float killer_weight;               // Weight for killer moves
    float history_weight;              // Weight for history heuristic
    
    // Output
    bool verbose;
    int info_interval;
    
    PUCTConfig()
        : c_puct(2.0f)                 // Higher than NN-based (1.5) because heuristics are noisier
        , c_puct_base(19652.0f)
        , c_puct_init(1.5f)
        , use_dynamic_cpuct(true)
        , dirichlet_alpha(0.3f)
        , dirichlet_epsilon(0.25f)
        , add_dirichlet_noise(true)
        , virtual_loss(3.0f)
        , use_virtual_loss(true)
        , temperature(1.0f)
        , num_simulations(1600)        // More sims needed without NN
        , batch_size(512)
        , fpu_reduction(0.25f)
        , use_fpu(true)
        , playout_mode(PlayoutMode::QUIESCENCE)  // Best tactical playout
        , quiescence_depth(4)
        , capture_weight(2.0f)
        , check_weight(1.5f)
        , killer_weight(1.2f)
        , history_weight(1.0f)
        , verbose(false)
        , info_interval(200)
    {}
    
    static PUCTConfig Fast() {
        PUCTConfig cfg;
        cfg.num_simulations = 400;
        cfg.batch_size = 256;
        return cfg;
    }
    
    static PUCTConfig Strong() {
        PUCTConfig cfg;
        cfg.num_simulations = 3200;
        cfg.batch_size = 512;
        return cfg;
    }
    
    static PUCTConfig Tactical() {
        PUCTConfig cfg;
        cfg.num_simulations = 1600;
        cfg.playout_mode = PlayoutMode::QUIESCENCE;
        cfg.quiescence_depth = 6;
        cfg.capture_weight = 3.0f;
        cfg.check_weight = 2.5f;
        return cfg;
    }
};

// PUCT MCTS Node
struct PUCTNode {
    BoardState state;
    Move move_from_parent;
    PUCTNode* parent;
    std::vector<std::unique_ptr<PUCTNode>> children;
    
    // Core statistics
    std::atomic<int> visits;           // N(s,a)
    std::atomic<float> total_value;    // W(s,a)
    float prior;                        // P(s,a) from HEURISTICS
    
    // Virtual loss
    std::atomic<int> virtual_losses;
    
    // Evaluation cache
    float value_estimate;               // From GPU playout
    bool evaluated;
    
    // Move info
    std::vector<Move> legal_moves;
    std::vector<float> move_priors;     // Heuristic priors
    bool moves_generated;
    bool is_terminal;
    
    // Depth tracking
    int depth;
    
    // Thread safety
    std::mutex expansion_mutex;
    
    PUCTNode(const BoardState& s, Move m = 0, PUCTNode* p = nullptr, float prior_prob = 1.0f)
        : state(s)
        , move_from_parent(m)
        , parent(p)
        , visits(0)
        , total_value(0.0f)
        , prior(prior_prob)
        , virtual_losses(0)
        , value_estimate(0.0f)
        , evaluated(false)
        , moves_generated(false)
        , is_terminal(false)
        , depth(p ? p->depth + 1 : 0)
    {}
    
    float Q() const {
        int n = visits.load(std::memory_order_relaxed);
        if (n == 0) return 0.0f;
        int vl = virtual_losses.load(std::memory_order_relaxed);
        float w = total_value.load(std::memory_order_relaxed);
        return w / (n + vl);
    }
    
    float puct_score(int parent_visits, float c_puct, float fpu_value) const {
        int n = visits.load(std::memory_order_relaxed);
        int vl = virtual_losses.load(std::memory_order_relaxed);
        
        float q_value = (n == 0 && vl == 0) ? fpu_value : Q();
        float u_value = c_puct * prior * std::sqrt((float)parent_visits) / (1.0f + n + vl);
        
        return q_value + u_value;
    }
    
    void add_virtual_loss(float loss = 1.0f) {
        virtual_losses.fetch_add((int)loss, std::memory_order_relaxed);
    }
    
    void remove_virtual_loss(float loss = 1.0f) {
        virtual_losses.fetch_sub((int)loss, std::memory_order_relaxed);
    }
    
    void update(float value) {
        visits.fetch_add(1, std::memory_order_relaxed);
        
        float old_value = total_value.load(std::memory_order_relaxed);
        float new_value;
        do {
            new_value = old_value + value;
        } while (!total_value.compare_exchange_weak(old_value, new_value, 
                                                     std::memory_order_relaxed));
    }
    
    bool is_leaf() const { return children.empty(); }
    bool is_fully_expanded() const {
        return moves_generated && (children.size() == legal_moves.size() || is_terminal);
    }
};

// PUCT MCTS Engine (Heuristic-based AlphaZero)
class PUCTEngine {
public:
    PUCTEngine(const PUCTConfig& config = PUCTConfig());
    ~PUCTEngine();
    
    // Initialize GPU resources
    void init();
    
    // Main search interface
    Move search(const BoardState& root_state);
    
    // Get move probabilities (visit count distribution)
    std::vector<float> get_move_probabilities(float temperature = 1.0f);
    
    // Get principal variation
    std::vector<Move> get_pv(int max_length = 10) const;
    
    // Statistics
    int get_total_visits() const { return root ? root->visits.load() : 0; }
    float get_root_value() const { return root ? root->Q() : 0.0f; }
    
    // Configuration
    void set_config(const PUCTConfig& cfg) { config = cfg; }
    const PUCTConfig& get_config() const { return config; }
    
private:
    PUCTConfig config;
    std::unique_ptr<PUCTNode> root;
    
    // Heuristic tables (shared across searches)
    KillerMoves killer_moves;
    HistoryTable history_table;
    
    // GPU resources
    BoardState* d_boards;
    float* d_results;
    BoardState* h_boards;
    float* h_results;
    int max_batch_size;
    
    // Random number generation
    std::mt19937 rng;
    
    // Statistics
    int total_simulations;
    
    // MCTS phases
    PUCTNode* select(PUCTNode* node);
    void expand_and_evaluate(PUCTNode* node);
    void expand_and_evaluate_batch(const std::vector<PUCTNode*>& nodes);
    void backpropagate(PUCTNode* node, float value);
    
    // Selection helpers
    PUCTNode* best_child_puct(PUCTNode* node);
    float get_dynamic_cpuct() const;
    
    // Move generation and heuristic evaluation
    void generate_legal_moves(PUCTNode* node);
    void compute_move_priors(PUCTNode* node);
    
    // Root handling
    void add_dirichlet_noise_to_root();
    Move select_move_by_temperature(float temperature);
    
    // GPU batch evaluation
    void evaluate_positions_gpu(const std::vector<PUCTNode*>& nodes);
    void ensure_batch_capacity(int size);
};

#endif // PUCT_MCTS_H
