#include "../include/puct_mcts.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>

// External GPU playout launchers
extern "C" void launch_quiescence_playout(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    unsigned int seed,
    int max_q_depth,
    cudaStream_t stream
);

extern "C" void launch_eval_playout(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    unsigned int seed,
    cudaStream_t stream
);

extern "C" void launch_static_eval(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    cudaStream_t stream
);

// External CPU move generation
namespace cpu_movegen {
    extern int generate_legal_moves_cpu(const BoardState* pos, Move* moves);
    extern void make_move_cpu(BoardState* pos, Move m);
    extern bool in_check_cpu(const BoardState* pos);
}

// ============================================================================
// HEURISTIC MOVE EVALUATION
// ============================================================================

// Simplified piece-square tables for move ordering
static const int PST_BONUS[6] = {
    10,   // Pawn
    30,   // Knight
    30,   // Bishop
    20,   // Rook
    10,   // Queen
    0     // King
};

int MoveHeuristics::piece_square_value(int piece, int square, int color) {
    // Simplified: bonus for advancing pieces
    int rank = square / 8;
    if (color == BLACK) rank = 7 - rank;
    
    return PST_BONUS[piece] * rank / 7;
}

int MoveHeuristics::mvv_lva_score(Move move, const BoardState& state) {
    static const int piece_values[6] = {100, 320, 330, 500, 900, 20000};
    
    if (!is_capture(move)) return 0;
    
    int to_sq = (move >> 6) & 0x3F;
    int from_sq = move & 0x3F;
    
    // Find captured piece
    int victim_value = 0;
    for (int piece = 0; piece < 6; piece++) {
        if (state.pieces[state.side_to_move ^ 1][piece] & (1ULL << to_sq)) {
            victim_value = piece_values[piece];
            break;
        }
    }
    
    // Find attacker piece
    int attacker_value = 0;
    for (int piece = 0; piece < 6; piece++) {
        if (state.pieces[state.side_to_move][piece] & (1ULL << from_sq)) {
            attacker_value = piece_values[piece];
            break;
        }
    }
    
    // MVV-LVA: prioritize high-value victims with low-value attackers
    return (victim_value * 10) - (attacker_value / 10);
}

int MoveHeuristics::see_score(Move move, const BoardState& state) {
    // Simplified SEE - just MVV-LVA for now
    return mvv_lva_score(move, state);
}

bool MoveHeuristics::is_killer_move(Move move, int ply) {
    // Will be checked against killer_moves table in engine
    return false;  // Placeholder
}

int MoveHeuristics::history_score(Move move, int color) {
    // Will be checked against history table in engine
    return 0;  // Placeholder
}

bool MoveHeuristics::is_check(Move move, const BoardState& state) {
    // Make move on copy and check if opponent is in check
    BoardState test_state = state;
    cpu_movegen::make_move_cpu(&test_state, move);
    return cpu_movegen::in_check_cpu(&test_state);
}

bool MoveHeuristics::is_capture(Move move) {
    int flags = (move >> 12) & 0xF;
    return (flags == MOVE_CAPTURE) || 
           (flags == MOVE_EP_CAPTURE) ||
           (flags >= MOVE_PROMO_CAP_N && flags <= MOVE_PROMO_CAP_Q);
}

bool MoveHeuristics::is_promotion(Move move) {
    int flags = (move >> 12) & 0xF;
    return flags >= MOVE_PROMO_N;
}

bool MoveHeuristics::is_passed_pawn_push(Move move, const BoardState& state) {
    // Simplified: check if pawn push to 7th rank
    int from_sq = move & 0x3F;
    int to_sq = (move >> 6) & 0x3F;
    
    if (state.pieces[state.side_to_move][PAWN] & (1ULL << from_sq)) {
        int to_rank = to_sq / 8;
        if (state.side_to_move == WHITE && to_rank == 6) return true;
        if (state.side_to_move == BLACK && to_rank == 1) return true;
    }
    
    return false;
}

float MoveHeuristics::heuristic_policy_prior(Move move, const BoardState& state, int ply,
                                              float capture_weight, float check_weight) {
    float score = 1.0f;  // Base score
    
    // Tactical bonuses
    if (is_capture(move)) {
        int mvv_lva = mvv_lva_score(move, state);
        score += (mvv_lva / 100.0f) * capture_weight;  // Use config weight
    }
    
    if (is_check(move, state)) {
        score += check_weight;  // Use config weight (CRITICAL for mate detection!)
    }
    
    if (is_promotion(move)) {
        score += 8.0f;  // Very strong bonus for promotions
    }
    
    if (is_passed_pawn_push(move, state)) {
        score += 2.0f;
    }
    
    // Ensure positive
    return std::max(0.1f, score);
}

// ============================================================================
// PUCT ENGINE IMPLEMENTATION
// ============================================================================

PUCTEngine::PUCTEngine(const PUCTConfig& cfg)
    : config(cfg)
    , root(nullptr)
    , d_boards(nullptr)
    , d_results(nullptr)
    , h_boards(nullptr)
    , h_results(nullptr)
    , max_batch_size(0)
    , rng(std::chrono::steady_clock::now().time_since_epoch().count())
    , total_simulations(0)
{
    killer_moves.clear();
    history_table.clear();
}

PUCTEngine::~PUCTEngine() {
    if (d_boards) cudaFree(d_boards);
    if (d_results) cudaFree(d_results);
    if (h_boards) cudaFreeHost(h_boards);
    if (h_results) cudaFreeHost(h_results);
}

void PUCTEngine::init() {
    // Allocate GPU memory
    max_batch_size = config.batch_size;
    
    CUDA_CHECK(cudaMalloc(&d_boards, max_batch_size * sizeof(BoardState)));
    CUDA_CHECK(cudaMalloc(&d_results, max_batch_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_boards, max_batch_size * sizeof(BoardState)));
    CUDA_CHECK(cudaMallocHost(&h_results, max_batch_size * sizeof(float)));
    
    std::cout << "PUCT Engine initialized (heuristic-based, NO neural networks)" << std::endl;
    std::cout << "Batch size: " << max_batch_size << std::endl;
}

Move PUCTEngine::search(const BoardState& root_state) {
    // Create root node
    root = std::make_unique<PUCTNode>(root_state);
    total_simulations = 0;
    
    // Generate legal moves
    generate_legal_moves(root.get());
    
    if (root->legal_moves.empty()) {
        return 0;  // No legal moves
    }
    
    if (root->legal_moves.size() == 1) {
        return root->legal_moves[0];  // Forced move
    }
    
    // Compute heuristic priors for root moves
    compute_move_priors(root.get());
    
    // Add Dirichlet noise for exploration
    if (config.add_dirichlet_noise) {
        add_dirichlet_noise_to_root();
    }
    
    // Main MCTS loop
    int simulations_done = 0;
    
    while (simulations_done < config.num_simulations) {
        int batch_size = std::min(config.batch_size, config.num_simulations - simulations_done);
        std::vector<PUCTNode*> leaf_nodes;
        leaf_nodes.reserve(batch_size);
        
        // SELECTION PHASE: Traverse tree using PUCT
        for (int i = 0; i < batch_size; i++) {
            PUCTNode* node = select(root.get());
            
            if (node) {
                leaf_nodes.push_back(node);
                
                // Add virtual loss
                if (config.use_virtual_loss) {
                    PUCTNode* current = node;
                    while (current) {
                        current->add_virtual_loss(config.virtual_loss);
                        current = current->parent;
                    }
                }
            }
        }
        
        if (leaf_nodes.empty()) break;
        
        // EXPANSION & EVALUATION PHASE: GPU batch evaluation
        expand_and_evaluate_batch(leaf_nodes);
        
        // BACKPROPAGATION PHASE
        for (PUCTNode* node : leaf_nodes) {
            float value = node->value_estimate;
            
            // Remove virtual loss and backpropagate
            PUCTNode* current = node;
            while (current) {
                if (config.use_virtual_loss) {
                    current->remove_virtual_loss(config.virtual_loss);
                }
                
                current->update(value);
                value = -value;  // Flip for opponent
                
                // Update history heuristic for good moves
                if (current->parent && value > 0.3f) {
                    history_table.update(current->move_from_parent, 
                                       current->state.side_to_move ^ 1, 
                                       current->depth);
                }
                
                current = current->parent;
            }
        }
        
        simulations_done += leaf_nodes.size();
        total_simulations += leaf_nodes.size();
        
        // Print progress
        if (config.verbose && config.info_interval > 0 && 
            simulations_done % config.info_interval == 0) {
            std::cout << "info sims " << simulations_done 
                      << " / " << config.num_simulations
                      << " value " << root->Q()
                      << " nps " << (simulations_done * 1000 / 
                                    (std::chrono::steady_clock::now().time_since_epoch().count() % 1000000))
                      << std::endl;
        }
    }
    
    // Select best move
    Move best_move = select_move_by_temperature(config.temperature);
    
    if (config.verbose) {
        std::cout << "Best move: " << ((best_move & 0x3F) % 8) << "," << ((best_move & 0x3F) / 8)
                  << " -> " << (((best_move >> 6) & 0x3F) % 8) << "," << (((best_move >> 6) & 0x3F) / 8)
                  << " | Value: " << root->Q()
                  << " | Visits: " << root->visits.load()
                  << std::endl;
    }
    
    return best_move;
}

// ============================================================================
// SELECTION PHASE (PUCT)
// ============================================================================

PUCTNode* PUCTEngine::select(PUCTNode* node) {
    while (true) {
        if (node->is_terminal) {
            return node;
        }
        
        if (!node->evaluated) {
            return node;
        }
        
        if (!node->is_fully_expanded()) {
            // Thread-safe expansion
            std::lock_guard<std::mutex> lock(node->expansion_mutex);
            
            if (!node->is_fully_expanded() && !node->is_terminal) {
                int child_idx = node->children.size();
                if (child_idx < (int)node->legal_moves.size()) {
                    Move move = node->legal_moves[child_idx];
                    float prior = node->move_priors[child_idx];
                    
                    BoardState child_state = node->state;
                    cpu_movegen::make_move_cpu(&child_state, move);
                    
                    auto child = std::make_unique<PUCTNode>(child_state, move, node, prior);
                    PUCTNode* child_ptr = child.get();
                    
                    // IMMEDIATE CHECKMATE DETECTION - critical for mate-in-1!
                    Move temp_moves[MAX_MOVES];
                    int num_child_moves = cpu_movegen::generate_legal_moves_cpu(&child_state, temp_moves);
                    if (num_child_moves == 0 && cpu_movegen::in_check_cpu(&child_state)) {
                        // Opponent has no legal moves and is in check = CHECKMATE!
                        child_ptr->is_terminal = true;
                        child_ptr->value_estimate = 1.0f;  // WIN for the player who made this move!
                        child_ptr->evaluated = true;
                        child_ptr->moves_generated = true;
                    }
                    
                    node->children.push_back(std::move(child));
                    
                    return child_ptr;
                }
            }
        }
        
        // Select best child using PUCT
        node = best_child_puct(node);
        
        if (!node) break;
    }
    
    return node;
}

PUCTNode* PUCTEngine::best_child_puct(PUCTNode* node) {
    if (node->children.empty()) return nullptr;
    
    float c_puct = config.use_dynamic_cpuct ? get_dynamic_cpuct() : config.c_puct;
    int parent_visits = node->visits.load(std::memory_order_relaxed);
    
    // FPU: First Play Urgency
    float fpu_value = config.use_fpu ? (node->Q() - config.fpu_reduction) : 0.0f;
    
    PUCTNode* best_child = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    
    for (auto& child : node->children) {
        float score = child->puct_score(parent_visits, c_puct, fpu_value);
        
        if (score > best_score) {
            best_score = score;
            best_child = child.get();
        }
    }
    
    return best_child;
}

float PUCTEngine::get_dynamic_cpuct() const {
    // AlphaGo Zero formula: c_puct = log((1 + N + c_base) / c_base) + c_init
    int N = root ? root->visits.load(std::memory_order_relaxed) : 0;
    return std::log((1.0f + N + config.c_puct_base) / config.c_puct_base) + config.c_puct_init;
}

// ============================================================================
// EXPANSION & EVALUATION PHASE
// ============================================================================

void PUCTEngine::expand_and_evaluate(PUCTNode* node) {
    generate_legal_moves(node);
    
    if (node->is_terminal) {
        // Terminal evaluation
        if (node->legal_moves.empty()) {
            if (cpu_movegen::in_check_cpu(&node->state)) {
                node->value_estimate = -1.0f;  // Checkmate
            } else {
                node->value_estimate = 0.0f;   // Stalemate
            }
        } else {
            node->value_estimate = 0.0f;  // Draw
        }
        node->evaluated = true;
        return;
    }
    
    // Compute heuristic priors
    compute_move_priors(node);
    
    // GPU evaluation via playout
    std::vector<PUCTNode*> batch = {node};
    evaluate_positions_gpu(batch);
}

void PUCTEngine::expand_and_evaluate_batch(const std::vector<PUCTNode*>& nodes) {
    if (nodes.empty()) return;
    
    // Generate moves and separate terminal/non-terminal
    std::vector<PUCTNode*> non_terminal;
    
    for (PUCTNode* node : nodes) {
        if (!node->moves_generated) {
            generate_legal_moves(node);
        }
        
        if (node->is_terminal) {
            if (node->legal_moves.empty()) {
                node->value_estimate = cpu_movegen::in_check_cpu(&node->state) ? -1.0f : 0.0f;
            } else {
                node->value_estimate = 0.0f;
            }
            node->evaluated = true;
        } else {
            compute_move_priors(node);
            non_terminal.push_back(node);
        }
    }
    
    if (!non_terminal.empty()) {
        evaluate_positions_gpu(non_terminal);
    }
}

void PUCTEngine::evaluate_positions_gpu(const std::vector<PUCTNode*>& nodes) {
    int count = nodes.size();
    if (count == 0) return;
    
    ensure_batch_capacity(count);
    
    // Copy to pinned memory
    for (int i = 0; i < count; i++) {
        h_boards[i] = nodes[i]->state;
    }
    
    // Transfer to GPU
    CUDA_CHECK(cudaMemcpy(d_boards, h_boards, count * sizeof(BoardState), cudaMemcpyHostToDevice));
    
    // Run GPU playout based on mode
    unsigned int seed = rng();
    
    switch (config.playout_mode) {
        case PlayoutMode::QUIESCENCE:
            launch_quiescence_playout(d_boards, d_results, count, seed, config.quiescence_depth, 0);
            break;
        case PlayoutMode::EVAL_HYBRID:
            launch_eval_playout(d_boards, d_results, count, seed, 0);
            break;
        case PlayoutMode::STATIC_EVAL:
            launch_static_eval(d_boards, d_results, count, 0);
            break;
        default:
            launch_quiescence_playout(d_boards, d_results, count, seed, config.quiescence_depth, 0);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Transfer results back
    CUDA_CHECK(cudaMemcpy(h_results, d_results, count * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Assign results
    for (int i = 0; i < count; i++) {
        nodes[i]->value_estimate = h_results[i];
        nodes[i]->evaluated = true;
    }
}

// ============================================================================
// MOVE GENERATION & HEURISTIC PRIORS
// ============================================================================

void PUCTEngine::generate_legal_moves(PUCTNode* node) {
    if (node->moves_generated) return;
    
    Move moves[MAX_MOVES];
    int num_moves = cpu_movegen::generate_legal_moves_cpu(&node->state, moves);
    
    node->legal_moves.clear();
    node->legal_moves.reserve(num_moves);
    for (int i = 0; i < num_moves; i++) {
        node->legal_moves.push_back(moves[i]);
    }
    
    node->is_terminal = (num_moves == 0) || (node->state.halfmove >= 100);
    node->moves_generated = true;
}

void PUCTEngine::compute_move_priors(PUCTNode* node) {
    if (node->legal_moves.empty()) return;
    
    node->move_priors.resize(node->legal_moves.size());
    
    float total_score = 0.0f;
    
    // Compute heuristic score for each move
    for (size_t i = 0; i < node->legal_moves.size(); i++) {
        Move move = node->legal_moves[i];
        float score = MoveHeuristics::heuristic_policy_prior(move, node->state, node->depth,
                                                              config.capture_weight, config.check_weight);
        
        // Add killer move bonus
        if (killer_moves.is_killer(move, node->depth)) {
            score *= config.killer_weight;
        }
        
        // Add history heuristic bonus
        int hist = history_table.get(move, node->state.side_to_move);
        score += (hist / 10000.0f) * config.history_weight;
        
        node->move_priors[i] = score;
        total_score += score;
    }
    
    // Normalize to probability distribution
    if (total_score > 0.0f) {
        for (size_t i = 0; i < node->move_priors.size(); i++) {
            node->move_priors[i] /= total_score;
        }
    } else {
        // Uniform if all scores are zero
        float uniform = 1.0f / node->legal_moves.size();
        for (size_t i = 0; i < node->move_priors.size(); i++) {
            node->move_priors[i] = uniform;
        }
    }
}

// ============================================================================
// ROOT NODE HANDLING
// ============================================================================

void PUCTEngine::add_dirichlet_noise_to_root() {
    if (!root || root->move_priors.empty()) return;
    
    int num_moves = root->move_priors.size();
    std::gamma_distribution<float> gamma(config.dirichlet_alpha, 1.0f);
    
    std::vector<float> noise(num_moves);
    float sum = 0.0f;
    
    for (int i = 0; i < num_moves; i++) {
        noise[i] = gamma(rng);
        sum += noise[i];
    }
    
    // Normalize
    for (int i = 0; i < num_moves; i++) {
        noise[i] /= sum;
    }
    
    // Mix: P = (1 - ε) * P + ε * noise
    float eps = config.dirichlet_epsilon;
    for (int i = 0; i < num_moves; i++) {
        root->move_priors[i] = (1.0f - eps) * root->move_priors[i] + eps * noise[i];
    }
}

Move PUCTEngine::select_move_by_temperature(float temperature) {
    if (!root || root->children.empty()) {
        return root ? (root->legal_moves.empty() ? 0 : root->legal_moves[0]) : 0;
    }
    
    // PRIORITY: Check for immediate checkmate (terminal win)
    for (auto& child : root->children) {
        if (child->is_terminal && child->evaluated) {
            // Terminal node with value > 0.9 = CHECKMATE for us!
            float child_q = child->Q();
            if (child_q < -0.9f) {  // Negative because it's opponent's loss = our win
                return child->move_from_parent;
            }
        }
    }
    
    std::vector<int> visits;
    std::vector<Move> moves;
    
    for (auto& child : root->children) {
        visits.push_back(child->visits.load(std::memory_order_relaxed));
        moves.push_back(child->move_from_parent);
    }
    
    if (temperature < 0.01f) {
        // Greedy - but also consider Q value for ties
        int max_visits = *std::max_element(visits.begin(), visits.end());
        float best_q = -999.0f;
        Move best_move = moves[0];
        
        for (size_t i = 0; i < visits.size(); i++) {
            if (visits[i] == max_visits) {
                float q = root->children[i]->Q();
                if (q > best_q) {
                    best_q = q;
                    best_move = moves[i];
                }
            }
        }
        return best_move;
    }
    
    // Temperature scaling
    std::vector<float> probs(visits.size());
    float sum = 0.0f;
    
    for (size_t i = 0; i < visits.size(); i++) {
        probs[i] = std::pow((float)visits[i], 1.0f / temperature);
        sum += probs[i];
    }
    
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum;
    }
    
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return moves[dist(rng)];
}

// ============================================================================
// UTILITY
// ============================================================================

std::vector<float> PUCTEngine::get_move_probabilities(float temperature) {
    std::vector<float> probs(root->legal_moves.size(), 0.0f);
    
    if (root->children.empty()) {
        float uniform = 1.0f / root->legal_moves.size();
        std::fill(probs.begin(), probs.end(), uniform);
        return probs;
    }
    
    std::vector<int> visits(root->legal_moves.size(), 0);
    
    for (auto& child : root->children) {
        for (size_t i = 0; i < root->legal_moves.size(); i++) {
            if (root->legal_moves[i] == child->move_from_parent) {
                visits[i] = child->visits.load(std::memory_order_relaxed);
                break;
            }
        }
    }
    
    if (temperature < 0.01f) {
        int max_idx = std::max_element(visits.begin(), visits.end()) - visits.begin();
        probs[max_idx] = 1.0f;
    } else {
        float sum = 0.0f;
        for (size_t i = 0; i < visits.size(); i++) {
            probs[i] = std::pow((float)visits[i], 1.0f / temperature);
            sum += probs[i];
        }
        
        if (sum > 0.0f) {
            for (size_t i = 0; i < probs.size(); i++) {
                probs[i] /= sum;
            }
        }
    }
    
    return probs;
}

std::vector<Move> PUCTEngine::get_pv(int max_length) const {
    std::vector<Move> pv;
    
    const PUCTNode* node = root.get();
    while (node && !node->children.empty() && (int)pv.size() < max_length) {
        const PUCTNode* best = nullptr;
        int max_visits = -1;
        
        for (auto& child : node->children) {
            int visits = child->visits.load(std::memory_order_relaxed);
            if (visits > max_visits) {
                max_visits = visits;
                best = child.get();
            }
        }
        
        if (!best || max_visits == 0) break;
        
        pv.push_back(best->move_from_parent);
        node = best;
    }
    
    return pv;
}

void PUCTEngine::ensure_batch_capacity(int size) {
    if (size <= max_batch_size) return;
    
    if (d_boards) cudaFree(d_boards);
    if (d_results) cudaFree(d_results);
    if (h_boards) cudaFreeHost(h_boards);
    if (h_results) cudaFreeHost(h_results);
    
    max_batch_size = size;
    CUDA_CHECK(cudaMalloc(&d_boards, size * sizeof(BoardState)));
    CUDA_CHECK(cudaMalloc(&d_results, size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_boards, size * sizeof(BoardState)));
    CUDA_CHECK(cudaMallocHost(&h_results, size * sizeof(float)));
}
