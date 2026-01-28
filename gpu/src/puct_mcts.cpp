// https://www.chessprogramming.org/Monte-Carlo_Tree_Search

#include "../include/puct_mcts.h"
#include "../include/kernel_launchers.h"
#include "../include/evaluation.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <chrono>
#include <limits>

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

#include "../include/cpu_movegen.h"

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

// MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for move ordering
int MoveHeuristics::mvv_lva_score(Move move, const BoardState& state) {
    static const int piece_values[6] = {100, 320, 330, 500, 900, 20000};
    
    if (!is_capture(move)) return 0;
    
    // Bits 0-5 src square, 6-11 destination
    int to_sq = (move >> 6) & 0x3F;
    int from_sq = move & 0x3F;
    
    // Find captured piece
    int victim_value = 0;
    for (int piece = 0; piece < 6; piece++) {
        // Get all piece position for a piece of 1 kind and probe it with a bitboard where only the to_sq is active
        if (state.pieces[state.side_to_move ^ 1][piece] & (1ULL << to_sq)) {
            victim_value = piece_values[piece];
            break;
        }
    }
    
    // Find attacker piece
    int attacker_value = 0;
    for (int piece = 0; piece < 6; piece++) {
        // Similar logic as previous
        if (state.pieces[state.side_to_move][piece] & (1ULL << from_sq)) {
            attacker_value = piece_values[piece];
            break;
        }
    }
    
    // MVV-LVA: prioritize high-value victims with low-value attackers
    return (victim_value * 10) - (attacker_value / 10);
}

int MoveHeuristics::see_score(Move move, const BoardState& state) {
    // Simplified SEE - MVV-LVA currently -> will be expanded
    return mvv_lva_score(move, state);
}

bool MoveHeuristics::is_check(Move move, const BoardState& state) {
    // Make move on copy and check if opponent is in check
    BoardState test_state = state;
    cpu_movegen::make_move_cpu(&test_state, move);
    return cpu_movegen::in_check_cpu(&test_state);
}

bool MoveHeuristics::is_capture(Move move) {
    // Bits 12-15 - special flags (capture, promotion, en passant, castling)
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
    
    // Tactical bonuses - eval bonus for types of moves - heuristic
    if (is_capture(move)) {
        int mvv_lva = mvv_lva_score(move, state);
        score += (mvv_lva / 100.0f) * capture_weight;  
    }
    
    if (is_check(move, state)) {
        score += check_weight;  
    }
    
    if (is_promotion(move)) {
        score += 8.0f;  
    }
    
    if (is_passed_pawn_push(move, state)) {
        score += 2.0f;
    }
    
    return std::max(0.1f, score);
}

PUCTEngine::PUCTEngine(const PUCTConfig& cfg)
    : config(cfg)
    , root(nullptr)
    , d_boards(nullptr) // Device variables
    , d_results(nullptr)
    , h_boards(nullptr) // Host variables
    , h_results(nullptr)
    , max_batch_size(0)
    , rng(std::chrono::steady_clock::now().time_since_epoch().count()) // Randomness
    , total_simulations(0)
{
    killer_moves.clear();
    history_table.clear();
    continuation_history.clear();
    stability_tracker.reset();
    aspiration_window.reset(config.aspiration_initial_window);

    // Initialize temperature schedule from config
    temp_schedule.initial_temp = config.temperature_initial;
    temp_schedule.final_temp = config.temperature_final;
    temp_schedule.warmup_sims = config.temperature_warmup_sims;
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
    
    std::cout << "PUCT Engine initialized (heuristic-based)" << std::endl;
    std::cout << "Batch size: " << max_batch_size << std::endl;
}

Move PUCTEngine::search(const BoardState& root_state) {
    // Create root node
    root = std::make_unique<PUCTNode>(root_state);
    total_simulations = 0;

    // Reset stability tracker
    stability_tracker.reset();

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

    // Add Dirichlet noise for exploration -- alpha zerp radi - ne znam zasto -- previse matematike trenutno
    if (config.add_dirichlet_noise) {
        add_dirichlet_noise_to_root();
    }

    // Start timer for time-based search
    auto search_start = std::chrono::steady_clock::now();
    bool use_time = config.use_time_limit && config.time_limit_ms > 0;

    // Main MCTS loop
    int simulations_done = 0;
    int max_sims = use_time ? INT_MAX : config.num_simulations;

    while (simulations_done < max_sims) {
        // Check time limit if enabled
        if (use_time) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start).count();
            if (elapsed >= config.time_limit_ms) {
                break;
            }
        }

        int batch_size = config.batch_size;
        if (!use_time) {
            batch_size = std::min(config.batch_size, config.num_simulations - simulations_done);
        }
        std::vector<PUCTNode*> leaf_nodes;
        std::vector<std::vector<Move>> tree_moves_batch;  // Track moves for RAVE - Rapid Action Value Estimation
        leaf_nodes.reserve(batch_size);
        tree_moves_batch.reserve(batch_size);

        // Traverse tree using PUCT (with move tracking for RAVE) - http://hal.inria.fr/inria-00485555/en/
        for (int i = 0; i < batch_size; i++) {
            std::vector<Move> tree_moves;
            PUCTNode* node;

            if (config.use_rave) {
                // Use RAVE-aware selection that tracks moves
                node = select_and_track(root.get(), tree_moves);
            } else {
                // Standard selection without move tracking
                node = select(root.get());
            }

            // Add virtual loss for parallel exploration - if no virtual loss then all threads go down same path
            if (config.use_virtual_loss && node) {
                PUCTNode* current = node;
                while (current) {
                    current->add_virtual_loss(config.virtual_loss);
                    current = current->parent;
                }
            }

            if (node) {
                leaf_nodes.push_back(node);
                tree_moves_batch.push_back(tree_moves);
            }
        }

        if (leaf_nodes.empty()) break;

        // Expansion and evaluation - GPU batch evaluation
        expand_and_evaluate_batch(leaf_nodes);

        // Backpropagation
        for (size_t i = 0; i < leaf_nodes.size(); i++) {
            PUCTNode* node = leaf_nodes[i];
            float value = node->value_estimate;

            if (config.use_rave) {
                // RAVE backpropagation with AMAF (all moves as first) updates
                backpropagate_with_rave(node, value, tree_moves_batch[i]);
            } else {
                // Standard backpropagation
                PUCTNode* current = node;
                while (current) {
                    if (config.use_virtual_loss) {
                        current->remove_virtual_loss(config.virtual_loss);
                    }

                    current->update(value);

                    // Update node dynamics for adaptive exploration
                    update_node_dynamics(current);

                    value = 1.0f - value;

                    // Update history heuristic for good moves
                    if (current->parent && value > 0.3f) {
                        history_table.update(current->move_from_parent,
                                           current->state.side_to_move ^ 1,
                                           current->depth);
                    }

                    current = current->parent;
                }
            }
        }

        simulations_done += leaf_nodes.size();
        total_simulations += leaf_nodes.size();

        // Track best move stability every 1000 simulations
        if (simulations_done % 1000 == 0 || simulations_done >= config.num_simulations) {
            Move current_best = get_best_move();
            stability_tracker.update(current_best, simulations_done);

            // Update aspiration window with current best Q value
            if (config.use_aspiration_windows && simulations_done >= config.aspiration_min_sims) {
                // Find best child Q value
                float best_q = -1.0f;
                for (auto& child : root->children) {
                    float q = child->Q();
                    if (q > best_q) best_q = q;
                }
                if (best_q >= 0.0f) {
                    update_aspiration_window(best_q);
                }
            }

            // Print progress if verbose
            if (config.verbose && simulations_done % config.info_interval == 0 && config.info_interval > 0) {
                std::cout << "Sims: " << simulations_done << "/" << config.num_simulations
                          << " | Best: " << move_to_string(current_best)
                          << " | Stability: " << stability_tracker.stability_score
                          << " | Changes: " << stability_tracker.changes_count;

                if (config.use_aspiration_windows && aspiration_window.active) {
                    std::cout << " | Aspiration: [" << aspiration_window.previous_best_q - aspiration_window.window_size
                              << ", " << aspiration_window.previous_best_q + aspiration_window.window_size << "]";
                }
                std::cout << std::endl;
            }

            // Early stopping if position is very stable (optional optimization)
            if (simulations_done >= config.num_simulations / 2 &&
                stability_tracker.stability_score > 0.95f &&
                stability_tracker.changes_count < 2) {
                if (config.verbose) {
                    std::cout << "Early stop: Position very stable" << std::endl;
                }
                break;
            }
        }
    }

    // Check if we should extend search due to instability
    if (stability_tracker.should_extend_search() && simulations_done < config.num_simulations * 2) {
        int extension_sims = config.num_simulations / 2;
        if (config.verbose) {
            std::cout << "Extending search by " << extension_sims
                      << " simulations (instability detected)" << std::endl;
        }

        // Continue search with more simulations
        int extended_target = simulations_done + extension_sims;
        while (simulations_done < extended_target) {
            int batch_size = std::min(config.batch_size, extended_target - simulations_done);
            std::vector<PUCTNode*> leaf_nodes;
            std::vector<std::vector<Move>> tree_moves_batch;
            leaf_nodes.reserve(batch_size);
            tree_moves_batch.reserve(batch_size);

            for (int i = 0; i < batch_size; i++) {
                std::vector<Move> tree_moves;
                PUCTNode* node;

                if (config.use_rave) {
                    node = select_and_track(root.get(), tree_moves);
                } else {
                    node = select(root.get());
                }

                if (config.use_virtual_loss && node) {
                    PUCTNode* current = node;
                    while (current) {
                        current->add_virtual_loss(config.virtual_loss);
                        current = current->parent;
                    }
                }

                if (node) {
                    leaf_nodes.push_back(node);
                    tree_moves_batch.push_back(tree_moves);
                }
            }

            if (leaf_nodes.empty()) break;

            expand_and_evaluate_batch(leaf_nodes);

            for (size_t i = 0; i < leaf_nodes.size(); i++) {
                PUCTNode* node = leaf_nodes[i];
                float value = node->value_estimate;

                if (config.use_rave) {
                    backpropagate_with_rave(node, value, tree_moves_batch[i]);
                } else {
                    PUCTNode* current = node;
                    while (current) {
                        if (config.use_virtual_loss) {
                            current->remove_virtual_loss(config.virtual_loss);
                        }

                        current->update(value);

                        // Update node dynamics for adaptive exploration
                        update_node_dynamics(current);

                        value = 1.0f - value;

                        if (current->parent && value > 0.3f) {
                            update_history_tables(current->move_from_parent, current->state, current->depth);
                        }

                        current = current->parent;
                    }
                }
            }

            simulations_done += leaf_nodes.size();
            total_simulations += leaf_nodes.size();
        }
    }

    // Select best move with temperature decay if enabled
    float final_temp = config.use_temperature_decay ?
        temp_schedule.get_temperature(simulations_done, config.num_simulations) :
        config.temperature;
    Move best_move = select_move_by_temperature(final_temp);

    return best_move;
}

// Selection phase 

PUCTNode* PUCTEngine::select(PUCTNode* node) {
    while (true) {
        if (node->is_terminal) {
            return node;
        }
        
        if (!node->evaluated) {
            return node;
        }
        
        if (!node->is_fully_expanded()) {
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
                    
                    Move temp_moves[MAX_MOVES];
                    int num_child_moves = cpu_movegen::generate_legal_moves_cpu(&child_state, temp_moves);
                    if (num_child_moves == 0 && cpu_movegen::in_check_cpu(&child_state)) {
                        // Opponent has no legal moves and is in check = checkmate
                        child_ptr->is_terminal = true;
                        child_ptr->value_estimate = 1.0f;  
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

    // FPU - first play urgency
    float fpu_value = config.use_fpu ? (node->Q() - config.fpu_reduction) : 0.0f;

    PUCTNode* best_child = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();

    int move_idx = 0;
    for (auto& child : node->children) {
        float score;

        // Use adaptive PUCT if enabled
        if (config.use_adaptive_cpuct) {
            score = child->adaptive_puct_score(parent_visits, c_puct, fpu_value, config);

            // Apply move number scaling if enabled
            if (config.use_move_number_scaling) {
                float move_factor = compute_move_number_factor(move_idx, node->children.size());
                score *= move_factor;
            }
        } else if (config.use_rave) {
            // RAVE-enhanced score
            score = child->rave_puct_score(parent_visits, c_puct, fpu_value, config.rave_k, true);
        } else {
            // Standard PUCT
            score = child->puct_score(parent_visits, c_puct, fpu_value);
        }

        if (score > best_score) {
            best_score = score;
            best_child = child.get();
        }

        move_idx++;
    }

    return best_child;
}

// Dynamic c_puct calculation - AlphaGo Zero formula for adaptive exploration
float PUCTEngine::get_dynamic_cpuct() const {
    // AlphaGo Zero formula: c_puct = log((1 + N + c_base) / c_base) + c_init
    int N = root ? root->visits.load(std::memory_order_relaxed) : 0;
    return std::log((1.0f + N + config.c_puct_base) / config.c_puct_base) + config.c_puct_init;
}

// Expanstion and evaluation

void PUCTEngine::expand_and_evaluate(PUCTNode* node) {
    generate_legal_moves(node);
    
    if (node->is_terminal) {
        if (node->legal_moves.empty()) {
            if (cpu_movegen::in_check_cpu(&node->state)) {
                node->value_estimate = 0.0f;   
            } else {
                node->value_estimate = 0.5f;   // Stalemate = DRAW
            }
        } else {
            node->value_estimate = 0.5f;  // Draw (
        }
        node->evaluated = true;
        return;
    }
    
    compute_move_priors(node);
    
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
                // Checkmate = 0.0 (loss), stalemate = 0.5 (draw)
                node->value_estimate = cpu_movegen::in_check_cpu(&node->state) ? 0.0f : 0.5f;
            } else {
                node->value_estimate = 0.5f;  // Draw
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

// Move generation & heuristics prior

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

        // Add continuation history bonus (Stockfish-style)
        if (config.use_continuation_history) {
            Piece piece = get_moved_piece(node->state, move);
            if (piece != NO_PIECE) {
                int cont_hist = continuation_history.get(move, node->state.side_to_move, piece);
                score += (cont_hist / 16384.0f) * config.continuation_weight;
            }
        }

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

    // Sort moves by prior (descending) so high-prior moves are expanded first
    // This is critical for PUCT efficiency - explore promising moves first!
    std::vector<std::pair<float, Move>> sorted_moves;
    sorted_moves.reserve(node->legal_moves.size());
    for (size_t i = 0; i < node->legal_moves.size(); i++) {
        sorted_moves.emplace_back(node->move_priors[i], node->legal_moves[i]);
    }
    std::sort(sorted_moves.begin(), sorted_moves.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (size_t i = 0; i < sorted_moves.size(); i++) {
        node->move_priors[i] = sorted_moves[i].first;
        node->legal_moves[i] = sorted_moves[i].second;
    }
}

// Root node handling

// Dirichlet Noise - AlphaZero-style root exploration for better move diversity
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
    
    //  Check for immediate checkmate 
    for (auto& child : root->children) {
        if (child->is_terminal && child->evaluated) {
            // Terminal node: if opponent is checkmated, child->value_estimate = 1.0
            float child_q = child->Q();
            if (child_q > 0.9f) {  
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

// Helpers/utility

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


Piece PUCTEngine::get_moved_piece(const BoardState& state, Move move) const {
    int from = move & 0x3F;
    Bitboard from_bb = 1ULL << from;

    // Find which piece is on the from square
    for (int pt = 0; pt < 6; pt++) {
        if (state.pieces[state.side_to_move][pt] & from_bb) {
            return (Piece)pt;
        }
    }

    return NO_PIECE;
}

// History Heuristic Update - updates main history and continuation history tables
void PUCTEngine::update_history_tables(Move move, const BoardState& state, int depth) {
    int color = state.side_to_move ^ 1;  // Opponent's color (who just made the move)

    // Update main history table
    history_table.update(move, color, depth);

    // Update continuation history if enabled
    if (config.use_continuation_history) {
        Piece piece = get_moved_piece(state, move);
        if (piece != NO_PIECE) {
            // Stockfish-style bonus: depth^2 for good moves
            int bonus = depth * depth;
            continuation_history.update(move, color, piece, bonus);
        }
    }
}

Move PUCTEngine::get_best_move() const {
    if (!root || root->children.empty()) return 0;

    const PUCTNode* best = nullptr;
    int max_visits = -1;

    for (auto& child : root->children) {
        int visits = child->visits.load(std::memory_order_relaxed);
        if (visits > max_visits) {
            max_visits = visits;
            best = child.get();
        }
    }

    return best ? best->move_from_parent : 0;
}

std::string PUCTEngine::move_to_string(Move move) const {
    if (move == 0) return "(none)";

    static const char* square_names[64] = {
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"
    };

    int from = move & 0x3F;
    int to = (move >> 6) & 0x3F;
    int flags = (move >> 12) & 0xF;

    std::string result = std::string(square_names[from]) + square_names[to];

    // Add promotion piece
    if (flags >= MOVE_PROMO_N) {
        const char promo_chars[] = "nbrq";
        result += promo_chars[flags & 0x3];
    }

    return result;
}


// Node Dynamics Tracking - updates Q-value history for adaptive exploration
void PUCTEngine::update_node_dynamics(PUCTNode* node) {
    if (!node || !config.use_adaptive_cpuct) return;

    // Update Q history and volatility
    float current_q = node->Q();
    node->dynamics.update_q_history(current_q);

    // Update improving flag
    if (node->parent) {
        float parent_q = node->parent->Q();
        node->dynamics.update_improving(current_q, parent_q);
    }

    // Cache game phase for this node
    node->dynamics.game_phase = calculate_phase(node->state);
}

// Move Number Scaling - reduces exploration for late moves in the ordering
// Based on Stockfish's logarithmic reduction: reduction = log(move_idx) * log(total_moves) / 2.0
float PUCTEngine::compute_move_number_factor(int move_idx, int total_moves) const {
    if (!config.use_move_number_scaling || move_idx < 3) {
        return 1.0f;  // First 3 moves get full exploration
    }

    // Stockfish-inspired logarithmic reduction
    // reduction = log(move_idx) * log(total_moves) / 2.0
    float reduction = std::log((float)move_idx) * std::log((float)total_moves) / 2.0f;

    // Convert to multiplier (lower = less exploration)
    float factor = 1.0f / (1.0f + reduction * 0.1f);

    return std::max(0.3f, factor);  // Don't reduce below 30%
}


// RAVE (Rapid Action Value Estimation)

// RAVE (Rapid Action Value Estimation) - selects leaf nodes while tracking moves for AMAF updates
PUCTNode* PUCTEngine::select_and_track(PUCTNode* node, std::vector<Move>& tree_moves) {
    // Select leaf node while tracking moves played during tree traversal
    tree_moves.clear();

    while (!node->is_leaf() && node->is_fully_expanded()) {
        // Select best child using PUCT
        PUCTNode* child = best_child_puct(node);

        if (!child) break;

        // Track the move taken
        tree_moves.push_back(child->move_from_parent);

        node = child;
    }

    return node;
}

// RAVE Backpropagation - updates both standard MCTS and AMAF statistics
void PUCTEngine::backpropagate_with_rave(PUCTNode* node, float value,
                                         const std::vector<Move>& tree_moves) {
    // Backpropagate value up the tree while updating AMAF statistics
    PUCTNode* current = node;
    int depth = 0;

    while (current) {
        // Remove virtual loss
        if (config.use_virtual_loss) {
            current->remove_virtual_loss(config.virtual_loss);
        }

        // Update standard PUCT statistics
        current->update(value);

        // Update node dynamics for adaptive exploration
        update_node_dynamics(current);

        // Update AMAF statistics if RAVE is enabled and within depth limit
        if (config.use_rave && depth < config.rave_update_depth && current->parent) {
            update_amaf_stats(current->parent, tree_moves, value);
        }

        // Flip value for opponent's perspective
        value = 1.0f - value;

        // Update history heuristic for good moves
        if (current->parent && value > 0.3f) {
            update_history_tables(current->move_from_parent, current->state, current->depth);
        }

        current = current->parent;
        depth++;
    }
}

// AMAF Statistics Update - updates RAVE values for all moves appearing in simulation path
void PUCTEngine::update_amaf_stats(PUCTNode* node, const std::vector<Move>& sim_moves,
                                   float value) {
    // Update AMAF statistics for all children whose moves appear in the simulation
    if (!node || node->children.empty()) return;

    for (auto& child : node->children) {
        Move child_move = child->move_from_parent;

        // Check if this move appeared anywhere in the simulation path
        bool move_played = false;
        for (Move sim_move : sim_moves) {
            if (sim_move == child_move) {
                move_played = true;
                break;
            }
        }

        // If the move was played in the simulation, update AMAF stats
        if (move_played) {
            child->update_amaf(value);
        }
    }
}


// Aspiration Window - initialize dynamic search window for faster convergence
void PUCTEngine::initialize_aspiration_window() {
    aspiration_window.reset(config.aspiration_initial_window);
}

// Aspiration Window - update window based on search results (fail-high/fail-low handling)
void PUCTEngine::update_aspiration_window(float best_q) {
    aspiration_window.update(best_q, config.aspiration_initial_window);
}

bool PUCTEngine::should_use_aspiration_window(int simulations_done) const {
    return config.use_aspiration_windows &&
           simulations_done >= config.aspiration_min_sims &&
           aspiration_window.active;
}

PUCTNode* PUCTEngine::select_child_in_window(PUCTNode* node, float q_lower, float q_upper) {
    if (!node || node->children.empty()) return nullptr;

    // Select child with highest visits among those within Q window
    PUCTNode* best_child = nullptr;
    int best_visits = -1;

    for (auto& child : node->children) {
        float child_q = child->Q();

        // Check if this child's Q is within the aspiration window
        if (child_q >= q_lower && child_q <= q_upper) {
            int visits = child->visits.load(std::memory_order_relaxed);
            if (visits > best_visits) {
                best_visits = visits;
                best_child = child.get();
            }
        }
    }

    // If no child in window, fall back to standard PUCT selection
    if (!best_child) {
        return best_child_puct(node);
    }

    return best_child;
}

float PUCTEngine::get_current_temperature(int simulations_done) const {
    if (config.use_temperature_decay) {
        return temp_schedule.get_temperature(simulations_done, config.num_simulations);
    }
    return config.temperature;
}

// Multi-PV Search - returns top-N best moves with PV lines for analysis
std::vector<MultiPVResult> PUCTEngine::get_multi_pv(int num_pvs) const {
    std::vector<MultiPVResult> results;

    if (!root || root->children.empty()) {
        return results;
    }

    // Create a list of all children with their stats
    std::vector<std::pair<int, PUCTNode*>> child_list;
    for (auto& child : root->children) {
        int visits = child->visits.load(std::memory_order_relaxed);
        child_list.emplace_back(visits, child.get());
    }

    // Sort by visits (descending)
    std::sort(child_list.begin(), child_list.end(),
              [](const auto& a, const auto& b) {
                  return a.first > b.first;
              });

    // Extract top N PVs
    int count = std::min(num_pvs, (int)child_list.size());
    for (int i = 0; i < count; i++) {
        PUCTNode* child = child_list[i].second;
        MultiPVResult pv_result;
        pv_result.move = child->move_from_parent;
        pv_result.q_value = child->Q();
        pv_result.visits = child_list[i].first;

        // Get PV line for this child
        const PUCTNode* current = child;
        int max_pv_depth = 10;
        while (current && !current->children.empty() && max_pv_depth-- > 0) {
            const PUCTNode* best = nullptr;
            int max_visits = -1;

            for (auto& c : current->children) {
                int v = c->visits.load(std::memory_order_relaxed);
                if (v > max_visits) {
                    max_visits = v;
                    best = c.get();
                }
            }

            if (!best || max_visits == 0) break;
            pv_result.pv.push_back(best->move_from_parent);
            current = best;
        }

        results.push_back(pv_result);
    }

    return results;
}
