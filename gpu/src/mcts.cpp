#include "../include/mcts.h"
#include <chrono>
#include <iostream>
#include <cassert>

// External GPU kernel launchers
extern "C" void launch_random_playout(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    unsigned int seed,
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

extern "C" void launch_quiescence_playout(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    unsigned int seed,
    int max_q_depth,
    cudaStream_t stream
);

// External initialization function
extern void init_attack_tables();
extern void init_startpos(BoardState* pos);

// CPU-side move generation (for tree building)
// We need a simple implementation that works without GPU
namespace cpu_movegen {

// Simple check detection (simplified version)
bool is_square_attacked_cpu(const BoardState* pos, int sq, int by_color);
int generate_legal_moves_cpu(const BoardState* pos, Move* moves);
void make_move_cpu(BoardState* pos, Move m);
bool in_check_cpu(const BoardState* pos);


} 

// MCTSEngine Implementation
MCTSEngine::MCTSEngine(int batch_size)
    : root(nullptr), total_nodes(0), total_simulations(0), max_depth_reached(0),
      d_boards(nullptr), d_results(nullptr),
      h_boards(nullptr), h_results(nullptr),
      batch_size(batch_size), max_batch_size(0),
      rng(std::chrono::steady_clock::now().time_since_epoch().count())
{
}

MCTSEngine::~MCTSEngine() {
    // Free GPU resources
    if (d_boards) cudaFree(d_boards);
    if (d_results) cudaFree(d_results);
    if (h_boards) cudaFreeHost(h_boards);
    if (h_results) cudaFreeHost(h_results);
}

void MCTSEngine::init() {
    // Initialize attack tables on GPU
    init_attack_tables();

    // Allocate GPU memory
    max_batch_size = batch_size;
    CUDA_CHECK(cudaMalloc(&d_boards, batch_size * sizeof(BoardState)));
    CUDA_CHECK(cudaMalloc(&d_results, batch_size * sizeof(float)));

    // Allocate pinned host memory for faster transfers
    CUDA_CHECK(cudaMallocHost(&h_boards, batch_size * sizeof(BoardState)));
    CUDA_CHECK(cudaMallocHost(&h_results, batch_size * sizeof(float)));

    std::cout << "MCTS Engine initialized with batch size: " << batch_size << std::endl;
}

Move MCTSEngine::search(const BoardState& root_state, int iterations) {
    // Use searchWithConfig with default config for consistency
    SearchConfig config;
    config.max_iterations = iterations;
    config.simulations_per_batch = batch_size;

    SearchResult result = searchWithConfig(root_state, config);
    return result.best_move;
}


MCTSNode* MCTSEngine::select(MCTSNode* node) {
    while (!node->is_leaf() && node->is_fully_expanded()) {
        node = best_child_ucb1(node);
    }
    return node;
}

MCTSNode* MCTSEngine::expand(MCTSNode* node) {
    // Generate moves if not done yet
    if (!node->moves_generated) {
        generate_moves_for_node(node);
    }

    if (node->is_terminal || node->legal_moves.empty()) {
        // Detect checkmate in terminal positions
        if (node->is_terminal && node->legal_moves.empty()) {
            if (cpu_movegen::in_check_cpu(&node->state)) {
                node->mate_distance = -1;  
            }
        }
        return node;
    }

    // Progressive widening - limit number of children based on visits
    int max_children = node->legal_moves.size();  // Default - all moves
    if (current_config.use_progressive_widening && node->parent != nullptr) {
        // max_children = base + alpha * sqrt(parent_visits)
        int parent_visits = node->parent->visits;
        max_children = current_config.progressive_widening_base +
                      (int)(current_config.progressive_widening_alpha * std::sqrt((float)parent_visits));

        // Clamp to legal moves count
        if (max_children > (int)node->legal_moves.size()) {
            max_children = node->legal_moves.size();
        }
    }

    // Check if we've already expanded enough children
    if ((int)node->children.size() >= max_children) {
        return node;  // Don't expand more children yet
    }

    // Find untried moves
    std::vector<Move> untried;
    for (const auto& move : node->legal_moves) {
        bool found = false;
        for (const auto& child : node->children) {
            if (child->move_from_parent == move) {
                found = true;
                break;
            }
        }
        if (!found) {
            untried.push_back(move);
        }
    }

    if (untried.empty()) {
        return node;
    }

    // Prioritize checking moves among untried moves
    // First, try to select a checking move if available
    Move selected_move = 0;
    bool found_checking_move = false;

    for (const auto& move : untried) {
        BoardState test_state = node->state;
        cpu_movegen::make_move_cpu(&test_state, move);
        if (cpu_movegen::in_check_cpu(&test_state)) {
            selected_move = move;
            found_checking_move = true;
            break;  // Prioritize first checking move found
        }
    }

    // If no checking move, select random untried move
    if (!found_checking_move) {
        selected_move = untried[rng() % untried.size()];
    }

    // Create child state
    BoardState child_state = node->state;
    cpu_movegen::make_move_cpu(&child_state, selected_move);

    // Create and add child node
    auto child = std::make_unique<MCTSNode>(child_state, selected_move, node);
    MCTSNode* child_ptr = child.get();

    // Detect if child position is in check
    child_ptr->is_check = cpu_movegen::in_check_cpu(&child_state);
    child_ptr->gives_check = child_ptr->is_check;  

    // Detect immediate checkmate
    Move test_moves[MAX_MOVES];
    int num_child_moves = cpu_movegen::generate_legal_moves_cpu(&child_state, test_moves);
    if (num_child_moves == 0 && child_ptr->is_check) {
        child_ptr->mate_distance = 1; 
    }

    node->children.push_back(std::move(child));
    total_nodes++;

    return child_ptr;
}

void MCTSEngine::simulate_batch(const std::vector<MCTSNode*>& nodes) {
    int count = nodes.size();
    if (count == 0) return;

    // Copy board states to pinned memory
    for (int i = 0; i < count; i++) {
        h_boards[i] = nodes[i]->state;
    }

    // Transfer to GPU
    CUDA_CHECK(cudaMemcpy(d_boards, h_boards, count * sizeof(BoardState), cudaMemcpyHostToDevice));

    // Run playouts based on configured mode
    unsigned int seed = rng();

    switch (current_config.playout_mode) {
        case PlayoutMode::RANDOM:
            // Pure random playouts (original)
            launch_random_playout(d_boards, d_results, count, seed, 0);
            break;

        case PlayoutMode::EVAL_HYBRID:
            // Short random playout + static evaluation
            launch_eval_playout(d_boards, d_results, count, seed, 0);
            break;

        case PlayoutMode::STATIC_EVAL:
            // Pure static evaluation (fastest)
            launch_static_eval(d_boards, d_results, count, 0);
            break;

        case PlayoutMode::QUIESCENCE:
            // Short random + quiescence search (tactical)
            launch_quiescence_playout(d_boards, d_results, count, seed,
                                     current_config.quiescence_depth, 0);
            break;
    }

    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Transfer results back
    CUDA_CHECK(cudaMemcpy(h_results, d_results, count * sizeof(float), cudaMemcpyDeviceToHost));
}

void MCTSEngine::backpropagate(MCTSNode* node, float result) {
    // Result is from the perspective of the starting side
    // We need to flip it as we go up the tree

    int starting_side = node->state.side_to_move ^ 1;  // Side that just moved

    while (node != nullptr) {
        node->visits++;

        // Determine if result is good for the side to move at this node
        // node->state.side_to_move is the side about to move
        // If starting_side won and it's their turn now, that's good
        if ((node->state.side_to_move ^ 1) == starting_side) {
            node->wins += result;
        } else {
            node->wins += (1.0f - result);
        }

        // Propagate mate distance information up the tree
        // If any child delivers checkmate, this node should reflect that
        if (!node->children.empty() && node->moves_generated) {
            int best_mate_distance = 0;
            bool found_mate = false;
            bool all_lose = true;

            for (const auto& child : node->children) {
                if (child->mate_distance > 0) {
                    // Child position is mate for opponent - good, we can force mate in (child->mate_distance + 1)
                    int our_mate = child->mate_distance + 1;
                    if (!found_mate || our_mate < best_mate_distance) {
                        best_mate_distance = our_mate;
                        found_mate = true;
                    }
                    all_lose = false;
                } else if (child->mate_distance < 0) {
                    // Child position is mate against us (we get mated) - bad
                    all_lose = all_lose && true;
                } else {
                    all_lose = false;
                }
            }

            if (found_mate) {
                // We found a forcing mate line
                node->mate_distance = best_mate_distance;
            } else if (all_lose && !node->children.empty()) {
                // All moves lead to us getting mated - find longest defense
                int longest_defense = -1000;
                for (const auto& child : node->children) {
                    if (child->mate_distance < longest_defense) {
                        longest_defense = child->mate_distance;
                    }
                }
                node->mate_distance = longest_defense - 1;
            }
        }

        node = node->parent;
    }
}

void MCTSEngine::generate_moves_for_node(MCTSNode* node) {
    if (node->moves_generated) return;

    Move moves[MAX_MOVES];
    int num_moves = cpu_movegen::generate_legal_moves_cpu(&node->state, moves);

    node->legal_moves.clear();
    for (int i = 0; i < num_moves; i++) {
        node->legal_moves.push_back(moves[i]);
    }

    node->is_terminal = (num_moves == 0) || (node->state.halfmove >= 100);
    node->moves_generated = true;
}

MCTSNode* MCTSEngine::best_child_ucb1(MCTSNode* node) {
    MCTSNode* best = nullptr;
    float best_value = -std::numeric_limits<float>::infinity();

    for (const auto& child : node->children) {
        float value = child->ucb1(node->visits);
        if (value > best_value) {
            best_value = value;
            best = child.get();
        }
    }

    return best;
}

Move MCTSEngine::best_move() const {
    if (!root || root->children.empty()) {
        return 0;
    }

    // Select child with most visits (more robust than highest win rate)
    MCTSNode* best = nullptr;
    int most_visits = -1;

    for (const auto& child : root->children) {
        if (child->visits > most_visits) {
            most_visits = child->visits;
            best = child.get();
        }
    }

    if (best) {
        if (current_config.verbose) {
            std::cout << "Best move visits: " << best->visits
                      << ", win rate: " << (best->wins / best->visits) << std::endl;
        }
        return best->move_from_parent;
    }

    return 0;
}

// New configurable search interface

SearchResult MCTSEngine::searchWithConfig(const BoardState& root_state, const SearchConfig& config) {
    SearchResult result;
    current_config = config;
    search_start_time = std::chrono::high_resolution_clock::now();

    // Ensure GPU buffers are large enough
    ensure_batch_capacity(config.simulations_per_batch);
    batch_size = config.simulations_per_batch;

    // Create root node
    root = std::make_unique<MCTSNode>(root_state);
    total_nodes = 1;
    total_simulations = 0;
    max_depth_reached = 0;

    // Generate moves for root
    generate_moves_for_node(root.get());

    if (root->legal_moves.empty()) {
        result.time_ms = elapsed_ms();
        return result;
    }

    // Main MCTS loop
    int iteration = 0;
    while (iteration < config.max_iterations && !should_stop()) {
        std::vector<MCTSNode*> leaf_nodes;
        leaf_nodes.reserve(batch_size);

        // Selection and Expansion phase
        int batch_count = std::min(batch_size, config.max_iterations - iteration);
        for (int i = 0; i < batch_count && !should_stop(); i++) {
            MCTSNode* node = select(root.get());

            // Check depth limit
            if (config.max_depth > 0 && node->depth >= config.max_depth) {
                leaf_nodes.push_back(node);
                continue;
            }

            // Expansion - add a child if not terminal
            if (!node->is_terminal && !node->is_fully_expanded()) {
                node = expand(node);
            }

            // Track max depth
            if (node->depth > max_depth_reached) {
                max_depth_reached = node->depth;
            }

            leaf_nodes.push_back(node);
        }

        if (leaf_nodes.empty()) break;

        // Simulation phase - run playouts on GPU
        simulate_batch(leaf_nodes);

        // Backpropagation phase
        for (size_t i = 0; i < leaf_nodes.size(); i++) {
            float playout_result = h_results[i];
            backpropagate(leaf_nodes[i], playout_result);
        }

        iteration += leaf_nodes.size();
        total_simulations += leaf_nodes.size();

        // Print info if requested
        if (config.verbose && config.info_interval > 0 &&
            (iteration % config.info_interval) < batch_size) {
            std::cout << "info depth " << max_depth_reached
                      << " nodes " << total_nodes
                      << " nps " << (int)(total_simulations * 1000.0f / elapsed_ms())
                      << " time " << (int)elapsed_ms() << std::endl;
        }
    }

    // Build result
    result.best_move = best_move();
    result.nodes_searched = total_nodes;
    result.simulations_run = total_simulations;
    result.time_ms = elapsed_ms();
    result.depth_reached = max_depth_reached;

    // Get score from best child
    if (!root->children.empty()) {
        MCTSNode* best = nullptr;
        int most_visits = -1;
        for (const auto& child : root->children) {
            if (child->visits > most_visits) {
                most_visits = child->visits;
                best = child.get();
            }
        }
        if (best && best->visits > 0) {
            result.score = best->wins / best->visits;
        }
    }

    // Extract PV
    get_pv(result.pv, result.pv_length, SearchResult::MAX_PV_LENGTH);

    return result;
}

Move MCTSEngine::searchToDepth(const BoardState& root_state, int depth) {
    SearchConfig config = SearchConfig::ForDepth(depth);
    config.verbose = false;
    SearchResult result = searchWithConfig(root_state, config);
    return result.best_move;
}

void MCTSEngine::get_pv(Move* pv, int& length, int max_length) const {
    length = 0;
    if (!root) return;

    const MCTSNode* node = root.get();
    while (node && !node->children.empty() && length < max_length) {
        // Find most visited child
        const MCTSNode* best = nullptr;
        int most_visits = -1;
        for (const auto& child : node->children) {
            if (child->visits > most_visits) {
                most_visits = child->visits;
                best = child.get();
            }
        }

        if (!best || best->visits == 0) break;

        pv[length++] = best->move_from_parent;
        node = best;
    }
}

bool MCTSEngine::should_stop() const {
    if (current_config.time_limit_ms <= 0) return false;
    return elapsed_ms() >= current_config.time_limit_ms;
}

float MCTSEngine::elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(now - search_start_time).count();
}

void MCTSEngine::ensure_batch_capacity(int required_size) {
    if (required_size <= max_batch_size) return;

    // Free old buffers
    if (d_boards) cudaFree(d_boards);
    if (d_results) cudaFree(d_results);
    if (h_boards) cudaFreeHost(h_boards);
    if (h_results) cudaFreeHost(h_results);

    // Allocate new buffers
    max_batch_size = required_size;
    CUDA_CHECK(cudaMalloc(&d_boards, required_size * sizeof(BoardState)));
    CUDA_CHECK(cudaMalloc(&d_results, required_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_boards, required_size * sizeof(BoardState)));
    CUDA_CHECK(cudaMallocHost(&h_results, required_size * sizeof(float)));
}

// Simple CPU-side move generation (for tree building) - this is a simplified version 

namespace cpu_movegen {

// Knight attack lookup 
static const Bitboard KNIGHT_ATTACKS_CPU[64] = {
    0x0000000000020400, 0x0000000000050800, 0x00000000000a1100, 0x0000000000142200,
    0x0000000000284400, 0x0000000000508800, 0x0000000000a01000, 0x0000000000402000,
    0x0000000002040004, 0x0000000005080008, 0x000000000a110011, 0x0000000014220022,
    0x0000000028440044, 0x0000000050880088, 0x00000000a0100010, 0x0000000040200020,
    0x0000000204000402, 0x0000000508000805, 0x0000000a1100110a, 0x0000001422002214,
    0x0000002844004428, 0x0000005088008850, 0x000000a0100010a0, 0x0000004020002040,
    0x0000020400040200, 0x0000050800080500, 0x00000a1100110a00, 0x0000142200221400,
    0x0000284400442800, 0x0000508800885000, 0x0000a0100010a000, 0x0000402000204000,
    0x0002040004020000, 0x0005080008050000, 0x000a1100110a0000, 0x0014220022140000,
    0x0028440044280000, 0x0050880088500000, 0x00a0100010a00000, 0x0040200020400000,
    0x0204000402000000, 0x0508000805000000, 0x0a1100110a000000, 0x1422002214000000,
    0x2844004428000000, 0x5088008850000000, 0xa0100010a0000000, 0x4020002040000000,
    0x0400040200000000, 0x0800080500000000, 0x1100110a00000000, 0x2200221400000000,
    0x4400442800000000, 0x8800885000000000, 0x100010a000000000, 0x2000204000000000,
    0x0004020000000000, 0x0008050000000000, 0x00110a0000000000, 0x0022140000000000,
    0x0044280000000000, 0x0088500000000000, 0x0010a00000000000, 0x0020400000000000
};

// King attack lookup 
static const Bitboard KING_ATTACKS_CPU[64] = {
    0x0000000000000302, 0x0000000000000705, 0x0000000000000e0a, 0x0000000000001c14,
    0x0000000000003828, 0x0000000000007050, 0x000000000000e0a0, 0x000000000000c040,
    0x0000000000030203, 0x0000000000070507, 0x00000000000e0a0e, 0x00000000001c141c,
    0x0000000000382838, 0x0000000000705070, 0x0000000000e0a0e0, 0x0000000000c040c0,
    0x0000000003020300, 0x0000000007050700, 0x000000000e0a0e00, 0x000000001c141c00,
    0x0000000038283800, 0x0000000070507000, 0x00000000e0a0e000, 0x00000000c040c000,
    0x0000000302030000, 0x0000000705070000, 0x0000000e0a0e0000, 0x0000001c141c0000,
    0x0000003828380000, 0x0000007050700000, 0x000000e0a0e00000, 0x000000c040c00000,
    0x0000030203000000, 0x0000070507000000, 0x00000e0a0e000000, 0x00001c141c000000,
    0x0000382838000000, 0x0000705070000000, 0x0000e0a0e0000000, 0x0000c040c0000000,
    0x0003020300000000, 0x0007050700000000, 0x000e0a0e00000000, 0x001c141c00000000,
    0x0038283800000000, 0x0070507000000000, 0x00e0a0e000000000, 0x00c040c000000000,
    0x0302030000000000, 0x0705070000000000, 0x0e0a0e0000000000, 0x1c141c0000000000,
    0x3828380000000000, 0x7050700000000000, 0xe0a0e00000000000, 0xc040c00000000000,
    0x0203000000000000, 0x0507000000000000, 0x0a0e000000000000, 0x141c000000000000,
    0x2838000000000000, 0x5070000000000000, 0xa0e0000000000000, 0x40c0000000000000
};

// Simple sliding attacks 
Bitboard rook_attacks_cpu(int sq, Bitboard occ) {
    Bitboard attacks = 0;
    int rank = sq / 8, file = sq % 8;

    // Up
    for (int r = rank + 1; r < 8; r++) {
        Bitboard bb = C64(1) << (r * 8 + file);
        attacks |= bb;
        if (occ & bb) break;
    }
    // Down
    for (int r = rank - 1; r >= 0; r--) {
        Bitboard bb = C64(1) << (r * 8 + file);
        attacks |= bb;
        if (occ & bb) break;
    }
    // Right
    for (int f = file + 1; f < 8; f++) {
        Bitboard bb = C64(1) << (rank * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    // Left
    for (int f = file - 1; f >= 0; f--) {
        Bitboard bb = C64(1) << (rank * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    return attacks;
}

Bitboard bishop_attacks_cpu(int sq, Bitboard occ) {
    Bitboard attacks = 0;
    int rank = sq / 8, file = sq % 8;

    for (int r = rank + 1, f = file + 1; r < 8 && f < 8; r++, f++) {
        Bitboard bb = C64(1) << (r * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    for (int r = rank + 1, f = file - 1; r < 8 && f >= 0; r++, f--) {
        Bitboard bb = C64(1) << (r * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    for (int r = rank - 1, f = file + 1; r >= 0 && f < 8; r--, f++) {
        Bitboard bb = C64(1) << (r * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; r--, f--) {
        Bitboard bb = C64(1) << (r * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    return attacks;
}

bool is_square_attacked_cpu(const BoardState* pos, int sq, int by_color) {
    Bitboard occ = pos->occupied();

    // Pawn attacks
    if (by_color == WHITE) {
        if (sq >= 9 && ((sq % 8) > 0) && (pos->pieces[WHITE][PAWN] & (C64(1) << (sq - 9)))) return true;
        if (sq >= 7 && ((sq % 8) < 7) && (pos->pieces[WHITE][PAWN] & (C64(1) << (sq - 7)))) return true;
    } else {
        if (sq <= 54 && ((sq % 8) > 0) && (pos->pieces[BLACK][PAWN] & (C64(1) << (sq + 7)))) return true;
        if (sq <= 56 && ((sq % 8) < 7) && (pos->pieces[BLACK][PAWN] & (C64(1) << (sq + 9)))) return true;
    }

    // Knight attacks
    if (KNIGHT_ATTACKS_CPU[sq] & pos->pieces[by_color][KNIGHT]) return true;

    // King attacks
    if (KING_ATTACKS_CPU[sq] & pos->pieces[by_color][KING]) return true;

    // Rook/Queen attacks
    Bitboard rook_attacks = rook_attacks_cpu(sq, occ);
    if (rook_attacks & (pos->pieces[by_color][ROOK] | pos->pieces[by_color][QUEEN])) return true;

    // Bishop/Queen attacks
    Bitboard bishop_attacks = bishop_attacks_cpu(sq, occ);
    if (bishop_attacks & (pos->pieces[by_color][BISHOP] | pos->pieces[by_color][QUEEN])) return true;

    return false;
}

bool in_check_cpu(const BoardState* pos) {
    int king_sq = lsb(pos->pieces[pos->side_to_move][KING]);
    return is_square_attacked_cpu(pos, king_sq, pos->side_to_move ^ 1);
}

void make_move_cpu(BoardState* pos, Move m) {
    Square from = move_from(m);
    Square to = move_to(m);
    uint8_t flags = move_flags(m);
    int us = pos->side_to_move;
    int them = us ^ 1;

    Bitboard from_bb = C64(1) << from;
    Bitboard to_bb = C64(1) << to;
    Bitboard from_to = from_bb | to_bb;

    // Find moving piece
    Piece moving_piece = NO_PIECE;
    for (int p = PAWN; p <= KING; p++) {
        if (pos->pieces[us][p] & from_bb) {
            moving_piece = p;
            break;
        }
    }

    // Handle captures
    if (flags == MOVE_EP_CAPTURE) {
        Square cap_sq = to + ((us == WHITE) ? -8 : 8);
        pos->pieces[them][PAWN] &= ~(C64(1) << cap_sq);
    } else if (is_capture(m)) {
        for (int p = PAWN; p <= QUEEN; p++) {
            if (pos->pieces[them][p] & to_bb) {
                pos->pieces[them][p] &= ~to_bb;
                break;
            }
        }
    }

    // Move piece
    pos->pieces[us][moving_piece] ^= from_to;

    // Promotions
    if (is_promotion(m)) {
        pos->pieces[us][PAWN] &= ~to_bb;
        pos->pieces[us][promotion_piece(m)] |= to_bb;
    }

    // Castling
    if (flags == MOVE_KING_CASTLE) {
        if (us == WHITE) pos->pieces[WHITE][ROOK] ^= C64(0xA0);
        else pos->pieces[BLACK][ROOK] ^= C64(0xA000000000000000);
    } else if (flags == MOVE_QUEEN_CASTLE) {
        if (us == WHITE) pos->pieces[WHITE][ROOK] ^= C64(0x09);
        else pos->pieces[BLACK][ROOK] ^= C64(0x0900000000000000);
    }

    // Update state
    pos->ep_square = -1;
    if (flags == MOVE_DOUBLE_PUSH) {
        pos->ep_square = (from + to) / 2;
    }

    // Update castling rights
    if (moving_piece == KING) {
        if (us == WHITE) pos->castling &= ~(CASTLE_WK | CASTLE_WQ);
        else pos->castling &= ~(CASTLE_BK | CASTLE_BQ);
    } else if (moving_piece == ROOK) {
        if (from == A1) pos->castling &= ~CASTLE_WQ;
        else if (from == H1) pos->castling &= ~CASTLE_WK;
        else if (from == A8) pos->castling &= ~CASTLE_BQ;
        else if (from == H8) pos->castling &= ~CASTLE_BK;
    }

    if (to == A1) pos->castling &= ~CASTLE_WQ;
    else if (to == H1) pos->castling &= ~CASTLE_WK;
    else if (to == A8) pos->castling &= ~CASTLE_BQ;
    else if (to == H8) pos->castling &= ~CASTLE_BK;

    if (moving_piece == PAWN || is_capture(m)) pos->halfmove = 0;
    else pos->halfmove++;

    pos->side_to_move ^= 1;
}

// Simplified legal move generation (enough for MCTS tree building)
int generate_legal_moves_cpu(const BoardState* pos, Move* moves) {
    int count = 0;
    int us = pos->side_to_move;
    int them = us ^ 1;
    Bitboard occ = pos->occupied();
    Bitboard our_pieces = pos->color_pieces(us);
    Bitboard their_pieces = pos->color_pieces(them);
    Bitboard empty = ~occ;

    // Pawn moves
    Bitboard pawns = pos->pieces[us][PAWN];
    int push_dir = (us == WHITE) ? 8 : -8;
    Bitboard start_rank = (us == WHITE) ? RANK_2 : RANK_7;
    Bitboard promo_rank = (us == WHITE) ? RANK_8 : RANK_1;

    while (pawns) {
        int from = pop_lsb_index(pawns);
        Bitboard from_bb = C64(1) << from;
        int to;

        // Single push
        to = from + push_dir;
        if (to >= 0 && to < 64 && (empty & (C64(1) << to))) {
            if ((C64(1) << to) & promo_rank) {
                moves[count++] = make_move(from, to, MOVE_PROMO_Q);
                moves[count++] = make_move(from, to, MOVE_PROMO_R);
                moves[count++] = make_move(from, to, MOVE_PROMO_B);
                moves[count++] = make_move(from, to, MOVE_PROMO_N);
            } else {
                moves[count++] = make_move(from, to, MOVE_QUIET);
            }

            // Double push
            if (from_bb & start_rank) {
                to = from + 2 * push_dir;
                if (empty & (C64(1) << to)) {
                    moves[count++] = make_move(from, to, MOVE_DOUBLE_PUSH);
                }
            }
        }

        // Captures
        int cap_left = (us == WHITE) ? 7 : -9;
        int cap_right = (us == WHITE) ? 9 : -7;

        to = from + cap_left;
        if (to >= 0 && to < 64 && (from % 8) > 0 && (their_pieces & (C64(1) << to))) {
            if ((C64(1) << to) & promo_rank) {
                // Generate all 4 promotion types for captures
                moves[count++] = make_move(from, to, MOVE_PROMO_CAP_Q);
                moves[count++] = make_move(from, to, MOVE_PROMO_CAP_R);
                moves[count++] = make_move(from, to, MOVE_PROMO_CAP_B);
                moves[count++] = make_move(from, to, MOVE_PROMO_CAP_N);
            } else {
                moves[count++] = make_move(from, to, MOVE_CAPTURE);
            }
        }

        to = from + cap_right;
        if (to >= 0 && to < 64 && (from % 8) < 7 && (their_pieces & (C64(1) << to))) {
            if ((C64(1) << to) & promo_rank) {
                // Generate all 4 promotion types for captures
                moves[count++] = make_move(from, to, MOVE_PROMO_CAP_Q);
                moves[count++] = make_move(from, to, MOVE_PROMO_CAP_R);
                moves[count++] = make_move(from, to, MOVE_PROMO_CAP_B);
                moves[count++] = make_move(from, to, MOVE_PROMO_CAP_N);
            } else {
                moves[count++] = make_move(from, to, MOVE_CAPTURE);
            }
        }

        // En passant - must check file boundaries to avoid wrap-around
        if (pos->ep_square >= 0) {
            int from_file = from % 8;
            // Can capture right (to higher file) if not on file H
            if (from_file < 7 && (from + cap_right) == pos->ep_square) {
                moves[count++] = make_move(from, pos->ep_square, MOVE_EP_CAPTURE);
            }
            // Can capture left (to lower file) if not on file A
            if (from_file > 0 && (from + cap_left) == pos->ep_square) {
                moves[count++] = make_move(from, pos->ep_square, MOVE_EP_CAPTURE);
            }
        }
    }

    // Knight moves
    Bitboard knights = pos->pieces[us][KNIGHT];
    while (knights) {
        int from = pop_lsb_index(knights);
        Bitboard attacks = KNIGHT_ATTACKS_CPU[from] & ~our_pieces;
        while (attacks) {
            int to = pop_lsb_index(attacks);
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(from, to, flags);
        }
    }

    // Bishop moves
    Bitboard bishops = pos->pieces[us][BISHOP];
    while (bishops) {
        int from = pop_lsb_index(bishops);
        Bitboard attacks = bishop_attacks_cpu(from, occ) & ~our_pieces;
        while (attacks) {
            int to = pop_lsb_index(attacks);
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(from, to, flags);
        }
    }

    // Rook moves
    Bitboard rooks = pos->pieces[us][ROOK];
    while (rooks) {
        int from = pop_lsb_index(rooks);
        Bitboard attacks = rook_attacks_cpu(from, occ) & ~our_pieces;
        while (attacks) {
            int to = pop_lsb_index(attacks);
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(from, to, flags);
        }
    }

    // Queen moves
    Bitboard queens = pos->pieces[us][QUEEN];
    while (queens) {
        int from = pop_lsb_index(queens);
        Bitboard attacks = (rook_attacks_cpu(from, occ) | bishop_attacks_cpu(from, occ)) & ~our_pieces;
        while (attacks) {
            int to = pop_lsb_index(attacks);
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(from, to, flags);
        }
    }

    // King moves
    int king_sq = lsb(pos->pieces[us][KING]);
    Bitboard king_attacks = KING_ATTACKS_CPU[king_sq] & ~our_pieces;
    while (king_attacks) {
        int to = pop_lsb_index(king_attacks);
        if (!is_square_attacked_cpu(pos, to, them)) {
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(king_sq, to, flags);
        }
    }

    // Castling
    if (!in_check_cpu(pos)) {
        if (us == WHITE) {
            if ((pos->castling & CASTLE_WK) &&
                !(occ & C64(0x60)) &&
                !is_square_attacked_cpu(pos, F1, them) &&
                !is_square_attacked_cpu(pos, G1, them)) {
                moves[count++] = make_move(E1, G1, MOVE_KING_CASTLE);
            }
            if ((pos->castling & CASTLE_WQ) &&
                !(occ & C64(0x0E)) &&
                !is_square_attacked_cpu(pos, D1, them) &&
                !is_square_attacked_cpu(pos, C1, them)) {
                moves[count++] = make_move(E1, C1, MOVE_QUEEN_CASTLE);
            }
        } else {
            if ((pos->castling & CASTLE_BK) &&
                !(occ & C64(0x6000000000000000)) &&
                !is_square_attacked_cpu(pos, F8, them) &&
                !is_square_attacked_cpu(pos, G8, them)) {
                moves[count++] = make_move(E8, G8, MOVE_KING_CASTLE);
            }
            if ((pos->castling & CASTLE_BQ) &&
                !(occ & C64(0x0E00000000000000)) &&
                !is_square_attacked_cpu(pos, D8, them) &&
                !is_square_attacked_cpu(pos, C8, them)) {
                moves[count++] = make_move(E8, C8, MOVE_QUEEN_CASTLE);
            }
        }
    }

    // Filter for legality (check if our king is safe after move)
    int legal_count = 0;
    for (int i = 0; i < count; i++) {
        BoardState copy = *pos;
        make_move_cpu(&copy, moves[i]);
        int our_king = lsb(copy.pieces[us][KING]);
        if (!is_square_attacked_cpu(&copy, our_king, them)) {
            moves[legal_count++] = moves[i];
        }
    }

    return legal_count;
}

}  
