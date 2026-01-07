#ifndef MCTS_H
#define MCTS_H

#include "chess_types.cuh"
#include "search_config.h"
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

// Default batch size for GPU operations
#define DEFAULT_BATCH_SIZE 512

// MCTS Node

struct MCTSNode {
    BoardState state;
    Move move_from_parent;  // Move that led to this node
    MCTSNode* parent;
    std::vector<std::unique_ptr<MCTSNode>> children;

    // Statistics
    float wins;        // Sum of results (1.0 for win, 0.5 for draw, 0.0 for loss)
    int visits;        // Number of times this node was visited
    int depth;         // Depth in tree (root = 0)

    // Tactical information 
    bool is_check;         // This position is in check
    bool gives_check;      // The move leading to this node gives check
    int mate_distance;     // >0 = we can force mate in N, <0 = opponent mates in N, 0 = unknown

    // Cached move list
    std::vector<Move> legal_moves;
    bool moves_generated;
    bool is_terminal;

    MCTSNode(const BoardState& s, Move m = 0, MCTSNode* p = nullptr)
        : state(s), move_from_parent(m), parent(p),
          wins(0.0f), visits(0), depth(p ? p->depth + 1 : 0),
          is_check(false), gives_check(false), mate_distance(0),
          moves_generated(false), is_terminal(false) {}

    // UCB1 value for selection with configurable exploration constant
    float ucb1(int parent_visits, float exploration_c = 1.414f) const {
        if (visits == 0) return std::numeric_limits<float>::infinity();

        float exploitation = wins / visits;
        float exploration = exploration_c * std::sqrt(std::log((float)parent_visits) / visits);
        float base_value = exploitation + exploration;

        // Phase 5.1: Boost checking moves by 50%
        if (gives_check) {
            base_value *= 1.5f;
        }

        if (mate_distance > 0) {
            // We can force mate - strongly prioritize this line
            base_value += 100.0f / mate_distance;  // Shorter mates get higher bonus
        } else if (mate_distance < 0) {
            // Opponent can force mate - strongly avoid
            base_value -= 100.0f / (-mate_distance);
        }

        return base_value;
    }

    // Win rate (from perspective of node's side to move)
    float win_rate() const {
        if (visits == 0) return 0.5f;
        return wins / visits;
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

// MCTS Engine

class MCTSEngine {
public:
    MCTSEngine(int batch_size = DEFAULT_BATCH_SIZE);
    ~MCTSEngine();

    // Initialize GPU resources
    void init();

    // Main search function (legacy interface)
    Move search(const BoardState& root_state, int iterations);

    // New configurable search interface
    SearchResult searchWithConfig(const BoardState& root_state, const SearchConfig& config);

    // Search for best move with specific depth
    Move searchToDepth(const BoardState& root_state, int depth);

    // Get statistics
    int get_total_nodes() const { return total_nodes; }
    int get_total_simulations() const { return total_simulations; }
    int get_max_depth() const { return max_depth_reached; }

    // Get the current tree root (for analysis)
    const MCTSNode* get_root() const { return root.get(); }

    // Extract principal variation from tree
    void get_pv(Move* pv, int& length, int max_length = 10) const;

private:
    // Tree management
    std::unique_ptr<MCTSNode> root;
    int total_nodes;
    int total_simulations;
    int max_depth_reached;

    // Current search config
    SearchConfig current_config;

    // GPU resources
    BoardState* d_boards;    // Device memory for batch boards
    float* d_results;        // Device memory for playout results
    BoardState* h_boards;    // Pinned host memory for boards
    float* h_results;        // Pinned host memory for results
    int batch_size;
    int max_batch_size;      // Maximum allocated batch size

    // Random number generator
    std::mt19937 rng;

    // Timing
    std::chrono::high_resolution_clock::time_point search_start_time;

    // MCTS phases
    MCTSNode* select(MCTSNode* node);
    MCTSNode* expand(MCTSNode* node);
    void simulate_batch(const std::vector<MCTSNode*>& nodes);
    void backpropagate(MCTSNode* node, float result);

    void generate_moves_for_node(MCTSNode* node);
    MCTSNode* best_child_ucb1(MCTSNode* node);
    Move best_move() const;

    // Time management
    bool should_stop() const;
    float elapsed_ms() const;

    // Reallocate GPU buffers if needed
    void ensure_batch_capacity(int required_size);
};

// CPU-side move generation (used by main.cpp for making moves)

namespace cpu_movegen {
    void make_move_cpu(BoardState* pos, Move m);
    int generate_legal_moves_cpu(const BoardState* pos, Move* moves);
    bool in_check_cpu(const BoardState* pos);
}

#endif 
