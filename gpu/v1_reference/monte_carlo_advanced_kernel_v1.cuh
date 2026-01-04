//N - v1: Complete rewrite of header with compact structures and MCTS support
#pragma once
#include <cuda_runtime.h>
#include "monte_carlo_advanced_types_v1.hpp"

// ============================================================================
// N - v1: Version identifier
// ============================================================================
#define GPU_MCTS_VERSION "1.0"

// ============================================================================
// GPU Constant Memory Declarations/Definitions
// ============================================================================
#ifdef GPU_CONST_DEF
__constant__ int d_piece_values[6] = {100, 300, 320, 500, 900, 20000};
__constant__ int d_pawn_table[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
__constant__ int d_knight_table[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};
__constant__ int d_bishop_table[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};
__constant__ int d_rook_table[64] = {
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
__constant__ int d_queen_table[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};
__constant__ int d_king_table[64] = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
};

//N - v1: Zobrist keys for hashing
__constant__ uint64_t d_zobrist_pieces[64][16];
__constant__ uint64_t d_zobrist_side;
__constant__ uint64_t d_zobrist_castling[16];
__constant__ uint64_t d_zobrist_ep[8];

#else
extern __constant__ int d_piece_values[6];
extern __constant__ int d_pawn_table[64];
extern __constant__ int d_knight_table[64];
extern __constant__ int d_bishop_table[64];
extern __constant__ int d_rook_table[64];
extern __constant__ int d_queen_table[64];
extern __constant__ int d_king_table[64];
extern __constant__ uint64_t d_zobrist_pieces[64][16];
extern __constant__ uint64_t d_zobrist_side;
extern __constant__ uint64_t d_zobrist_castling[16];
extern __constant__ uint64_t d_zobrist_ep[8];
#endif

// ============================================================================
//N - v1: Global device memory declarations/definitions
// ============================================================================
#ifdef GPU_CONST_DEF
__device__ GPUTTEntry* d_transposition_table = nullptr;
__device__ int d_history_table[2][64][64];
__device__ MCTSNode* d_mcts_nodes = nullptr;
__device__ int d_mcts_node_count = 0;
#else
extern __device__ GPUTTEntry* d_transposition_table;
extern __device__ int d_history_table[2][64][64];
extern __device__ MCTSNode* d_mcts_nodes;
extern __device__ int d_mcts_node_count;
#endif

// ============================================================================
// Kernel Function Declarations
// ============================================================================

//N - v1: Legacy kernel (single move evaluation)
__global__ void monte_carlo_simulate_kernel(
    const Position root_position,
    const Move root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed
);

//N - v1: New batched kernel (all moves in one launch)
__global__ void monte_carlo_simulate_batch_kernel(
    const Position root_position,
    const Move* all_moves,
    int num_moves,
    int num_simulations_per_move,
    float* results,
    unsigned long long seed
);

//N - v1: MCTS kernel with UCB tree policy
__global__ void mcts_tree_kernel(
    const Position root_position,
    MCTSNode* tree_nodes,
    int* node_count,
    GPUTTEntry* transposition_table,
    int num_iterations,
    unsigned long long seed
);

//N - v1: Reduction kernel for collecting results
__global__ void reduce_simulation_results(
    float* block_results,
    float* final_results,
    int num_blocks_per_move,
    int num_moves
);

// ============================================================================
// Launch Function Declarations
// ============================================================================

extern "C" void launch_monte_carlo_simulate_kernel(
    const Position* root_position,
    const Move* root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed,
    int blocks,
    int threads_per_block
);

//N - v1: New batched launch function with CUDA streams
extern "C" void launch_monte_carlo_batch_kernel(
    const Position* root_position,
    const Move* all_moves,
    int num_moves,
    int simulations_per_move,
    float* results,
    unsigned long long seed
);

//N - v1: MCTS launch function
extern "C" void launch_mcts_kernel(
    const Position* root_position,
    int num_iterations,
    float* move_scores,
    int* best_move_idx
);

//N - v1: Initialize GPU resources (TT, history, Zobrist)
extern "C" void initialize_gpu_resources();

//N - v1: Cleanup GPU resources
extern "C" void cleanup_gpu_resources();

//N - v1: Clear transposition table
extern "C" void clear_gpu_transposition_table();

//N - v1: Clear history table
extern "C" void clear_gpu_history_table();
