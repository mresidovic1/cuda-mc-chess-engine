#ifndef KERNEL_LAUNCHERS_H
#define KERNEL_LAUNCHERS_H

#include "chess_types.cuh"
#include <cuda_runtime.h>

// Kernel launcher function declarations

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

extern "C" void launch_tactical_solver(
    const BoardState* d_positions,
    Move* d_best_moves,
    int* d_scores,
    int numPositions,
    int depth,
    cudaStream_t stream
);

#endif 
