#include "../../include/chess_types.cuh"

// Constants needed for launchers
#define BLOCK_SIZE 256

// Forward declarations of kernels
__global__ void RandomPlayout(
    const BoardState* __restrict__ boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed
);

__global__ void EvalPlayout(
    const BoardState* __restrict__ boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed
);

__global__ void StaticEval(
    const BoardState* __restrict__ boards,
    float* __restrict__ results,
    int numBoards
);

__global__ void QuiescencePlayout(
    const BoardState* __restrict__ boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed,
    int max_q_depth
);

__global__ void TacticalSolver(
    const BoardState* __restrict__ positions,
    Move* __restrict__ best_moves,
    int* __restrict__ scores,
    int numPositions,
    int depth
);

// Host-side kernel launchers

extern "C" void launch_random_playout(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    unsigned int seed,
    cudaStream_t stream
) {
    int blocks = (numBoards + BLOCK_SIZE - 1) / BLOCK_SIZE;
    RandomPlayout<<<blocks, BLOCK_SIZE, 0, stream>>>(d_boards, d_results, numBoards, seed);
}

extern "C" void launch_eval_playout(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    unsigned int seed,
    cudaStream_t stream
) {
    int blocks = (numBoards + BLOCK_SIZE - 1) / BLOCK_SIZE;
    EvalPlayout<<<blocks, BLOCK_SIZE, 0, stream>>>(d_boards, d_results, numBoards, seed);
}

extern "C" void launch_static_eval(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    cudaStream_t stream
) {
    int blocks = (numBoards + BLOCK_SIZE - 1) / BLOCK_SIZE;
    StaticEval<<<blocks, BLOCK_SIZE, 0, stream>>>(d_boards, d_results, numBoards);
}

extern "C" void launch_quiescence_playout(
    const BoardState* d_boards,
    float* d_results,
    int numBoards,
    unsigned int seed,
    int max_q_depth,
    cudaStream_t stream
) {
    int blocks = (numBoards + BLOCK_SIZE - 1) / BLOCK_SIZE;
    QuiescencePlayout<<<blocks, BLOCK_SIZE, 0, stream>>>(d_boards, d_results, numBoards, seed, max_q_depth);
}
extern "C" void launch_tactical_solver(
    const BoardState* d_positions,
    Move* d_best_moves,
    int* d_scores,
    int numPositions,
    int depth,
    cudaStream_t stream
) {
    const int BLOCK_SIZE_TACTICAL = 64;  
    int blocks = (numPositions + BLOCK_SIZE_TACTICAL - 1) / BLOCK_SIZE_TACTICAL;
    TacticalSolver<<<blocks, BLOCK_SIZE_TACTICAL, 0, stream>>>(
        d_positions, d_best_moves, d_scores, numPositions, depth
    );
}
