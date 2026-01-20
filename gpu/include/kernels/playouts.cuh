#ifndef KERNELS_PLAYOUTS_CUH
#define KERNELS_PLAYOUTS_CUH

#include "common.cuh"

// Helper functions - implemented in playouts.cu
__device__ bool is_tactical_move(const BoardState* pos, Move m);
__device__ int generate_tactical_moves(const BoardState* pos, Move* moves);
__device__ int mvv_lva_score(const BoardState* pos, Move m);
__device__ int see_capture(const BoardState* pos, Move m);
__device__ int quiescence_search_simple(const BoardState* pos, int max_depth);

__global__ void RandomPlayout(const BoardState* __restrict__ starting_boards, float* __restrict__ results, int numBoards, unsigned int seed);
__global__ void EvalPlayout(const BoardState* __restrict__ starting_boards, float* __restrict__ results, int numBoards, unsigned int seed);
__global__ void StaticEval(const BoardState* __restrict__ boards, float* __restrict__ results, int numBoards);
__global__ void QuiescencePlayout(const BoardState* __restrict__ starting_boards, float* __restrict__ results, int numBoards, unsigned int seed, int max_q_depth);

#endif
