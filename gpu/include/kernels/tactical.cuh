#ifndef KERNELS_TACTICAL_CUH
#define KERNELS_TACTICAL_CUH

#include "common.cuh"

__device__ __forceinline__ bool gives_check_simple(BoardState* pos, Move m);
__device__ __forceinline__ int tactical_move_score(const BoardState* pos, Move m, bool gives_check);
__device__ __noinline__ int tactical_depth2(BoardState* pos, int alpha, int beta, int ply);
__device__ __noinline__ int tactical_depth4(BoardState* pos, int alpha, int beta, int ply);
__device__ __noinline__ int tactical_depth6(BoardState* pos, int alpha, int beta, int ply);

__global__ void TacticalSolver(const BoardState* __restrict__ positions, Move* __restrict__ best_moves, int* __restrict__ scores, int numPositions, int depth);

#endif
