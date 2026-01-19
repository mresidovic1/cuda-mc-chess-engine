#ifndef KERNELS_TACTICAL_CUH
#define KERNELS_TACTICAL_CUH

#include "common.cuh"
#include "movegen.cuh"
#include "playouts.cuh"

// Inline helper functions defined in header
__device__ __forceinline__ bool gives_check_simple(BoardState* pos, Move m) {
    BoardState temp = *pos;
    make_move(&temp, m);
    return in_check(&temp);
}

__device__ __forceinline__ int tactical_move_score(const BoardState* pos, Move m, bool gives_check) {
    int move_type = (m >> 12) & 0xF;
    if (gives_check) return 1000000;
    if (move_type >= MOVE_PROMO_CAP_N && move_type <= MOVE_PROMO_CAP_Q) {
        return 100000 + see_capture(pos, m);
    }
    if (move_type >= MOVE_PROMO_N && move_type <= MOVE_PROMO_Q) {
        return 50000 + ((move_type - MOVE_PROMO_N) * 1000);
    }
    if (move_type == MOVE_CAPTURE || move_type == MOVE_EP_CAPTURE) {
        int see = see_capture(pos, m);
        return 10000 + see * 10 + mvv_lva_score(pos, m);
    }
    return 0;
}

// Non-inline tactical search functions
__device__ __noinline__ int tactical_depth2(BoardState* pos, int alpha, int beta, int ply);
__device__ __noinline__ int tactical_depth4(BoardState* pos, int alpha, int beta, int ply);
__device__ __noinline__ int tactical_depth6(BoardState* pos, int alpha, int beta, int ply);

// Kernel declaration
__global__ void TacticalSolver(const BoardState* __restrict__ positions, Move* __restrict__ best_moves, int* __restrict__ scores, int numPositions, int depth);

#endif
