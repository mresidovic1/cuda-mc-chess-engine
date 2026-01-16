#ifndef KERNELS_EVALUATION_CUH
#define KERNELS_EVALUATION_CUH

#include "common.cuh"

#define EVAL_PAWN   100
#define EVAL_KNIGHT 320
#define EVAL_BISHOP 330
#define EVAL_ROOK   500
#define EVAL_QUEEN  900

__device__ int gpu_evaluate(const BoardState* pos);
__device__ float score_to_winprob(int score, int side_to_move);

#endif
