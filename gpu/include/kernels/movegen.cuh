#ifndef KERNELS_MOVEGEN_CUH
#define KERNELS_MOVEGEN_CUH

#include "common.cuh"

// Direction shifts
__device__ __forceinline__ Bitboard shift_north(Bitboard b);
__device__ __forceinline__ Bitboard shift_south(Bitboard b);
__device__ __forceinline__ Bitboard shift_east(Bitboard b);
__device__ __forceinline__ Bitboard shift_west(Bitboard b);
__device__ __forceinline__ Bitboard shift_ne(Bitboard b);
__device__ __forceinline__ Bitboard shift_nw(Bitboard b);
__device__ __forceinline__ Bitboard shift_se(Bitboard b);
__device__ __forceinline__ Bitboard shift_sw(Bitboard b);

// Magic bitboard lookups
__device__ __forceinline__ Bitboard rook_attacks(Square sq, Bitboard occ);
__device__ __forceinline__ Bitboard bishop_attacks(Square sq, Bitboard occ);
__device__ __forceinline__ Bitboard queen_attacks(Square sq, Bitboard occ);

// Move generation
__device__ int generate_pawn_moves(const BoardState* pos, Move* moves, Bitboard target);
__device__ int generate_knight_moves(const BoardState* pos, Move* moves, Bitboard target);
__device__ int generate_bishop_moves(const BoardState* pos, Move* moves, Bitboard target);
__device__ int generate_rook_moves(const BoardState* pos, Move* moves, Bitboard target);
__device__ int generate_queen_moves(const BoardState* pos, Move* moves, Bitboard target);
__device__ int generate_king_moves(const BoardState* pos, Move* moves, Bitboard target);
__device__ int generate_pseudo_legal_moves(const BoardState* pos, Move* moves);
__device__ int generate_legal_moves(const BoardState* pos, Move* moves);

// Make move
__device__ void make_move(BoardState* pos, Move m);

// Attack detection
__device__ bool is_attacked(const BoardState* pos, Square sq, int by_color);
__device__ __forceinline__ bool in_check(const BoardState* pos);

// Table initialization functions
extern "C" cudaError_t copy_knight_attacks(const Bitboard* data);
extern "C" cudaError_t copy_king_attacks(const Bitboard* data);
extern "C" cudaError_t copy_pawn_attacks(const Bitboard* data);
extern "C" cudaError_t copy_rook_magics(const Bitboard* data);
extern "C" cudaError_t copy_bishop_magics(const Bitboard* data);
extern "C" cudaError_t copy_rook_masks(const Bitboard* data);
extern "C" cudaError_t copy_bishop_masks(const Bitboard* data);
extern "C" cudaError_t copy_rook_attacks(const Bitboard* data);
extern "C" cudaError_t copy_bishop_attacks(const Bitboard* data);

#endif
