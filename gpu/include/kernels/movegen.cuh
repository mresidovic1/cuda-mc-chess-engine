#ifndef KERNELS_MOVEGEN_CUH
#define KERNELS_MOVEGEN_CUH

#include "common.cuh"

// Direction shifts - defined inline in header
__device__ __forceinline__ Bitboard shift_north(Bitboard b) { return b << 8; }
__device__ __forceinline__ Bitboard shift_south(Bitboard b) { return b >> 8; }
__device__ __forceinline__ Bitboard shift_east(Bitboard b) { return (b << 1) & ~FILE_A; }
__device__ __forceinline__ Bitboard shift_west(Bitboard b) { return (b >> 1) & ~FILE_H; }
__device__ __forceinline__ Bitboard shift_ne(Bitboard b) { return (b << 9) & ~FILE_A; }
__device__ __forceinline__ Bitboard shift_nw(Bitboard b) { return (b << 7) & ~FILE_H; }
__device__ __forceinline__ Bitboard shift_se(Bitboard b) { return (b >> 7) & ~FILE_A; }
__device__ __forceinline__ Bitboard shift_sw(Bitboard b) { return (b >> 9) & ~FILE_H; }

// Forward declarations for attack tables
extern __constant__ Bitboard g_ROOK_MAGICS[64];
extern __constant__ Bitboard g_BISHOP_MAGICS[64];
extern __constant__ Bitboard g_ROOK_MASKS[64];
extern __constant__ Bitboard g_BISHOP_MASKS[64];
extern __device__ Bitboard g_ROOK_ATTACKS[64][4096];
extern __device__ Bitboard g_BISHOP_ATTACKS[64][512];

// Magic bitboard lookups - defined inline in header
__device__ __forceinline__ Bitboard rook_attacks(Square sq, Bitboard occ) {
    occ &= g_ROOK_MASKS[sq];
    occ *= g_ROOK_MAGICS[sq];
    occ >>= (64 - 12); // ROOK_MAGIC_BITS
    return g_ROOK_ATTACKS[sq][occ];
}

__device__ __forceinline__ Bitboard bishop_attacks(Square sq, Bitboard occ) {
    occ &= g_BISHOP_MASKS[sq];
    occ *= g_BISHOP_MAGICS[sq];
    occ >>= (64 - 9); // BISHOP_MAGIC_BITS
    return g_BISHOP_ATTACKS[sq][occ];
}

__device__ __forceinline__ Bitboard queen_attacks(Square sq, Bitboard occ) {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

// Forward declarations for attack detection helpers
extern __constant__ Bitboard g_PAWN_ATTACKS[2][64];
extern __constant__ Bitboard g_KNIGHT_ATTACKS[64];
extern __constant__ Bitboard g_KING_ATTACKS[64];

// Attack detection - defined inline in header
__device__ __forceinline__ bool is_attacked(const BoardState* pos, Square sq, int by_color) {
    Bitboard occ = pos->occupied();
    Bitboard attackers =
        (g_PAWN_ATTACKS[by_color ^ 1][sq] & pos->pieces[by_color][PAWN]) |
        (g_KNIGHT_ATTACKS[sq] & pos->pieces[by_color][KNIGHT]) |
        (g_KING_ATTACKS[sq] & pos->pieces[by_color][KING]) |
        (rook_attacks(sq, occ) & (pos->pieces[by_color][ROOK] | pos->pieces[by_color][QUEEN])) |
        (bishop_attacks(sq, occ) & (pos->pieces[by_color][BISHOP] | pos->pieces[by_color][QUEEN]));
    return attackers != 0;
}

__device__ __forceinline__ bool in_check(const BoardState* pos) {
    Square king_sq = lsb(pos->pieces[pos->side_to_move][KING]);
    return is_attacked(pos, king_sq, pos->side_to_move ^ 1);
}

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
