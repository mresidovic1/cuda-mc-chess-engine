#pragma once
#include "gpu_chess_types.cuh"

// ============================================================================
// CUDA Intrinsics for Bitboards
// ============================================================================

__device__ __forceinline__ int pop_lsb(Bitboard* b) {
    int idx = __ffsll(*b) - 1;
    *b &= *b - 1;
    return idx;
}

__device__ __forceinline__ int lsb(Bitboard b) {
    return __ffsll(b) - 1;
}

__device__ __forceinline__ int count_bits(Bitboard b) {
    return __popcll(b);
}

// ============================================================================
// Bitboard Shifts with Edge Protection
// ============================================================================

__device__ __forceinline__ Bitboard north(Bitboard b) {
    return b << 8;
}

__device__ __forceinline__ Bitboard south(Bitboard b) {
    return b >> 8;
}

__device__ __forceinline__ Bitboard east(Bitboard b) {
    return (b << 1) & ~FILE_A;
}

__device__ __forceinline__ Bitboard west(Bitboard b) {
    return (b >> 1) & ~FILE_H;
}

__device__ __forceinline__ Bitboard north_east(Bitboard b) {
    return (b << 9) & ~FILE_A;
}

__device__ __forceinline__ Bitboard north_west(Bitboard b) {
    return (b << 7) & ~FILE_H;
}

__device__ __forceinline__ Bitboard south_east(Bitboard b) {
    return (b >> 7) & ~FILE_A;
}

__device__ __forceinline__ Bitboard south_west(Bitboard b) {
    return (b >> 9) & ~FILE_H;
}

// ============================================================================
// Leaper Attacks (Knight, King, Pawn)
// ============================================================================

__device__ __forceinline__ Bitboard knight_attacks(Bitboard knights) {
    Bitboard l1 = (knights >> 1) & ~FILE_H;
    Bitboard l2 = (knights >> 2) & ~(FILE_G | FILE_H);
    Bitboard r1 = (knights << 1) & ~FILE_A;
    Bitboard r2 = (knights << 2) & ~(FILE_A | FILE_B);
    
    Bitboard h1 = l1 | r1;
    Bitboard h2 = l2 | r2;
    
    return (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8);
}

__device__ __forceinline__ Bitboard king_attacks(Bitboard king) {
    Bitboard attacks = east(king) | west(king);
    king |= attacks;
    return attacks | north(king) | south(king);
}

__device__ __forceinline__ Bitboard pawn_attacks(Bitboard pawns, Color c) {
    if (c == WHITE) {
        return north_east(pawns) | north_west(pawns);
    } else {
        return south_east(pawns) | south_west(pawns);
    }
}

// ============================================================================
// Sliding Piece Attacks (Kogge-Stone Fill)
// ============================================================================

// Generic ray fill in a direction
__device__ __forceinline__ Bitboard ray_fill(Bitboard gen, Bitboard empty, 
                                              Bitboard (*shift_fn)(Bitboard)) {
    Bitboard result = gen;
    for (int i = 0; i < 7; ++i) {
        result |= shift_fn(result) & empty;
    }
    return result;
}

__device__ __forceinline__ Bitboard ray_attacks(Bitboard pieces, Bitboard empty,
                                                 Bitboard (*shift_fn)(Bitboard)) {
    Bitboard fill = ray_fill(pieces, empty, shift_fn);
    return shift_fn(fill);
}

__device__ Bitboard get_rook_attacks(Bitboard rooks, Bitboard empty) {
    return ray_attacks(rooks, empty, north) |
           ray_attacks(rooks, empty, south) |
           ray_attacks(rooks, empty, east) |
           ray_attacks(rooks, empty, west);
}

__device__ Bitboard get_bishop_attacks(Bitboard bishops, Bitboard empty) {
    return ray_attacks(bishops, empty, north_east) |
           ray_attacks(bishops, empty, north_west) |
           ray_attacks(bishops, empty, south_east) |
           ray_attacks(bishops, empty, south_west);
}

__device__ Bitboard get_queen_attacks(Bitboard queens, Bitboard empty) {
    return get_rook_attacks(queens, empty) | get_bishop_attacks(queens, empty);
}
