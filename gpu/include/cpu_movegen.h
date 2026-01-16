#ifndef CPU_MOVEGEN_H
#define CPU_MOVEGEN_H

#include "chess_types.cuh"

// CPU-side move generation (used by both MCTS and PUCT engines for tree building/validation)
// This implementation does not require GPU resources

namespace cpu_movegen {

// Constants for attack lookups
extern const Bitboard KNIGHT_ATTACKS_CPU[64];
extern const Bitboard KING_ATTACKS_CPU[64];

// Core move generation functions
void make_move_cpu(BoardState* pos, Move m);
int generate_legal_moves_cpu(const BoardState* pos, Move* moves);
bool in_check_cpu(const BoardState* pos);

// Helper functions usually internal but exposed for testing/completeness
bool is_square_attacked_cpu(const BoardState* pos, int sq, int by_color);
Bitboard rook_attacks_cpu(int sq, Bitboard occ);
Bitboard bishop_attacks_cpu(int sq, Bitboard occ);

} // namespace cpu_movegen

#endif // CPU_MOVEGEN_H
