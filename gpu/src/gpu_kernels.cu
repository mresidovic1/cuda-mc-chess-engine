#include "../include/chess_types.cuh"
#include <curand_kernel.h>
#include <cstdio>

// Constants for MCTS

#define MAX_PLAYOUT_MOVES 500  // Maximum moves per playout
#define BLOCK_SIZE 256         // Threads per block
#define ROOK_MAGIC_BITS   12
#define BISHOP_MAGIC_BITS 9

// Attack tables

__constant__ Bitboard g_KNIGHT_ATTACKS[64];
__constant__ Bitboard g_KING_ATTACKS[64];
__constant__ Bitboard g_PAWN_ATTACKS[2][64];

__constant__ Bitboard g_ROOK_MAGICS[64];
__constant__ Bitboard g_BISHOP_MAGICS[64];
__constant__ Bitboard g_ROOK_MASKS[64];
__constant__ Bitboard g_BISHOP_MASKS[64];

// Large attack tables in global memory
__device__ Bitboard g_ROOK_ATTACKS[64][1 << ROOK_MAGIC_BITS];
__device__ Bitboard g_BISHOP_ATTACKS[64][1 << BISHOP_MAGIC_BITS];

// Direction shifts

__device__ __forceinline__
Bitboard shift_north(Bitboard b) { return b << 8; }

__device__ __forceinline__
Bitboard shift_south(Bitboard b) { return b >> 8; }

__device__ __forceinline__
Bitboard shift_east(Bitboard b) { return (b << 1) & ~FILE_A; }

__device__ __forceinline__
Bitboard shift_west(Bitboard b) { return (b >> 1) & ~FILE_H; }

__device__ __forceinline__
Bitboard shift_ne(Bitboard b) { return (b << 9) & ~FILE_A; }

__device__ __forceinline__
Bitboard shift_nw(Bitboard b) { return (b << 7) & ~FILE_H; }

__device__ __forceinline__
Bitboard shift_se(Bitboard b) { return (b >> 7) & ~FILE_A; }

__device__ __forceinline__
Bitboard shift_sw(Bitboard b) { return (b >> 9) & ~FILE_H; }

// Magic bitboard lookups

__device__ __forceinline__
Bitboard rook_attacks(Square sq, Bitboard occ) {
    occ &= g_ROOK_MASKS[sq];
    occ *= g_ROOK_MAGICS[sq];
    occ >>= (64 - ROOK_MAGIC_BITS);
    return g_ROOK_ATTACKS[sq][occ];
}

__device__ __forceinline__
Bitboard bishop_attacks(Square sq, Bitboard occ) {
    occ &= g_BISHOP_MASKS[sq];
    occ *= g_BISHOP_MAGICS[sq];
    occ >>= (64 - BISHOP_MAGIC_BITS);
    return g_BISHOP_ATTACKS[sq][occ];
}

__device__ __forceinline__
Bitboard queen_attacks(Square sq, Bitboard occ) {
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ);
}

// Attack detection

__device__ __forceinline__
bool is_attacked(const BoardState* pos, Square sq, int by_color) {
    Bitboard occ = pos->occupied();
    Bitboard attackers =
        (g_PAWN_ATTACKS[by_color ^ 1][sq] & pos->pieces[by_color][PAWN]) |
        (g_KNIGHT_ATTACKS[sq] & pos->pieces[by_color][KNIGHT]) |
        (g_KING_ATTACKS[sq] & pos->pieces[by_color][KING]) |
        (rook_attacks(sq, occ) & (pos->pieces[by_color][ROOK] | pos->pieces[by_color][QUEEN])) |
        (bishop_attacks(sq, occ) & (pos->pieces[by_color][BISHOP] | pos->pieces[by_color][QUEEN]));
    return attackers != 0;
}

__device__ __forceinline__
bool in_check(const BoardState* pos) {
    Square king_sq = lsb(pos->pieces[pos->side_to_move][KING]);
    return is_attacked(pos, king_sq, pos->side_to_move ^ 1);
}

// Move generation helpers

__device__
int generate_pawn_moves(const BoardState* pos, Move* moves, Bitboard target) {
    int count = 0;
    int us = pos->side_to_move;
    int them = us ^ 1;

    Bitboard pawns = pos->pieces[us][PAWN];
    Bitboard occ = pos->occupied();
    Bitboard empty = ~occ;
    Bitboard enemies = pos->color_pieces(them);

    int push_dir = (us == WHITE) ? 8 : -8;
    Bitboard promo_rank = (us == WHITE) ? RANK_8 : RANK_1;

    // Single pushes
    Bitboard single_push = (us == WHITE) ? shift_north(pawns) : shift_south(pawns);
    single_push &= empty;

    // Double pushes
    Bitboard double_push = (us == WHITE) ? shift_north(single_push & RANK_3) :
                                           shift_south(single_push & RANK_6);
    double_push &= empty;

    // Captures
    Bitboard left_cap = (us == WHITE) ? shift_nw(pawns) : shift_sw(pawns);
    Bitboard right_cap = (us == WHITE) ? shift_ne(pawns) : shift_se(pawns);
    left_cap &= enemies;
    right_cap &= enemies;

    // En passant
    Bitboard ep_target = (pos->ep_square >= 0) ? (C64(1) << pos->ep_square) : 0;
    Bitboard ep_left = (us == WHITE) ? shift_nw(pawns) : shift_sw(pawns);
    Bitboard ep_right = (us == WHITE) ? shift_ne(pawns) : shift_se(pawns);
    ep_left &= ep_target;
    ep_right &= ep_target;

    // Non-promotion moves
    Bitboard non_promo_push = single_push & ~promo_rank & target;
    while (non_promo_push) {
        Square to = pop_lsb_index(non_promo_push);
        Square from = to - push_dir;
        moves[count++] = make_move(from, to, MOVE_QUIET);
    }

    Bitboard dbl = double_push & target;
    while (dbl) {
        Square to = pop_lsb_index(dbl);
        Square from = to - 2 * push_dir;
        moves[count++] = make_move(from, to, MOVE_DOUBLE_PUSH);
    }

    Bitboard lc = left_cap & ~promo_rank & target;
    while (lc) {
        Square to = pop_lsb_index(lc);
        Square from = to - push_dir + 1;
        moves[count++] = make_move(from, to, MOVE_CAPTURE);
    }

    Bitboard rc = right_cap & ~promo_rank & target;
    while (rc) {
        Square to = pop_lsb_index(rc);
        Square from = to - push_dir - 1;
        moves[count++] = make_move(from, to, MOVE_CAPTURE);
    }

    // Promotions
    Bitboard promo_push = single_push & promo_rank & target;
    while (promo_push) {
        Square to = pop_lsb_index(promo_push);
        Square from = to - push_dir;
        moves[count++] = make_move(from, to, MOVE_PROMO_Q);
        moves[count++] = make_move(from, to, MOVE_PROMO_R);
        moves[count++] = make_move(from, to, MOVE_PROMO_B);
        moves[count++] = make_move(from, to, MOVE_PROMO_N);
    }

    Bitboard promo_lc = left_cap & promo_rank & target;
    while (promo_lc) {
        Square to = pop_lsb_index(promo_lc);
        Square from = to - push_dir + 1;
        moves[count++] = make_move(from, to, MOVE_PROMO_CAP_Q);
        moves[count++] = make_move(from, to, MOVE_PROMO_CAP_R);
        moves[count++] = make_move(from, to, MOVE_PROMO_CAP_B);
        moves[count++] = make_move(from, to, MOVE_PROMO_CAP_N);
    }

    Bitboard promo_rc = right_cap & promo_rank & target;
    while (promo_rc) {
        Square to = pop_lsb_index(promo_rc);
        Square from = to - push_dir - 1;
        moves[count++] = make_move(from, to, MOVE_PROMO_CAP_Q);
        moves[count++] = make_move(from, to, MOVE_PROMO_CAP_R);
        moves[count++] = make_move(from, to, MOVE_PROMO_CAP_B);
        moves[count++] = make_move(from, to, MOVE_PROMO_CAP_N);
    }

    // En passant
    while (ep_left) {
        Square to = pop_lsb_index(ep_left);
        Square from = to - push_dir + 1;
        moves[count++] = make_move(from, to, MOVE_EP_CAPTURE);
    }

    while (ep_right) {
        Square to = pop_lsb_index(ep_right);
        Square from = to - push_dir - 1;
        moves[count++] = make_move(from, to, MOVE_EP_CAPTURE);
    }

    return count;
}

__device__
int generate_knight_moves(const BoardState* pos, Move* moves, Bitboard target) {
    int count = 0;
    int us = pos->side_to_move;
    Bitboard knights = pos->pieces[us][KNIGHT];
    Bitboard our_pieces = pos->us();
    Bitboard enemy = pos->them();

    while (knights) {
        Square from = pop_lsb_index(knights);
        Bitboard attacks = g_KNIGHT_ATTACKS[from] & ~our_pieces & target;

        while (attacks) {
            Square to = pop_lsb_index(attacks);
            uint8_t flags = (enemy & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(from, to, flags);
        }
    }
    return count;
}

__device__
int generate_bishop_moves(const BoardState* pos, Move* moves, Bitboard target) {
    int count = 0;
    int us = pos->side_to_move;
    Bitboard bishops = pos->pieces[us][BISHOP];
    Bitboard occ = pos->occupied();
    Bitboard our_pieces = pos->us();
    Bitboard enemy = pos->them();

    while (bishops) {
        Square from = pop_lsb_index(bishops);
        Bitboard attacks = bishop_attacks(from, occ) & ~our_pieces & target;

        while (attacks) {
            Square to = pop_lsb_index(attacks);
            uint8_t flags = (enemy & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(from, to, flags);
        }
    }
    return count;
}

__device__
int generate_rook_moves(const BoardState* pos, Move* moves, Bitboard target) {
    int count = 0;
    int us = pos->side_to_move;
    Bitboard rooks = pos->pieces[us][ROOK];
    Bitboard occ = pos->occupied();
    Bitboard our_pieces = pos->us();
    Bitboard enemy = pos->them();

    while (rooks) {
        Square from = pop_lsb_index(rooks);
        Bitboard attacks = rook_attacks(from, occ) & ~our_pieces & target;

        while (attacks) {
            Square to = pop_lsb_index(attacks);
            uint8_t flags = (enemy & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(from, to, flags);
        }
    }
    return count;
}

__device__
int generate_queen_moves(const BoardState* pos, Move* moves, Bitboard target) {
    int count = 0;
    int us = pos->side_to_move;
    Bitboard queens = pos->pieces[us][QUEEN];
    Bitboard occ = pos->occupied();
    Bitboard our_pieces = pos->us();
    Bitboard enemy = pos->them();

    while (queens) {
        Square from = pop_lsb_index(queens);
        Bitboard attacks = queen_attacks(from, occ) & ~our_pieces & target;

        while (attacks) {
            Square to = pop_lsb_index(attacks);
            uint8_t flags = (enemy & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(from, to, flags);
        }
    }
    return count;
}

__device__
int generate_king_moves(const BoardState* pos, Move* moves, Bitboard target) {
    int count = 0;
    int us = pos->side_to_move;
    int them = us ^ 1;
    Square king_sq = lsb(pos->pieces[us][KING]);
    Bitboard our_pieces = pos->us();
    Bitboard enemy = pos->them();
    Bitboard occ = pos->occupied();

    // Normal king moves
    Bitboard attacks = g_KING_ATTACKS[king_sq] & ~our_pieces & target;
    while (attacks) {
        Square to = pop_lsb_index(attacks);
        if (!is_attacked(pos, to, them)) {
            uint8_t flags = (enemy & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = make_move(king_sq, to, flags);
        }
    }

    // Castling
    if (!in_check(pos)) {
        if (us == WHITE) {
            if ((pos->castling & CASTLE_WK) &&
                !(occ & C64(0x60)) &&
                !is_attacked(pos, F1, them) &&
                !is_attacked(pos, G1, them)) {
                moves[count++] = make_move(E1, G1, MOVE_KING_CASTLE);
            }
            if ((pos->castling & CASTLE_WQ) &&
                !(occ & C64(0x0E)) &&
                !is_attacked(pos, D1, them) &&
                !is_attacked(pos, C1, them)) {
                moves[count++] = make_move(E1, C1, MOVE_QUEEN_CASTLE);
            }
        } else {
            if ((pos->castling & CASTLE_BK) &&
                !(occ & C64(0x6000000000000000)) &&
                !is_attacked(pos, F8, them) &&
                !is_attacked(pos, G8, them)) {
                moves[count++] = make_move(E8, G8, MOVE_KING_CASTLE);
            }
            if ((pos->castling & CASTLE_BQ) &&
                !(occ & C64(0x0E00000000000000)) &&
                !is_attacked(pos, D8, them) &&
                !is_attacked(pos, C8, them)) {
                moves[count++] = make_move(E8, C8, MOVE_QUEEN_CASTLE);
            }
        }
    }

    return count;
}

// Generate all pseudo-legal moves

__device__
int generate_pseudo_legal_moves(const BoardState* pos, Move* moves) {
    Bitboard target = ALL_SQUARES;
    int count = 0;
    count += generate_pawn_moves(pos, moves + count, target);
    count += generate_knight_moves(pos, moves + count, target);
    count += generate_bishop_moves(pos, moves + count, target);
    count += generate_rook_moves(pos, moves + count, target);
    count += generate_queen_moves(pos, moves + count, target);
    count += generate_king_moves(pos, moves + count, target);
    return count;
}

// Make move

__device__
void make_move(BoardState* pos, Move m) {
    Square from = move_from(m);
    Square to = move_to(m);
    uint8_t flags = move_flags(m);
    int us = pos->side_to_move;
    int them = us ^ 1;

    Bitboard from_bb = C64(1) << from;
    Bitboard to_bb = C64(1) << to;
    Bitboard from_to = from_bb | to_bb;

    // Find the moving piece
    Piece moving_piece = NO_PIECE;
    for (int p = PAWN; p <= KING; p++) {
        if (pos->pieces[us][p] & from_bb) {
            moving_piece = p;
            break;
        }
    }

    // Remove captured piece - if any
    if (flags == MOVE_EP_CAPTURE) {
        Square cap_sq = to + ((us == WHITE) ? -8 : 8);
        pos->pieces[them][PAWN] &= ~(C64(1) << cap_sq);
    } else if (is_capture(m)) {
        for (int p = PAWN; p <= QUEEN; p++) {
            if (pos->pieces[them][p] & to_bb) {
                pos->pieces[them][p] &= ~to_bb;
                break;
            }
        }
    }

    // Move the piece
    pos->pieces[us][moving_piece] ^= from_to;

    // Handle promotions
    if (is_promotion(m)) {
        pos->pieces[us][PAWN] &= ~to_bb;
        pos->pieces[us][promotion_piece(m)] |= to_bb;
    }

    // Handle castling
    if (flags == MOVE_KING_CASTLE) {
        if (us == WHITE) {
            pos->pieces[WHITE][ROOK] ^= C64(0xA0);
        } else {
            pos->pieces[BLACK][ROOK] ^= C64(0xA000000000000000);
        }
    } else if (flags == MOVE_QUEEN_CASTLE) {
        if (us == WHITE) {
            pos->pieces[WHITE][ROOK] ^= C64(0x09);
        } else {
            pos->pieces[BLACK][ROOK] ^= C64(0x0900000000000000);
        }
    }

    // Update en passant square
    pos->ep_square = -1;
    if (flags == MOVE_DOUBLE_PUSH) {
        pos->ep_square = (from + to) / 2;
    }

    // Update castling rights
    if (moving_piece == KING) {
        if (us == WHITE) {
            pos->castling &= ~(CASTLE_WK | CASTLE_WQ);
        } else {
            pos->castling &= ~(CASTLE_BK | CASTLE_BQ);
        }
    } else if (moving_piece == ROOK) {
        if (from == A1) pos->castling &= ~CASTLE_WQ;
        else if (from == H1) pos->castling &= ~CASTLE_WK;
        else if (from == A8) pos->castling &= ~CASTLE_BQ;
        else if (from == H8) pos->castling &= ~CASTLE_BK;
    }

    // If a rook is captured, remove castling rights
    if (to == A1) pos->castling &= ~CASTLE_WQ;
    else if (to == H1) pos->castling &= ~CASTLE_WK;
    else if (to == A8) pos->castling &= ~CASTLE_BQ;
    else if (to == H8) pos->castling &= ~CASTLE_BK;

    // Update halfmove clock
    if (moving_piece == PAWN || is_capture(m)) {
        pos->halfmove = 0;
    } else {
        pos->halfmove++;
    }

    // Switch side to move
    pos->side_to_move ^= 1;
}

// Legal move generation

__device__
int generate_legal_moves(const BoardState* pos, Move* moves) {
    Move pseudo_moves[MAX_MOVES];
    int num_pseudo = generate_pseudo_legal_moves(pos, pseudo_moves);

    int num_legal = 0;
    for (int i = 0; i < num_pseudo; i++) {
        BoardState copy = *pos;
        make_move(&copy, pseudo_moves[i]);

        // Check if our king is in check after the move
        Square our_king = lsb(copy.pieces[pos->side_to_move][KING]);
        if (!is_attacked(&copy, our_king, copy.side_to_move)) {
            moves[num_legal++] = pseudo_moves[i];
        }
    }

    return num_legal;
}

// Static Evaluation

// Material values in centipawns
#define EVAL_PAWN   100
#define EVAL_KNIGHT 320
#define EVAL_BISHOP 330
#define EVAL_ROOK   500
#define EVAL_QUEEN  900

// Piece-square tables in constant memory
__constant__ int8_t g_PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
};

__constant__ int8_t g_PST_KNIGHT[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};

__constant__ int8_t g_PST_BISHOP[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

__constant__ int8_t g_PST_ROOK[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

__constant__ int8_t g_PST_KING_MG[64] = {
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30
};

__device__ __forceinline__
int gpu_evaluate(const BoardState* pos) {
    int score = 0;

    // Material
    score += popcount(pos->pieces[WHITE][PAWN])   * EVAL_PAWN;
    score += popcount(pos->pieces[WHITE][KNIGHT]) * EVAL_KNIGHT;
    score += popcount(pos->pieces[WHITE][BISHOP]) * EVAL_BISHOP;
    score += popcount(pos->pieces[WHITE][ROOK])   * EVAL_ROOK;
    score += popcount(pos->pieces[WHITE][QUEEN])  * EVAL_QUEEN;

    score -= popcount(pos->pieces[BLACK][PAWN])   * EVAL_PAWN;
    score -= popcount(pos->pieces[BLACK][KNIGHT]) * EVAL_KNIGHT;
    score -= popcount(pos->pieces[BLACK][BISHOP]) * EVAL_BISHOP;
    score -= popcount(pos->pieces[BLACK][ROOK])   * EVAL_ROOK;
    score -= popcount(pos->pieces[BLACK][QUEEN])  * EVAL_QUEEN;

    // Piece-square tables - simplified
    Bitboard bb;

    // White pieces
    bb = pos->pieces[WHITE][PAWN];
    while (bb) { int sq = pop_lsb_index(bb); score += g_PST_PAWN[sq]; }

    bb = pos->pieces[WHITE][KNIGHT];
    while (bb) { int sq = pop_lsb_index(bb); score += g_PST_KNIGHT[sq]; }

    bb = pos->pieces[WHITE][BISHOP];
    while (bb) { int sq = pop_lsb_index(bb); score += g_PST_BISHOP[sq]; }

    bb = pos->pieces[WHITE][ROOK];
    while (bb) { int sq = pop_lsb_index(bb); score += g_PST_ROOK[sq]; }

    // Black pieces 
    bb = pos->pieces[BLACK][PAWN];
    while (bb) { int sq = pop_lsb_index(bb); score -= g_PST_PAWN[sq ^ 56]; }

    bb = pos->pieces[BLACK][KNIGHT];
    while (bb) { int sq = pop_lsb_index(bb); score -= g_PST_KNIGHT[sq ^ 56]; }

    bb = pos->pieces[BLACK][BISHOP];
    while (bb) { int sq = pop_lsb_index(bb); score -= g_PST_BISHOP[sq ^ 56]; }

    bb = pos->pieces[BLACK][ROOK];
    while (bb) { int sq = pop_lsb_index(bb); score -= g_PST_ROOK[sq ^ 56]; }

    // King safety
    int wk = lsb(pos->pieces[WHITE][KING]);
    int bk = lsb(pos->pieces[BLACK][KING]);
    score += g_PST_KING_MG[wk];
    score -= g_PST_KING_MG[bk ^ 56];

    return score;
}

// Convert centipawn score to win probability using sigmoid
__device__ __forceinline__
float score_to_winprob(int score, int side_to_move) {
    // Adjust for side to move
    if (side_to_move == BLACK) score = -score;

    // Sigmoid: 1 / (1 + exp(-score/400))
    // Approximation for GPU
    float x = (float)score / 400.0f;
    float ex = expf(-x);
    return 1.0f / (1.0f + ex);
}

// Kernel: Random Playout (original)

// ============================================================================
// BATCHED ROLLOUT KERNELS - Massively parallel GPU playouts
// ============================================================================

// Kernel: Pure random playout (baseline, fast)
__global__ void RandomPlayout(
    const BoardState* __restrict__ starting_boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) return;

    curandState rng;
    curand_init(seed, idx, 0, &rng);

    BoardState pos = starting_boards[idx];
    int starting_side = pos.side_to_move;

    Move moves[MAX_MOVES];

    // Random playout up to MAX_PLAYOUT_MOVES
    for (int ply = 0; ply < MAX_PLAYOUT_MOVES; ply++) {
        int num_moves = generate_legal_moves(&pos, moves);

        // Terminal position - checkmate or stalemate
        if (num_moves == 0) {
            if (in_check(&pos)) {
                int winner = pos.side_to_move ^ 1;
                results[idx] = (winner == starting_side) ? 1.0f : 0.0f;
            } else {
                results[idx] = 0.5f; // Stalemate = draw
            }
            return;
        }

        // Draw by 50-move rule
        if (pos.halfmove >= 100) {
            results[idx] = 0.5f;
            return;
        }

        // Select random move
        int move_idx = curand(&rng) % num_moves;
        make_move(&pos, moves[move_idx]);
    }

    // Max depth reached - draw
    results[idx] = 0.5f;
}

// Kernel: HYBRID PLAYOUT - 10 random moves + static evaluation (OPTIMIZED)
// Perfect balance: randomness for variety + eval for accuracy

#define HYBRID_RANDOM_DEPTH 10  // Exactly 10 random moves as requested

__global__ void EvalPlayout(
    const BoardState* __restrict__ starting_boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) return;

    curandState rng;
    curand_init(seed, idx, 0, &rng);

    BoardState pos = starting_boards[idx];
    int starting_side = pos.side_to_move;

    Move moves[MAX_MOVES];

    // EXACTLY 10 random moves for variety (as requested)
    for (int ply = 0; ply < HYBRID_RANDOM_DEPTH; ply++) {
        int num_moves = generate_legal_moves(&pos, moves);

        if (num_moves == 0) {
            if (in_check(&pos)) {
                int winner = pos.side_to_move ^ 1;
                results[idx] = (winner == starting_side) ? 1.0f : 0.0f;
            } else {
                results[idx] = 0.5f;
            }
            return;
        }

        if (pos.halfmove >= 100) {
            results[idx] = 0.5f;
            return;
        }

        int move_idx = curand(&rng) % num_moves;
        make_move(&pos, moves[move_idx]);
    }

    // Evaluate final position with optimized static eval
    int eval = gpu_evaluate(&pos);

    // Convert to win probability using sigmoid (centipawns -> probability)
    float winprob = score_to_winprob(eval, pos.side_to_move);

    // Adjust perspective for starting side
    if (starting_side == BLACK) {
        winprob = 1.0f - winprob;
    }

    results[idx] = winprob;
}

// Kernel: STATIC EVALUATION ONLY (no playout, fastest)
// Pure material + positional eval, instant results

__global__ void StaticEval(
    const BoardState* __restrict__ boards,
    float* __restrict__ results,
    int numBoards
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) return;

    BoardState pos = boards[idx];

    // Check for terminal positions (mate/stalemate)
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(&pos, moves);

    if (num_moves == 0) {
        if (in_check(&pos)) {
            // Checkmate - loss
            results[idx] = 0.0f;
        } else {
            // Stalemate - draw
            results[idx] = 0.5f;
        }
        return;
    }

    // PST + Material evaluation -> win probability
    int eval = gpu_evaluate(&pos);
    results[idx] = score_to_winprob(eval, pos.side_to_move);
}

// ============================================================================
// QUIESCENCE PLAYOUT - Tactical extension for horizon effect
// ============================================================================

// Check if a move is tactical (capture, check, or promotion)
__device__ __forceinline__
bool is_tactical_move(const BoardState* pos, Move m) {
    int move_type = (m >> 12) & 0xF;

    // Promotions are always tactical
    if (move_type >= MOVE_PROMO_N && move_type <= MOVE_PROMO_CAP_Q) {
        return true;
    }

    // Captures are tactical
    if (move_type == MOVE_CAPTURE || move_type == MOVE_EP_CAPTURE) {
        return true;
    }

    // TODO: Add lightweight check detection

    return false;
}

// Generate only tactical moves (captures and promotions)
__device__ __forceinline__
int generate_tactical_moves(const BoardState* pos, Move* moves) {
    // Generate all legal moves first
    Move all_moves[MAX_MOVES];
    int total = generate_legal_moves(pos, all_moves);

    // Filter for tactical moves only
    int count = 0;
    for (int i = 0; i < total; i++) {
        if (is_tactical_move(pos, all_moves[i])) {
            moves[count++] = all_moves[i];
        }
    }

    return count;
}

// MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for move ordering
__device__ __forceinline__
int mvv_lva_score(const BoardState* pos, Move m) {
    int from = m & 0x3F;
    int to = (m >> 6) & 0x3F;
    int move_type = (m >> 12) & 0xF;

    // Promotion captures: highest priority
    if (move_type >= MOVE_PROMO_CAP_N && move_type <= MOVE_PROMO_CAP_Q) {
        return 10000 + ((move_type - MOVE_PROMO_CAP_N) * 100);
    }

    // Promotions without capture
    if (move_type >= MOVE_PROMO_N && move_type <= MOVE_PROMO_Q) {
        return 9000 + ((move_type - MOVE_PROMO_N) * 100);
    }

    // Captures: victim value - attacker value
    if (move_type == MOVE_CAPTURE || move_type == MOVE_EP_CAPTURE) {
        int side = pos->side_to_move;
        int opp = side ^ 1;

        // Find victim piece type
        int victim_value = 0;
        Bitboard to_bb = C64(1) << to;
        for (int pt = PAWN; pt <= QUEEN; pt++) {
            if (pos->pieces[opp][pt] & to_bb) {
                static const int values[6] = {100, 320, 330, 500, 900, 0};
                victim_value = values[pt];
                break;
            }
        }

        // Find attacker piece type
        int attacker_value = 0;
        Bitboard from_bb = C64(1) << from;
        for (int pt = PAWN; pt <= QUEEN; pt++) {
            if (pos->pieces[side][pt] & from_bb) {
                static const int values[6] = {100, 320, 330, 500, 900, 0};
                attacker_value = values[pt];
                break;
            }
        }

        // MVV-LVA: high victim value - low attacker value
        return victim_value * 10 - attacker_value;
    }

    return 0;
}

// OPTIMIZED quiescence search with delta pruning and SEE
__device__ __forceinline__
int quiescence_search_simple(const BoardState* pos, int max_depth) {
    // Stand-pat - current static evaluation
    int stand_pat = gpu_evaluate(pos);

    if (max_depth <= 0) {
        return stand_pat;
    }

    // Generate tactical moves (captures, promotions)
    Move moves[MAX_MOVES];
    int num_moves = generate_tactical_moves(pos, moves);

    // No tactical moves - quiet position
    if (num_moves == 0) {
        return stand_pat;
    }

    // Score moves with SEE for better ordering
    int scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        int see = see_capture(pos, moves[i]);
        scores[i] = see * 10 + mvv_lva_score(pos, moves[i]);
    }

    // Sort top 8 tactical moves (increased from 5 for better accuracy)
    int sort_limit = (num_moves < 8) ? num_moves : 8;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
        }
    }

    int best_score = stand_pat;

    // Try top tactical moves with delta pruning
    int try_limit = (num_moves < 6) ? num_moves : 6;
    for (int i = 0; i < try_limit; i++) {
        // Delta pruning: skip captures that can't improve position
        // Even capturing a queen won't help if we're too far behind
        if (scores[i] < 0 && stand_pat + 1200 < best_score) {
            continue; // Losing capture in hopeless position
        }
        
        BoardState next_pos = *pos;
        make_move(&next_pos, moves[i]);

        // Evaluate after capture
        int score = -gpu_evaluate(&next_pos);

        if (score > best_score) {
            best_score = score;
        }

        // Cutoff if we're already winning by a lot
        if (best_score > stand_pat + 900) { // ~Queen up
            break;
        }
    }

    return best_score;
}

// Kernel: Quiescence Playout - Random moves + tactical search
// Combines randomness with capture resolution for accuracy
__global__ void QuiescencePlayout(
    const BoardState* __restrict__ starting_boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed,
    int max_q_depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) return;

    curandState rng;
    curand_init(seed, idx, 0, &rng);

    BoardState pos = starting_boards[idx];
    int starting_side = pos.side_to_move;

    Move moves[MAX_MOVES];

    // Short random playout (5 moves) for positional variety
    for (int ply = 0; ply < 5; ply++) {  
        int num_moves = generate_legal_moves(&pos, moves);

        if (num_moves == 0) {
            if (in_check(&pos)) {
                int winner = pos.side_to_move ^ 1;
                results[idx] = (winner == starting_side) ? 1.0f : 0.0f;
            } else {
                results[idx] = 0.5f; // Stalemate
            }
            return;
        }

        if (pos.halfmove >= 100) {
            results[idx] = 0.5f; // Draw
            return;
        }

        int move_idx = curand(&rng) % num_moves;
        make_move(&pos, moves[move_idx]);
    }

    // Quiescence search - resolve tactical sequences (captures)
    int eval = quiescence_search_simple(&pos, max_q_depth);

    // Convert centipawns to win probability (sigmoid)
    float winprob = score_to_winprob(eval, pos.side_to_move);

    // Adjust for which side started
    if (starting_side == BLACK) {
        winprob = 1.0f - winprob;
    }

    results[idx] = winprob;
}

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

// Kernel: Tactical Solver (Negamax for Mate Detection)

#define MATE_SCORE 30000
#define INF_SCORE 32000

// ============================================================================
// TACTICAL OPTIMIZATIONS - SEE, Pruning, Enhanced Move Ordering
// ============================================================================

// Static Exchange Evaluation (SEE) - evaluates capture profitability
__device__ __forceinline__
int see_capture(const BoardState* pos, Move m) {
    int from = m & 0x3F;
    int to = (m >> 6) & 0x3F;
    int move_type = (m >> 12) & 0xF;
    
    constexpr int piece_values[6] = {100, 320, 330, 500, 900, 0};
    
    // Find moving piece
    int attacker_value = 0;
    int moving_piece = -1;
    for (int pt = 0; pt < 5; pt++) {
        if (pos->pieces[pos->side_to_move][pt] & (1ULL << from)) {
            moving_piece = pt;
            attacker_value = piece_values[pt];
            break;
        }
    }
    
    if (moving_piece < 0) return 0;
    
    // Find captured piece
    int victim_value = 0;
    if (move_type == MOVE_CAPTURE || move_type >= MOVE_PROMO_CAP_N) {
        for (int pt = 0; pt < 5; pt++) {
            if (pos->pieces[pos->side_to_move ^ 1][pt] & (1ULL << to)) {
                victim_value = piece_values[pt];
                break;
            }
        }
    } else if (move_type == MOVE_EP_CAPTURE) {
        victim_value = 100;
    }
    
    // Simple SEE: victim - (attacker if recaptured)
    // Positive = good capture, Negative = losing capture
    return victim_value - attacker_value;
}

// Check if move gives check
__device__ __forceinline__
bool gives_check_simple(BoardState* pos, Move m) {
    BoardState temp = *pos;
    make_move(&temp, m);
    return in_check(&temp);
}

// ULTIMATE tactical move ordering with SEE
__device__ __forceinline__
int tactical_move_score(const BoardState* pos, Move m, bool gives_check) {
    int move_type = (m >> 12) & 0xF;

    // Checks - highest priority
    if (gives_check) return 1000000;

    // Promotion captures - use SEE
    if (move_type >= MOVE_PROMO_CAP_N && move_type <= MOVE_PROMO_CAP_Q) {
        int see = see_capture(pos, m);
        return 100000 + see;
    }

    // Promotions (non-capture)
    if (move_type >= MOVE_PROMO_N && move_type <= MOVE_PROMO_Q) {
        return 50000 + ((move_type - MOVE_PROMO_N) * 1000);
    }

    // Captures - SEE + MVV-LVA for fine-grained ordering
    if (move_type == MOVE_CAPTURE || move_type == MOVE_EP_CAPTURE) {
        int see = see_capture(pos, m);
        // SEE dominates, MVV-LVA breaks ties
        return 10000 + see * 10 + mvv_lva_score(pos, m);
    }

    return 0;  // Quiet moves
}

// OPTIMIZED depth-2 tactical solver with futility pruning
__device__ __noinline__
int tactical_depth2(BoardState* pos, int alpha, int beta, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    if (num_moves == 0) {
        return in_check(pos) ? -(MATE_SCORE - ply) : 0;
    }

    // Static eval for futility pruning
    int static_eval = gpu_evaluate(pos);
    
    // Futility pruning: if position is very bad, skip quiet moves
    bool futility_prune = false;
    if (!in_check(pos) && static_eval + 900 < alpha) { // ~Queen down
        futility_prune = true;
    }

    // Score and sort moves with SEE
    int scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        bool gives_check = gives_check_simple(pos, moves[i]);
        scores[i] = tactical_move_score(pos, moves[i], gives_check);
    }

    // Sort top 25 moves for better move ordering
    int sort_limit = (num_moves < 25) ? num_moves : 25;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
        }
    }

    int best = -(MATE_SCORE + 1);
    for (int i = 0; i < num_moves; i++) {
        // Futility pruning: skip quiet moves if hopeless
        if (futility_prune && scores[i] < 10000) { // Skip non-tactical moves
            continue;
        }
        
        BoardState pos2 = *pos;
        make_move(&pos2, moves[i]);

        int score;
        Move moves2[MAX_MOVES];
        int num_moves2 = generate_legal_moves(&pos2, moves2);

        if (num_moves2 == 0) {
            score = in_check(&pos2) ? (MATE_SCORE - ply - 1) : 0;
        } else {
            int scores2[MAX_MOVES];
            for (int j = 0; j < num_moves2; j++) {
                bool gives_check2 = gives_check_simple(&pos2, moves2[j]);
                scores2[j] = tactical_move_score(&pos2, moves2[j], gives_check2);
            }

            int sort_limit2 = (num_moves2 < 40) ? num_moves2 : 40;
            for (int j = 0; j < sort_limit2; j++) {
                int best_idx = j;
                for (int k = j + 1; k < num_moves2; k++) {
                    if (scores2[k] > scores2[best_idx]) best_idx = k;
                }
                if (best_idx != j) {
                    Move tmp_m = moves2[j]; moves2[j] = moves2[best_idx]; moves2[best_idx] = tmp_m;
                    int tmp_s = scores2[j]; scores2[j] = scores2[best_idx]; scores2[best_idx] = tmp_s;
                }
            }

            int worst = MATE_SCORE + 1;
            for (int j = 0; j < num_moves2; j++) {
                BoardState pos3 = pos2;
                make_move(&pos3, moves2[j]);

                Move moves3[MAX_MOVES];
                int num_moves3 = generate_legal_moves(&pos3, moves3);

                int eval;
                if (num_moves3 == 0) {
                    eval = in_check(&pos3) ? -(MATE_SCORE - ply - 2) : 0;
                } else {
                    eval = gpu_evaluate(&pos3);
                }

                if (eval < worst) worst = eval;
                if (worst <= -beta) break;
                if (j >= 40 && worst < alpha - 200) break;
            }
            score = -worst;
        }

        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    return best;
}

// ============================================================================
// TACTICAL DEPTH 4 - Iterative mate-in-4 solver (NO RECURSION)
// ============================================================================

__device__ __noinline__
int tactical_depth4(BoardState* pos, int alpha, int beta, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);
    if (num_moves == 0) return in_check(pos) ? -(MATE_SCORE - ply) : 0;

    // Aggressive futility pruning for depth 4
    int static_eval = gpu_evaluate(pos);
    bool futility_prune = !in_check(pos) && static_eval + 1200 < alpha;

    // Score & sort top 20 moves only (reduced for speed)
    int scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        scores[i] = tactical_move_score(pos, moves[i], gives_check_simple(pos, moves[i]));
    }
    int sort_limit = (num_moves < 20) ? num_moves : 20;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
        }
    }

    int best = -(MATE_SCORE + 1);
    
    // Ply 1
    for (int i = 0; i < num_moves && i < 20; i++) {
        if (futility_prune && scores[i] < 10000) continue;
        
        BoardState pos2 = *pos;
        make_move(&pos2, moves[i]);
        Move moves2[MAX_MOVES];
        int num_moves2 = generate_legal_moves(&pos2, moves2);
        
        int score;
        if (num_moves2 == 0) {
            score = in_check(&pos2) ? (MATE_SCORE - ply - 1) : 0;
        } else {
            // Score ply 2 moves
            int scores2[MAX_MOVES];
            for (int j = 0; j < num_moves2; j++) {
                scores2[j] = tactical_move_score(&pos2, moves2[j], gives_check_simple(&pos2, moves2[j]));
            }
            int sort_limit2 = (num_moves2 < 20) ? num_moves2 : 20;
            for (int j = 0; j < sort_limit2; j++) {
                int best_idx = j;
                for (int k = j + 1; k < num_moves2; k++) {
                    if (scores2[k] > scores2[best_idx]) best_idx = k;
                }
                if (best_idx != j) {
                    Move tm = moves2[j]; moves2[j] = moves2[best_idx]; moves2[best_idx] = tm;
                    int ts = scores2[j]; scores2[j] = scores2[best_idx]; scores2[best_idx] = ts;
                }
            }
            
            int worst2 = MATE_SCORE + 1;
            
            // Ply 2
            for (int j = 0; j < num_moves2 && j < 20; j++) {
                BoardState pos3 = pos2;
                make_move(&pos3, moves2[j]);
                Move moves3[MAX_MOVES];
                int num_moves3 = generate_legal_moves(&pos3, moves3);
                
                int s3;
                if (num_moves3 == 0) {
                    s3 = in_check(&pos3) ? -(MATE_SCORE - ply - 2) : 0;
                } else {
                    // Score ply 3 moves
                    int scores3[MAX_MOVES];
                    for (int k = 0; k < num_moves3; k++) {
                        scores3[k] = tactical_move_score(&pos3, moves3[k], gives_check_simple(&pos3, moves3[k]));
                    }
                    int sort_limit3 = (num_moves3 < 15) ? num_moves3 : 15;
                    for (int k = 0; k < sort_limit3; k++) {
                        int best_idx = k;
                        for (int m = k + 1; m < num_moves3; m++) {
                            if (scores3[m] > scores3[best_idx]) best_idx = m;
                        }
                        if (best_idx != k) {
                            Move tm = moves3[k]; moves3[k] = moves3[best_idx]; moves3[best_idx] = tm;
                            int ts = scores3[k]; scores3[k] = scores3[best_idx]; scores3[best_idx] = ts;
                        }
                    }
                    
                    int best3 = -(MATE_SCORE + 1);
                    
                    // Ply 3
                    for (int k = 0; k < num_moves3 && k < 15; k++) {
                        BoardState pos4 = pos3;
                        make_move(&pos4, moves3[k]);
                        Move moves4[MAX_MOVES];
                        int num_moves4 = generate_legal_moves(&pos4, moves4);
                        
                        int s4;
                        if (num_moves4 == 0) {
                            s4 = in_check(&pos4) ? (MATE_SCORE - ply - 3) : 0;
                        } else {
                            // Ply 4 - evaluate only
                            int worst4 = MATE_SCORE + 1;
                            for (int m = 0; m < num_moves4 && m < 10; m++) {
                                BoardState pos5 = pos4;
                                make_move(&pos5, moves4[m]);
                                int eval = gpu_evaluate(&pos5);
                                if (eval < worst4) worst4 = eval;
                            }
                            s4 = -worst4;
                        }
                        
                        if (s4 > best3) best3 = s4;
                        if (best3 >= -worst2 + 200) break; // Alpha-beta style cutoff
                    }
                    s3 = best3;
                }
                
                if (s3 < worst2) worst2 = s3;
                if (worst2 <= -beta) break;
            }
            score = -worst2;
        }
        
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    
    return best;
}

// ============================================================================
// TACTICAL DEPTH 6 - Iterative mate-in-6 solver (NO RECURSION)
// Very aggressive pruning for acceptable speed
// ============================================================================

__device__ __noinline__
int tactical_depth6(BoardState* pos, int alpha, int beta, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);
    if (num_moves == 0) return in_check(pos) ? -(MATE_SCORE - ply) : 0;

    // Very aggressive pruning
    int static_eval = gpu_evaluate(pos);
    bool futility_prune = !in_check(pos) && static_eval + 1500 < alpha;

    // Top 15 moves only
    int scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        scores[i] = tactical_move_score(pos, moves[i], gives_check_simple(pos, moves[i]));
    }
    int sort_limit = (num_moves < 15) ? num_moves : 15;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
        }
    }

    int best = -(MATE_SCORE + 1);
    
    for (int i = 0; i < num_moves && i < 15; i++) {
        if (futility_prune && scores[i] < 10000) continue;
        
        BoardState pos2 = *pos;
        make_move(&pos2, moves[i]);
        
        // Use tactical_depth4 for remaining plies (depth 6 = 1 + depth 4 of 5)
        int score = -tactical_depth4(&pos2, -beta, -alpha, ply + 1);
        
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    
    return best;
}

// Simple mate-in-1 detection (called after making our move)
// Returns score from the side-to-move's perspective in pos
__device__ __forceinline__
int eval_position_simple(BoardState* pos, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    // Checkmate or stalemate
    if (num_moves == 0) {
        return in_check(pos) ? -(MATE_SCORE - ply) : 0;
    }

    // Not mate, just evaluate
    return gpu_evaluate(pos);
}

__global__ void TacticalSolver(
    const BoardState* __restrict__ positions,
    Move* __restrict__ best_moves,
    int* __restrict__ scores,
    int numPositions,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPositions) return;

    BoardState pos = positions[idx];

    // Generate legal moves
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(&pos, moves);

    if (num_moves == 0) {
        best_moves[idx] = 0;
        scores[idx] = 0;
        return;
    }

    // Score and sort moves
    int move_scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        bool gives_check = gives_check_simple(&pos, moves[i]);
        move_scores[i] = tactical_move_score(&pos, moves[i], gives_check);
    }

    // Sort top 20 moves
    int sort_limit = (num_moves < 20) ? num_moves : 20;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (move_scores[j] > move_scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = move_scores[i]; move_scores[i] = move_scores[best_idx]; move_scores[best_idx] = ts;
        }
    }

    // Find best move
    Move best_move = moves[0];
    int best_score = -(MATE_SCORE + 1);

    // Search ALL moves (sorted moves first, then rest)
    for (int i = 0; i < num_moves; i++) {
        BoardState next_pos = pos;
        make_move(&next_pos, moves[i]);

        int score;
        int alpha = -(MATE_SCORE + 1);
        int beta = MATE_SCORE + 1;
        
        // Route to appropriate depth function (all iterative, no recursion)
        if (depth <= 1) {
            // Depth 1: immediate evaluation (mate-in-1 detection)
            score = -eval_position_simple(&next_pos, 1);
        } else if (depth <= 2) {
            // Depth 2: mate-in-2 (proven stable)
            score = -tactical_depth2(&next_pos, -beta, -alpha, 1);
        } else if (depth <= 4) {
            // Depth 4: mate-in-4 (iterative, no stack overflow)
            score = -tactical_depth4(&next_pos, -beta, -alpha, 1);
        } else {
            // Depth 6+: mate-in-5/6 (aggressive pruning for speed)
            score = -tactical_depth6(&next_pos, -beta, -alpha, 1);
        }

        // Sanity check for mate scores
        if (score >= MATE_SCORE - 10) {
            Move test_moves[MAX_MOVES];
            int test_count = generate_legal_moves(&next_pos, test_moves);
            bool is_check = in_check(&next_pos);
            if (test_count != 0 || !is_check) {
                // Not actually mate, use regular eval
                score = -gpu_evaluate(&next_pos);
            }
        }

        if (score > best_score) {
            best_score = score;
            best_move = moves[i];
        }
    }

    best_moves[idx] = best_move;
    scores[idx] = best_score;
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

// Table initialization (called from init_tables.cu)

// Host-side symbol access for initialization
cudaError_t copy_knight_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_KNIGHT_ATTACKS, data, 64 * sizeof(Bitboard));
}

cudaError_t copy_king_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_KING_ATTACKS, data, 64 * sizeof(Bitboard));
}

cudaError_t copy_pawn_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_PAWN_ATTACKS, data, 2 * 64 * sizeof(Bitboard));
}

cudaError_t copy_rook_magics(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_ROOK_MAGICS, data, 64 * sizeof(Bitboard));
}

cudaError_t copy_bishop_magics(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_BISHOP_MAGICS, data, 64 * sizeof(Bitboard));
}

cudaError_t copy_rook_masks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_ROOK_MASKS, data, 64 * sizeof(Bitboard));
}

cudaError_t copy_bishop_masks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_BISHOP_MASKS, data, 64 * sizeof(Bitboard));
}

cudaError_t copy_rook_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_ROOK_ATTACKS, data, 64 * (1 << ROOK_MAGIC_BITS) * sizeof(Bitboard));
}

cudaError_t copy_bishop_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_BISHOP_ATTACKS, data, 64 * (1 << BISHOP_MAGIC_BITS) * sizeof(Bitboard));
}
