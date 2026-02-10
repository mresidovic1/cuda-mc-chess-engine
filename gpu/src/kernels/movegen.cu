#include "../../include/chess_types.cuh"
#include "../../include/kernels/movegen.cuh"

// Constants for MCTS
#define MAX_PLAYOUT_MOVES 500  
#define BLOCK_SIZE 256         
#define ROOK_MAGIC_BITS   12
#define BISHOP_MAGIC_BITS 9

// Safe move append (prevents overflow)
#define PUSH_MOVE_SAFE(m) do { \
    if (count >= max_moves) return count; \
    moves[count++] = (m); \
} while(0)

// Attack tables - precalculate to improve speed
__constant__ Bitboard g_KNIGHT_ATTACKS[64];
__constant__ Bitboard g_KING_ATTACKS[64];
__constant__ Bitboard g_PAWN_ATTACKS[2][64];

__constant__ Bitboard g_ROOK_MAGICS[64];
__constant__ Bitboard g_BISHOP_MAGICS[64];
__constant__ Bitboard g_ROOK_MASKS[64];
__constant__ Bitboard g_BISHOP_MASKS[64];

// Large attack tables must go in the global memory - slider follow different logic 
// https://www.chessprogramming.org/Magic_Bitboards

__device__ Bitboard g_ROOK_ATTACKS[64][1 << ROOK_MAGIC_BITS];
__device__ Bitboard g_BISHOP_ATTACKS[64][1 << BISHOP_MAGIC_BITS];

// Inline functions are now in movegen.cuh header

// Table initialization functions for symbol access (called from init_tables.cu)

extern "C" cudaError_t copy_knight_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_KNIGHT_ATTACKS, data, 64 * sizeof(Bitboard));
}

extern "C" cudaError_t copy_king_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_KING_ATTACKS, data, 64 * sizeof(Bitboard));
}

extern "C" cudaError_t copy_pawn_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_PAWN_ATTACKS, data, 2 * 64 * sizeof(Bitboard));
}

extern "C" cudaError_t copy_rook_magics(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_ROOK_MAGICS, data, 64 * sizeof(Bitboard));
}

extern "C" cudaError_t copy_bishop_magics(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_BISHOP_MAGICS, data, 64 * sizeof(Bitboard));
}

extern "C" cudaError_t copy_rook_masks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_ROOK_MASKS, data, 64 * sizeof(Bitboard));
}

extern "C" cudaError_t copy_bishop_masks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_BISHOP_MASKS, data, 64 * sizeof(Bitboard));
}

extern "C" cudaError_t copy_rook_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_ROOK_ATTACKS, data, 64 * (1 << ROOK_MAGIC_BITS) * sizeof(Bitboard));
}

extern "C" cudaError_t copy_bishop_attacks(const Bitboard* data) {
    return cudaMemcpyToSymbol(g_BISHOP_ATTACKS, data, 64 * (1 << BISHOP_MAGIC_BITS) * sizeof(Bitboard));
}

// Move generation helpers

__device__
int generate_pawn_moves_cap(const BoardState* pos, Move* moves, Bitboard target, int max_moves) {
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
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_QUIET));
    }

    Bitboard dbl = double_push & target;
    while (dbl) {
        Square to = pop_lsb_index(dbl);
        Square from = to - 2 * push_dir;
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_DOUBLE_PUSH));
    }

    Bitboard lc = left_cap & ~promo_rank & target;
    while (lc) {
        Square to = pop_lsb_index(lc);
        Square from = to - push_dir + 1;
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_CAPTURE));
    }

    Bitboard rc = right_cap & ~promo_rank & target;
    while (rc) {
        Square to = pop_lsb_index(rc);
        Square from = to - push_dir - 1;
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_CAPTURE));
    }

    // Promotions
    Bitboard promo_push = single_push & promo_rank & target;
    while (promo_push) {
        Square to = pop_lsb_index(promo_push);
        Square from = to - push_dir;
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_Q));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_R));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_B));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_N));
    }

    Bitboard promo_lc = left_cap & promo_rank & target;
    while (promo_lc) {
        Square to = pop_lsb_index(promo_lc);
        Square from = to - push_dir + 1;
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_CAP_Q));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_CAP_R));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_CAP_B));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_CAP_N));
    }

    Bitboard promo_rc = right_cap & promo_rank & target;
    while (promo_rc) {
        Square to = pop_lsb_index(promo_rc);
        Square from = to - push_dir - 1;
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_CAP_Q));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_CAP_R));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_CAP_B));
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_PROMO_CAP_N));
    }

    // En passant
    while (ep_left) {
        Square to = pop_lsb_index(ep_left);
        Square from = to - push_dir + 1;
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_EP_CAPTURE));
    }

    while (ep_right) {
        Square to = pop_lsb_index(ep_right);
        Square from = to - push_dir - 1;
        PUSH_MOVE_SAFE(encode_move(from, to, MOVE_EP_CAPTURE));
    }

    return count;
}

__device__
int generate_knight_moves_cap(const BoardState* pos, Move* moves, Bitboard target, int max_moves) {
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
            PUSH_MOVE_SAFE(encode_move(from, to, flags));
        }
    }
    return count;
}

__device__
int generate_bishop_moves_cap(const BoardState* pos, Move* moves, Bitboard target, int max_moves) {
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
            PUSH_MOVE_SAFE(encode_move(from, to, flags));
        }
    }
    return count;
}

__device__
int generate_rook_moves_cap(const BoardState* pos, Move* moves, Bitboard target, int max_moves) {
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
            PUSH_MOVE_SAFE(encode_move(from, to, flags));
        }
    }
    return count;
}

__device__
int generate_queen_moves_cap(const BoardState* pos, Move* moves, Bitboard target, int max_moves) {
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
            PUSH_MOVE_SAFE(encode_move(from, to, flags));
        }
    }
    return count;
}

__device__
int generate_king_moves_cap(const BoardState* pos, Move* moves, Bitboard target, int max_moves) {
    int count = 0;
    int us = pos->side_to_move;
    int them = us ^ 1;
    Bitboard king_bb = pos->pieces[us][KING];
    if (king_bb == 0) return 0;
    Square king_sq = lsb(king_bb);
    Bitboard our_pieces = pos->us();
    Bitboard enemy = pos->them();
    Bitboard occ = pos->occupied();

    // Normal king moves
    Bitboard attacks = g_KING_ATTACKS[king_sq] & ~our_pieces & target;
    while (attacks) {
        Square to = pop_lsb_index(attacks);
        if (!is_attacked(pos, to, them)) {
            uint8_t flags = (enemy & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            PUSH_MOVE_SAFE(encode_move(king_sq, to, flags));
        }
    }

    // Castling
    if (!in_check(pos)) {
        if (us == WHITE) {
            if ((pos->castling & CASTLE_WK) &&
                !(occ & C64(0x60)) &&
                !is_attacked(pos, F1, them) &&
                !is_attacked(pos, G1, them)) {
                PUSH_MOVE_SAFE(encode_move(E1, G1, MOVE_KING_CASTLE));
            }
            if ((pos->castling & CASTLE_WQ) &&
                !(occ & C64(0x0E)) &&
                !is_attacked(pos, D1, them) &&
                !is_attacked(pos, C1, them)) {
                PUSH_MOVE_SAFE(encode_move(E1, C1, MOVE_QUEEN_CASTLE));
            }
        } else {
            if ((pos->castling & CASTLE_BK) &&
                !(occ & C64(0x6000000000000000)) &&
                !is_attacked(pos, F8, them) &&
                !is_attacked(pos, G8, them)) {
                PUSH_MOVE_SAFE(encode_move(E8, G8, MOVE_KING_CASTLE));
            }
            if ((pos->castling & CASTLE_BQ) &&
                !(occ & C64(0x0E00000000000000)) &&
                !is_attacked(pos, D8, them) &&
                !is_attacked(pos, C8, them)) {
                PUSH_MOVE_SAFE(encode_move(E8, C8, MOVE_QUEEN_CASTLE));
            }
        }
    }

    return count;
}

// Generate all pseudo-legal moves

__device__
int generate_pseudo_legal_moves_cap(const BoardState* pos, Move* moves, int max_moves) {
    if (max_moves <= 0) return 0;
    Bitboard target = ALL_SQUARES;
    int count = 0;
    if (count < max_moves) count += generate_pawn_moves_cap(pos, moves + count, target, max_moves - count);
    if (count < max_moves) count += generate_knight_moves_cap(pos, moves + count, target, max_moves - count);
    if (count < max_moves) count += generate_bishop_moves_cap(pos, moves + count, target, max_moves - count);
    if (count < max_moves) count += generate_rook_moves_cap(pos, moves + count, target, max_moves - count);
    if (count < max_moves) count += generate_queen_moves_cap(pos, moves + count, target, max_moves - count);
    if (count < max_moves) count += generate_king_moves_cap(pos, moves + count, target, max_moves - count);
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
    if (moving_piece == NO_PIECE) {
        return;
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
int generate_legal_moves_cap(const BoardState* pos, Move* moves, int max_moves) {
    if (max_moves <= 0) return 0;
    Move pseudo_moves[MAX_MOVES];
    int num_pseudo = generate_pseudo_legal_moves_cap(pos, pseudo_moves, MAX_MOVES);

    int num_legal = 0;
    for (int i = 0; i < num_pseudo; i++) {
        // Guard against invalid pseudo-moves (from square must contain our piece)
        Square from = move_from(pseudo_moves[i]);
        if ((pos->color_pieces(pos->side_to_move) & (C64(1) << from)) == 0) {
            continue;
        }

        BoardState copy = *pos;
        make_move(&copy, pseudo_moves[i]);

        // Check if our king is in check after the move
        Bitboard king_bb = copy.pieces[pos->side_to_move][KING];
        if (king_bb == 0) {
            continue;
        }
        Square our_king = lsb(king_bb);
        if (!is_attacked(&copy, our_king, copy.side_to_move)) {
            if (num_legal >= max_moves) break;
            moves[num_legal++] = pseudo_moves[i];
        }
    }

    return num_legal;
}
