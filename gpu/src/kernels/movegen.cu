#include "../../include/chess_types.cuh"

// Constants for MCTS
#define MAX_PLAYOUT_MOVES 500  
#define BLOCK_SIZE 256         
#define ROOK_MAGIC_BITS   12
#define BISHOP_MAGIC_BITS 9

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

__device__ __forceinline__
Bitboard rook_attacks(Square sq, Bitboard occ) {
    occ &= g_ROOK_MASKS[sq];
    occ *= g_ROOK_MAGICS[sq];
    occ >>= (64 - ROOK_MAGIC_BITS);
    return g_ROOK_ATTACKS[sq][occ];
}

__device__ __forceinline__
Bitboard bishop_attacks(Square sq, Bitboard occ) {
    // And-anje da dobijemo blockere
    occ &= g_BISHOP_MASKS[sq];
    // Magic dio iz gore linka, kompresija svih bitova u jedan broj
    occ *= g_BISHOP_MAGICS[sq];
    // Shift da dobijemo odgovarajuce bitove
    occ >>= (64 - BISHOP_MAGIC_BITS);
    // Da dobijemo odgovarajuce poteze iz hard-codanih maski
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
    // Provjera da li pjesak napada neko mjesto tako sto provjeravamo da li crni napada njega - obrnuta logika
    // Slicno za kralja i skakaca
    // Topovi i lovci prvo generisu maske napada sa blokerima i onda provjera enemy piecova
    Bitboard attackers =
        (g_PAWN_ATTACKS[by_color ^ 1][sq] & pos->pieces[by_color][PAWN]) |
        (g_KNIGHT_ATTACKS[sq] & pos->pieces[by_color][KNIGHT]) |
        (g_KING_ATTACKS[sq] & pos->pieces[by_color][KING]) |
        (rook_attacks(sq, occ) & (pos->pieces[by_color][ROOK] | pos->pieces[by_color][QUEEN])) |
        (bishop_attacks(sq, occ) & (pos->pieces[by_color][BISHOP] | pos->pieces[by_color][QUEEN]));
    return attackers != 0;
}

__device__
bool in_check(const BoardState* pos) {
    // Find king -> check if king attacked
    Square king_sq = lsb(pos->pieces[pos->side_to_move][KING]);
    return is_attacked(pos, king_sq, pos->side_to_move ^ 1);
}

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
        moves[count++] = encode_move(from, to, MOVE_QUIET);
    }

    Bitboard dbl = double_push & target;
    while (dbl) {
        Square to = pop_lsb_index(dbl);
        Square from = to - 2 * push_dir;
        moves[count++] = encode_move(from, to, MOVE_DOUBLE_PUSH);
    }

    Bitboard lc = left_cap & ~promo_rank & target;
    while (lc) {
        Square to = pop_lsb_index(lc);
        Square from = to - push_dir + 1;
        moves[count++] = encode_move(from, to, MOVE_CAPTURE);
    }

    Bitboard rc = right_cap & ~promo_rank & target;
    while (rc) {
        Square to = pop_lsb_index(rc);
        Square from = to - push_dir - 1;
        moves[count++] = encode_move(from, to, MOVE_CAPTURE);
    }

    // Promotions
    Bitboard promo_push = single_push & promo_rank & target;
    while (promo_push) {
        Square to = pop_lsb_index(promo_push);
        Square from = to - push_dir;
        moves[count++] = encode_move(from, to, MOVE_PROMO_Q);
        moves[count++] = encode_move(from, to, MOVE_PROMO_R);
        moves[count++] = encode_move(from, to, MOVE_PROMO_B);
        moves[count++] = encode_move(from, to, MOVE_PROMO_N);
    }

    Bitboard promo_lc = left_cap & promo_rank & target;
    while (promo_lc) {
        Square to = pop_lsb_index(promo_lc);
        Square from = to - push_dir + 1;
        moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_Q);
        moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_R);
        moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_B);
        moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_N);
    }

    Bitboard promo_rc = right_cap & promo_rank & target;
    while (promo_rc) {
        Square to = pop_lsb_index(promo_rc);
        Square from = to - push_dir - 1;
        moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_Q);
        moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_R);
        moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_B);
        moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_N);
    }

    // En passant
    while (ep_left) {
        Square to = pop_lsb_index(ep_left);
        Square from = to - push_dir + 1;
        moves[count++] = encode_move(from, to, MOVE_EP_CAPTURE);
    }

    while (ep_right) {
        Square to = pop_lsb_index(ep_right);
        Square from = to - push_dir - 1;
        moves[count++] = encode_move(from, to, MOVE_EP_CAPTURE);
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
            moves[count++] = encode_move(from, to, flags);
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
            moves[count++] = encode_move(from, to, flags);
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
            moves[count++] = encode_move(from, to, flags);
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
            moves[count++] = encode_move(from, to, flags);
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
            moves[count++] = encode_move(king_sq, to, flags);
        }
    }

    // Castling
    if (!in_check(pos)) {
        if (us == WHITE) {
            if ((pos->castling & CASTLE_WK) &&
                !(occ & C64(0x60)) &&
                !is_attacked(pos, F1, them) &&
                !is_attacked(pos, G1, them)) {
                moves[count++] = encode_move(E1, G1, MOVE_KING_CASTLE);
            }
            if ((pos->castling & CASTLE_WQ) &&
                !(occ & C64(0x0E)) &&
                !is_attacked(pos, D1, them) &&
                !is_attacked(pos, C1, them)) {
                moves[count++] = encode_move(E1, C1, MOVE_QUEEN_CASTLE);
            }
        } else {
            if ((pos->castling & CASTLE_BK) &&
                !(occ & C64(0x6000000000000000)) &&
                !is_attacked(pos, F8, them) &&
                !is_attacked(pos, G8, them)) {
                moves[count++] = encode_move(E8, G8, MOVE_KING_CASTLE);
            }
            if ((pos->castling & CASTLE_BQ) &&
                !(occ & C64(0x0E00000000000000)) &&
                !is_attacked(pos, D8, them) &&
                !is_attacked(pos, C8, them)) {
                moves[count++] = encode_move(E8, C8, MOVE_QUEEN_CASTLE);
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
