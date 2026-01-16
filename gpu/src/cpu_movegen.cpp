#include "../include/cpu_movegen.h"
#include <cstring> // For memset/memcpy if needed

namespace cpu_movegen {

// Knight attack lookup 
const Bitboard KNIGHT_ATTACKS_CPU[64] = {
    0x0000000000020400, 0x0000000000050800, 0x00000000000a1100, 0x0000000000142200,
    0x0000000000284400, 0x0000000000508800, 0x0000000000a01000, 0x0000000000402000,
    0x0000000002040004, 0x0000000005080008, 0x000000000a110011, 0x0000000014220022,
    0x0000000028440044, 0x0000000050880088, 0x00000000a0100010, 0x0000000040200020,
    0x0000000204000402, 0x0000000508000805, 0x0000000a1100110a, 0x0000001422002214,
    0x0000002844004428, 0x0000005088008850, 0x000000a0100010a0, 0x0000004020002040,
    0x0000020400040200, 0x0000050800080500, 0x00000a1100110a00, 0x0000142200221400,
    0x0000284400442800, 0x0000508800885000, 0x0000a0100010a000, 0x0000402000204000,
    0x0002040004020000, 0x0005080008050000, 0x000a1100110a0000, 0x0014220022140000,
    0x0028440044280000, 0x0050880088500000, 0x00a0100010a00000, 0x0040200020400000,
    0x0204000402000000, 0x0508000805000000, 0x0a1100110a000000, 0x1422002214000000,
    0x2844004428000000, 0x5088008850000000, 0xa0100010a0000000, 0x4020002040000000,
    0x0400040200000000, 0x0800080500000000, 0x1100110a00000000, 0x2200221400000000,
    0x4400442800000000, 0x8800885000000000, 0x100010a000000000, 0x2000204000000000,
    0x0004020000000000, 0x0008050000000000, 0x00110a0000000000, 0x0022140000000000,
    0x0044280000000000, 0x0088500000000000, 0x0010a00000000000, 0x0020400000000000
};

// King attack lookup 
const Bitboard KING_ATTACKS_CPU[64] = {
    0x0000000000000302, 0x0000000000000705, 0x0000000000000e0a, 0x0000000000001c14,
    0x0000000000003828, 0x0000000000007050, 0x000000000000e0a0, 0x000000000000c040,
    0x0000000000030203, 0x0000000000070507, 0x00000000000e0a0e, 0x00000000001c141c,
    0x0000000000382838, 0x0000000000705070, 0x0000000000e0a0e0, 0x0000000000c040c0,
    0x0000000003020300, 0x0000000007050700, 0x000000000e0a0e00, 0x000000001c141c00,
    0x0000000038283800, 0x0000000070507000, 0x00000000e0a0e000, 0x00000000c040c000,
    0x0000000302030000, 0x0000000705070000, 0x0000000e0a0e0000, 0x0000001c141c0000,
    0x0000003828380000, 0x0000007050700000, 0x000000e0a0e00000, 0x000000c040c00000,
    0x0000030203000000, 0x0000070507000000, 0x00000e0a0e000000, 0x00001c141c000000,
    0x0000382838000000, 0x0000705070000000, 0x0000e0a0e0000000, 0x0000c040c0000000,
    0x0003020300000000, 0x0007050700000000, 0x000e0a0e00000000, 0x001c141c00000000,
    0x0038283800000000, 0x0070507000000000, 0x00e0a0e000000000, 0x00c040c000000000,
    0x0302030000000000, 0x0705070000000000, 0x0e0a0e0000000000, 0x1c141c0000000000,
    0x3828380000000000, 0x7050700000000000, 0xe0a0e00000000000, 0xc040c00000000000,
    0x0203000000000000, 0x0507000000000000, 0x0a0e000000000000, 0x141c000000000000,
    0x2838000000000000, 0x5070000000000000, 0xa0e0000000000000, 0x40c0000000000000
};

// Simple sliding attacks 
Bitboard rook_attacks_cpu(int sq, Bitboard occ) {
    Bitboard attacks = 0;
    int rank = sq / 8, file = sq % 8;

    // Up
    for (int r = rank + 1; r < 8; r++) {
        Bitboard bb = C64(1) << (r * 8 + file);
        attacks |= bb;
        if (occ & bb) break;
    }
    // Down
    for (int r = rank - 1; r >= 0; r--) {
        Bitboard bb = C64(1) << (r * 8 + file);
        attacks |= bb;
        if (occ & bb) break;
    }
    // Right
    for (int f = file + 1; f < 8; f++) {
        Bitboard bb = C64(1) << (rank * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    // Left
    for (int f = file - 1; f >= 0; f--) {
        Bitboard bb = C64(1) << (rank * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    return attacks;
}

Bitboard bishop_attacks_cpu(int sq, Bitboard occ) {
    Bitboard attacks = 0;
    int rank = sq / 8, file = sq % 8;

    for (int r = rank + 1, f = file + 1; r < 8 && f < 8; r++, f++) {
        Bitboard bb = C64(1) << (r * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    for (int r = rank + 1, f = file - 1; r < 8 && f >= 0; r++, f--) {
        Bitboard bb = C64(1) << (r * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    for (int r = rank - 1, f = file + 1; r >= 0 && f < 8; r--, f++) {
        Bitboard bb = C64(1) << (r * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; r--, f--) {
        Bitboard bb = C64(1) << (r * 8 + f);
        attacks |= bb;
        if (occ & bb) break;
    }
    return attacks;
}

bool is_square_attacked_cpu(const BoardState* pos, int sq, int by_color) {
    Bitboard occ = pos->occupied();

    // Pawn attacks
    if (by_color == WHITE) {
        if (sq >= 9 && ((sq % 8) > 0) && (pos->pieces[WHITE][PAWN] & (C64(1) << (sq - 9)))) return true;
        if (sq >= 7 && ((sq % 8) < 7) && (pos->pieces[WHITE][PAWN] & (C64(1) << (sq - 7)))) return true;
    } else {
        if (sq <= 54 && ((sq % 8) > 0) && (pos->pieces[BLACK][PAWN] & (C64(1) << (sq + 7)))) return true;
        if (sq <= 56 && ((sq % 8) < 7) && (pos->pieces[BLACK][PAWN] & (C64(1) << (sq + 9)))) return true;
    }

    // Knight attacks
    if (KNIGHT_ATTACKS_CPU[sq] & pos->pieces[by_color][KNIGHT]) return true;

    // King attacks
    if (KING_ATTACKS_CPU[sq] & pos->pieces[by_color][KING]) return true;

    // Rook/Queen attacks
    Bitboard rook_attacks = rook_attacks_cpu(sq, occ);
    if (rook_attacks & (pos->pieces[by_color][ROOK] | pos->pieces[by_color][QUEEN])) return true;

    // Bishop/Queen attacks
    Bitboard bishop_attacks = bishop_attacks_cpu(sq, occ);
    if (bishop_attacks & (pos->pieces[by_color][BISHOP] | pos->pieces[by_color][QUEEN])) return true;

    return false;
}

bool in_check_cpu(const BoardState* pos) {
    int king_sq = lsb(pos->pieces[pos->side_to_move][KING]);
    return is_square_attacked_cpu(pos, king_sq, pos->side_to_move ^ 1);
}

void make_move_cpu(BoardState* pos, Move m) {
    Square from = move_from(m);
    Square to = move_to(m);
    uint8_t flags = move_flags(m);
    int us = pos->side_to_move;
    int them = us ^ 1;

    Bitboard from_bb = C64(1) << from;
    Bitboard to_bb = C64(1) << to;
    Bitboard from_to = from_bb | to_bb;

    // Find moving piece
    Piece moving_piece = NO_PIECE;
    for (int p = PAWN; p <= KING; p++) {
        if (pos->pieces[us][p] & from_bb) {
            moving_piece = p;
            break;
        }
    }

    // Handle captures
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

    // Move piece
    pos->pieces[us][moving_piece] ^= from_to;

    // Promotions
    if (is_promotion(m)) {
        pos->pieces[us][PAWN] &= ~to_bb;
        pos->pieces[us][promotion_piece(m)] |= to_bb;
    }

    // Castling
    if (flags == MOVE_KING_CASTLE) {
        if (us == WHITE) pos->pieces[WHITE][ROOK] ^= C64(0xA0);
        else pos->pieces[BLACK][ROOK] ^= C64(0xA000000000000000);
    } else if (flags == MOVE_QUEEN_CASTLE) {
        if (us == WHITE) pos->pieces[WHITE][ROOK] ^= C64(0x09);
        else pos->pieces[BLACK][ROOK] ^= C64(0x0900000000000000);
    }

    // Update state
    pos->ep_square = -1;
    if (flags == MOVE_DOUBLE_PUSH) {
        pos->ep_square = (from + to) / 2;
    }

    // Update castling rights
    if (moving_piece == KING) {
        if (us == WHITE) pos->castling &= ~(CASTLE_WK | CASTLE_WQ);
        else pos->castling &= ~(CASTLE_BK | CASTLE_BQ);
    } else if (moving_piece == ROOK) {
        if (from == A1) pos->castling &= ~CASTLE_WQ;
        else if (from == H1) pos->castling &= ~CASTLE_WK;
        else if (from == A8) pos->castling &= ~CASTLE_BQ;
        else if (from == H8) pos->castling &= ~CASTLE_BK;
    }

    if (to == A1) pos->castling &= ~CASTLE_WQ;
    else if (to == H1) pos->castling &= ~CASTLE_WK;
    else if (to == A8) pos->castling &= ~CASTLE_BQ;
    else if (to == H8) pos->castling &= ~CASTLE_BK;

    if (moving_piece == PAWN || is_capture(m)) pos->halfmove = 0;
    else pos->halfmove++;

    pos->side_to_move ^= 1;
}

// Simplified legal move generation (enough for MCTS tree building)
int generate_legal_moves_cpu(const BoardState* pos, Move* moves) {
    int count = 0;
    int us = pos->side_to_move;
    int them = us ^ 1;
    Bitboard occ = pos->occupied();
    Bitboard our_pieces = pos->color_pieces(us);
    Bitboard their_pieces = pos->color_pieces(them);
    Bitboard empty = ~occ;

    // Pawn moves
    Bitboard pawns = pos->pieces[us][PAWN];
    int push_dir = (us == WHITE) ? 8 : -8;
    Bitboard start_rank = (us == WHITE) ? RANK_2 : RANK_7;
    Bitboard promo_rank = (us == WHITE) ? RANK_8 : RANK_1;

    while (pawns) {
        int from = pop_lsb_index(pawns);
        Bitboard from_bb = C64(1) << from;
        int to;

        // Single push
        to = from + push_dir;
        if (to >= 0 && to < 64 && (empty & (C64(1) << to))) {
            if ((C64(1) << to) & promo_rank) {
                moves[count++] = encode_move(from, to, MOVE_PROMO_Q);
                moves[count++] = encode_move(from, to, MOVE_PROMO_R);
                moves[count++] = encode_move(from, to, MOVE_PROMO_B);
                moves[count++] = encode_move(from, to, MOVE_PROMO_N);
            } else {
                moves[count++] = encode_move(from, to, MOVE_QUIET);
            }

            // Double push
            if (from_bb & start_rank) {
                to = from + 2 * push_dir;
                if (empty & (C64(1) << to)) {
                    moves[count++] = encode_move(from, to, MOVE_DOUBLE_PUSH);
                }
            }
        }

        // Captures
        int cap_left = (us == WHITE) ? 7 : -9;
        int cap_right = (us == WHITE) ? 9 : -7;

        to = from + cap_left;
        if (to >= 0 && to < 64 && (from % 8) > 0 && (their_pieces & (C64(1) << to))) {
            if ((C64(1) << to) & promo_rank) {
                // Generate all 4 promotion types for captures
                moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_Q);
                moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_R);
                moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_B);
                moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_N);
            } else {
                moves[count++] = encode_move(from, to, MOVE_CAPTURE);
            }
        }

        to = from + cap_right;
        if (to >= 0 && to < 64 && (from % 8) < 7 && (their_pieces & (C64(1) << to))) {
            if ((C64(1) << to) & promo_rank) {
                // Generate all 4 promotion types for captures
                moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_Q);
                moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_R);
                moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_B);
                moves[count++] = encode_move(from, to, MOVE_PROMO_CAP_N);
            } else {
                moves[count++] = encode_move(from, to, MOVE_CAPTURE);
            }
        }

        // En passant - must check file boundaries to avoid wrap-around
        if (pos->ep_square >= 0) {
            int from_file = from % 8;
            // Can capture right (to higher file) if not on file H
            if (from_file < 7 && (from + cap_right) == pos->ep_square) {
                moves[count++] = encode_move(from, pos->ep_square, MOVE_EP_CAPTURE);
            }
            // Can capture left (to lower file) if not on file A
            if (from_file > 0 && (from + cap_left) == pos->ep_square) {
                moves[count++] = encode_move(from, pos->ep_square, MOVE_EP_CAPTURE);
            }
        }
    }

    // Knight moves
    Bitboard knights = pos->pieces[us][KNIGHT];
    while (knights) {
        int from = pop_lsb_index(knights);
        Bitboard attacks = KNIGHT_ATTACKS_CPU[from] & ~our_pieces;
        while (attacks) {
            int to = pop_lsb_index(attacks);
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = encode_move(from, to, flags);
        }
    }

    // Bishop moves
    Bitboard bishops = pos->pieces[us][BISHOP];
    while (bishops) {
        int from = pop_lsb_index(bishops);
        Bitboard attacks = bishop_attacks_cpu(from, occ) & ~our_pieces;
        while (attacks) {
            int to = pop_lsb_index(attacks);
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = encode_move(from, to, flags);
        }
    }

    // Rook moves
    Bitboard rooks = pos->pieces[us][ROOK];
    while (rooks) {
        int from = pop_lsb_index(rooks);
        Bitboard attacks = rook_attacks_cpu(from, occ) & ~our_pieces;
        while (attacks) {
            int to = pop_lsb_index(attacks);
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = encode_move(from, to, flags);
        }
    }

    // Queen moves
    Bitboard queens = pos->pieces[us][QUEEN];
    while (queens) {
        int from = pop_lsb_index(queens);
        Bitboard attacks = (rook_attacks_cpu(from, occ) | bishop_attacks_cpu(from, occ)) & ~our_pieces;
        while (attacks) {
            int to = pop_lsb_index(attacks);
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = encode_move(from, to, flags);
        }
    }

    // King moves
    int king_sq = lsb(pos->pieces[us][KING]);
    Bitboard king_attacks = KING_ATTACKS_CPU[king_sq] & ~our_pieces;
    while (king_attacks) {
        int to = pop_lsb_index(king_attacks);
        if (!is_square_attacked_cpu(pos, to, them)) {
            uint8_t flags = (their_pieces & (C64(1) << to)) ? MOVE_CAPTURE : MOVE_QUIET;
            moves[count++] = encode_move(king_sq, to, flags);
        }
    }

    // Castling
    if (!in_check_cpu(pos)) {
        if (us == WHITE) {
            if ((pos->castling & CASTLE_WK) &&
                !(occ & C64(0x60)) &&
                !is_square_attacked_cpu(pos, F1, them) &&
                !is_square_attacked_cpu(pos, G1, them)) {
                moves[count++] = encode_move(E1, G1, MOVE_KING_CASTLE);
            }
            if ((pos->castling & CASTLE_WQ) &&
                !(occ & C64(0x0E)) &&
                !is_square_attacked_cpu(pos, D1, them) &&
                !is_square_attacked_cpu(pos, C1, them)) {
                moves[count++] = encode_move(E1, C1, MOVE_QUEEN_CASTLE);
            }
        } else {
            if ((pos->castling & CASTLE_BK) &&
                !(occ & C64(0x6000000000000000)) &&
                !is_square_attacked_cpu(pos, F8, them) &&
                !is_square_attacked_cpu(pos, G8, them)) {
                moves[count++] = encode_move(E8, G8, MOVE_KING_CASTLE);
            }
            if ((pos->castling & CASTLE_BQ) &&
                !(occ & C64(0x0E00000000000000)) &&
                !is_square_attacked_cpu(pos, D8, them) &&
                !is_square_attacked_cpu(pos, C8, them)) {
                moves[count++] = encode_move(E8, C8, MOVE_QUEEN_CASTLE);
            }
        }
    }

    // Filter for legality (check if our king is safe after move)
    int legal_count = 0;
    for (int i = 0; i < count; i++) {
        BoardState copy = *pos;
        make_move_cpu(&copy, moves[i]);
        int our_king = lsb(copy.pieces[us][KING]);
        if (!is_square_attacked_cpu(&copy, our_king, them)) {
            moves[legal_count++] = moves[i];
        }
    }

    return legal_count;
}

} // namespace cpu_movegen
