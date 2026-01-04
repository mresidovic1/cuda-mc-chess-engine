#pragma once
#include "gpu_chess_types.cuh"
#include "gpu_chess_bitops.cuh"
#include "gpu_chess_position.cuh"
#include <cstdio>

// ============================================================================
// Attack Detection
// ============================================================================

__device__ bool is_square_attacked(const Position* pos, int sq, Color by_color) {
    Bitboard sq_bb = 1ULL << sq;
    Bitboard empty = ~pos->occupied[2];
    int offset = by_color * 6;
    
    // Pawn attacks (reverse direction)
    if (pawn_attacks(sq_bb, (Color)!by_color) & pos->pieces[offset + PAWN]) return true;
    
    // Knight attacks
    if (knight_attacks(sq_bb) & pos->pieces[offset + KNIGHT]) return true;
    
    // King attacks
    if (king_attacks(sq_bb) & pos->pieces[offset + KING]) return true;
    
    // Sliding pieces
    Bitboard bishops_queens = pos->pieces[offset + BISHOP] | pos->pieces[offset + QUEEN];
    if (get_bishop_attacks(sq_bb, empty) & bishops_queens) return true;
    
    Bitboard rooks_queens = pos->pieces[offset + ROOK] | pos->pieces[offset + QUEEN];
    if (get_rook_attacks(sq_bb, empty) & rooks_queens) return true;
    
    return false;
}

// ============================================================================
// Check Detection
// ============================================================================

__device__ Bitboard get_checkers(const Position* pos) {
    Color us = (Color)pos->side_to_move;
    Color them = (Color)!us;
    int our_offset = us * 6;
    int their_offset = them * 6;
    
    Bitboard king = pos->pieces[our_offset + KING];
    int king_sq = lsb(king);
    Bitboard empty = ~pos->occupied[2];
    
    Bitboard checkers = 0;
    
    // Pawn checks
    checkers |= pawn_attacks(king, us) & pos->pieces[their_offset + PAWN];
    
    // Knight checks
    checkers |= knight_attacks(king) & pos->pieces[their_offset + KNIGHT];
    
    // Sliding checks
    Bitboard their_bishops_queens = pos->pieces[their_offset + BISHOP] | 
                                    pos->pieces[their_offset + QUEEN];
    checkers |= get_bishop_attacks(king, empty) & their_bishops_queens;
    
    Bitboard their_rooks_queens = pos->pieces[their_offset + ROOK] | 
                                  pos->pieces[their_offset + QUEEN];
    checkers |= get_rook_attacks(king, empty) & their_rooks_queens;
    
    return checkers;
}

// ============================================================================
// Pin Detection
// ============================================================================

struct PinInfo {
    Bitboard pinned_pieces;
    Bitboard pin_rays[64];  // Bitboard of valid moves for each square if pinned
};

__device__ void calc_pins(const Position* pos, PinInfo* pin_info) {
    Color us = (Color)pos->side_to_move;
    Color them = (Color)!us;
    int our_offset = us * 6;
    int their_offset = them * 6;
    
    Bitboard king = pos->pieces[our_offset + KING];
    int king_sq = lsb(king);
    Bitboard our_pieces = pos->occupied[us];
    Bitboard their_pieces = pos->occupied[them];
    
    pin_info->pinned_pieces = 0;
    for (int i = 0; i < 64; i++) pin_info->pin_rays[i] = ~0ULL;
    
    Bitboard their_diag = pos->pieces[their_offset + BISHOP] | 
                          pos->pieces[their_offset + QUEEN];
    printf("GPU DEBUG: their_offset=%d, bishops=0x%llx, queens=0x%llx, their_diag=0x%llx\n",
       their_offset, pos->pieces[their_offset + BISHOP], pos->pieces[their_offset + QUEEN], their_diag);                      
    Bitboard their_ortho = pos->pieces[their_offset + ROOK] | 
                           pos->pieces[their_offset + QUEEN];
    
    // North-East
    {
        Bitboard ray = 0;
        Bitboard bb = king;
        int pieces_found = 0;
        int pinned_sq = -1;
        
        for (int i = 0; i < 7; i++) {
            bb = north_east(bb);
            if (!bb) break;
            ray |= bb;
            
            if (bb & our_pieces) {
                pieces_found++;
                if (pieces_found == 1) pinned_sq = lsb(bb);
                else break;
            } else if (bb & their_pieces) {
                if (pieces_found == 1 && (bb & their_diag)) {
                    int attacker_sq = lsb(bb);
                    printf("GPU DEBUG NE: King:%d Pinned:%d Attacker:%d | their_diag:0x%llx bb:0x%llx overlap:0x%llx\n",
           king_sq, pinned_sq, attacker_sq, their_diag, bb, bb & their_diag);
    printf("  our_pieces:0x%llx their_pieces:0x%llx\n", our_pieces, their_pieces);
                    pin_info->pinned_pieces |= (1ULL << pinned_sq);
                    pin_info->pin_rays[pinned_sq] = ray | bb;
                }
                break;
            }
        }
    }
    
    // North-West
    {
        Bitboard ray = 0;
        Bitboard bb = king;
        int pieces_found = 0;
        int pinned_sq = -1;
        
        for (int i = 0; i < 7; i++) {
            bb = north_west(bb);
            if (!bb) break;
            ray |= bb;
            
            if (bb & our_pieces) {
                pieces_found++;
                if (pieces_found == 1) pinned_sq = lsb(bb);
                else break;
            } else if (bb & their_pieces) {
                if (pieces_found == 1 && (bb & their_diag)) {
    int attacker_sq = lsb(bb);
    printf("GPU DEBUG: Marking pin - King:%d, Pinned:%d, Attacker:%d, Direction:NE\n",
           king_sq, pinned_sq, attacker_sq);
    pin_info->pinned_pieces |= (1ULL << pinned_sq);
    pin_info->pin_rays[pinned_sq] = ray | bb;
}
                break;
            }
        }
    }
    
    // South-East
    {
        Bitboard ray = 0;
        Bitboard bb = king;
        int pieces_found = 0;
        int pinned_sq = -1;
        
        for (int i = 0; i < 7; i++) {
            bb = south_east(bb);
            if (!bb) break;
            ray |= bb;
            
            if (bb & our_pieces) {
                pieces_found++;
                if (pieces_found == 1) pinned_sq = lsb(bb);
                else break;
            } else if (bb & their_pieces) {
                if (pieces_found == 1 && (bb & their_diag)) {
                    int attacker_sq = lsb(bb);
                    printf("GPU DEBUG: Marking pin - King:%d, Pinned:%d, Attacker:%d, Direction:NE\n",
                        king_sq, pinned_sq, attacker_sq);
                    pin_info->pinned_pieces |= (1ULL << pinned_sq);
                    pin_info->pin_rays[pinned_sq] = ray | bb;
                }
                break;
            }
        }
    }
    
    // South-West
    {
        Bitboard ray = 0;
        Bitboard bb = king;
        int pieces_found = 0;
        int pinned_sq = -1;
        
        for (int i = 0; i < 7; i++) {
            bb = south_west(bb);
            if (!bb) break;
            ray |= bb;
            
            if (bb & our_pieces) {
                pieces_found++;
                if (pieces_found == 1) pinned_sq = lsb(bb);
                else break;
            } else if (bb & their_pieces) {
                if (pieces_found == 1 && (bb & their_diag)) {
                    int attacker_sq = lsb(bb);
                    printf("GPU DEBUG: Marking pin - King:%d, Pinned:%d, Attacker:%d, Direction:NE\n",
                        king_sq, pinned_sq, attacker_sq);
                    pin_info->pinned_pieces |= (1ULL << pinned_sq);
                    pin_info->pin_rays[pinned_sq] = ray | bb;
                }
                break;
            }
        }
    }
    
    // North
    {
        Bitboard ray = 0;
        Bitboard bb = king;
        int pieces_found = 0;
        int pinned_sq = -1;
        
        for (int i = 0; i < 7; i++) {
            bb = north(bb);
            if (!bb) break;
            ray |= bb;
            
            if (bb & our_pieces) {
                pieces_found++;
                if (pieces_found == 1) pinned_sq = lsb(bb);
                else break;
            } else if (bb & their_pieces) {
                if (pieces_found == 1 && (bb & their_ortho)) {
                    int attacker_sq = lsb(bb);
                    printf("GPU DEBUG: Marking pin - King:%d, Pinned:%d, Attacker:%d, Direction:NE\n",
                        king_sq, pinned_sq, attacker_sq);
                    pin_info->pinned_pieces |= (1ULL << pinned_sq);
                    pin_info->pin_rays[pinned_sq] = ray | bb;
                }
                break;
            }
        }
    }
    
    // South
    {
        Bitboard ray = 0;
        Bitboard bb = king;
        int pieces_found = 0;
        int pinned_sq = -1;
        
        for (int i = 0; i < 7; i++) {
            bb = south(bb);
            if (!bb) break;
            ray |= bb;
            
            if (bb & our_pieces) {
                pieces_found++;
                if (pieces_found == 1) pinned_sq = lsb(bb);
                else break;
            } else if (bb & their_pieces) {
                if (pieces_found == 1 && (bb & their_ortho)) {
                    int attacker_sq = lsb(bb);
                    printf("GPU DEBUG: Marking pin - King:%d, Pinned:%d, Attacker:%d, Direction:NE\n",
                        king_sq, pinned_sq, attacker_sq);
                    pin_info->pinned_pieces |= (1ULL << pinned_sq);
                    pin_info->pin_rays[pinned_sq] = ray | bb;
                }
                break;
            }
        }
    }
    
    // East
    {
        Bitboard ray = 0;
        Bitboard bb = king;
        int pieces_found = 0;
        int pinned_sq = -1;
        
        for (int i = 0; i < 7; i++) {
            bb = east(bb);
            if (!bb) break;
            ray |= bb;
            
            if (bb & our_pieces) {
                pieces_found++;
                if (pieces_found == 1) pinned_sq = lsb(bb);
                else break;
            } else if (bb & their_pieces) {
                if (pieces_found == 1 && (bb & their_ortho)) {
                    int attacker_sq = lsb(bb);
                    printf("GPU DEBUG: Marking pin - King:%d, Pinned:%d, Attacker:%d, Direction:NE\n",
                        king_sq, pinned_sq, attacker_sq);
                    pin_info->pinned_pieces |= (1ULL << pinned_sq);
                    pin_info->pin_rays[pinned_sq] = ray | bb;
                }
                break;
            }
        }
    }
    
    // West
    {
        Bitboard ray = 0;
        Bitboard bb = king;
        int pieces_found = 0;
        int pinned_sq = -1;
        
        for (int i = 0; i < 7; i++) {
            bb = west(bb);
            if (!bb) break;
            ray |= bb;
            
            if (bb & our_pieces) {
                pieces_found++;
                if (pieces_found == 1) pinned_sq = lsb(bb);
                else break;
            } else if (bb & their_pieces) {
                if (pieces_found == 1 && (bb & their_ortho)) {
                    int attacker_sq = lsb(bb);
                    printf("GPU DEBUG: Marking pin - King:%d, Pinned:%d, Attacker:%d, Direction:NE\n",
                        king_sq, pinned_sq, attacker_sq);
                    pin_info->pinned_pieces |= (1ULL << pinned_sq);
                    pin_info->pin_rays[pinned_sq] = ray | bb;
                }
                break;
            }
        }
    }
}

// ============================================================================
// Legal Move Generation
// ============================================================================

__device__ int generate_moves(const Position* pos, Move* moves) {
    Color us = (Color)pos->side_to_move;
    Color them = (Color)!us;
    int our_offset = us * 6;
    int their_offset = them * 6;
    
    printf("\n=== GENERATE_MOVES START ===\n");
    printf("Side to move: %s (us=%d them=%d)\n", us == WHITE ? "WHITE" : "BLACK", us, them);
    printf("Offsets: our=%d their=%d\n", our_offset, their_offset);
    printf("Our pieces - Pawns: 0x%016llx\n", pos->pieces[our_offset + PAWN]);
    printf("Our pieces - King: 0x%016llx\n", pos->pieces[our_offset + KING]);
    printf("Their pieces - Pawns: 0x%016llx\n", pos->pieces[their_offset + PAWN]);
    printf("Their pieces - King: 0x%016llx\n", pos->pieces[their_offset + KING]);
    printf("Occupied[WHITE]: 0x%016llx\n", pos->occupied[WHITE]);
    printf("Occupied[BLACK]: 0x%016llx\n", pos->occupied[BLACK]);
    
    Bitboard our_pieces = pos->occupied[us];
    Bitboard their_pieces = pos->occupied[them];
    Bitboard empty_squares = ~pos->occupied[2];
    
    Move* move_ptr = moves;
    
    // Get checkers and pins
    Bitboard checkers = get_checkers(pos);
    int num_checkers = count_bits(checkers);
    
    PinInfo pin_info;
    calc_pins(pos, &pin_info);

    if (pin_info.pinned_pieces) {
    printf("GPU DEBUG: Pinned pieces bitboard: 0x%llx\n", pin_info.pinned_pieces);
    Bitboard temp = pin_info.pinned_pieces;
    while (temp) {
        int sq = pop_lsb(&temp);
        printf("  Pinned square %d, pin_ray: 0x%llx\n", sq, pin_info.pin_rays[sq]);
    }
}
    
    Bitboard king = pos->pieces[our_offset + KING];
    int king_sq = lsb(king);
    
    // King moves (always valid even in double check)
    {
        Bitboard king_moves = king_attacks(king) & ~our_pieces;
        while (king_moves) {
            int to = pop_lsb(&king_moves);
            // King cannot move to attacked square
            if (!is_square_attacked(pos, to, them)) {
                Move flags = (1ULL << to) & their_pieces ? CAPTURE : QUIET_MOVE;
                *move_ptr++ = make_move(king_sq, to, flags);
            }
        }
    }
    
    // If double check, only king moves are legal
    if (num_checkers > 1) {
        return move_ptr - moves;
    }
    
    // Calculate check mask (squares that block/capture checker)
    Bitboard check_mask = ~0ULL;
    if (num_checkers == 1) {
        int checker_sq = lsb(checkers);
        check_mask = checkers; // Can capture checker
        
        // Add blocking squares if it's a sliding piece check
        if ((1ULL << checker_sq) & (pos->pieces[their_offset + BISHOP] | pos->pieces[their_offset + ROOK] | pos->pieces[their_offset + QUEEN])) {
            // Generate ray between king and checker
            int kx = sq_file(king_sq), ky = sq_rank(king_sq);
            int cx = sq_file(checker_sq), cy = sq_rank(checker_sq);
            
            int dx = cx - kx;
            int dy = cy - ky;
            
            // Normalize direction
            int sx = (dx > 0) - (dx < 0);
            int sy = (dy > 0) - (dy < 0);
            
            int x = kx + sx;
            int y = ky + sy;
            
            while (x != cx || y != cy) {
                check_mask |= (1ULL << make_square(x, y));
                x += sx;
                y += sy;
            }
        }
    }
    
    // Generate moves for other pieces
    // Pawns
    {
        Bitboard pawns = pos->pieces[our_offset + PAWN];
        Bitboard unpinned_pawns = pawns & ~pin_info.pinned_pieces;
        
        // Single pushes
        Bitboard push_targets = us == WHITE ? north(unpinned_pawns) : south(unpinned_pawns);
        push_targets &= empty_squares & check_mask;
        
        while (push_targets) {
            int to = pop_lsb(&push_targets);
            int from = us == WHITE ? to - 8 : to + 8;
            
            // Check for promotion
            if ((to >= 56 && us == WHITE) || (to <= 7 && us == BLACK)) {
                *move_ptr++ = make_move(from, to, QUEEN_PROMO);
                *move_ptr++ = make_move(from, to, KNIGHT_PROMO);
                *move_ptr++ = make_move(from, to, ROOK_PROMO);
                *move_ptr++ = make_move(from, to, BISHOP_PROMO);
            } else {
                *move_ptr++ = make_move(from, to, QUIET_MOVE);
            }
        }
        
        // Double pushes
        Bitboard double_push_src = unpinned_pawns & (us == WHITE ? RANK_2 : RANK_7);
        Bitboard single_push = us == WHITE ? north(double_push_src) : south(double_push_src);
        single_push &= empty_squares;
        Bitboard double_push = us == WHITE ? north(single_push) : south(single_push);
        double_push &= empty_squares & check_mask;
        
        while (double_push) {
            int to = pop_lsb(&double_push);
            int from = us == WHITE ? to - 16 : to + 16;
            *move_ptr++ = make_move(from, to, DOUBLE_PUSH);
        }
        
        // Captures
        Bitboard pawn_attacks_left = us == WHITE ? north_west(unpinned_pawns) : south_west(unpinned_pawns);
        Bitboard pawn_attacks_right = us == WHITE ? north_east(unpinned_pawns) : south_east(unpinned_pawns);
        
        pawn_attacks_left &= their_pieces & check_mask;
        pawn_attacks_right &= their_pieces & check_mask;
        
        while (pawn_attacks_left) {
            int to = pop_lsb(&pawn_attacks_left);
            int from = us == WHITE ? to - 9 : to + 7;
            if ((to >= 56 && us == WHITE) || (to <= 7 && us == BLACK)) {
                *move_ptr++ = make_move(from, to, QUEEN_PROMO_CAP);
                *move_ptr++ = make_move(from, to, KNIGHT_PROMO_CAP);
                *move_ptr++ = make_move(from, to, ROOK_PROMO_CAP);
                *move_ptr++ = make_move(from, to, BISHOP_PROMO_CAP);
            } else {
                *move_ptr++ = make_move(from, to, CAPTURE);
            }
        }
        
        while (pawn_attacks_right) {
            int to = pop_lsb(&pawn_attacks_right);
            int from = us == WHITE ? to - 7 : to + 9;
            if ((to >= 56 && us == WHITE) || (to <= 7 && us == BLACK)) {
                *move_ptr++ = make_move(from, to, QUEEN_PROMO_CAP);
                *move_ptr++ = make_move(from, to, KNIGHT_PROMO_CAP);
                *move_ptr++ = make_move(from, to, ROOK_PROMO_CAP);
                *move_ptr++ = make_move(from, to, BISHOP_PROMO_CAP);
            } else {
                *move_ptr++ = make_move(from, to, CAPTURE);
            }
        }
        
        // En passant
        if (pos->ep_square >= 0 && pos->ep_square < 64) {
            // En passant square should be on rank 3 (index 16-23) or rank 6 (index 40-47) in flipped position
            // If it's outside these ranges after flipping, there's a bug
            int ep_rank = pos->ep_square / 8;
            if (ep_rank == 2 || ep_rank == 5) {  // Only valid EP ranks (3rd or 6th rank, 0-indexed)
                Bitboard ep_bb = 1ULL << pos->ep_square;

                // Find pawns that can capture en passant
                // pawn_attacks(ep_bb, them) gives squares where 'them' colored pawns would be
                // to attack ep_bb. Since we want 'us' colored pawns to capture, we need the inverse.
                Bitboard ep_capturers = pawn_attacks(ep_bb, them) & pawns & ~pin_info.pinned_pieces;
                // Only use unpinned pawns to avoid complex pin-along-rank validation

                while (ep_capturers) {
                    int from = pop_lsb(&ep_capturers);

                    // Verify by simulation (handles discovered checks)
                    Position temp_pos = *pos;
                    Move m = make_move(from, pos->ep_square, EP_CAPTURE);
                    apply_move(&temp_pos, m);

                    // After apply_move, position is flipped
                    // Our king is now the BLACK king in temp_pos
                    Bitboard black_king = temp_pos.pieces[6 + KING];
                    if (black_king) {  // Verify king exists
                        int king_sq = lsb(black_king);
                        if (!is_square_attacked(&temp_pos, king_sq, WHITE)) {
                            *move_ptr++ = m;
                        }
                    }
                }
            }
        }
        
        // Pinned pawn moves
        Bitboard pinned_pawns = pawns & pin_info.pinned_pieces;
        while (pinned_pawns) {
            int from = pop_lsb(&pinned_pawns);
            Bitboard pin_ray = pin_info.pin_rays[from];

            // Can only move along pin ray
            int to_push = us == WHITE ? from + 8 : from - 8;
            if ((1ULL << to_push) & pin_ray & empty_squares & check_mask) {
                if ((to_push >= 56 && us == WHITE) || (to_push <= 7 && us == BLACK)) {
                    *move_ptr++ = make_move(from, to_push, QUEEN_PROMO);
                    *move_ptr++ = make_move(from, to_push, KNIGHT_PROMO);
                    *move_ptr++ = make_move(from, to_push, ROOK_PROMO);
                    *move_ptr++ = make_move(from, to_push, BISHOP_PROMO);
                } else {
                    *move_ptr++ = make_move(from, to_push, QUIET_MOVE);

                    // Double push for pinned pawns on starting rank (only if pinned vertically)
                    // Since we're inside the block where to_push is empty and on pin_ray,
                    // we can safely check for double push
                    bool on_start_rank = (us == WHITE && from >= 8 && from <= 15) ||
                                       (us == BLACK && from >= 48 && from <= 55);
                    if (on_start_rank) {
                        int to_double = us == WHITE ? from + 16 : from - 16;
                        // to_push is already verified as empty, now check to_double
                        if ((1ULL << to_double) & pin_ray & empty_squares & check_mask) {
                            *move_ptr++ = make_move(from, to_double, DOUBLE_PUSH);
                        }
                    }
                }
            }

            // Captures along pin ray
            Bitboard attacks = pawn_attacks(1ULL << from, us);
            attacks &= pin_ray & their_pieces & check_mask;
            while (attacks) {
                int to = pop_lsb(&attacks);
                if ((to >= 56 && us == WHITE) || (to <= 7 && us == BLACK)) {
                    *move_ptr++ = make_move(from, to, QUEEN_PROMO_CAP);
                    *move_ptr++ = make_move(from, to, KNIGHT_PROMO_CAP);
                    *move_ptr++ = make_move(from, to, ROOK_PROMO_CAP);
                    *move_ptr++ = make_move(from, to, BISHOP_PROMO_CAP);
                } else {
                    *move_ptr++ = make_move(from, to, CAPTURE);
                }
            }
        }
    }
    
    // Knights
    {
        Bitboard knights = pos->pieces[our_offset + KNIGHT] & ~pin_info.pinned_pieces;
        while (knights) {
            int from = pop_lsb(&knights);
            Bitboard moves_bb = knight_attacks(1ULL << from) & ~our_pieces & check_mask;
            while (moves_bb) {
                int to = pop_lsb(&moves_bb);
                Move flags = (1ULL << to) & their_pieces ? CAPTURE : QUIET_MOVE;
                *move_ptr++ = make_move(from, to, flags);
            }
        }
    }
    
    // Bishops
    {
        Bitboard bishops = pos->pieces[our_offset + BISHOP];
        Bitboard unpinned = bishops & ~pin_info.pinned_pieces;
        while (unpinned) {
            int from = pop_lsb(&unpinned);
            
            #ifndef __CUDA_ARCH__
            if (from == 2) { // Bishop at C1
                bool d2_empty = (empty_squares >> 11) & 1;
                printf("DEBUG: Bishop C1 from %d. D2 EmptyBit: %d. Occupied[2] D2: %d.\n", 
                    from, d2_empty, (int)((pos->occupied[2] >> 11) & 1));
            }
            #endif

            Bitboard moves_bb = get_bishop_attacks(1ULL << from, empty_squares) & ~our_pieces & check_mask;
            while (moves_bb) {
                int to = pop_lsb(&moves_bb);
                Move flags = (1ULL << to) & their_pieces ? CAPTURE : QUIET_MOVE;
                *move_ptr++ = make_move(from, to, flags);
            }
        }
        
        // Pinned bishops
        Bitboard pinned = bishops & pin_info.pinned_pieces;
        while (pinned) {
            int from = pop_lsb(&pinned);
            Bitboard pin_ray = pin_info.pin_rays[from];
            Bitboard moves_bb = get_bishop_attacks(1ULL << from, empty_squares) & ~our_pieces & check_mask & pin_ray;
            while (moves_bb) {
                int to = pop_lsb(&moves_bb);
                Move flags = (1ULL << to) & their_pieces ? CAPTURE : QUIET_MOVE;
                *move_ptr++ = make_move(from, to, flags);
            }
        }
    }
    
    // Rooks
    {
        Bitboard rooks = pos->pieces[our_offset + ROOK];
        Bitboard unpinned = rooks & ~pin_info.pinned_pieces;
        while (unpinned) {
            int from = pop_lsb(&unpinned);
            Bitboard moves_bb = get_rook_attacks(1ULL << from, empty_squares) & ~our_pieces & check_mask;
            while (moves_bb) {
                int to = pop_lsb(&moves_bb);
                Move flags = (1ULL << to) & their_pieces ? CAPTURE : QUIET_MOVE;
                *move_ptr++ = make_move(from, to, flags);
            }
        }
        
        // Pinned rooks
        Bitboard pinned = rooks & pin_info.pinned_pieces;
        while (pinned) {
            int from = pop_lsb(&pinned);
            Bitboard pin_ray = pin_info.pin_rays[from];
            Bitboard moves_bb = get_rook_attacks(1ULL << from, empty_squares) & ~our_pieces & check_mask & pin_ray;
            while (moves_bb) {
                int to = pop_lsb(&moves_bb);
                Move flags = (1ULL << to) & their_pieces ? CAPTURE : QUIET_MOVE;
                *move_ptr++ = make_move(from, to, flags);
            }
        }
    }
    
    // Queens
    {
        Bitboard queens = pos->pieces[our_offset + QUEEN];
        Bitboard unpinned = queens & ~pin_info.pinned_pieces;
        while (unpinned) {
            int from = pop_lsb(&unpinned);
            Bitboard moves_bb = get_queen_attacks(1ULL << from, empty_squares) & ~our_pieces & check_mask;
            while (moves_bb) {
                int to = pop_lsb(&moves_bb);
                Move flags = (1ULL << to) & their_pieces ? CAPTURE : QUIET_MOVE;
                *move_ptr++ = make_move(from, to, flags);
            }
        }
        
        // Pinned queens
        Bitboard pinned = queens & pin_info.pinned_pieces;
        while (pinned) {
            int from = pop_lsb(&pinned);
            Bitboard pin_ray = pin_info.pin_rays[from];
            Bitboard moves_bb = get_queen_attacks(1ULL << from, empty_squares) & ~our_pieces & check_mask & pin_ray;
            while (moves_bb) {
                int to = pop_lsb(&moves_bb);
                Move flags = (1ULL << to) & their_pieces ? CAPTURE : QUIET_MOVE;
                *move_ptr++ = make_move(from, to, flags);
            }
        }
    }
    
    // Castling (only if not in check)
    if (num_checkers == 0) {
        if (us == WHITE) {
            // Kingside
            // Ensure rook is actually at H1 (7)
            if ((pos->castling & CASTLE_WK) && 
                (pos->pieces[our_offset + ROOK] & 0x80ULL) && 
                (empty_squares & 0x60ULL) == 0x60ULL &&
                !is_square_attacked(pos, 5, BLACK) &&
                !is_square_attacked(pos, 6, BLACK)) {
                *move_ptr++ = make_move(4, 6, KING_CASTLE);
            }
            // Queenside
            // Ensure rook is actually at A1 (0)
            if ((pos->castling & CASTLE_WQ) && 
                (pos->pieces[our_offset + ROOK] & 0x01ULL) &&
                (empty_squares & 0x0EULL) == 0x0EULL &&
                !is_square_attacked(pos, 3, BLACK) &&
                !is_square_attacked(pos, 2, BLACK)) {
                *move_ptr++ = make_move(4, 2, QUEEN_CASTLE);
            }
        } else {
            // Kingside
            // Ensure rook is actually at H8 (63)
            if ((pos->castling & CASTLE_BK) && 
                (pos->pieces[our_offset + ROOK] & 0x8000000000000000ULL) &&
                (empty_squares & 0x6000000000000000ULL) == 0x6000000000000000ULL &&
                !is_square_attacked(pos, 61, WHITE) &&
                !is_square_attacked(pos, 62, WHITE)) {
                *move_ptr++ = make_move(60, 62, KING_CASTLE);
            }
            // Queenside
            // Ensure rook is actually at A8 (56)
            if ((pos->castling & CASTLE_BQ) && 
                (pos->pieces[our_offset + ROOK] & 0x0100000000000000ULL) &&
                (empty_squares & 0x0E00000000000000ULL) == 0x0E00000000000000ULL &&
                !is_square_attacked(pos, 59, WHITE) &&
                !is_square_attacked(pos, 58, WHITE)) {
                *move_ptr++ = make_move(60, 58, QUEEN_CASTLE);
            }
        }
    }
    
    int move_count = move_ptr - moves;
    printf("Generated %d total moves\n", move_count);
    printf("=== GENERATE_MOVES END ===\n\n");
    
    return move_count;
}

// ============================================================================
// Game State Detection
// ============================================================================

__device__ bool has_legal_moves(const Position* pos) {
    Move moves[256];
    return generate_moves(pos, moves) > 0;
}

__device__ bool is_checkmate(const Position* pos) {
    Bitboard checkers = get_checkers(pos);
    if (checkers == 0) return false;  // Not in check
    return !has_legal_moves(pos);
}

__device__ bool is_stalemate(const Position* pos) {
    Bitboard checkers = get_checkers(pos);
    if (checkers != 0) return false;  // In check
    return !has_legal_moves(pos);
}

__device__ bool is_draw(const Position* pos) {
    // 50-move rule
    if (pos->halfmove >= 100) return true;
    
    // Insufficient material (simplified - just check major pieces)
    Color us = (Color)pos->side_to_move;
    Color them = (Color)!us;
    int our_offset = us * 6;
    int their_offset = them * 6;
    
    bool our_has_major = pos->pieces[our_offset + PAWN] ||
                         pos->pieces[our_offset + ROOK] ||
                         pos->pieces[our_offset + QUEEN];
    bool their_has_major = pos->pieces[their_offset + PAWN] ||
                           pos->pieces[their_offset + ROOK] ||
                           pos->pieces[their_offset + QUEEN];
    
    // If both sides have only king + minor pieces or less
    if (!our_has_major && !their_has_major) {
        int our_minors = count_bits(pos->pieces[our_offset + KNIGHT] | 
                                    pos->pieces[our_offset + BISHOP]);
        int their_minors = count_bits(pos->pieces[their_offset + KNIGHT] | 
                                      pos->pieces[their_offset + BISHOP]);
        
        // K vs K, KN vs K, KB vs K, etc.
        if (our_minors <= 1 && their_minors <= 1) return true;
    }
    
    return false;
}

__device__ void update_game_result(Position* pos) {
    if (is_checkmate(pos)) {
        pos->result = pos->side_to_move == WHITE ? BLACK_WINS : WHITE_WINS;
    } else if (is_stalemate(pos) || is_draw(pos)) {
        pos->result = DRAW;
    } else {
        pos->result = ONGOING;
    }
}
