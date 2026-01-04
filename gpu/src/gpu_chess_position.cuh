#pragma once
#include "gpu_chess_types.cuh"
#include "gpu_chess_types.cuh"
#include "gpu_chess_bitops.cuh"
#include <cstdio>

// ============================================================================
// Forward Declarations
// ============================================================================

__device__ void flip_position(Position* pos);

// ============================================================================
// Make Move
// ============================================================================

__device__ void apply_move(Position* pos, Move move) {
    Color us = (Color)pos->side_to_move;
    Color them = (Color)!us;
    
    int from = move_from(move);
    int to = move_to(move);
    Move flags = move_flags(move);
    
    printf("\n=== APPLY_MOVE START ===\n");
    printf("Move: from=%d to=%d flags=0x%04x\n", from, to, flags);
    printf("Side to move: %s (us=%d them=%d)\n", us == WHITE ? "WHITE" : "BLACK", us, them);
    printf("Offsets: our=%d their=%d\n", us * 6, them * 6);
    
    Bitboard from_bb = 1ULL << from;
    Bitboard to_bb = 1ULL << to;
    
    int our_offset = us * 6;
    int their_offset = them * 6;
    
    // Find piece type at 'from'
    PieceType moving_piece = NONE;
    for (int pt = PAWN; pt <= KING; pt++) {
        if (pos->pieces[our_offset + pt] & from_bb) {
            moving_piece = (PieceType)pt;
            printf("Found moving piece: type=%d at square=%d\n", pt, from);
            break;
        }
    }
    
    if (moving_piece == NONE) {
        printf("WARNING: No piece found at from square %d!\n", from);
    }
    
    // Update halfmove clock (reset on pawn move or capture)
    if (moving_piece == PAWN || is_capture(move)) {
        pos->halfmove = 0;
    } else {
        pos->halfmove++;
    }
    
    // Clear EP square (will be set again if double push)
    pos->ep_square = -1;
    
    // Handle special moves
    if (flags == KING_CASTLE) {
        // Move king
        pos->pieces[our_offset + KING] ^= from_bb | to_bb;
        pos->occupied[us] ^= from_bb | to_bb;
        
        // Move rook
        if (us == WHITE) {
            pos->pieces[our_offset + ROOK] ^= 0xA0ULL;  // H1 to F1
            pos->occupied[us] ^= 0xA0ULL;
        } else {
            pos->pieces[our_offset + ROOK] ^= 0xA000000000000000ULL;  // H8 to F8
            pos->occupied[us] ^= 0xA000000000000000ULL;
        }
        
        // Update castling rights
        pos->castling &= us == WHITE ? ~(CASTLE_WK | CASTLE_WQ) : ~(CASTLE_BK | CASTLE_BQ);
        
    } else if (flags == QUEEN_CASTLE) {
        // Move king
        pos->pieces[our_offset + KING] ^= from_bb | to_bb;
        pos->occupied[us] ^= from_bb | to_bb;
        
        // Move rook
        if (us == WHITE) {
            pos->pieces[our_offset + ROOK] ^= 0x09ULL;  // A1 to D1
            pos->occupied[us] ^= 0x09ULL;
        } else {
            pos->pieces[our_offset + ROOK] ^= 0x0900000000000000ULL;  // A8 to D8
            pos->occupied[us] ^= 0x0900000000000000ULL;
        }
        
        // Update castling rights
        pos->castling &= us == WHITE ? ~(CASTLE_WK | CASTLE_WQ) : ~(CASTLE_BK | CASTLE_BQ);
        
    } else if (flags == EP_CAPTURE) {
        // Move pawn
        pos->pieces[our_offset + PAWN] ^= from_bb | to_bb;
        pos->occupied[us] ^= from_bb | to_bb;
        
        // Remove captured pawn
        int captured_sq = us == WHITE ? to - 8 : to + 8;
        Bitboard captured_bb = 1ULL << captured_sq;
        pos->pieces[their_offset + PAWN] ^= captured_bb;
        pos->occupied[them] ^= captured_bb;
        
    } else if (is_promotion(move)) {
        // Remove pawn from 'from'
        pos->pieces[our_offset + PAWN] ^= from_bb;
        pos->occupied[us] ^= from_bb;
        
        // Handle capture
        if (is_capture(move)) {
            for (int pt = PAWN; pt <= QUEEN; pt++) {
                if (pos->pieces[their_offset + pt] & to_bb) {
                    pos->pieces[their_offset + pt] ^= to_bb;
                    pos->occupied[them] ^= to_bb;
                    break;
                }
            }
        }
        
        // Add promoted piece to 'to'
        PieceType promo = promotion_type(move);
        pos->pieces[our_offset + promo] |= to_bb;
        pos->occupied[us] |= to_bb;
        
    } else {
        // Normal move or capture
        
        // Handle capture
        if (is_capture(move)) {
            for (int pt = PAWN; pt <= QUEEN; pt++) {
                if (pos->pieces[their_offset + pt] & to_bb) {
                    pos->pieces[their_offset + pt] ^= to_bb;
                    pos->occupied[them] ^= to_bb;
                    break;
                }
            }
        }
        
        // Move our piece
        pos->pieces[our_offset + moving_piece] ^= from_bb | to_bb;
        pos->occupied[us] ^= from_bb | to_bb;
        
        // Set EP square if double push
        if (flags == DOUBLE_PUSH) {
            pos->ep_square = us == WHITE ? from + 8 : from - 8;
        }
    }
    
    // Update occupancy for all pieces
    pos->occupied[2] = pos->occupied[WHITE] | pos->occupied[BLACK];
    
    // Update castling rights
    if (moving_piece == KING) {
        uint8_t old_castling = pos->castling;
        pos->castling &= us == WHITE ? ~(CASTLE_WK | CASTLE_WQ) : ~(CASTLE_BK | CASTLE_BQ);
        printf("King moved: castling 0x%02x -> 0x%02x (us=%d)\n", old_castling, pos->castling, us);
    } else if (moving_piece == ROOK) {
        uint8_t old_castling = pos->castling;
        // Check specific rook starting squares for the side to move
        if (us == WHITE) {
            if (from == 0) pos->castling &= ~CASTLE_WQ;
            else if (from == 7) pos->castling &= ~CASTLE_WK;
        } else {
            if (from == 0) pos->castling &= ~CASTLE_BQ;
            else if (from == 7) pos->castling &= ~CASTLE_BK;
        }
        printf("Rook moved from %d: castling 0x%02x -> 0x%02x (us=%d)\n", from, old_castling, pos->castling, us);
    }

    // If opponent's rook was captured on a corner
    if (is_capture(move)) {
        if (them == WHITE) {
            if (to == 0) pos->castling &= ~CASTLE_WQ;
            else if (to == 7) pos->castling &= ~CASTLE_WK;
        } else {
            // "them" is BLACK: Rooks are at 56 (A8) and 63 (H8)
            if (to == 56) pos->castling &= ~CASTLE_BQ;
            else if (to == 63) pos->castling &= ~CASTLE_BK;
        }
    }
    
    printf("Before flip - side_to_move: %d\n", pos->side_to_move);
    
    // Flip side to move (Always stay White in relative perspective)
    pos->side_to_move = WHITE;
    
    printf("Set side_to_move to WHITE (0)\n");
    printf("=== APPLY_MOVE calling flip_position ===\n");
    
    // Perform flip
    flip_position(pos);
    
    printf("=== APPLY_MOVE END ===\n\n");
}

// ============================================================================
// Flip Position (Jaglavak Technique)
// ============================================================================

__device__ Bitboard flip_bitboard(Bitboard bb) {
    // Perform VERTICAL flip only (like Jaglavak's ByteSwap): mirror ranks, keep files same
    // Square A1 (0) -> A8 (56), E1 (4) -> E8 (60), etc.
    // Swap bytes to reverse rank order while preserving file order within each rank
    Bitboard result = ((bb & 0x00000000000000FFULL) << 56) |
                      ((bb & 0x000000000000FF00ULL) << 40) |
                      ((bb & 0x0000000000FF0000ULL) << 24) |
                      ((bb & 0x00000000FF000000ULL) <<  8) |
                      ((bb & 0x000000FF00000000ULL) >>  8) |
                      ((bb & 0x0000FF0000000000ULL) >> 24) |
                      ((bb & 0x00FF000000000000ULL) >> 40) |
                      ((bb & 0xFF00000000000000ULL) >> 56);
    return result;
}

__device__ void flip_position(Position* pos) {
    printf("\n=== FLIP_POSITION START ===\n");
    printf("Before flip - Side to move: %d\n", pos->side_to_move);
    printf("Before flip - White pieces[PAWN]: 0x%016llx\n", pos->pieces[WHITE * 6 + PAWN]);
    printf("Before flip - Black pieces[PAWN]: 0x%016llx\n", pos->pieces[BLACK * 6 + PAWN]);
    printf("Before flip - White King: 0x%016llx\n", pos->pieces[WHITE * 6 + KING]);
    printf("Before flip - Black King: 0x%016llx\n", pos->pieces[BLACK * 6 + KING]);
    printf("Before flip - EP square: %d\n", pos->ep_square);
    printf("Before flip - Castling: 0x%02x\n", pos->castling);
    
    // Swap colors
    for (int pt = 0; pt < 6; pt++) {
        Bitboard white_before = pos->pieces[pt];
        Bitboard black_before = pos->pieces[6 + pt];
        
        Bitboard temp = flip_bitboard(pos->pieces[pt]);
        pos->pieces[pt] = flip_bitboard(pos->pieces[6 + pt]);
        pos->pieces[6 + pt] = temp;
        
        if (pt == PAWN || pt == KING) {
            printf("Piece type %d: White 0x%llx->0x%llx, Black 0x%llx->0x%llx\n",
                   pt, white_before, pos->pieces[pt], black_before, pos->pieces[6 + pt]);
        }
    }
    
    // Swap occupancy
    Bitboard temp_occ = flip_bitboard(pos->occupied[WHITE]);
    pos->occupied[WHITE] = flip_bitboard(pos->occupied[BLACK]);
    pos->occupied[BLACK] = temp_occ;
    pos->occupied[2] = pos->occupied[WHITE] | pos->occupied[BLACK];
    
    printf("After swap - White occupied: 0x%016llx\n", pos->occupied[WHITE]);
    printf("After swap - Black occupied: 0x%016llx\n", pos->occupied[BLACK]);
    
    // Flip castling rights (WK <-> BK, WQ <-> BQ)
    uint8_t old_castling = pos->castling;
    uint8_t new_castling = 0;
    if (pos->castling & CASTLE_WK) new_castling |= CASTLE_BK;
    if (pos->castling & CASTLE_WQ) new_castling |= CASTLE_BQ;
    if (pos->castling & CASTLE_BK) new_castling |= CASTLE_WK;
    if (pos->castling & CASTLE_BQ) new_castling |= CASTLE_WQ;
    pos->castling = new_castling;
    printf("Castling rights: 0x%02x -> 0x%02x\n", old_castling, new_castling);
    
    // Flip EP square
    if (pos->ep_square >= 0) {
        int old_ep = pos->ep_square;
        pos->ep_square ^= 56;
        printf("EP square flipped: %d -> %d\n", old_ep, pos->ep_square);
    }
    
    printf("After flip - White King: 0x%016llx\n", pos->pieces[WHITE * 6 + KING]);
    printf("After flip - Black King: 0x%016llx\n", pos->pieces[BLACK * 6 + KING]);
    printf("=== FLIP_POSITION END ===\n\n");
    
    // Side to move stays the same (semantic meaning changes)
}

// (Function moved to gpu_chess_movegen.cuh to avoid circular dependency)
