#include "../src/gpu_chess_types.cuh"
#include <iostream>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

// External kernel launch functions
extern "C" unsigned long long run_perft(Position* h_pos, int depth);
extern "C" int test_move_generation(Position* h_pos, Move* h_moves);
extern "C" void run_perft_divide(Position* h_pos, int depth, unsigned long long* h_counts, Move* h_moves, int* h_num_moves);

// Move category names for debugging
const char* get_move_type_name(Move flags) {
    switch (flags) {
        case QUIET_MOVE:       return "Quiet";
        case DOUBLE_PUSH:      return "DoublePush";
        case KING_CASTLE:      return "KingCastle";
        case QUEEN_CASTLE:     return "QueenCastle";
        case CAPTURE:          return "Capture";
        case EP_CAPTURE:       return "EnPassant";
        case KNIGHT_PROMO:     return "PromoN";
        case BISHOP_PROMO:     return "PromoB";
        case ROOK_PROMO:       return "PromoR";
        case QUEEN_PROMO:      return "PromoQ";
        case KNIGHT_PROMO_CAP: return "PromoCapN";
        case BISHOP_PROMO_CAP: return "PromoCapB";
        case ROOK_PROMO_CAP:   return "PromoCapR";
        case QUEEN_PROMO_CAP:  return "PromoCapQ";
        default:               return "Unknown";
    }
}

// ============================================================================
// Position Initialization
// ============================================================================

void init_startpos(Position* pos) {
    memset(pos, 0, sizeof(Position));
    
    // White pieces
    pos->pieces[0 * 6 + PAWN] = RANK_2;
    pos->pieces[0 * 6 + KNIGHT] = 0x42ULL;  // B1, G1
    pos->pieces[0 * 6 + BISHOP] = 0x24ULL;  // C1, F1
    pos->pieces[0 * 6 + ROOK] = 0x81ULL;    // A1, H1
    pos->pieces[0 * 6 + QUEEN] = 0x08ULL;   // D1
    pos->pieces[0 * 6 + KING] = 0x10ULL;    // E1
    
    // Black pieces
    pos->pieces[1 * 6 + PAWN] = RANK_7;
    pos->pieces[1 * 6 + KNIGHT] = 0x4200000000000000ULL;  // B8, G8
    pos->pieces[1 * 6 + BISHOP] = 0x2400000000000000ULL;  // C8, F8
    pos->pieces[1 * 6 + ROOK] = 0x8100000000000000ULL;    // A8, H8
    pos->pieces[1 * 6 + QUEEN] = 0x0800000000000000ULL;   // D8
    pos->pieces[1 * 6 + KING] = 0x1000000000000000ULL;    // E8
    
    // Occupancy
    pos->occupied[WHITE] = 0xFFFFULL;
    pos->occupied[BLACK] = 0xFFFF000000000000ULL;
    pos->occupied[2] = pos->occupied[WHITE] | pos->occupied[BLACK];
    
    // Game state
    pos->side_to_move = WHITE;
    pos->castling = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;
    pos->ep_square = -1;
    pos->halfmove = 0;
    pos->result = ONGOING;
}

void init_d2d3_pos(Position* pos) {
    init_startpos(pos);
    
    // Apply d2d3 (White Pawn 11 -> 19)
    // Clear d2 (11)
    pos->pieces[0 * 6 + PAWN] &= ~(1ULL << 11);
    pos->occupied[WHITE] &= ~(1ULL << 11);
    
    // Set d3 (19)
    pos->pieces[0 * 6 + PAWN] |= (1ULL << 19);
    pos->occupied[WHITE] |= (1ULL << 19);
    
    // Update total occupied
    pos->occupied[2] = pos->occupied[WHITE] | pos->occupied[BLACK];
    
    // Side to move -> Black
    pos->side_to_move = BLACK;
    // EP square -1
    pos->ep_square = -1;
    // Halfmove increment
    pos->halfmove = 0; // Pawn move resets
}

void init_d3d6_pos(Position* pos) {
    init_startpos(pos);
    
    // Apply d2d3 (White P 11->19)
    pos->pieces[0 * 6 + PAWN] &= ~(1ULL << 11);
    pos->occupied[WHITE] &= ~(1ULL << 11);
    pos->pieces[0 * 6 + PAWN] |= (1ULL << 19);
    pos->occupied[WHITE] |= (1ULL << 19);
    
    // Apply d7d6 (Black P 51->43)
    pos->pieces[1 * 6 + PAWN] &= ~(1ULL << 51);
    pos->occupied[BLACK] &= ~(1ULL << 51);
    pos->pieces[1 * 6 + PAWN] |= (1ULL << 43);
    pos->occupied[BLACK] |= (1ULL << 43);
    
    // Update total occupied
    pos->occupied[2] = pos->occupied[WHITE] | pos->occupied[BLACK];
    
    // Side to move -> White
    pos->side_to_move = WHITE;
    // EP square -1
    pos->ep_square = -1;
}

// ============================================================================
// Utility Functions
// ============================================================================

std::string square_name(int sq) {
    char name[3];
    name[0] = 'a' + (sq & 7);
    name[1] = '1' + (sq >> 3);
    name[2] = '\0';
    return std::string(name);
}

void print_move(Move move) {
    int from = move_from(move);
    int to = move_to(move);
    Move flags = move_flags(move);
    
    std::cout << square_name(from) << square_name(to);
    
    if (is_promotion(move)) {
        PieceType pt = promotion_type(move);
        const char* promo_chars = "nbrq";
        if (pt >= KNIGHT && pt <= QUEEN) {
            std::cout << promo_chars[pt - KNIGHT];
        }
    }
}

// ============================================================================
// Test Functions
// ============================================================================

void test_perft() {
    std::cout << "=== Perft Test ===" << std::endl;

    Position pos;
    init_startpos(&pos);

    // Expected perft values for starting position
    unsigned long long expected[] = {1, 20, 400, 8902, 197281, 4865609};

    for (int depth = 0; depth <= 4; depth++) {
        unsigned long long result = run_perft(&pos, depth);
        bool passed = result == expected[depth];

        std::cout << "Depth " << depth << ": " << result
                  << (passed ? " PASS" : " FAIL (expected " + std::to_string(expected[depth]) + ")")
                  << std::endl;
    }
}

void test_perft_divide() {
    std::cout << "\n=== Perft Divide (Depth 4 from start position) ===" << std::endl;

    Position pos;
    init_startpos(&pos);

    Move moves[256];
    unsigned long long counts[256];
    int num_moves;

    run_perft_divide(&pos, 4, counts, moves, &num_moves);

    // Expected counts for each move at depth 4
    // From https://www.chessprogramming.org/Perft_Results
    // Order: single pushes (a3-h3), double pushes (a4-h4), knights (Na3, Nc3, Nf3, Nh3)
    unsigned long long expected_counts[] = {
        9893,  // a2a3
        9345,  // b2b3
        9272,  // c2c3
        8073,  // d2d3
        9726,  // e2e3
        8457,  // f2f3
        9345,  // g2g3
        9893,  // h2h3
        9467,  // a2a4
        9332,  // b2b4
        9744,  // c2c4
        12435, // d2d4
        13134, // e2e4
        8929,  // f2f4
        9328,  // g2g4
        9467,  // h2h4
        8885,  // b1a3
        9755,  // b1c3
        9748,  // g1f3
        8881   // g1h3
    };

    unsigned long long total = 0;
    std::cout << "\nMove counts:" << std::endl;
    for (int i = 0; i < num_moves; i++) {
        std::cout << "  ";
        print_move(moves[i]);
        std::cout << ": " << counts[i];
        if (i < 20 && counts[i] != expected_counts[i]) {
            std::cout << " (expected " << expected_counts[i] << ", diff " << ((long long)counts[i] - (long long)expected_counts[i]) << ")";
        }
        std::cout << std::endl;
        total += counts[i];
    }

    std::cout << "\nTotal: " << total << " (expected 197281)" << std::endl;
    if (total != 197281) {
        std::cout << "Difference: " << ((long long)total - 197281) << " extra moves" << std::endl;
    }
}

void test_move_generation() {
    std::cout << "\n=== Move Generation Test ===" << std::endl;

    Position pos;
    init_startpos(&pos);

    Move moves[256];
    int num_moves = test_move_generation(&pos, moves);

    std::cout << "Starting position has " << num_moves << " legal moves (expected: 20)" << std::endl;

    // Count moves by category
    int quiet_count = 0, double_push_count = 0, capture_count = 0;
    int castle_count = 0, ep_count = 0, promo_count = 0;

    std::cout << "\n--- Moves by Category ---" << std::endl;

    // Print quiet moves (pawn single pushes, knight/king/etc)
    std::cout << "\nQuiet Moves:" << std::endl;
    for (int i = 0; i < num_moves; i++) {
        Move flags = move_flags(moves[i]);
        if (flags == QUIET_MOVE) {
            std::cout << "  ";
            print_move(moves[i]);
            std::cout << std::endl;
            quiet_count++;
        }
    }

    // Print double pushes
    std::cout << "\nDouble Pushes:" << std::endl;
    for (int i = 0; i < num_moves; i++) {
        Move flags = move_flags(moves[i]);
        if (flags == DOUBLE_PUSH) {
            std::cout << "  ";
            print_move(moves[i]);
            std::cout << std::endl;
            double_push_count++;
        }
    }

    // Print captures
    std::cout << "\nCaptures:" << std::endl;
    for (int i = 0; i < num_moves; i++) {
        Move flags = move_flags(moves[i]);
        if (flags == CAPTURE) {
            std::cout << "  ";
            print_move(moves[i]);
            std::cout << std::endl;
            capture_count++;
        }
    }

    // Print castling
    std::cout << "\nCastling:" << std::endl;
    for (int i = 0; i < num_moves; i++) {
        Move flags = move_flags(moves[i]);
        if (flags == KING_CASTLE || flags == QUEEN_CASTLE) {
            std::cout << "  ";
            print_move(moves[i]);
            std::cout << " (" << get_move_type_name(flags) << ")" << std::endl;
            castle_count++;
        }
    }

    // Print en passant
    std::cout << "\nEn Passant:" << std::endl;
    for (int i = 0; i < num_moves; i++) {
        Move flags = move_flags(moves[i]);
        if (flags == EP_CAPTURE) {
            std::cout << "  ";
            print_move(moves[i]);
            std::cout << std::endl;
            ep_count++;
        }
    }

    // Print promotions
    std::cout << "\nPromotions:" << std::endl;
    for (int i = 0; i < num_moves; i++) {
        if (is_promotion(moves[i])) {
            std::cout << "  ";
            print_move(moves[i]);
            std::cout << " (" << get_move_type_name(move_flags(moves[i])) << ")" << std::endl;
            promo_count++;
        }
    }

    // Summary
    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "Quiet moves:   " << quiet_count << " (expected: 12 = 8 pawn + 4 knight)" << std::endl;
    std::cout << "Double pushes: " << double_push_count << " (expected: 8)" << std::endl;
    std::cout << "Captures:      " << capture_count << " (expected: 0)" << std::endl;
    std::cout << "Castling:      " << castle_count << " (expected: 0)" << std::endl;
    std::cout << "En passant:    " << ep_count << " (expected: 0)" << std::endl;
    std::cout << "Promotions:    " << promo_count << " (expected: 0)" << std::endl;
    std::cout << "TOTAL:         " << num_moves << " (expected: 20)" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

// ============================================================================
// Main
// ============================================================================

void init_a2a3_pos(Position* pos) {
    init_startpos(pos);
    
    // Apply a2a3 (White P 8->16)
    pos->pieces[0 * 6 + PAWN] &= ~(1ULL << 8);
    pos->occupied[WHITE] &= ~(1ULL << 8);
    pos->pieces[0 * 6 + PAWN] |= (1ULL << 16);
    pos->occupied[WHITE] |= (1ULL << 16);
    
    // Update total occupied
    pos->occupied[2] = pos->occupied[WHITE] | pos->occupied[BLACK];
    
    // Side to move -> Black
    pos->side_to_move = BLACK;
    // EP square -1
    pos->ep_square = -1;
    // Halfmove increment
    pos->halfmove = 0;
}

void test_a2a3_perft() {
    std::cout << "\n=== Perft Divide (Depth 2 from 'a2a3' position) ===" << std::endl;
    std::cout << "Expected count for most moves: ~22 (if missing moves)." << std::endl;

    Position pos;
    init_a2a3_pos(&pos);

    Move moves[256];
    unsigned long long counts[256];
    int num_moves;

    // Run depth 2 to see leaf counts of Black moves (i.e. number of White responses)
    run_perft_divide(&pos, 2, counts, moves, &num_moves);

    // Debug D2 occupancy
    int d2 = 11;
    bool d2_occ = (pos.occupied[WHITE] >> d2) & 1;
    std::cout << "D2 Occupancy (Expected 1): " << d2_occ << std::endl;

    unsigned long long total = 0;
    std::cout << "\nMove counts:" << std::endl;
    for (int i = 0; i < num_moves; i++) {
        std::cout << "  ";
        print_move(moves[i]);
        std::cout << ": " << counts[i] << std::endl;
        total += counts[i];
    }
    std::cout << "\nTotal: " << total << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "GPU Chess Logic Test" << std::endl;
    std::cout << "=====================\n" << std::endl;

    // Increase stack size for recursion
    // Frame size approx: 128 (Position) + 512 (moves) + overhead ~= 700 bytes
    // Depth 5 needs ~3.5KB. Default is 1KB.
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, 16384); // 16KB
    if (err != cudaSuccess) {
        std::cout << "Failed to set stack limit: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    test_move_generation();
    test_perft();
    test_perft_divide();
    test_a2a3_perft();

    return 0;
}
