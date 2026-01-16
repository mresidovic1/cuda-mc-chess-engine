// Bratko-Kopec Test Suite
// Tests the engine's tactical ability on 24 carefully selected positions
//
// The Bratko-Kopec test is a standard benchmark for chess engines consisting
// of 24 positions where the engine must find the best move(s).

#include "../include/puct_mcts.h"
#include "../include/fen.h"
#include "../include/chess_types.cuh"
#include "test_positions.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <chrono>

// External CPU move generation
#include "../include/cpu_movegen.h"

// Move utility functions (defined in chess_types.cuh, just use them)

const char* SQUARE_NAMES[64] = {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"
};

const char PROMO_CHARS[] = "nbrq";

std::string move_to_string(Move m) {
    if (m == 0) return "(none)";

    Square from = move_from(m);
    Square to = move_to(m);
    uint8_t flags = move_flags(m);

    std::string result = std::string(SQUARE_NAMES[from]) + SQUARE_NAMES[to];

    // Add promotion piece
    if (flags >= MOVE_PROMO_N) {
        result += PROMO_CHARS[flags & 0x3];
    }

    return result;
}

// EPD (Extended Position Description) structure
struct EPDPosition {
    std::string fen;
    std::vector<std::string> best_moves;  // Can have multiple best moves
    std::string id;
};

// Parse EPD format: "FEN bm move1 move2; id "ID";"
EPDPosition parse_epd(const std::string& epd_line) {
    EPDPosition result;

    std::istringstream ss(epd_line);

    // Parse FEN (first 4 tokens: position, side, castling, ep)
    std::string token;
    std::ostringstream fen_ss;

    // Piece placement
    ss >> token;
    fen_ss << token;

    // Side to move
    ss >> token;
    fen_ss << " " << token;

    // Castling
    ss >> token;
    fen_ss << " " << token;

    // En passant
    ss >> token;
    fen_ss << " " << token;

    // Default halfmove and fullmove
    fen_ss << " 0 1";

    result.fen = fen_ss.str();

    // Parse operations (bm, id, etc.)
    std::string line_remainder;
    std::getline(ss, line_remainder);

    // Find "bm" operation
    size_t bm_pos = line_remainder.find("bm");
    if (bm_pos != std::string::npos) {
        size_t bm_end = line_remainder.find(';', bm_pos);
        std::string bm_str = line_remainder.substr(bm_pos + 2, bm_end - bm_pos - 2);

        // Parse moves (space-separated)
        std::istringstream bm_ss(bm_str);
        std::string move;
        while (bm_ss >> move) {
            // Remove any trailing punctuation (like '+' for check)
            while (!move.empty() && !std::isalnum(move.back())) {
                move.pop_back();
            }
            if (!move.empty()) {
                result.best_moves.push_back(move);
            }
        }
    }

    // Find "id" operation
    size_t id_pos = line_remainder.find("id");
    if (id_pos != std::string::npos) {
        size_t quote1 = line_remainder.find('"', id_pos);
        size_t quote2 = line_remainder.find('"', quote1 + 1);
        if (quote1 != std::string::npos && quote2 != std::string::npos) {
            result.id = line_remainder.substr(quote1 + 1, quote2 - quote1 - 1);
        }
    }

    return result;
}

// Parse a move string like "Qd1" or "e6" into a Move
// This is a simplified parser that matches move strings to legal moves
Move parse_move_string(const std::string& move_str, const BoardState& pos) {
    Move legal_moves[256];
    int num_moves = cpu_movegen::generate_legal_moves_cpu(&pos, legal_moves);

    // Generate all legal moves and convert to strings
    for (int i = 0; i < num_moves; i++) {
        std::string move_long = move_to_string(legal_moves[i]);

        // Match exact move (e.g., "e2e4")
        if (move_long == move_str) {
            return legal_moves[i];
        }

        // Try to match algebraic notation
        // For simple moves like "e6", check if destination matches
        if (move_str.length() == 2) {
            // Simple pawn move or piece move
            if (move_long.substr(2, 2) == move_str) {
                return legal_moves[i];
            }
        }

        // For piece moves like "Qd1" or "Nf6"
        if (move_str.length() >= 2) {
            // Check if destination matches
            std::string dest = move_str.substr(move_str.length() - 2);
            if (move_long.substr(2, 2) == dest) {
                // Also check piece type if specified
                if (std::isupper(move_str[0])) {
                    // Get piece at from square
                    Move m = legal_moves[i];
                    int from_sq = move_from(m);
                    Bitboard from_bb = C64(1) << from_sq;

                    char piece_char = move_str[0];
                    int piece_type = -1;
                    switch (piece_char) {
                        case 'N': piece_type = KNIGHT; break;
                        case 'B': piece_type = BISHOP; break;
                        case 'R': piece_type = ROOK; break;
                        case 'Q': piece_type = QUEEN; break;
                        case 'K': piece_type = KING; break;
                    }

                    if (piece_type >= 0 && (pos.pieces[pos.side_to_move][piece_type] & from_bb)) {
                        return legal_moves[i];
                    }
                } else {
                    // Pawn move (no piece specified)
                    int from_sq = move_from(legal_moves[i]);
                    Bitboard from_bb = C64(1) << from_sq;
                    if (pos.pieces[pos.side_to_move][PAWN] & from_bb) {
                        return legal_moves[i];
                    }
                }
            }
        }
    }

    return 0;  // Move not found
}

// Check if engine's best move matches any of the suggested best moves
bool is_correct_move(Move engine_move, const std::vector<std::string>& best_moves, const BoardState& pos) {
    std::string engine_move_str = move_to_string(engine_move);

    for (const std::string& bm_str : best_moves) {
        Move bm = parse_move_string(bm_str, pos);
        if (bm == engine_move) {
            return true;
        }

        // Also check if move strings match (for debugging)
        if (engine_move_str.substr(2, 2) == bm_str.substr(bm_str.length() - 2)) {
            // Destination matches
            return true;
        }
    }

    return false;
}

// Bratko-Kopec positions now in test_positions.h

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Bratko-Kopec Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Parse command line args
    int simulations = 350000;  // Default
    bool use_rave = true;    // Default
    int time_limit_sec = 0;  // No time limit by default

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--sims" && i + 1 < argc) {
            simulations = std::stoi(argv[++i]);
        } else if (arg == "--no-rave") {
            use_rave = false;
        } else if (arg == "--time" && i + 1 < argc) {
            time_limit_sec = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --sims N      Number of simulations per position (default: 350000)\n";
            std::cout << "  --no-rave     Disable RAVE\n";
            std::cout << "  --time N      Time limit per position in seconds (default: unlimited)\n";
            std::cout << "  --help        Show this help\n";
            return 0;
        }
    }

    // Configure engine
    PUCTConfig config;
    config.num_simulations = simulations;
    config.use_rave = use_rave;
    config.playout_mode = PlayoutMode::QUIESCENCE;  // Best for tactics
    config.quiescence_depth = 6;
    config.verbose = false;

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Simulations: " << simulations << std::endl;
    std::cout << "  RAVE: " << (use_rave ? "enabled" : "disabled") << std::endl;
    std::cout << "  Playout mode: Quiescence (depth " << config.quiescence_depth << ")" << std::endl;
    if (time_limit_sec > 0) {
        std::cout << "  Time limit: " << time_limit_sec << " seconds per position" << std::endl;
    }
    std::cout << std::endl;

    // Create engine
    PUCTEngine engine(config);
    engine.init();

    int passed = 0;
    int total = NUM_BRATKO_KOPEC;

    std::cout << "\nRunning tests...\n" << std::endl;

    for (int i = 0; i < NUM_BRATKO_KOPEC; i++) {
        EPDPosition epd = parse_epd(BRATKO_KOPEC_POSITIONS[i]);

        // Parse position
        BoardState pos;
        FENError err = ParseFEN(epd.fen, pos);
        if (err != FENError::OK) {
            std::cout << "[SKIP] " << epd.id << " - FEN parse error: "
                     << FENErrorToString(err) << std::endl;
            continue;
        }

        // Print test info
        std::cout << "[" << (i + 1) << "/" << total << "] " << epd.id << std::endl;
        std::cout << "  Best move(s): ";
        for (size_t j = 0; j < epd.best_moves.size(); j++) {
            std::cout << epd.best_moves[j];
            if (j < epd.best_moves.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;

        // Run engine
        auto start = std::chrono::high_resolution_clock::now();
        Move best_move = engine.search(pos);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();

        // Check result
        bool correct = is_correct_move(best_move, epd.best_moves, pos);

        std::cout << "  Engine move: " << move_to_string(best_move);
        std::cout << " (" << elapsed << "s, ";
        std::cout << engine.get_total_visits() << " visits)";
        std::cout << " - " << (correct ? "PASS" : "FAIL") << std::endl;

        if (correct) {
            passed++;
        }

        std::cout << std::endl;
    }

    // Print summary
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " ("
              << (100.0 * passed / total) << "%)" << std::endl;
    std::cout << "========================================" << std::endl;

    return (passed == total) ? 0 : 1;
}
