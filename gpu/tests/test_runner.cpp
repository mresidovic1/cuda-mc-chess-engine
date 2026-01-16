// test_runner.cpp - Main test harness for GPU PUCT MCTS Chess Engine
// Tests FEN parsing, move generation, and tactical solver

#include "../include/chess_types.cuh"
#include "../include/fen.h"
#include "../include/cpu_movegen.h"
#include "test_positions.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <cstring>

// External functions
extern void init_attack_tables();
extern void init_startpos(BoardState* pos);

// Tactical solver
extern "C" void launch_tactical_solver(
    const BoardState* d_positions,
    Move* d_best_moves,
    int* d_scores,
    int numPositions,
    int depth,
    cudaStream_t stream
);

// ============================================================================
// Test Result Tracking
// ============================================================================

struct TestStats {
    int total;
    int passed;
    int failed;
    double total_time_ms;

    TestStats() : total(0), passed(0), failed(0), total_time_ms(0) {}

    void add_pass(double time_ms) {
        total++;
        passed++;
        total_time_ms += time_ms;
    }

    void add_fail(double time_ms) {
        total++;
        failed++;
        total_time_ms += time_ms;
    }

    double pass_rate() const {
        return total > 0 ? (100.0 * passed / total) : 0;
    }
};

// ============================================================================
// Move Notation Helpers
// ============================================================================

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

std::string move_to_uci(Move m) {
    if (m == 0) return "(none)";

    Square from = move_from(m);
    Square to = move_to(m);
    uint8_t flags = move_flags(m);

    std::string result = std::string(SQUARE_NAMES[from]) + SQUARE_NAMES[to];

    if (flags >= MOVE_PROMO_N) {
        result += PROMO_CHARS[flags & 0x3];
    }

    return result;
}

Move uci_to_move(const std::string& uci, const BoardState& pos) {
    if (uci.length() < 4) return 0;

    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int to_file = uci[2] - 'a';
    int to_rank = uci[3] - '1';

    if (from_file < 0 || from_file > 7 || from_rank < 0 || from_rank > 7) return 0;
    if (to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) return 0;

    Square from = from_rank * 8 + from_file;
    Square to = to_rank * 8 + to_file;

    // Generate all legal moves and find the matching one
    Move moves[MAX_MOVES];
    int num_moves = cpu_movegen::generate_legal_moves_cpu(&pos, moves);

    for (int i = 0; i < num_moves; i++) {
        if (move_from(moves[i]) == from && move_to(moves[i]) == to) {
            // Check promotion if specified
            if (uci.length() > 4) {
                uint8_t flags = move_flags(moves[i]);
                if (flags >= MOVE_PROMO_N) {
                    char promo = uci[4];
                    int expected_promo = -1;
                    switch (promo) {
                        case 'n': expected_promo = 0; break;
                        case 'b': expected_promo = 1; break;
                        case 'r': expected_promo = 2; break;
                        case 'q': expected_promo = 3; break;
                    }
                    if (expected_promo >= 0 && (flags & 0x3) == expected_promo) {
                        return moves[i];
                    }
                    continue;
                }
            }
            return moves[i];
        }
    }

    return 0;
}

// ============================================================================
// FEN Parser Tests
// ============================================================================

TestStats run_fen_tests() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "FEN PARSER TESTS\n";
    std::cout << std::string(60, '=') << "\n\n";

    TestStats stats;

    for (int i = 0; i < NUM_FEN_TESTS; i++) {
        const auto& test = FEN_TESTS[i];

        auto start = std::chrono::high_resolution_clock::now();

        BoardState board;
        FENError err = ParseFEN(test.fen, board);
        bool is_valid = (err == FENError::OK);

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        bool passed = (is_valid == test.should_be_valid);

        std::cout << (passed ? "[PASS]" : "[FAIL]") << " ";
        std::cout << test.description << "\n";
        std::cout << "       FEN: " << test.fen << "\n";
        std::cout << "       Expected: " << (test.should_be_valid ? "valid" : "invalid");
        std::cout << ", Got: " << (is_valid ? "valid" : "invalid");
        if (!is_valid && !test.should_be_valid) {
            std::cout << " (" << FENErrorToString(err) << ")";
        }
        std::cout << "\n\n";

        if (passed) {
            stats.add_pass(time_ms);
        } else {
            stats.add_fail(time_ms);
        }
    }

    // Test round-trip (FEN -> Board -> FEN)
    std::cout << "Round-trip test:\n";
    const char* test_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    BoardState board;
    ParseFEN(test_fen, board);
    std::string result_fen = BoardToFEN(board);
    std::cout << "  Input:  " << test_fen << "\n";
    std::cout << "  Output: " << result_fen << "\n";

    // Parse result and compare
    BoardState board2;
    ParseFEN(result_fen, board2);
    bool round_trip_ok = (memcmp(&board, &board2, sizeof(BoardState)) == 0);
    std::cout << "  Round-trip: " << (round_trip_ok ? "[PASS]" : "[FAIL]") << "\n\n";

    if (round_trip_ok) stats.add_pass(0);
    else stats.add_fail(0);

    return stats;
}

// ============================================================================
// Perft Tests (Move Generation Validation)
// ============================================================================

unsigned long long perft(const BoardState& pos, int depth) {
    if (depth == 0) return 1;

    Move moves[MAX_MOVES];
    int num_moves = cpu_movegen::generate_legal_moves_cpu(&pos, moves);

    if (depth == 1) return num_moves;

    unsigned long long nodes = 0;
    for (int i = 0; i < num_moves; i++) {
        BoardState copy = pos;
        cpu_movegen::make_move_cpu(&copy, moves[i]);
        nodes += perft(copy, depth - 1);
    }

    return nodes;
}

TestStats run_perft_tests() {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "PERFT TESTS (Move Generation Validation)\n";
    std::cout << std::string(60, '=') << "\n\n";

    TestStats stats;

    for (int i = 0; i < NUM_PERFT_TESTS; i++) {
        const auto& test = PERFT_TESTS[i];

        BoardState board;
        if (ParseFEN(test.fen, board) != FENError::OK) {
            std::cout << "[ERROR] Failed to parse FEN: " << test.fen << "\n";
            stats.add_fail(0);
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();
        unsigned long long nodes = perft(board, test.depth);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        bool passed = (nodes == test.expected_nodes);

        std::cout << (passed ? "[PASS]" : "[FAIL]") << " ";
        std::cout << test.name << "\n";
        std::cout << "       Depth: " << test.depth;
        std::cout << ", Expected: " << test.expected_nodes;
        std::cout << ", Got: " << nodes;
        std::cout << " (" << std::fixed << std::setprecision(2) << time_ms << " ms)\n\n";

        if (passed) {
            stats.add_pass(time_ms);
        } else {
            stats.add_fail(time_ms);
        }
    }

    return stats;
}

// ============================================================================
// Tactical Tests
// ============================================================================

// Run tactical tests using the GPU negamax solver
TestStats run_tactical_tests(const TestPosition* tests, int num_tests,
                              const std::string& category, int depth = 2) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TACTICAL TESTS (GPU Negamax) - " << category << "\n";
    std::cout << std::string(60, '=') << "\n\n";

    TestStats stats;

    // Allocate GPU memory
    BoardState* d_positions;
    Move* d_best_moves;
    int* d_scores;

    cudaMalloc(&d_positions, sizeof(BoardState));
    cudaMalloc(&d_best_moves, sizeof(Move));
    cudaMalloc(&d_scores, sizeof(int));

    for (int i = 0; i < num_tests; i++) {
        const auto& test = tests[i];

        // Clear any previous CUDA errors
        cudaGetLastError();

        std::cout << "Test " << (i + 1) << "/" << num_tests << ": " << test.name << "\n";
        std::cout << "FEN: " << test.fen << "\n";

        BoardState board;
        if (ParseFEN(test.fen, board) != FENError::OK) {
            std::cout << "[ERROR] Failed to parse FEN\n\n";
            stats.add_fail(0);
            continue;
        }

        // Copy position to GPU
        cudaError_t copyErr = cudaMemcpy(d_positions, &board, sizeof(BoardState), cudaMemcpyHostToDevice);
        if (copyErr != cudaSuccess) {
            std::cout << "[ERROR] CUDA memcpy error: " << cudaGetErrorString(copyErr) << "\n\n";
            stats.add_fail(0);
            continue;
        }

        // Run tactical solver
        auto start = std::chrono::high_resolution_clock::now();
        launch_tactical_solver(d_positions, d_best_moves, d_scores, 1, depth, 0);
        cudaError_t err = cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (err != cudaSuccess) {
            std::cout << "[ERROR] CUDA kernel error: " << cudaGetErrorString(err) << "\n\n";
            stats.add_fail(time_ms);
            continue;
        }

        // Get results
        Move best_move;
        int score;
        cudaMemcpy(&best_move, d_best_moves, sizeof(Move), cudaMemcpyDeviceToHost);
        cudaMemcpy(&score, d_scores, sizeof(int), cudaMemcpyDeviceToHost);

        std::string engine_move = move_to_uci(best_move);
        bool passed = (engine_move == test.expected_move);

        std::cout << "Expected: " << test.expected_move << "\n";
        std::cout << "Got:      " << engine_move;
        std::cout << " (score: " << score;
        std::cout << ", depth: " << depth;
        std::cout << ", " << std::setprecision(0) << time_ms << " ms)\n";
        std::cout << "Result:   " << (passed ? "[PASS]" : "[FAIL]") << "\n\n";

        if (passed) {
            stats.add_pass(time_ms);
        } else {
            stats.add_fail(time_ms);
        }
    }

    // Free GPU memory
    cudaFree(d_positions);
    cudaFree(d_best_moves);
    cudaFree(d_scores);

    return stats;
}

// ============================================================================
// Print Summary
// ============================================================================

void print_summary(const std::string& name, const TestStats& stats) {
    std::cout << std::left << std::setw(25) << name;
    std::cout << " " << std::right << std::setw(3) << stats.passed << "/" << stats.total;
    std::cout << " (" << std::fixed << std::setprecision(1) << stats.pass_rate() << "%)";
    std::cout << "  [" << std::setprecision(0) << stats.total_time_ms << " ms]\n";
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    std::cout << "GPU MCTS Chess Engine Test Runner\n\n";
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --all       Run all tests (default)\n";
    std::cout << "  --fen       Run FEN parser tests only\n";
    std::cout << "  --perft     Run perft tests only\n";
    std::cout << "  --easy      Run easy engine tests only\n";
    std::cout << "  --medium    Run medium engine tests only\n";
    std::cout << "  --hard      Run hard engine tests only\n";
    std::cout << "  --engine    Run all engine tests\n";
    std::cout << "  --help      Show this help\n";
}

int main(int argc, char** argv) {
    bool run_fen = false;
    bool run_perft = false;
    bool run_easy = false;
    bool run_medium = false;
    bool run_hard = false;
    bool run_all = true;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--fen") {
            run_fen = true; run_all = false;
        } else if (arg == "--perft") {
            run_perft = true; run_all = false;
        } else if (arg == "--easy") {
            run_easy = true; run_all = false;
        } else if (arg == "--medium") {
            run_medium = true; run_all = false;
        } else if (arg == "--hard") {
            run_hard = true; run_all = false;
        } else if (arg == "--engine") {
            run_easy = run_medium = run_hard = true; run_all = false;
        } else if (arg == "--all") {
            run_all = true;
        }
    }

    if (run_all) {
        run_fen = run_perft = run_easy = run_medium = run_hard = true;
    }

    std::cout << std::string(60, '=') << "\n";
    std::cout << "GPU TACTICAL CHESS ENGINE TEST SUITE\n";
    std::cout << std::string(60, '=') << "\n";

    // Initialize attack tables for CPU movegen
    std::cout << "\nInitializing attack tables...\n";
    init_attack_tables();
    std::cout << "Ready.\n";

    TestStats fen_stats, perft_stats, easy_stats, medium_stats, hard_stats;

    // Run tests
    if (run_fen) {
        fen_stats = run_fen_tests();
    }

    if (run_perft) {
        perft_stats = run_perft_tests();
    }

    if (run_easy) {
        // Depth 2 for mate-in-1/2 (proven stable)
        easy_stats = run_tactical_tests(EASY_TESTS, NUM_EASY_TESTS, "EASY", 2);
    }

    if (run_medium) {
        // Depth 4 for mate-in-4 (iterative solver, no crashes)
        medium_stats = run_tactical_tests(MEDIUM_TESTS, NUM_MEDIUM_TESTS, "MEDIUM", 4);
    }

    if (run_hard) {
        // Depth 6 for mate-in-5/6 using tactical solver (faster than MCTS)
        hard_stats = run_tactical_tests(HARD_TESTS, NUM_HARD_TESTS, "HARD", 6);
    }

    // Print summary
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST SUMMARY\n";
    std::cout << std::string(60, '=') << "\n\n";

    TestStats overall;

    if (run_fen) {
        print_summary("FEN Parser Tests", fen_stats);
        overall.total += fen_stats.total;
        overall.passed += fen_stats.passed;
        overall.failed += fen_stats.failed;
        overall.total_time_ms += fen_stats.total_time_ms;
    }

    if (run_perft) {
        print_summary("Perft Tests", perft_stats);
        overall.total += perft_stats.total;
        overall.passed += perft_stats.passed;
        overall.failed += perft_stats.failed;
        overall.total_time_ms += perft_stats.total_time_ms;
    }

    if (run_easy) {
        print_summary("Easy Tactical Tests", easy_stats);
        overall.total += easy_stats.total;
        overall.passed += easy_stats.passed;
        overall.failed += easy_stats.failed;
        overall.total_time_ms += easy_stats.total_time_ms;
    }

    if (run_medium) {
        print_summary("Medium Tactical Tests", medium_stats);
        overall.total += medium_stats.total;
        overall.passed += medium_stats.passed;
        overall.failed += medium_stats.failed;
        overall.total_time_ms += medium_stats.total_time_ms;
    }

    if (run_hard) {
        print_summary("Hard Tactical Tests", hard_stats);
        overall.total += hard_stats.total;
        overall.passed += hard_stats.passed;
        overall.failed += hard_stats.failed;
        overall.total_time_ms += hard_stats.total_time_ms;
    }

    std::cout << std::string(50, '-') << "\n";
    print_summary("OVERALL", overall);

    std::cout << "\n";

    // Return exit code based on test results
    return (overall.failed > 0) ? 1 : 0;
}
