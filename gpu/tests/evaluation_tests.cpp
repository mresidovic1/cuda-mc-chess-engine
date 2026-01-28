#include "../include/chess_types.cuh"
#include "../include/fen.h"
#include "../include/evaluation.h"
#include "../include/cpu_movegen.h"
#include "../include/puct_mcts.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

extern void init_attack_tables();

// Test Position Structure

struct EvalTestPosition {
    std::string fen;
    std::string best_move;
    std::string id;
};

// Test Positions (2830 BM Problem Set)

const EvalTestPosition EVAL_TESTS[] = {
    { "4r1k1/p1pb1ppp/Qbp1r3/8/1P6/2Pq1B2/R2P1PPP/2B2RK1 b - - 0", "d3f3", "BS2830-01" },
    { "7r/2qpkp2/p3p3/6P1/1p2b2r/7P/PPP2QP1/R2N1RK1 b - - 0", "f7f5", "BS2830-02" },
    { "r1bq1rk1/pp4bp/2np4/2p1p1p1/P1N1P3/1P1P1NP1/1BP1QPKP/1R3R2 b - - 0", "c8h3", "BS2830-03" },
    { "8/2kPR3/5q2/5N2/8/1p1P4/1p6/1K6 w - - 0", "f5d4", "BS2830-04" },
    { "2r1r3/p3bk1p/1pnqpppB/3n4/3P2Q1/PB3N2/1P3PPP/3RR1K1 w - - 0", "g4e6", "BS2830-05" },
    { "8/2p5/7p/pP2k1pP/5pP1/8/1P2PPK1/8 w - - 0", "f2f3", "BS2830-06" },
    { "8/5p1p/1p2pPk1/p1p1P3/P1P1K2b/4B3/1P5P/8 w - - 0", "b2b4", "BS2830-07" },
    { "rn2r1k1/pp3ppp/8/1qNp4/3BnQb1/5N2/PPP2PPP/2KR3R b - - 0", "d7h3", "BS2830-08" },
    { "r3kb1r/1p1b1p2/p1nppp2/7p/4PP2/qNN5/P1PQB1PP/R4R1K w kq - 0", "b3a1", "BS2830-09" },
    { "r3r1k1/pp1bp2p/1n2q1P1/6b1/1B2B3/5Q2/5PPP/1R3RK1 w - - 0", "c2d2", "BS2830-10" },
    { "r3k2r/pb3pp1/2p1qnnp/1pp1P3/Q1N4B/2PB1P2/P5PP/R4RK1 w kq - 0", "e5f6", "BS2830-11" },
    { "r1b1r1k1/ppp2ppp/2nb1q2/8/2B5/1P1Q1N2/P1PP1PPP/R1B2RK1 w - - 0", "c1b2", "BS2830-12" },
    { "rnb1kb1r/1p3ppp/p5q1/4p3/3N4/4BB2/PPPQ1P1P/R3K2R w KQkq - 0", "e1c1", "BS2830-13" },
    { "r1bqr1k1/pp1n1ppp/5b2/4N1B1/3p3P/8/PPPQ1PP1/2K1RB1R w - - 0", "g5f7", "BS2830-14" },
    { "2r2rk1/1bpR1p2/1pq1pQp1/p3P2p/P1PR3P/5N2/2P2PPK/8 w - - 0", "h2g3", "BS2830-15" },
    { "8/pR4pk/1b6/2p5/N1p5/8/PP1r2PP/6K1 b - - 0", "b4b2", "BS2830-16" },
    { "r1b1qrk1/ppBnppb1/2n4p/1NN1P1p1/3p4/8/PPP1BPPP/R2Q1R1K w - - 0", "c5e6", "BS2830-17" },
    { "8/8/4b1p1/2Bp3p/5P1P/1pK1Pk2/8/8 b - - 0", "g7g5", "BS2830-18" },
    { "r3k2r/pp1n1ppp/1qpnp3/3bN1PP/3P2Q1/2B1R3/PPP2P2/2KR1B2 w kq - 0", "c2e1", "BS2830-19" },
    { "r1bqk2r/pppp1Npp/8/2bnP3/8/6K1/PB4PP/RN1Q3R b kq - 0", "e8g8", "BS2830-20" },
    { "r4r1k/pbnq1ppp/np3b2/3p1N2/5B2/2N3PB/PP3P1P/R2QR1K1 w - - 0", "f3e4", "BS2830-21" },
    { "r2qr2k/pbp3pp/1p2Bb2/2p5/2P2P2/3R2P1/PP2Q1NP/5RK1 b - - 0", "d6d3", "BS2830-22" },
    { "5r2/1p4r1/3kp1b1/1Pp1p2p/2PpP3/q2B1PP1/3Q2K1/1R5R b - - 0", "f4f3", "BS2830-23" },
    { "8/7p/8/7P/1p6/1p5P/1P2Q1pk/1K6 w - - 0", "c1c2", "BS2830-24" },
    { "r5k1/p4n1p/6p1/2qPp3/2p1P1Q1/8/1rB3PP/R4R1K b - - 0", "f8f4", "BS2830-25" },
    { "1r4k1/1q2pN1p/3pPnp1/8/2pQ4/P5PP/5P2/3R2K1 b - - 0", "d6d5", "BS2830-26" },
    { "2rq1rk1/pb3ppp/1p2pn2/4N3/1b1PPB2/4R1P1/P4PB", "e8e1", "BS2830-27" }
};

const int NUM_EVAL_TESTS = sizeof(EVAL_TESTS) / sizeof(EvalTestPosition);

// Move Notation Helpers

const char* SQUARE_NAMES[64] = {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "f8", "g8", "h8"
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


struct TestResult {
    std::string id;
    std::string engine_move;
    std::string expected_move;
    int visits;
    float q_value;
    double time_ms;
    bool passed;
};

int run_mcts_tests(int time_limit_seconds) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "MCTS EVALUATION TESTS - " << time_limit_seconds << " seconds per position\n";
    std::cout << std::string(70, '=') << "\n\n";

    // Initialize PUCT engine with time-based search
    PUCTConfig config = PUCTConfig::Advanced();  // All Phase 1+2+3 features
    config.use_time_limit = true;
    config.time_limit_ms = time_limit_seconds * 1000;
    config.verbose = false;  // Set to true for detailed progress
    config.playout_mode = PlayoutMode::STATIC_EVAL;  // Use simpler eval mode to test

    PUCTEngine engine(config);
    engine.init();

    std::vector<TestResult> results;

    int total = NUM_EVAL_TESTS;
    int time_limit_ms = time_limit_seconds * 1000;

    for (int i = 0; i < total; i++) {
        const auto& test = EVAL_TESTS[i];

        std::cout << "Test " << std::setw(2) << (i + 1) << "/" << total << ": " << test.id << "\n";
        std::cout << "FEN: " << test.fen << "\n";
        std::cout << "Expected: " << test.best_move << "\n";

        BoardState board;
        if (ParseFEN(test.fen, board) != FENError::OK) {
            std::cout << "[ERROR] Failed to parse FEN\n\n";
            continue;
        }

        // Run MCTS search with time limit
        auto start = std::chrono::high_resolution_clock::now();
        Move best_move = engine.search(board);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Get move info
        std::string engine_uci = move_to_uci(best_move);

        // Get root value
        float root_value = engine.get_root_value();
        int total_visits = engine.get_total_visits();

        std::cout << "Got:      " << engine_uci;
        std::cout << " (visits: " << std::setw(6) << total_visits;
        std::cout << ", Q: " << std::fixed << std::setprecision(3) << root_value;
        std::cout << ", time: " << std::setprecision(1) << time_ms << " ms)\n";

        bool passed = (engine_uci == test.best_move);
        std::cout << "Result:   " << (passed ? "[PASS]" : "[FAIL]") << "\n\n";

        TestResult result;
        result.id = test.id;
        result.engine_move = engine_uci;
        result.expected_move = test.best_move;
        result.visits = total_visits;
        result.q_value = root_value;
        result.time_ms = time_ms;
        result.passed = passed;
        results.push_back(result);
    }

    // Print summary
    std::cout << std::string(70, '=') << "\n";
    std::cout << "RESULTS SUMMARY\n";
    std::cout << std::string(70, '=') << "\n\n";

    int passed = 0;
    double total_time = 0;
    int total_visits = 0;

    for (const auto& r : results) {
        if (r.passed) passed++;
        total_time += r.time_ms;
        total_visits += r.visits;
    }

    std::cout << "Total tests: " << results.size() << "\n";
    std::cout << "Passed:      " << passed << " (" << std::fixed << std::setprecision(1)
              << (100.0 * passed / results.size()) << "%)\n";
    std::cout << "Failed:      " << (results.size() - passed) << "\n";
    std::cout << "Total time:  " << std::setprecision(1) << total_time / 1000.0 << " seconds\n";
    std::cout << "Avg time:    " << total_time / results.size() << " ms per test\n";
    std::cout << "Total sims:  " << total_visits << "\n";
    std::cout << "Avg sims:    " << total_visits / results.size() << " per test\n";
    std::cout << "Sims/sec:    " << (int)(total_visits / (total_time / 1000.0)) << "\n\n";

    // Show failed tests
    if (passed < (int)results.size()) {
        std::cout << "Failed tests:\n";
        for (const auto& r : results) {
            if (!r.passed) {
                std::cout << "  " << r.id << ": expected " << r.expected_move
                          << ", got " << r.engine_move << "\n";
            }
        }
        std::cout << "\n";
    }

    std::cout << std::string(70, '=') << "\n\n";

    return passed;
}


void run_static_eval_tests() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "STATIC EVALUATION TESTS\n";
    std::cout << std::string(70, '=') << "\n\n";

    int total = NUM_EVAL_TESTS;

    for (int i = 0; i < total; i++) {
        const auto& test = EVAL_TESTS[i];

        BoardState board;
        if (ParseFEN(test.fen, board) != FENError::OK) {
            continue;
        }

        int score = evaluate(board);
        int phase = calculate_phase(board);

        std::cout << std::setw(12) << test.id << ": ";
        std::cout << "Eval=" << std::setw(6) << score;
        std::cout << " Phase=" << std::setw(3) << phase;

        if (board.side_to_move == WHITE) {
            std::cout << " (to move)";
        } else {
            std::cout << " (Black to move, score from White's POV)";
        }
        std::cout << "\n";
    }

    std::cout << "\n";
}


int main(int argc, char** argv) {
    std::cout << "GPU Evaluation Test Suite\n";
    std::cout << "Testing Phase 1+2+3 MCTS features\n\n";

    // Initialize attack tables
    std::cout << "Initializing attack tables...\n";
    init_attack_tables();
    std::cout << "Ready.\n\n";

    // Default: 10 seconds per test
    int time_limit_seconds = 10;
    bool run_static_only = false;

    // Check for command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--static" || arg == "-s") {
            run_static_only = true;
        } else if (arg == "--time" || arg == "-t") {
            if (i + 1 < argc) {
                time_limit_seconds = std::atoi(argv[++i]);
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: evaluation_tests.exe [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --time N, -t N   Set time limit per test (seconds, default=10)\n";
            std::cout << "  --static, -s     Run static evaluation only (no MCTS)\n";
            std::cout << "  --help, -h       Show this help\n";
            return 0;
        }
    }

    if (run_static_only) {
        run_static_eval_tests();
    } else {
        run_mcts_tests(time_limit_seconds);
    }

    return 0;
}
