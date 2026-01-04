//N - v1: Test driver for improved Monte Carlo engine
#include "monte_carlo_advanced_v1.hpp"
#include "../include/chess.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>

void print_gpu_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "ERROR: No CUDA-capable devices found!\n";
        exit(1);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "=== GPU Information ===\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Total global memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "=======================\n\n";
}

//N - v1: Compare evaluation modes
void benchmark_modes(const std::string& fen, int simulations) {
    std::cout << "\n=== Benchmarking Evaluation Modes ===\n";
    std::cout << "FEN: " << fen << "\n";
    std::cout << "Simulations per move: " << simulations << "\n\n";

    chess::Board board(fen);

    using monte_carlo_advanced_v1::EvaluationMode;
    using monte_carlo_advanced_v1::find_best_move;

    struct ModeInfo {
        EvaluationMode mode;
        const char* name;
    };

    ModeInfo modes[] = {
        {EvaluationMode::LEGACY, "Legacy (sequential)"},
        {EvaluationMode::BATCHED, "Batched (parallel)"},
        {EvaluationMode::STREAMS, "Streams (async)"}
    };

    for (const auto& mode_info : modes) {
        std::cout << "Testing: " << mode_info.name << "\n";

        auto start = std::chrono::high_resolution_clock::now();
        auto best_move = find_best_move(board, simulations, 256, false, mode_info.mode);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Best move: " << chess::uci::moveToUci(best_move) << "\n";
        std::cout << "  Time: " << duration.count() << " ms\n\n";
    }
}

void test_position(const std::string& fen, const std::string& description,
                   int simulations = 50000,
                   monte_carlo_advanced_v1::EvaluationMode mode = monte_carlo_advanced_v1::EvaluationMode::BATCHED) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Testing: " << description << "\n";
    std::cout << "FEN: " << fen << "\n";
    std::cout << std::string(70, '=') << "\n";

    chess::Board board(fen);
    std::cout << board << "\n";

    auto best_move = monte_carlo_advanced_v1::find_best_move(
        board,
        simulations,
        256,
        true,
        mode
    );

    if (best_move != chess::Move::NO_MOVE) {
        std::cout << "Recommended move: " << chess::uci::moveToUci(best_move) << "\n";

        board.makeMove(best_move);
        std::cout << "\nPosition after move:\n";
        std::cout << board << "\n";
    }
}

int main(int argc, char* argv[]) {
    print_gpu_info();

    int simulations = 50000;
    std::string custom_fen = "";
    bool benchmark = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--benchmark" || arg == "-b") {
            benchmark = true;
        } else if (arg == "--sims" || arg == "-s") {
            if (i + 1 < argc) {
                simulations = std::atoi(argv[++i]);
            }
        } else if (arg == "--fen" || arg == "-f") {
            if (i + 1 < argc) {
                custom_fen = argv[++i];
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  -s, --sims N     Set simulations per move (default: 50000)\n";
            std::cout << "  -f, --fen FEN    Test custom FEN position\n";
            std::cout << "  -b, --benchmark  Run mode comparison benchmark\n";
            std::cout << "  -h, --help       Show this help\n";
            return 0;
        } else {
            // Try parsing as number (legacy behavior)
            int val = std::atoi(arg.c_str());
            if (val > 0) simulations = val;
        }
    }

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Advanced Monte Carlo Chess Engine v1.0                      ║\n";
    std::cout << "║  With MCTS, Transposition Table, History Heuristic           ║\n";
    std::cout << "║  P1 & P2 Improvements Implemented                            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    if (benchmark) {
        benchmark_modes(
            "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
            simulations
        );
    } else if (!custom_fen.empty()) {
        test_position(custom_fen, "Custom Position", simulations);
    } else {
        // Default test suite
        using monte_carlo_advanced_v1::EvaluationMode;

        // Test 1: Starting position
        test_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Starting Position",
            simulations,
            EvaluationMode::BATCHED
        );

        // Clear caches between tests
        monte_carlo_advanced_v1::new_game();

        // Test 2: Tactical position
        test_position(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "Scholar's Mate Threat",
            simulations,
            EvaluationMode::BATCHED
        );

        monte_carlo_advanced_v1::new_game();

        // Test 3: Endgame
        test_position(
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            "King and Pawn Endgame",
            simulations,
            EvaluationMode::BATCHED
        );

        monte_carlo_advanced_v1::new_game();

        // Test 4: Mate in 2
        test_position(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
            "Mate Threat Position",
            simulations,
            EvaluationMode::BATCHED
        );

        monte_carlo_advanced_v1::new_game();

        // Test 5: Complex middlegame
        test_position(
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
            "Complex Middlegame",
            simulations,
            EvaluationMode::BATCHED
        );
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "All tests completed!\n";
    std::cout << std::string(70, '=') << "\n\n";

    // Cleanup
    monte_carlo_advanced_v1::shutdown();

    return 0;
}
