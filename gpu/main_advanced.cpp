#include "monte_carlo_advanced.hpp"
#include "../include/chess.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

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
    std::cout << "=======================\n\n";
}

void test_position(const std::string& fen, const std::string& description, int simulations = 50000) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Testing: " << description << "\n";
    std::cout << "FEN: " << fen << "\n";
    std::cout << std::string(70, '=') << "\n";
    
    chess::Board board(fen);
    std::cout << board << "\n";
    
    auto best_move = monte_carlo_advanced::find_best_move(
        board,
        simulations,  // simulations per move
        256,          // threads per move (GPU threads)
        true          // verbose
    );
    
    if (best_move != chess::Move::NO_MOVE) {
        std::cout << "Recommended move: " << chess::uci::moveToUci(best_move) << "\n";
        
        // Make move and show resulting position
        board.makeMove(best_move);
        std::cout << "\nPosition after move:\n";
        std::cout << board << "\n";
    }
}

int main(int argc, char* argv[]) {
    // Initialize CUDA
    print_gpu_info();
    
    // Default parameters
    int simulations = 50000;
    std::string custom_fen = "";
    
    // Parse command line arguments
    if (argc > 1) {
        simulations = std::atoi(argv[1]);
    }
    if (argc > 2) {
        custom_fen = argv[2];
    }
    
    if (!custom_fen.empty()) {
        // Test custom position
        test_position(custom_fen, "Custom Position", simulations);
    } else {
        // Test suite of positions
        
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║  Advanced Monte Carlo Chess Engine - GPU Version            ║\n";
        std::cout << "║  With Heuristic-Guided Playouts and Position Evaluation     ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        // Test 1: Starting position
        test_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Starting Position",
            simulations
        );
        
        // Test 2: Tactical position (Scholar's Mate threat)
        test_position(
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "Scholar's Mate Threat",
            simulations
        );
        
        // Test 3: Endgame (King and Pawn vs King)
        test_position(
            "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
            "King and Pawn Endgame",
            simulations
        );
        
        // Test 4: Mate in 2
        test_position(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
            "Mate in 2 for White",
            simulations
        );
        
        // Test 5: Complex middlegame
        test_position(
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
            "Complex Middlegame",
            simulations
        );
    }
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "All tests completed!\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    return 0;
}
