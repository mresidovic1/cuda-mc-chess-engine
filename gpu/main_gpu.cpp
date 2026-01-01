#include "monte_carlo_gpu.hpp"
#include <iostream>
#include <chrono>
#include <string>

void test_position(const std::string& fen, int num_simulations, const std::string& description) {
    std::cout << "\n=== Testing Position: " << description << " ===" << std::endl;
    std::cout << "FEN: " << fen << std::endl;
    std::cout << "Number of simulations: " << num_simulations << std::endl;
    
    GPUBoard board = MonteCarloGPU::convert_to_gpu_board(fen);
    
    MonteCarloGPU mc_gpu;
    
    auto start = std::chrono::high_resolution_clock::now();
    MCTSResults results = mc_gpu.run_simulations(board, num_simulations);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    MonteCarloGPU::print_results(results);
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    std::cout << "Simulations per second: " 
              << (num_simulations * 1000.0 / duration.count()) << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Monte Carlo Chess Engine - GPU Version ===" << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA-capable device found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using CUDA device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    
    // Default number of simulations
    int num_simulations = 10000;
    
    // Parse command line arguments
    if (argc > 1) {
        num_simulations = std::stoi(argv[1]);
    }
    
    // Test various positions
    
    // 1. Starting position
    test_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        num_simulations,
        "Starting Position"
    );
    
    // 2. Mid-game position
    test_position(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        num_simulations,
        "Italian Game"
    );
    
    // 3. Endgame position
    test_position(
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        num_simulations,
        "King and Pawn Endgame"
    );
    
    // 4. Complex mid-game
    test_position(
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
        num_simulations,
        "Complex Mid-game"
    );
    
    // 5. User-provided FEN (if available)
    if (argc > 2) {
        std::string fen = argv[2];
        test_position(fen, num_simulations, "Custom Position");
    }
    
    std::cout << "\n=== All tests completed ===" << std::endl;
    
    return 0;
}
