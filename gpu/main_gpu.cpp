#include "monte_carlo_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <string>

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
    
    // Default number of simulations per move
    int simulations_per_move = 5000;
    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"; // Starting position
    
    // Parse command line arguments
    if (argc > 1) {
        simulations_per_move = std::stoi(argv[1]);
    }
    
    if (argc > 2) {
        fen = argv[2];
    }
    
    std::cout << "\n=== Finding Best Move ===" << std::endl;
    std::cout << "Position: " << fen << std::endl;
    
    MonteCarloGPU mc_gpu;
    
    auto start = std::chrono::high_resolution_clock::now();
    BestMove best = mc_gpu.find_best_move(fen, simulations_per_move);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "\n=== BEST MOVE ===" << std::endl;
    std::cout << "Move: " << best.from_sq << " -> " << best.to_sq << std::endl;
    std::cout << "Score: " << (best.score * 100.0) << "%" << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "=================" << std::endl;
    
    return 0;
}
