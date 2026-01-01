#include "monte_carlo_gpu.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <random>
#include <chrono>

// External kernel launch function
extern "C" void launch_monte_carlo_kernel(
    GPUBoard* d_initial_board,
    int* d_results,
    int num_simulations,
    unsigned long long seed
);

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

MonteCarloGPU::MonteCarloGPU() : d_board_(nullptr), d_results_(nullptr), max_simulations_(0) {
}

MonteCarloGPU::~MonteCarloGPU() {
    free_device_memory();
}

void MonteCarloGPU::allocate_device_memory(int num_simulations) {
    if (num_simulations <= max_simulations_) {
        return; // Already allocated enough memory
    }
    
    free_device_memory();
    
    CUDA_CHECK(cudaMalloc(&d_board_, sizeof(GPUBoard)));
    CUDA_CHECK(cudaMalloc(&d_results_, num_simulations * sizeof(int)));
    
    max_simulations_ = num_simulations;
}

void MonteCarloGPU::free_device_memory() {
    if (d_board_) {
        CUDA_CHECK(cudaFree(d_board_));
        d_board_ = nullptr;
    }
    if (d_results_) {
        CUDA_CHECK(cudaFree(d_results_));
        d_results_ = nullptr;
    }
    max_simulations_ = 0;
}

MCTSResults MonteCarloGPU::run_simulations(const GPUBoard& board, int num_simulations) {
    allocate_device_memory(num_simulations);
    
    // Copy board to device
    CUDA_CHECK(cudaMemcpy(d_board_, &board, sizeof(GPUBoard), cudaMemcpyHostToDevice));
    
    // Initialize results to 0
    CUDA_CHECK(cudaMemset(d_results_, 0, num_simulations * sizeof(int)));
    
    // Generate random seed
    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    
    // Launch kernel
    launch_monte_carlo_kernel(d_board_, d_results_, num_simulations, seed);
    
    // Copy results back to host
    std::vector<int> h_results(num_simulations);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results_, num_simulations * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Count results
    MCTSResults results = {0, 0, 0, num_simulations};
    for (int result : h_results) {
        switch (result) {
            case WHITE_WIN:
                results.white_wins++;
                break;
            case BLACK_WIN:
                results.black_wins++;
                break;
            case DRAW:
                results.draws++;
                break;
        }
    }
    
    return results;
}

GPUBoard MonteCarloGPU::convert_to_gpu_board(const std::string& fen) {
    GPUBoard board;
    std::memset(&board, 0, sizeof(GPUBoard));
    
    // Initialize empty board
    for (int i = 0; i < 64; i++) {
        board.squares[i] = 0; // EMPTY
    }
    
    // Parse FEN string
    size_t pos = 0;
    int rank = 7; // Start from rank 8 (index 7)
    int file = 0;
    
    // Parse piece placement
    while (pos < fen.length() && fen[pos] != ' ') {
        char c = fen[pos];
        
        if (c == '/') {
            rank--;
            file = 0;
        } else if (c >= '1' && c <= '8') {
            file += (c - '0'); // Skip empty squares
        } else {
            // Parse piece
            uint8_t piece_type = 0;
            uint8_t color_mask = 0;
            
            switch (std::tolower(c)) {
                case 'p': piece_type = 1; break; // PAWN
                case 'n': piece_type = 2; break; // KNIGHT
                case 'b': piece_type = 3; break; // BISHOP
                case 'r': piece_type = 4; break; // ROOK
                case 'q': piece_type = 5; break; // QUEEN
                case 'k': piece_type = 6; break; // KING
            }
            
            if (std::isupper(c)) {
                color_mask = 0x00; // WHITE
            } else {
                color_mask = 0x80; // BLACK
            }
            
            int square = rank * 8 + file;
            board.squares[square] = piece_type | color_mask;
            file++;
        }
        
        pos++;
    }
    
    // Parse side to move
    pos++; // Skip space
    if (pos < fen.length()) {
        board.side_to_move = (fen[pos] == 'w') ? 0 : 1;
        pos += 2; // Skip color and space
    }
    
    // Parse castling rights (simplified)
    board.castling_rights = 0;
    while (pos < fen.length() && fen[pos] != ' ') {
        switch (fen[pos]) {
            case 'K': board.castling_rights |= 0x01; break;
            case 'Q': board.castling_rights |= 0x02; break;
            case 'k': board.castling_rights |= 0x04; break;
            case 'q': board.castling_rights |= 0x08; break;
        }
        pos++;
    }
    
    // Parse en passant (simplified)
    pos++; // Skip space
    board.en_passant_file = 255; // No en passant
    if (pos < fen.length() && fen[pos] != '-') {
        board.en_passant_file = fen[pos] - 'a';
    }
    
    // Skip to halfmove clock
    while (pos < fen.length() && fen[pos] != ' ') pos++;
    pos++;
    
    // Parse halfmove clock
    board.halfmove_clock = 0;
    while (pos < fen.length() && fen[pos] >= '0' && fen[pos] <= '9') {
        board.halfmove_clock = board.halfmove_clock * 10 + (fen[pos] - '0');
        pos++;
    }
    
    // Parse fullmove number
    pos++; // Skip space
    board.fullmove_number = 1;
    while (pos < fen.length() && fen[pos] >= '0' && fen[pos] <= '9') {
        board.fullmove_number = board.fullmove_number * 10 + (fen[pos] - '0');
        pos++;
    }
    
    return board;
}

void MonteCarloGPU::print_results(const MCTSResults& results) {
    std::cout << "\n=== Monte Carlo Simulation Results ===" << std::endl;
    std::cout << "Total simulations: " << results.total_simulations << std::endl;
    std::cout << "White wins: " << results.white_wins 
              << " (" << (results.white_win_rate() * 100.0) << "%)" << std::endl;
    std::cout << "Black wins: " << results.black_wins 
              << " (" << (results.black_win_rate() * 100.0) << "%)" << std::endl;
    std::cout << "Draws: " << results.draws 
              << " (" << (results.draw_rate() * 100.0) << "%)" << std::endl;
    std::cout << "=======================================" << std::endl;
}
