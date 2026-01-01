#ifndef MONTE_CARLO_GPU_HPP
#define MONTE_CARLO_GPU_HPP

#include <cstdint>
#include <vector>
#include <string>

// Forward declarations matching CUDA kernel structures
struct GPUBoard {
    uint8_t squares[64];
    uint8_t side_to_move;
    uint8_t castling_rights;
    uint8_t en_passant_file;
    uint16_t halfmove_clock;
    uint16_t fullmove_number;
};

enum GameResult {
    ONGOING = 0,
    WHITE_WIN = 1,
    BLACK_WIN = 2,
    DRAW = 3
};

// Monte Carlo results
struct MCTSResults {
    int white_wins;
    int black_wins;
    int draws;
    int total_simulations;
    
    double white_win_rate() const {
        return total_simulations > 0 ? (double)white_wins / total_simulations : 0.0;
    }
    
    double draw_rate() const {
        return total_simulations > 0 ? (double)draws / total_simulations : 0.0;
    }
    
    double black_win_rate() const {
        return total_simulations > 0 ? (double)black_wins / total_simulations : 0.0;
    }
};

class MonteCarloGPU {
public:
    MonteCarloGPU();
    ~MonteCarloGPU();
    
    // Run Monte Carlo simulations from a given board position
    MCTSResults run_simulations(const GPUBoard& board, int num_simulations);
    
    // Helper to convert chess library board to GPU board
    static GPUBoard convert_to_gpu_board(const std::string& fen);
    
    // Print results
    static void print_results(const MCTSResults& results);
    
private:
    // CUDA device pointers
    GPUBoard* d_board_;
    int* d_results_;
    
    int max_simulations_;
    
    void allocate_device_memory(int num_simulations);
    void free_device_memory();
};

#endif // MONTE_CARLO_GPU_HPP
