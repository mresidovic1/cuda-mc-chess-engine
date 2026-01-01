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

// Simple move representation for best move selection
struct BestMove {
    std::string from_sq; // e.g., "e2"
    std::string to_sq;   // e.g., "e4"
    double score;        // Win rate from perspective of side to move
    int simulations;
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
    
    // Score from perspective of side to move (higher is better for current player)
    double score_for_side_to_move(bool white_to_move) const {
        if (white_to_move) {
            return white_win_rate() + 0.5 * draw_rate();
        } else {
            return black_win_rate() + 0.5 * draw_rate();
        }
    }
};

class MonteCarloGPU {
public:
    MonteCarloGPU();
    ~MonteCarloGPU();
    
    // Run Monte Carlo simulations from a given board position
    MCTSResults run_simulations(const GPUBoard& board, int num_simulations);
    
    // Find best move using Monte Carlo simulations
    // Returns the best move with its score
    BestMove find_best_move(const std::string& fen, int simulations_per_move);
    
    // Helper to convert chess library board to GPU board
    static GPUBoard convert_to_gpu_board(const std::string& fen);
    
    // Helper to apply a move to FEN and get new FEN
    static std::string apply_move_to_fen(const std::string& fen, int from_sq, int to_sq);
    
    // Helper to get all legal moves from a position
    static std::vector<std::pair<int, int>> get_legal_moves(const GPUBoard& board);
    
    // Convert square index to algebraic notation
    static std::string square_to_string(int sq);
    
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
