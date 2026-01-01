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

std::string MonteCarloGPU::square_to_string(int sq) {
    int rank = sq / 8;
    int file = sq % 8;
    std::string result = "";
    result += char('a' + file);
    result += char('1' + rank);
    return result;
}

// Simple move generation for finding best move
std::vector<std::pair<int, int>> MonteCarloGPU::get_legal_moves(const GPUBoard& board) {
    std::vector<std::pair<int, int>> moves;
    
    uint8_t my_color = board.side_to_move == 0 ? 0x00 : 0x80; // WHITE_MASK : BLACK_MASK
    int direction = (my_color == 0x00) ? 1 : -1;
    
    for (int sq = 0; sq < 64; sq++) {
        uint8_t piece = board.squares[sq];
        uint8_t piece_type = piece & 0x0F;
        uint8_t piece_color = piece & 0x80;
        
        if (piece_type == 0) continue; // Empty
        if (piece_color != my_color) continue;
        
        int rank = sq / 8;
        int file = sq % 8;
        
        // Pawn moves (simplified)
        if (piece_type == 1) { // PAWN
            int forward_rank = rank + direction;
            int forward_sq = forward_rank * 8 + file;
            
            if (forward_rank >= 0 && forward_rank < 8 && 
                (board.squares[forward_sq] & 0x0F) == 0) {
                moves.push_back({sq, forward_sq});
                
                // Double push
                int start_rank = (my_color == 0x00) ? 1 : 6;
                if (rank == start_rank) {
                    int double_sq = (rank + 2 * direction) * 8 + file;
                    if ((board.squares[double_sq] & 0x0F) == 0) {
                        moves.push_back({sq, double_sq});
                    }
                }
            }
            
            // Pawn captures
            for (int df = -1; df <= 1; df += 2) {
                if (file + df < 0 || file + df >= 8) continue;
                int capture_sq = forward_rank * 8 + (file + df);
                if (forward_rank >= 0 && forward_rank < 8) {
                    uint8_t target = board.squares[capture_sq];
                    if ((target & 0x0F) != 0 && (target & 0x80) != my_color) {
                        moves.push_back({sq, capture_sq});
                    }
                }
            }
        }
        
        // Knight moves
        else if (piece_type == 2) { // KNIGHT
            int offsets[8][2] = {{2,1}, {2,-1}, {-2,1}, {-2,-1}, {1,2}, {1,-2}, {-1,2}, {-1,-2}};
            for (int i = 0; i < 8; i++) {
                int new_rank = rank + offsets[i][0];
                int new_file = file + offsets[i][1];
                if (new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8) {
                    int to_sq = new_rank * 8 + new_file;
                    uint8_t target = board.squares[to_sq];
                    if ((target & 0x0F) == 0 || (target & 0x80) != my_color) {
                        moves.push_back({sq, to_sq});
                    }
                }
            }
        }
        
        // King moves
        else if (piece_type == 6) { // KING
            for (int dr = -1; dr <= 1; dr++) {
                for (int df = -1; df <= 1; df++) {
                    if (dr == 0 && df == 0) continue;
                    int new_rank = rank + dr;
                    int new_file = file + df;
                    if (new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8) {
                        int to_sq = new_rank * 8 + new_file;
                        uint8_t target = board.squares[to_sq];
                        if ((target & 0x0F) == 0 || (target & 0x80) != my_color) {
                            moves.push_back({sq, to_sq});
                        }
                    }
                }
            }
        }
        
        // Sliding pieces (Bishop, Rook, Queen)
        else if (piece_type >= 3 && piece_type <= 5) {
            int directions[8][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}, {1,1}, {1,-1}, {-1,1}, {-1,-1}};
            int start_dir = (piece_type == 3) ? 4 : 0; // Bishop starts at diagonals
            int end_dir = (piece_type == 4) ? 4 : 8;   // Rook ends at straights
            if (piece_type == 5) { start_dir = 0; end_dir = 8; } // Queen uses all
            
            for (int d = start_dir; d < end_dir; d++) {
                for (int dist = 1; dist <= 7; dist++) {
                    int new_rank = rank + dist * directions[d][0];
                    int new_file = file + dist * directions[d][1];
                    if (new_rank < 0 || new_rank >= 8 || new_file < 0 || new_file >= 8) break;
                    
                    int to_sq = new_rank * 8 + new_file;
                    uint8_t target = board.squares[to_sq];
                    
                    if ((target & 0x0F) != 0 && (target & 0x80) == my_color) break; // Blocked by friendly
                    
                    moves.push_back({sq, to_sq});
                    
                    if ((target & 0x0F) != 0) break; // Blocked by enemy
                }
            }
        }
    }
    
    return moves;
}

// Apply move to FEN (simplified - doesn't handle all edge cases)
std::string MonteCarloGPU::apply_move_to_fen(const std::string& fen, int from_sq, int to_sq) {
    GPUBoard board = convert_to_gpu_board(fen);
    
    // Make the move
    board.squares[to_sq] = board.squares[from_sq];
    board.squares[from_sq] = 0;
    
    // Switch side to move
    board.side_to_move = 1 - board.side_to_move;
    
    // Increment move counters (simplified)
    board.halfmove_clock++;
    if (board.side_to_move == 0) board.fullmove_number++;
    
    // Convert back to FEN (simplified version)
    std::string new_fen = "";
    
    // Build position part
    for (int rank = 7; rank >= 0; rank--) {
        int empty_count = 0;
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            uint8_t piece = board.squares[sq];
            uint8_t type = piece & 0x0F;
            uint8_t color = piece & 0x80;
            
            if (type == 0) {
                empty_count++;
            } else {
                if (empty_count > 0) {
                    new_fen += char('0' + empty_count);
                    empty_count = 0;
                }
                
                char piece_char = ' ';
                switch (type) {
                    case 1: piece_char = 'p'; break;
                    case 2: piece_char = 'n'; break;
                    case 3: piece_char = 'b'; break;
                    case 4: piece_char = 'r'; break;
                    case 5: piece_char = 'q'; break;
                    case 6: piece_char = 'k'; break;
                }
                
                if (color == 0x00) piece_char = std::toupper(piece_char);
                new_fen += piece_char;
            }
        }
        if (empty_count > 0) new_fen += char('0' + empty_count);
        if (rank > 0) new_fen += '/';
    }
    
    // Add side to move
    new_fen += (board.side_to_move == 0) ? " w " : " b ";
    
    // Add castling rights (simplified - keep original)
    size_t pos = fen.find(' ');
    pos = fen.find(' ', pos + 1);
    size_t end_pos = fen.find(' ', pos + 1);
    new_fen += fen.substr(pos + 1, end_pos - pos - 1) + " ";
    
    // Add en passant and move counters
    new_fen += "- ";
    new_fen += std::to_string(board.halfmove_clock) + " ";
    new_fen += std::to_string(board.fullmove_number);
    
    return new_fen;
}

BestMove MonteCarloGPU::find_best_move(const std::string& fen, int simulations_per_move) {
    GPUBoard board = convert_to_gpu_board(fen);
    auto legal_moves = get_legal_moves(board);
    
    if (legal_moves.empty()) {
        return {"--", "--", 0.0, 0};
    }
    
    BestMove best;
    best.score = -1.0;
    best.simulations = simulations_per_move;
    
    bool white_to_move = (board.side_to_move == 0);
    
    std::cout << "\n=== Analyzing " << legal_moves.size() << " possible moves ===" << std::endl;
    std::cout << "Simulations per move: " << simulations_per_move << std::endl;
    std::cout << "Side to move: " << (white_to_move ? "White" : "Black") << std::endl;
    
    for (size_t i = 0; i < legal_moves.size(); i++) {
        auto [from, to] = legal_moves[i];
        
        // Apply move to get new position
        std::string new_fen = apply_move_to_fen(fen, from, to);
        GPUBoard new_board = convert_to_gpu_board(new_fen);
        
        // Run simulations from this position
        MCTSResults results = run_simulations(new_board, simulations_per_move);
        
        // Calculate score from the perspective of the side that moved
        // After the move, it's opponent's turn, so we want the opposite of their score
        double opponent_score = results.score_for_side_to_move(!white_to_move);
        double our_score = 1.0 - opponent_score;
        
        std::string move_str = square_to_string(from) + square_to_string(to);
        
        std::cout << "Move " << (i+1) << "/" << legal_moves.size() << ": " << move_str 
                  << " - Score: " << (our_score * 100.0) << "% "
                  << "(W:" << results.white_wins << " D:" << results.draws 
                  << " B:" << results.black_wins << ")" << std::endl;
        
        if (our_score > best.score) {
            best.score = our_score;
            best.from_sq = square_to_string(from);
            best.to_sq = square_to_string(to);
        }
    }
    
    return best;
}
