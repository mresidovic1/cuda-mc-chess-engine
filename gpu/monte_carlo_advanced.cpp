#include "monte_carlo_advanced.hpp"
#ifdef __cplusplus
extern "C" {
#endif
#include "monte_carlo_advanced_launcher.cu"
#ifdef __cplusplus
}
#endif

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>

namespace monte_carlo_advanced {

// ============================================================================
// Conversion Functions: chess.hpp Board <-> GPU Position
// ============================================================================

Position board_to_gpu_position(const chess::Board& board) {
    Position pos;
    
    // Initialize board
    for (int sq = 0; sq < 64; sq++) {
        auto square = chess::Square(sq);
        auto piece = board.at(square);
        if (piece == chess::Piece::NONE) {
            pos.board[sq] = EMPTY;
        } else {
            int piece_type = static_cast<int>(piece.type());
            int color = (piece.color() == chess::Color::WHITE) ? 0 : 8;
            pos.board[sq] = (piece_type + 1) + color;
        }
    }
    // Side to move
    pos.side_to_move = (board.sideToMove() == chess::Color::WHITE) ? 0 : 1;
    
    // Castling rights (simplified)
    pos.castling_rights[0] = false;
    pos.castling_rights[1] = false;
    pos.castling_rights[2] = false;
    pos.castling_rights[3] = false;
    
    // En passant
    pos.en_passant = -1;
    
    // Move counters
    pos.halfmove_clock = 0;
    pos.fullmove_number = 1;
    
    return pos;
}

Move chess_move_to_gpu_move(const chess::Move& move, const chess::Board& board) {
    Move gpu_move;
    
    gpu_move.from = move.from().index();
    gpu_move.to = move.to().index();
    
    // Promotion
    if (move.typeOf() == chess::Move::PROMOTION) {
        auto promo_piece = move.promotionType();
        // Map to GPU piece type (2=knight, 3=bishop, 4=rook, 5=queen)
        gpu_move.promotion = static_cast<int>(promo_piece) + 1;
    } else {
        gpu_move.promotion = 0;
    }
    
    // Capture
    auto captured = board.at(move.to());
    if (captured == chess::Piece::NONE) {
        gpu_move.capture = EMPTY;
    } else {
        int piece_type = static_cast<int>(captured.type());
        int color = (captured.color() == chess::Color::WHITE) ? 0 : 8;
        gpu_move.capture = (piece_type + 1) + color;
    }
    
    // Moving piece
    auto piece = board.at(move.from());
    int piece_type = static_cast<int>(piece.type());
    int color = (piece.color() == chess::Color::WHITE) ? 0 : 8;
    gpu_move.piece = (piece_type + 1) + color;
    
    gpu_move.score = 0.0f;
    
    return gpu_move;
}

// ============================================================================
// Main Functions
// ============================================================================

std::vector<MoveEvaluation> evaluate_all_moves(
    const chess::Board& board,
    int simulations_per_move,
    int threads_per_move
) {
    std::vector<MoveEvaluation> evaluations;
    
    // Get all legal moves
    chess::Movelist movelist;
    chess::movegen::legalmoves(movelist, board);
    
    if (movelist.empty()) {
        return evaluations;
    }
    
    // Convert board to GPU position
    Position root_position = board_to_gpu_position(board);
    
    // CUDA setup
    int blocks = (simulations_per_move + threads_per_move - 1) / threads_per_move;
    int total_threads = blocks * threads_per_move;
    int sims_per_thread = (simulations_per_move + total_threads - 1) / total_threads;
    
    // Allocate device memory for results
    float* d_results;
    cudaMalloc(&d_results, total_threads * sizeof(float));
    
    // Host results
    std::vector<float> h_results(total_threads);
    
    // Evaluate each move
    for (const auto& move : movelist) {
        Move gpu_move = chess_move_to_gpu_move(move, board);
        
        // Generate random seed
        unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        
        // Launch kernel via extern "C" wrapper
        launch_monte_carlo_simulate_kernel(
            &root_position,
            &gpu_move,
            sims_per_thread,
            d_results,
            seed,
            blocks,
            threads_per_move
        );
        
        // Copy results back
        cudaMemcpy(h_results.data(), d_results, total_threads * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Calculate average score
        float total_score = 0.0f;
        for (int i = 0; i < total_threads; i++) {
            total_score += h_results[i];
        }
        float avg_score = total_score / total_threads;
        
        evaluations.push_back({move, avg_score, simulations_per_move});
    }
    
    // Cleanup
    cudaFree(d_results);
    
    return evaluations;
}

chess::Move find_best_move(
    const chess::Board& board,
    int simulations_per_move,
    int threads_per_move,
    bool verbose
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (verbose) {
        std::cout << "\n=== Advanced Monte Carlo Engine ===\n";
        std::cout << "Simulations per move: " << simulations_per_move << "\n";
        std::cout << "GPU threads per move: " << threads_per_move << "\n\n";
    }
    
    // Evaluate all moves
    auto evaluations = evaluate_all_moves(board, simulations_per_move, threads_per_move);
    
    if (evaluations.empty()) {
        if (verbose) {
            std::cout << "No legal moves available!\n";
        }
        return chess::Move::NO_MOVE;
    }
    
    // Sort by score (descending)
    std::sort(evaluations.begin(), evaluations.end(), 
        [](const MoveEvaluation& a, const MoveEvaluation& b) {
            return a.average_score > b.average_score;
        });
    
    if (verbose) {
        std::cout << "Top 10 moves:\n";
        std::cout << "Rank | Move  | Average Score | Simulations\n";
        std::cout << "-----+-------+---------------+------------\n";
        
        int count = std::min(10, static_cast<int>(evaluations.size()));
        for (int i = 0; i < count; i++) {
            std::cout << std::setw(4) << (i + 1) << " | "
                      << std::setw(5) << chess::uci::moveToUci(evaluations[i].move) << " | "
                      << std::setw(13) << std::fixed << std::setprecision(2) << evaluations[i].average_score << " | "
                      << std::setw(10) << evaluations[i].simulations << "\n";
        }
        std::cout << "\n";
    }
    
    auto best_move = evaluations[0].move;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (verbose) {
        std::cout << "Best move: " << chess::uci::moveToUci(best_move) << "\n";
        std::cout << "Score: " << evaluations[0].average_score << "\n";
        std::cout << "Time taken: " << duration.count() << " ms\n";
        std::cout << "Total simulations: " << (simulations_per_move * evaluations.size()) << "\n";
        
        if (duration.count() > 0) {
            int sims_per_sec = (simulations_per_move * evaluations.size() * 1000) / duration.count();
            std::cout << "Simulations per second: " << sims_per_sec << "\n";
        }
        
        std::cout << "===================================\n\n";
    }
    
    return best_move;
}

} // namespace monte_carlo_advanced
