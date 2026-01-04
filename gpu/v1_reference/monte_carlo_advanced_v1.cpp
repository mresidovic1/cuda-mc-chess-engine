//N - v1: Updated wrapper with CUDA streams and batched evaluation
#include "monte_carlo_advanced_v1.hpp"
#include "monte_carlo_advanced_types_v1.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstring>

// ============================================================================
// CUDA Error Checking Macro
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
    } \
} while(0)

extern "C" void launch_monte_carlo_simulate_kernel(
    const Position* root_position,
    const Move* root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed,
    int blocks,
    int threads_per_block
);

//N - v1: New batched launch function
extern "C" void launch_monte_carlo_batch_kernel(
    const Position* root_position,
    const Move* all_moves,
    int num_moves,
    int simulations_per_move,
    float* results,
    unsigned long long seed
);

extern "C" void initialize_gpu_resources();
extern "C" void cleanup_gpu_resources();
extern "C" void clear_gpu_transposition_table();
extern "C" void clear_gpu_history_table();

namespace monte_carlo_advanced_v1 {

//N - v1: Static initialization flag
static bool s_gpu_initialized = false;

// ============================================================================
// Conversion Functions
// ============================================================================
Position board_to_gpu_position(const chess::Board& board) {
    Position pos;

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

    pos.side_to_move = (board.sideToMove() == chess::Color::WHITE) ? 0 : 1;

    //N - v1: Initialize castling rights
    pos.castling_rights[0] = false;
    pos.castling_rights[1] = false;
    pos.castling_rights[2] = false;
    pos.castling_rights[3] = false;

    pos.en_passant = -1;
    pos.halfmove_clock = 0;
    pos.fullmove_number = 1;

    return pos;
}

Move chess_move_to_gpu_move(const chess::Move& move, const chess::Board& board) {
    Move gpu_move;

    gpu_move.from = move.from().index();
    gpu_move.to = move.to().index();

    if (move.typeOf() == chess::Move::PROMOTION) {
        auto promo_piece = move.promotionType();
        gpu_move.promotion = static_cast<int>(promo_piece) + 1;
    } else {
        gpu_move.promotion = 0;
    }

    auto captured = board.at(move.to());
    if (captured == chess::Piece::NONE) {
        gpu_move.capture = EMPTY;
    } else {
        int piece_type = static_cast<int>(captured.type());
        int color = (captured.color() == chess::Color::WHITE) ? 0 : 8;
        gpu_move.capture = (piece_type + 1) + color;
    }

    auto piece = board.at(move.from());
    int piece_type = static_cast<int>(piece.type());
    int color = (piece.color() == chess::Color::WHITE) ? 0 : 8;
    gpu_move.piece = (piece_type + 1) + color;

    gpu_move.score = 0.0f;

    return gpu_move;
}

// ============================================================================
//N - v1: Legacy evaluation (sequential, for compatibility)
// ============================================================================
std::vector<MoveEvaluation> evaluate_all_moves_legacy(
    const chess::Board& board,
    int simulations_per_move,
    int threads_per_move
) {
    std::vector<MoveEvaluation> evaluations;

    chess::Movelist movelist;
    chess::movegen::legalmoves(movelist, board);

    if (movelist.empty()) {
        return evaluations;
    }

    Position root_position = board_to_gpu_position(board);

    int blocks = (simulations_per_move + threads_per_move - 1) / threads_per_move;
    int total_threads = blocks * threads_per_move;
    int sims_per_thread = (simulations_per_move + total_threads - 1) / total_threads;

    float* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, total_threads * sizeof(float)));

    std::vector<float> h_results(total_threads);

    // Calculate actual total simulations for this mode
    int actual_total_sims = total_threads * sims_per_thread;

    for (const auto& move : movelist) {
        Move gpu_move = chess_move_to_gpu_move(move, board);

        // Clear results before each move
        CUDA_CHECK(cudaMemset(d_results, 0, total_threads * sizeof(float)));

        unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

        launch_monte_carlo_simulate_kernel(
            &root_position,
            &gpu_move,
            sims_per_thread,
            d_results,
            seed,
            blocks,
            threads_per_move
        );

        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, total_threads * sizeof(float), cudaMemcpyDeviceToHost));

        float total_score = 0.0f;
        for (int i = 0; i < total_threads; i++) {
            total_score += h_results[i];
        }
        float avg_score = total_score / total_threads;

        evaluations.push_back({move, avg_score, actual_total_sims});
    }

    CUDA_CHECK(cudaFree(d_results));

    return evaluations;
}

// ============================================================================
//N - v1: Batched evaluation (all moves in one kernel launch)
// ============================================================================
std::vector<MoveEvaluation> evaluate_all_moves_batched(
    const chess::Board& board,
    int simulations_per_move
) {
    std::vector<MoveEvaluation> evaluations;

    chess::Movelist movelist;
    chess::movegen::legalmoves(movelist, board);

    if (movelist.empty()) {
        return evaluations;
    }

    int num_moves = static_cast<int>(movelist.size());
    Position root_position = board_to_gpu_position(board);

    // Convert all moves to GPU format
    std::vector<Move> gpu_moves(num_moves);
    for (int i = 0; i < num_moves; i++) {
        gpu_moves[i] = chess_move_to_gpu_move(movelist[i], board);
    }

    // Allocate device results
    float* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, num_moves * sizeof(float)));

    unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Launch batched kernel
    launch_monte_carlo_batch_kernel(
        &root_position,
        gpu_moves.data(),
        num_moves,
        simulations_per_move,
        d_results,
        seed
    );

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back
    std::vector<float> h_results(num_moves);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, num_moves * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_results));

    // Calculate actual simulations (blocks_per_move * 256 threads)
    int threads_per_block = 256;
    int blocks_per_move = (simulations_per_move + threads_per_block - 1) / threads_per_block;
    int actual_sims = blocks_per_move * threads_per_block;

    // Build evaluations
    for (int i = 0; i < num_moves; i++) {
        evaluations.push_back({movelist[i], h_results[i], actual_sims});
    }

    return evaluations;
}

// ============================================================================
//N - v1: CUDA Streams based evaluation
// ============================================================================
std::vector<MoveEvaluation> evaluate_all_moves_streams(
    const chess::Board& board,
    int simulations_per_move,
    int threads_per_move,
    int num_streams
) {
    std::vector<MoveEvaluation> evaluations;

    chess::Movelist movelist;
    chess::movegen::legalmoves(movelist, board);

    if (movelist.empty()) {
        return evaluations;
    }

    int num_moves = static_cast<int>(movelist.size());
    Position root_position = board_to_gpu_position(board);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    int blocks = (simulations_per_move + threads_per_move - 1) / threads_per_move;
    int total_threads = blocks * threads_per_move;
    int sims_per_thread = (simulations_per_move + total_threads - 1) / total_threads;
    int actual_total_sims = total_threads * sims_per_thread;

    // Allocate device memory for all moves
    std::vector<float*> d_results(num_moves);
    for (int i = 0; i < num_moves; i++) {
        CUDA_CHECK(cudaMalloc(&d_results[i], total_threads * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_results[i], 0, total_threads * sizeof(float)));
    }

    // Convert moves and launch kernels asynchronously
    std::vector<Move> gpu_moves(num_moves);
    for (int i = 0; i < num_moves; i++) {
        gpu_moves[i] = chess_move_to_gpu_move(movelist[i], board);
    }

    unsigned long long base_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();

    // Launch all kernels
    for (int i = 0; i < num_moves; i++) {
        int stream_idx = i % num_streams;

        // Note: We need to use cudaLaunchKernel or a wrapper that supports streams
        // For now, we use the synchronous version but interleave with streams
        CUDA_CHECK(cudaStreamSynchronize(streams[stream_idx]));

        launch_monte_carlo_simulate_kernel(
            &root_position,
            &gpu_moves[i],
            sims_per_thread,
            d_results[i],
            base_seed + i,
            blocks,
            threads_per_move
        );
    }

    // Wait for all streams
    CUDA_CHECK(cudaDeviceSynchronize());

    // Collect results
    std::vector<float> h_results(total_threads);
    for (int i = 0; i < num_moves; i++) {
        CUDA_CHECK(cudaMemcpy(h_results.data(), d_results[i], total_threads * sizeof(float), cudaMemcpyDeviceToHost));

        float total_score = 0.0f;
        for (int j = 0; j < total_threads; j++) {
            total_score += h_results[j];
        }
        float avg_score = total_score / total_threads;

        evaluations.push_back({movelist[i], avg_score, actual_total_sims});

        CUDA_CHECK(cudaFree(d_results[i]));
    }

    // Destroy streams
    for (int i = 0; i < num_streams; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return evaluations;
}

// ============================================================================
// Main Functions
// ============================================================================
std::vector<MoveEvaluation> evaluate_all_moves(
    const chess::Board& board,
    int simulations_per_move,
    int threads_per_move,
    EvaluationMode mode
) {
    //N - v1: Initialize GPU resources on first call
    if (!s_gpu_initialized) {
        initialize_gpu_resources();
        s_gpu_initialized = true;
    }

    switch (mode) {
        case EvaluationMode::LEGACY:
            return evaluate_all_moves_legacy(board, simulations_per_move, threads_per_move);

        case EvaluationMode::BATCHED:
            return evaluate_all_moves_batched(board, simulations_per_move);

        case EvaluationMode::STREAMS:
            return evaluate_all_moves_streams(board, simulations_per_move, threads_per_move, 8);

        default:
            // Default to batched (best performance)
            return evaluate_all_moves_batched(board, simulations_per_move);
    }
}

chess::Move find_best_move(
    const chess::Board& board,
    int simulations_per_move,
    int threads_per_move,
    bool verbose,
    EvaluationMode mode
) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (verbose) {
        std::cout << "\n=== Advanced Monte Carlo Engine v1 ===\n";
        std::cout << "Simulations per move: " << simulations_per_move << "\n";
        std::cout << "Mode: ";
        switch (mode) {
            case EvaluationMode::LEGACY: std::cout << "Legacy (sequential)\n"; break;
            case EvaluationMode::BATCHED: std::cout << "Batched (parallel)\n"; break;
            case EvaluationMode::STREAMS: std::cout << "Streams (async)\n"; break;
        }
        std::cout << "\n";
    }

    auto evaluations = evaluate_all_moves(board, simulations_per_move, threads_per_move, mode);

    // Check for game over conditions (checkmate/stalemate)
    if (evaluations.empty()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (verbose) {
            // Check if it's checkmate or stalemate
            bool in_check = board.inCheck();
            if (in_check) {
                std::cout << "CHECKMATE! ";
                if (board.sideToMove() == chess::Color::WHITE) {
                    std::cout << "Black wins!\n";
                } else {
                    std::cout << "White wins!\n";
                }
                std::cout << "Score: " << (board.sideToMove() == chess::Color::WHITE ? "-" : "+") << "10000.00\n";
            } else {
                std::cout << "STALEMATE! Game is drawn.\n";
                std::cout << "Score: 0.00\n";
            }
            std::cout << "Time taken: " << duration.count() << " ms\n";
            std::cout << "======================================\n\n";
        }
        return chess::Move::NO_MOVE;
    }

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
        // Calculate total simulations from actual simulation counts
        long long total_sims = 0;
        for (const auto& eval : evaluations) {
            total_sims += eval.simulations;
        }

        std::cout << "Best move: " << chess::uci::moveToUci(best_move) << "\n";
        std::cout << "Score: " << std::fixed << std::setprecision(2) << evaluations[0].average_score << "\n";
        std::cout << "Time taken: " << duration.count() << " ms\n";
        std::cout << "Total simulations: " << total_sims << "\n";

        if (duration.count() > 0) {
            long long sims_per_sec = (total_sims * 1000) / duration.count();
            std::cout << "Simulations per second: " << sims_per_sec << "\n";
        }

        std::cout << "======================================\n\n";
    }

    return best_move;
}

//N - v1: Cleanup function
void shutdown() {
    if (s_gpu_initialized) {
        cleanup_gpu_resources();
        s_gpu_initialized = false;
    }
}

//N - v1: Clear caches between games
void new_game() {
    if (s_gpu_initialized) {
        clear_gpu_transposition_table();
        clear_gpu_history_table();
    }
}

} // namespace monte_carlo_advanced_v1
