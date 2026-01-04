//N - v1: Updated header with new features
#ifndef MONTE_CARLO_ADVANCED_V1_HPP
#define MONTE_CARLO_ADVANCED_V1_HPP

#include "../include/chess.hpp"
#include <string>
#include <vector>

namespace monte_carlo_advanced_v1 {

//N - v1: Evaluation mode selection
enum class EvaluationMode {
    LEGACY,     // Original sequential kernel launches
    BATCHED,    // All moves in single kernel (recommended)
    STREAMS     // Multiple streams for async evaluation
};

struct MoveEvaluation {
    chess::Move move;
    float average_score;
    int simulations;
};

// Find best move using advanced Monte Carlo with improvements
chess::Move find_best_move(
    const chess::Board& board,
    int simulations_per_move = 10000,
    int threads_per_move = 256,
    bool verbose = true,
    EvaluationMode mode = EvaluationMode::BATCHED
);

// Get detailed evaluations for all legal moves
std::vector<MoveEvaluation> evaluate_all_moves(
    const chess::Board& board,
    int simulations_per_move = 10000,
    int threads_per_move = 256,
    EvaluationMode mode = EvaluationMode::BATCHED
);

//N - v1: Resource management
void shutdown();      // Clean up GPU resources
void new_game();      // Clear caches for new game

} // namespace monte_carlo_advanced_v1

#endif // MONTE_CARLO_ADVANCED_V1_HPP
