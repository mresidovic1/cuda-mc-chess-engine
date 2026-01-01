#ifndef MONTE_CARLO_ADVANCED_HPP
#define MONTE_CARLO_ADVANCED_HPP

#include "../include/chess.hpp"
#include <string>
#include <vector>

namespace monte_carlo_advanced {

struct MoveEvaluation {
    chess::Move move;
    float average_score;
    int simulations;
};

// Find best move using advanced Monte Carlo with heuristics
chess::Move find_best_move(
    const chess::Board& board,
    int simulations_per_move = 10000,
    int threads_per_move = 256,
    bool verbose = true
);

// Get detailed evaluations for all legal moves
std::vector<MoveEvaluation> evaluate_all_moves(
    const chess::Board& board,
    int simulations_per_move = 10000,
    int threads_per_move = 256
);

} // namespace monte_carlo_advanced

#endif // MONTE_CARLO_ADVANCED_HPP
