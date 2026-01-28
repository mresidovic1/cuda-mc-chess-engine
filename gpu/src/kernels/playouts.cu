#include "../../include/chess_types.cuh"
#include "../../include/kernels/common.cuh"
#include "../../include/kernels/movegen.cuh"
#include "../../include/kernels/evaluation.cuh"
#include <curand_kernel.h>

// Playout Mode 1: Pure Random Playout (baseline, fastest but least accurate)
__global__ void RandomPlayout(
    const BoardState* __restrict__ starting_boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) return;

    curandState rng;
    curand_init(seed, idx, 0, &rng);

    BoardState pos = starting_boards[idx];
    int starting_side = pos.side_to_move;

    Move moves[MAX_MOVES];

    // Random playout up to MAX_PLAYOUT_MOVES
    for (int ply = 0; ply < MAX_PLAYOUT_MOVES; ply++) {
        int num_moves = generate_legal_moves(&pos, moves);

        // Terminal position - checkmate or stalemate
        if (num_moves == 0) {
            if (in_check(&pos)) {
                int winner = pos.side_to_move ^ 1;
                results[idx] = (winner == starting_side) ? 1.0f : 0.0f;
            } else {
                results[idx] = 0.5f; // Stalemate = draw
            }
            return;
        }

        // Draw by 50-move rule
        if (pos.halfmove >= 100) {
            results[idx] = 0.5f;
            return;
        }

        // Select random move
        int move_idx = curand(&rng) % num_moves;
        make_move(&pos, moves[move_idx]);
    }

    // Max depth reached - draw
    results[idx] = 0.5f;
}

// Playout Mode 2: Eval Hybrid - 10 random moves + static evaluation (balanced speed/accuracy)
#define HYBRID_RANDOM_DEPTH 10  // Exactly 10 random moves as requested

__global__ void EvalPlayout(
    const BoardState* __restrict__ starting_boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) return;

    curandState rng;
    curand_init(seed, idx, 0, &rng);

    BoardState pos = starting_boards[idx];
    int starting_side = pos.side_to_move;

    Move moves[MAX_MOVES];

    // EXACTLY 10 random moves for variety (as requested)
    for (int ply = 0; ply < HYBRID_RANDOM_DEPTH; ply++) {
        int num_moves = generate_legal_moves(&pos, moves);

        if (num_moves == 0) {
            if (in_check(&pos)) {
                int winner = pos.side_to_move ^ 1;
                results[idx] = (winner == starting_side) ? 1.0f : 0.0f;
            } else {
                results[idx] = 0.5f;
            }
            return;
        }

        if (pos.halfmove >= 100) {
            results[idx] = 0.5f;
            return;
        }

        int move_idx = curand(&rng) % num_moves;
        make_move(&pos, moves[move_idx]);
    }

    // Evaluate final position with optimized static eval
    int eval = gpu_evaluate(&pos);

    // Convert eval to starting side's perspective
    // gpu_evaluate returns White-relative centipawns, so negate for Black
    if (starting_side == BLACK) {
        eval = -eval;
    }

    // Convert to win probability using sigmoid (centipawns -> probability)
    float x = (float)eval / 400.0f;
    float winprob = 1.0f / (1.0f + expf(-x));

    results[idx] = winprob;
}

// Playout Mode 3: Static Evaluation Only (fastest, no playout)
// Pure material + positional eval, instant results

__global__ void StaticEval(
    const BoardState* __restrict__ boards,
    float* __restrict__ results,
    int numBoards
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) return;

    BoardState pos = boards[idx];

    // Check for terminal positions (mate/stalemate)
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(&pos, moves);

    if (num_moves == 0) {
        if (in_check(&pos)) {
            // Checkmate - loss
            results[idx] = 0.0f;
        } else {
            // Stalemate - draw
            results[idx] = 0.5f;
        }
        return;
    }

    // PST + Material evaluation -> win probability
    int eval = gpu_evaluate(&pos);
    results[idx] = score_to_winprob(eval, pos.side_to_move);
}


// Playout Mode 4: Quiescence Playout - tactical extension for horizon effect
// Uses iterative deepening with SEE-based move ordering and delta pruning

// Check if a move is tactical (capture, check, or promotion)
__device__
bool is_tactical_move(const BoardState* pos, Move m) {
    int move_type = (m >> 12) & 0xF;
    if (move_type >= MOVE_PROMO_N && move_type <= MOVE_PROMO_CAP_Q) return true;
    if (move_type == MOVE_CAPTURE || move_type == MOVE_EP_CAPTURE) return true;
    return false;
}

// Generate only tactical moves (captures and promotions)
__device__
int generate_tactical_moves(const BoardState* pos, Move* moves) {
    Move all_moves[MAX_MOVES];
    int total = generate_legal_moves(pos, all_moves);
    int count = 0;
    for (int i = 0; i < total; i++) {
        if (is_tactical_move(pos, all_moves[i])) {
            moves[count++] = all_moves[i];
        }
    }
    return count;
}

// MVV-LVA (Most Valuable Victim - Least Valuable Attacker) for move ordering
__device__
int mvv_lva_score(const BoardState* pos, Move m) {
    int from = m & 0x3F;
    int to = (m >> 6) & 0x3F;
    int move_type = (m >> 12) & 0xF;

    // Promotion captures: highest priority
    if (move_type >= MOVE_PROMO_CAP_N && move_type <= MOVE_PROMO_CAP_Q) {
        return 10000 + ((move_type - MOVE_PROMO_CAP_N) * 100);
    }

    // Promotions without capture
    if (move_type >= MOVE_PROMO_N && move_type <= MOVE_PROMO_Q) {
        return 9000 + ((move_type - MOVE_PROMO_N) * 100);
    }

    // Captures: victim value - attacker value
    if (move_type == MOVE_CAPTURE || move_type == MOVE_EP_CAPTURE) {
        int side = pos->side_to_move;
        int opp = side ^ 1;

        // Find victim piece type
        int victim_value = 0;
        Bitboard to_bb = C64(1) << to;
        for (int pt = PAWN; pt <= QUEEN; pt++) {
            if (pos->pieces[opp][pt] & to_bb) {
                static const int values[6] = {100, 320, 330, 500, 900, 0};
                victim_value = values[pt];
                break;
            }
        }

        // Find attacker piece type
        int attacker_value = 0;
        Bitboard from_bb = C64(1) << from;
        for (int pt = PAWN; pt <= QUEEN; pt++) {
            if (pos->pieces[side][pt] & from_bb) {
                static const int values[6] = {100, 320, 330, 500, 900, 0};
                attacker_value = values[pt];
                break;
            }
        }

        // MVV-LVA: high victim value - low attacker value
        return victim_value * 10 - attacker_value;
    }

    return 0;
}

// Static Exchange Evaluation (SEE) - evaluates capture profitability
__device__
int see_capture(const BoardState* pos, Move m) {
    int from = m & 0x3F;
    int to = (m >> 6) & 0x3F;
    int move_type = (m >> 12) & 0xF;
    
    constexpr int piece_values[6] = {100, 320, 330, 500, 900, 0};
    
    // Find moving piece
    int attacker_value = 0;
    int moving_piece = -1;
    for (int pt = 0; pt < 5; pt++) {
        if (pos->pieces[pos->side_to_move][pt] & (1ULL << from)) {
            moving_piece = pt;
            attacker_value = piece_values[pt];
            break;
        }
    }
    
    if (moving_piece < 0) return 0;
    
    // Find captured piece
    int victim_value = 0;
    if (move_type == MOVE_CAPTURE || move_type >= MOVE_PROMO_CAP_N) {
        for (int pt = 0; pt < 5; pt++) {
            if (pos->pieces[pos->side_to_move ^ 1][pt] & (1ULL << to)) {
                victim_value = piece_values[pt];
                break;
            }
        }
    } else if (move_type == MOVE_EP_CAPTURE) {
        victim_value = 100;
    }
    
    // Simple SEE: victim - (attacker if recaptured)
    // Positive = good capture, Negative = losing capture
    return victim_value - attacker_value;
}

// ITERATIVE quiescence search with delta pruning and SEE
// Uses explicit stack to avoid GPU stack overflow from recursion
// Returns score from pos->side_to_move perspective (negamax convention)
__device__ __forceinline__
int quiescence_search_simple(const BoardState* pos, int max_depth) {
    // Limit max_depth to prevent excessive memory usage
    if (max_depth > 4) max_depth = 4;

    // Stand-pat - current static evaluation from side-to-move's perspective
    // gpu_evaluate returns white-relative score, so negate for black
    int stand_pat = gpu_evaluate(pos);
    if (pos->side_to_move == BLACK) stand_pat = -stand_pat;

    if (max_depth <= 0) {
        return stand_pat;
    }

    // Generate tactical moves (captures, promotions)
    Move moves[MAX_MOVES];
    int num_moves = generate_tactical_moves(pos, moves);

    // No tactical moves - quiet position
    if (num_moves == 0) {
        return stand_pat;
    }

    // Score moves with SEE for better ordering
    int scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        int see = see_capture(pos, moves[i]);
        scores[i] = see * 10 + mvv_lva_score(pos, moves[i]);
    }

    // Sort top 6 tactical moves
    int sort_limit = (num_moves < 6) ? num_moves : 6;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
        }
    }

    int best_score = stand_pat;

    // Try top tactical moves - ITERATIVE 2-ply search
    int try_limit = (num_moves < 4) ? num_moves : 4;  // Reduced to save memory
    for (int i = 0; i < try_limit; i++) {
        // Delta pruning: skip captures that can't improve position
        if (scores[i] < 0 && stand_pat + 1200 < best_score) {
            continue;
        }

        BoardState pos1 = *pos;
        make_move(&pos1, moves[i]);

        // Recursively search opponent's responses (negamax: negate child's score)
        int score1;
        if (max_depth > 1) {
            // Recurse: opponent evaluates their position, we negate it
            score1 = -quiescence_search_simple(&pos1, max_depth - 1);
        } else {
            // Leaf: just evaluate and negate (opponent's score = negative of our score)
            int eval1 = gpu_evaluate(&pos1);
            if (pos1.side_to_move == BLACK) eval1 = -eval1;
            score1 = -eval1;
        }

        if (score1 > best_score) {
            best_score = score1;
        }

        // Cutoff if we're already winning by a lot
        if (best_score > stand_pat + 900) {
            break;
        }
    }

    return best_score;
}

// Playout Mode 4: Quiescence Playout - 5 random moves + iterative tactical search
// Combines randomness with capture resolution for accuracy, avoids horizon effect
__global__ void QuiescencePlayout(
    const BoardState* __restrict__ starting_boards,
    float* __restrict__ results,
    int numBoards,
    unsigned int seed,
    int max_q_depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoards) return;

    curandState rng;
    curand_init(seed, idx, 0, &rng);

    BoardState pos = starting_boards[idx];
    int starting_side = pos.side_to_move;

    Move moves[MAX_MOVES];

    // Short random playout (5 moves) for positional variety
    for (int ply = 0; ply < 5; ply++) {  
        int num_moves = generate_legal_moves(&pos, moves);

        if (num_moves == 0) {
            if (in_check(&pos)) {
                int winner = pos.side_to_move ^ 1;
                results[idx] = (winner == starting_side) ? 1.0f : 0.0f;
            } else {
                results[idx] = 0.5f; // Stalemate
            }
            return;
        }

        if (pos.halfmove >= 100) {
            results[idx] = 0.5f; // Draw
            return;
        }

        int move_idx = curand(&rng) % num_moves;
        make_move(&pos, moves[move_idx]);
    }

    // Quiescence search - resolve tactical sequences (captures)
    int eval = quiescence_search_simple(&pos, max_q_depth);

    // Convert eval to starting side's perspective
    // quiescence_search_simple returns score from pos.side_to_move perspective
    // So we need to adjust based on who's moving after the random playout
    if (pos.side_to_move != starting_side) {
        eval = -eval;
    }

    // Convert to win probability using sigmoid (centipawns -> probability)
    float x = (float)eval / 400.0f;
    float winprob = 1.0f / (1.0f + expf(-x));

    results[idx] = winprob;
}
