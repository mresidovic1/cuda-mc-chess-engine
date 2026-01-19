#include "../../include/chess_types.cuh"
#include "../../include/kernels/common.cuh"
#include "../../include/kernels/movegen.cuh"
#include "../../include/kernels/evaluation.cuh"
#include <curand_kernel.h>

// Kernel: Random Playout (original)

// ============================================================================
// BATCHED ROLLOUT KERNELS - Massively parallel GPU playouts
// ============================================================================

// Kernel: Pure random playout (baseline, fast)
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

// Kernel: HYBRID PLAYOUT - 10 random moves + static evaluation (OPTIMIZED)
// Perfect balance: randomness for variety + eval for accuracy

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

    // Convert to win probability using sigmoid (centipawns -> probability)
    float winprob = score_to_winprob(eval, pos.side_to_move);

    // Adjust perspective for starting side
    if (starting_side == BLACK) {
        winprob = 1.0f - winprob;
    }

    results[idx] = winprob;
}

// Kernel: STATIC EVALUATION ONLY (no playout, fastest)
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

// ============================================================================
// QUIESCENCE PLAYOUT - Tactical extension for horizon effect
// ============================================================================

// Inline helper functions (is_tactical_move, generate_tactical_moves) now in playouts.cuh

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

    // For depth 1, just do a simple 1-ply tactical search (no recursion needed)
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

    // Try top tactical moves - ITERATIVE 2-ply search (no recursion)
    int try_limit = (num_moves < 4) ? num_moves : 4;  // Reduced to save memory
    for (int i = 0; i < try_limit; i++) {
        // Delta pruning: skip captures that can't improve position
        if (scores[i] < 0 && stand_pat + 1200 < best_score) {
            continue;
        }
        
        BoardState pos1 = *pos;
        make_move(&pos1, moves[i]);

        // Evaluate position after our capture (from opponent's perspective)
        int eval1 = gpu_evaluate(&pos1);
        if (pos1.side_to_move == BLACK) eval1 = -eval1;
        int score1 = -eval1;  // Negate for our perspective
        
        // If max_depth > 1, do another ply of tactical search for opponent
        if (max_depth > 1) {
            Move moves2[MAX_MOVES];
            int num_moves2 = generate_tactical_moves(&pos1, moves2);
            
            if (num_moves2 > 0) {
                // Opponent has tactical replies - find best one (worst for us)
                int best_reply = score1;  // If opponent has no good captures, we keep score1
                
                // Score and sort opponent's moves
                int scores2[MAX_MOVES];
                for (int j = 0; j < num_moves2; j++) {
                    scores2[j] = see_capture(&pos1, moves2[j]) * 10 + mvv_lva_score(&pos1, moves2[j]);
                }
                
                int sort_limit2 = (num_moves2 < 4) ? num_moves2 : 4;
                for (int j = 0; j < sort_limit2; j++) {
                    int best_idx = j;
                    for (int k = j + 1; k < num_moves2; k++) {
                        if (scores2[k] > scores2[best_idx]) best_idx = k;
                    }
                    if (best_idx != j) {
                        Move tm = moves2[j]; moves2[j] = moves2[best_idx]; moves2[best_idx] = tm;
                        int ts = scores2[j]; scores2[j] = scores2[best_idx]; scores2[best_idx] = ts;
                    }
                }
                
                // Try opponent's top captures
                int try_limit2 = (num_moves2 < 3) ? num_moves2 : 3;
                for (int j = 0; j < try_limit2; j++) {
                    if (scores2[j] < 0) continue;  // Skip losing captures for opponent
                    
                    BoardState pos2 = pos1;
                    make_move(&pos2, moves2[j]);
                    
                    // Evaluate after opponent's capture (from our perspective again)
                    int eval2 = gpu_evaluate(&pos2);
                    if (pos2.side_to_move == BLACK) eval2 = -eval2;
                    // pos2.side_to_move is now us again, so eval2 is from our perspective
                    int opp_score = -eval2;  // Score from opponent's view (negated = our loss)
                    
                    // Opponent wants to maximize their score (minimize ours)
                    if (-opp_score < best_reply) {
                        best_reply = -opp_score;  // This capture is worse for us
                    }
                }
                score1 = best_reply;
            }
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

// Kernel: Quiescence Playout - Random moves + tactical search
// Combines randomness with capture resolution for accuracy
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

    // Convert centipawns to win probability (sigmoid)
    float winprob = score_to_winprob(eval, pos.side_to_move);

    // Adjust for which side started
    if (starting_side == BLACK) {
        winprob = 1.0f - winprob;
    }

    results[idx] = winprob;
}
