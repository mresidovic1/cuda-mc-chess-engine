#include "../../include/chess_types.cuh"
#include "../../include/kernels/common.cuh"
#include "../../include/kernels/movegen.cuh"
#include "../../include/kernels/evaluation.cuh"
#include "../../include/kernels/playouts.cuh"

#define MATE_SCORE 30000
#define INF_SCORE 32000

// ============================================================================
// TACTICAL OPTIMIZATIONS - Pruning, Enhanced Move Ordering
// ============================================================================

// TACTICAL OPTIMIZATIONS - Pruning, Enhanced Move Ordering
// ============================================================================

// Check if move gives check
__device__ __forceinline__
bool gives_check_simple(BoardState* pos, Move m) {
    BoardState temp = *pos;
    make_move(&temp, m);
    return in_check(&temp);
}

// ============================================================================
// TACTICAL MOVE ORDERING - Enhanced with SEE
// ============================================================================

__device__ __forceinline__
int tactical_move_score(const BoardState* pos, Move m, bool gives_check) {
    int move_type = (m >> 12) & 0xF;

    // Checks - highest priority
    if (gives_check) return 1000000;

    // Promotion captures - use SEE
    if (move_type >= MOVE_PROMO_CAP_N && move_type <= MOVE_PROMO_CAP_Q) {
        int see = see_capture(pos, m);
        return 100000 + see;
    }

    // Promotions (non-capture)
    if (move_type >= MOVE_PROMO_N && move_type <= MOVE_PROMO_Q) {
        return 50000 + ((move_type - MOVE_PROMO_N) * 1000);
    }

    // Captures - SEE + MVV-LVA
    if (move_type == MOVE_CAPTURE || move_type == MOVE_EP_CAPTURE) {
        int see = see_capture(pos, m);
        return 10000 + see * 10 + mvv_lva_score(pos, m);
    }

    return 0;  // Quiet moves
}

// OPTIMIZED depth-2 tactical solver - FULLY ITERATIVE
__device__ __noinline__
int tactical_depth2(BoardState* pos, int alpha, int beta, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    if (num_moves == 0) {
        return in_check(pos) ? -(MATE_SCORE - ply) : 0;
    }

    // Static eval for futility pruning
    int static_eval = gpu_evaluate(pos);
    
    // Futility pruning: if position is very bad, skip quiet moves
    bool futility_prune = false;
    if (!in_check(pos) && static_eval + 900 < alpha) { // ~Queen down
        futility_prune = true;
    }

    // Score and sort moves with SEE
    int scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        bool gives_check = gives_check_simple(pos, moves[i]);
        scores[i] = tactical_move_score(pos, moves[i], gives_check);
    }

    // Sort top 25 moves for better move ordering
    int sort_limit = (num_moves < 25) ? num_moves : 25;
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

    int best = -(MATE_SCORE + 1);
    for (int i = 0; i < num_moves; i++) {
        // Futility pruning: skip quiet moves if hopeless
        if (futility_prune && scores[i] < 10000) { // Skip non-tactical moves
            continue;
        }
        
        BoardState pos2 = *pos;
        make_move(&pos2, moves[i]);

        int score;
        Move moves2[MAX_MOVES];
        int num_moves2 = generate_legal_moves(&pos2, moves2);

        if (num_moves2 == 0) {
            score = in_check(&pos2) ? (MATE_SCORE - ply - 1) : 0;
        } else {
            int scores2[MAX_MOVES];
            for (int j = 0; j < num_moves2; j++) {
                bool gives_check2 = gives_check_simple(&pos2, moves2[j]);
                scores2[j] = tactical_move_score(&pos2, moves2[j], gives_check2);
            }

            int sort_limit2 = (num_moves2 < 40) ? num_moves2 : 40;
            for (int j = 0; j < sort_limit2; j++) {
                int best_idx = j;
                for (int k = j + 1; k < num_moves2; k++) {
                    if (scores2[k] > scores2[best_idx]) best_idx = k;
                }
                if (best_idx != j) {
                    Move tmp_m = moves2[j]; moves2[j] = moves2[best_idx]; moves2[best_idx] = tmp_m;
                    int tmp_s = scores2[j]; scores2[j] = scores2[best_idx]; scores2[best_idx] = tmp_s;
                }
            }

            int worst = MATE_SCORE + 1;
            for (int j = 0; j < num_moves2; j++) {
                BoardState pos3 = pos2;
                make_move(&pos3, moves2[j]);

                Move moves3[MAX_MOVES];
                int num_moves3 = generate_legal_moves(&pos3, moves3);

                int eval;
                if (num_moves3 == 0) {
                    eval = in_check(&pos3) ? -(MATE_SCORE - ply - 2) : 0;
                } else {
                    eval = gpu_evaluate(&pos3);
                }

                if (eval < worst) worst = eval;
                if (worst <= -beta) break;
                if (j >= 40 && worst < alpha - 200) break;
            }
            score = -worst;
        }

        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    return best;
}

// ============================================================================
// TACTICAL DEPTH 4 - ADVANCED ITERATIVE (NO RECURSION!)
// Features: SEE ordering, Selective Extensions, LMR, Futility, Null-Move
// ============================================================================

__device__ __noinline__
int tactical_depth4(BoardState* pos, int alpha, int beta, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);
    if (num_moves == 0) return in_check(pos) ? -(MATE_SCORE - ply) : 0;

    bool in_check_root = in_check(pos);
    int static_eval = gpu_evaluate(pos);
    
    // Futility pruning threshold
    bool futility_prune = !in_check_root && static_eval + 1200 < alpha;

    // Null-Move Pruning (R=2) - skip 2 plies
    bool can_null = !in_check_root && static_eval > beta && ply < 4;
    if (can_null) {
        BoardState pos_null = *pos;
        pos_null.side_to_move ^= 1; // Switch side
        int null_score = -gpu_evaluate(&pos_null); // Rough 2-ply reduction
        if (null_score >= beta) {
            return beta; // Prune this branch
        }
    }

    // Score & sort moves with SEE
    int scores[MAX_MOVES];
    bool move_gives_check[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        move_gives_check[i] = gives_check_simple(pos, moves[i]);
        scores[i] = tactical_move_score(pos, moves[i], move_gives_check[i]);
    }
    
    // Sort top 20 moves
    int sort_limit = (num_moves < 20) ? num_moves : 20;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
            bool tc = move_gives_check[i]; move_gives_check[i] = move_gives_check[best_idx]; move_gives_check[best_idx] = tc;
        }
    }

    int best = -(MATE_SCORE + 1);
    
    // ========================================================================
    // PLY 1
    // ========================================================================
    for (int i = 0; i < num_moves && i < 20; i++) {
        // Futility: Skip quiet moves when far behind
        if (futility_prune && scores[i] < 10000) continue;
        
        // LMR: Reduce search width for late quiet moves
        int max_ply2 = 20;
        if (i >= 6 && scores[i] < 10000) { // Late quiet move
            max_ply2 = 12; // Reduced branching
        }
        
        // Selective Extension: +1 ply for checks
        bool extend_ply1 = move_gives_check[i];
        
        BoardState pos2 = *pos;
        make_move(&pos2, moves[i]);
        Move moves2[MAX_MOVES];
        int num_moves2 = generate_legal_moves(&pos2, moves2);
        
        int score;
        if (num_moves2 == 0) {
            score = in_check(&pos2) ? (MATE_SCORE - ply - 1) : 0;
        } else {
            // Score ply 2 moves
            int scores2[MAX_MOVES];
            bool move_gives_check2[MAX_MOVES];
            for (int j = 0; j < num_moves2; j++) {
                move_gives_check2[j] = gives_check_simple(&pos2, moves2[j]);
                scores2[j] = tactical_move_score(&pos2, moves2[j], move_gives_check2[j]);
            }
            int sort_limit2 = (num_moves2 < max_ply2) ? num_moves2 : max_ply2;
            for (int j = 0; j < sort_limit2; j++) {
                int best_idx = j;
                for (int k = j + 1; k < num_moves2; k++) {
                    if (scores2[k] > scores2[best_idx]) best_idx = k;
                }
                if (best_idx != j) {
                    Move tm = moves2[j]; moves2[j] = moves2[best_idx]; moves2[best_idx] = tm;
                    int ts = scores2[j]; scores2[j] = scores2[best_idx]; scores2[best_idx] = ts;
                    bool tc = move_gives_check2[j]; move_gives_check2[j] = move_gives_check2[best_idx]; move_gives_check2[best_idx] = tc;
                }
            }
            
            int worst2 = MATE_SCORE + 1;
            
            // ====================================================================
            // PLY 2
            // ====================================================================
            for (int j = 0; j < num_moves2 && j < max_ply2; j++) {
                int max_ply3 = 15;
                if (j >= 6 && scores2[j] < 10000) max_ply3 = 10; // LMR
                
                bool extend_ply2 = move_gives_check2[j];
                
                BoardState pos3 = pos2;
                make_move(&pos3, moves2[j]);
                Move moves3[MAX_MOVES];
                int num_moves3 = generate_legal_moves(&pos3, moves3);
                
                int s3;
                if (num_moves3 == 0) {
                    s3 = in_check(&pos3) ? -(MATE_SCORE - ply - 2) : 0;
                } else {
                    // Score ply 3 moves
                    int scores3[MAX_MOVES];
                    bool move_gives_check3[MAX_MOVES];
                    for (int k = 0; k < num_moves3; k++) {
                        move_gives_check3[k] = gives_check_simple(&pos3, moves3[k]);
                        scores3[k] = tactical_move_score(&pos3, moves3[k], move_gives_check3[k]);
                    }
                    int sort_limit3 = (num_moves3 < max_ply3) ? num_moves3 : max_ply3;
                    for (int k = 0; k < sort_limit3; k++) {
                        int best_idx = k;
                        for (int m = k + 1; m < num_moves3; m++) {
                            if (scores3[m] > scores3[best_idx]) best_idx = m;
                        }
                        if (best_idx != k) {
                            Move tm = moves3[k]; moves3[k] = moves3[best_idx]; moves3[best_idx] = tm;
                            int ts = scores3[k]; scores3[k] = scores3[best_idx]; scores3[best_idx] = ts;
                            bool tc = move_gives_check3[k]; move_gives_check3[k] = move_gives_check3[best_idx]; move_gives_check3[best_idx] = tc;
                        }
                    }
                    
                    int best3 = -(MATE_SCORE + 1);
                    
                    // ================================================================
                    // PLY 3
                    // ================================================================
                    for (int k = 0; k < num_moves3 && k < max_ply3; k++) {
                        int max_ply4 = 10;
                        if (k >= 5 && scores3[k] < 10000) max_ply4 = 7; // LMR
                        
                        // Extension: If ply1 or ply2 was check, search deeper here
                        if (extend_ply1 || extend_ply2 || move_gives_check3[k]) {
                            max_ply4 += 3; // +1 effective extension
                        }
                        
                        BoardState pos4 = pos3;
                        make_move(&pos4, moves3[k]);
                        Move moves4[MAX_MOVES];
                        int num_moves4 = generate_legal_moves(&pos4, moves4);
                        
                        int s4;
                        if (num_moves4 == 0) {
                            s4 = in_check(&pos4) ? (MATE_SCORE - ply - 3) : 0;
                        } else {
                            // ============================================================
                            // PLY 4 - Leaf evaluation
                            // ============================================================
                            int worst4 = MATE_SCORE + 1;
                            for (int m = 0; m < num_moves4 && m < max_ply4; m++) {
                                BoardState pos5 = pos4;
                                make_move(&pos5, moves4[m]);
                                int eval = gpu_evaluate(&pos5);
                                if (eval < worst4) worst4 = eval;
                            }
                            s4 = -worst4;
                        }
                        
                        if (s4 > best3) best3 = s4;
                        if (best3 >= -worst2 + 200) break; // Beta cutoff approximation
                    }
                    s3 = best3;
                }
                
                if (s3 < worst2) worst2 = s3;
                if (worst2 <= -beta) break;
            }
            score = -worst2;
        }
        
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    
    return best;
}

// ============================================================================
// TACTICAL DEPTH 6 - ADVANCED ITERATIVE (NO RECURSION!)
// Uses depth4 as subroutine (which is fully iterative)
// ============================================================================

__device__ __noinline__
int tactical_depth6(BoardState* pos, int alpha, int beta, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);
    if (num_moves == 0) return in_check(pos) ? -(MATE_SCORE - ply) : 0;

    bool in_check_root = in_check(pos);
    int static_eval = gpu_evaluate(pos);
    
    // Aggressive futility for depth 6
    bool futility_prune = !in_check_root && static_eval + 1500 < alpha;

    // Null-Move Pruning (R=3 for depth 6)
    bool can_null = !in_check_root && static_eval > beta + 100 && ply < 3;
    if (can_null) {
        BoardState pos_null = *pos;
        pos_null.side_to_move ^= 1;
        // Approximate 3-ply reduction by calling depth4 at reduced window
        int null_score = -tactical_depth4(&pos_null, -beta, -beta + 1, ply + 1);
        if (null_score >= beta) {
            return beta; // Prune
        }
    }

    // Score & sort moves (top 15 only for depth 6)
    int scores[MAX_MOVES];
    bool move_gives_check[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        move_gives_check[i] = gives_check_simple(pos, moves[i]);
        scores[i] = tactical_move_score(pos, moves[i], move_gives_check[i]);
    }
    
    int sort_limit = (num_moves < 15) ? num_moves : 15;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (scores[j] > scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = scores[i]; scores[i] = scores[best_idx]; scores[best_idx] = ts;
            bool tc = move_gives_check[i]; move_gives_check[i] = move_gives_check[best_idx]; move_gives_check[best_idx] = tc;
        }
    }

    int best = -(MATE_SCORE + 1);
    
    for (int i = 0; i < num_moves && i < 15; i++) {
        // Futility: Skip quiet moves when far behind
        if (futility_prune && scores[i] < 10000) continue;
        
        // LMR: Late quiet moves get reduced search
        int depth_reduction = 0;
        if (i >= 5 && scores[i] < 10000 && !move_gives_check[i]) {
            depth_reduction = 1; // Effectively search to depth 5 instead of 6
        }
        
        BoardState pos2 = *pos;
        make_move(&pos2, moves[i]);
        
        int score;
        if (depth_reduction > 0) {
            // Reduced search: call shallower depth
            score = -tactical_depth4(&pos2, -beta, -alpha, ply + 1);
            
            // If reduced search shows promise, re-search at full depth
            if (score > alpha) {
                // Re-search at depth 5 (manually iterate 1 more ply)
                Move moves2[MAX_MOVES];
                int num_moves2 = generate_legal_moves(&pos2, moves2);
                if (num_moves2 == 0) {
                    score = in_check(&pos2) ? (MATE_SCORE - ply - 1) : 0;
                } else {
                    int worst2 = MATE_SCORE + 1;
                    for (int j = 0; j < num_moves2 && j < 12; j++) {
                        BoardState pos3 = pos2;
                        make_move(&pos3, moves2[j]);
                        int s = -tactical_depth4(&pos3, -beta, -alpha, ply + 2);
                        if (s < worst2) worst2 = s;
                        if (worst2 <= -beta) break;
                    }
                    score = -worst2;
                }
            }
        } else {
            // Full depth: Iterate 2 plies then call depth4
            Move moves2[MAX_MOVES];
            int num_moves2 = generate_legal_moves(&pos2, moves2);
            
            if (num_moves2 == 0) {
                score = in_check(&pos2) ? (MATE_SCORE - ply - 1) : 0;
            } else {
                // Score ply 2 moves
                int scores2[MAX_MOVES];
                for (int j = 0; j < num_moves2; j++) {
                    scores2[j] = tactical_move_score(&pos2, moves2[j], gives_check_simple(&pos2, moves2[j]));
                }
                int sort_limit2 = (num_moves2 < 12) ? num_moves2 : 12;
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
                
                int worst2 = MATE_SCORE + 1;
                for (int j = 0; j < num_moves2 && j < 12; j++) {
                    BoardState pos3 = pos2;
                    make_move(&pos3, moves2[j]);
                    
                    // Call depth4 for remaining 4 plies (total 6 plies)
                    int s = -tactical_depth4(&pos3, -beta, -alpha, ply + 2);
                    
                    if (s < worst2) worst2 = s;
                    if (worst2 <= -beta) break;
                }
                score = -worst2;
            }
        }
        
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    
    return best;
}

// OLD ITERATIVE implementations kept as reference below
//
__device__ __noinline__
int tactical_depth4_old(BoardState* pos, int alpha, int beta, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);
    if (num_moves == 0) return in_check(pos) ? -(MATE_SCORE - ply) : 0;

    // Aggressive futility pruning for depth 4
    int static_eval = gpu_evaluate(pos);
    bool futility_prune = !in_check(pos) && static_eval + 1200 < alpha;

    // Score & sort top 20 moves only (reduced for speed)
    int scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        scores[i] = tactical_move_score(pos, moves[i], gives_check_simple(pos, moves[i]));
    }
    int sort_limit = (num_moves < 20) ? num_moves : 20;
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

    int best = -(MATE_SCORE + 1);
    
    // Ply 1
    for (int i = 0; i < num_moves && i < 20; i++) {
        if (futility_prune && scores[i] < 10000) continue;
        
        BoardState pos2 = *pos;
        make_move(&pos2, moves[i]);
        Move moves2[MAX_MOVES];
        int num_moves2 = generate_legal_moves(&pos2, moves2);
        
        int score;
        if (num_moves2 == 0) {
            score = in_check(&pos2) ? (MATE_SCORE - ply - 1) : 0;
        } else {
            // Score ply 2 moves
            int scores2[MAX_MOVES];
            for (int j = 0; j < num_moves2; j++) {
                scores2[j] = tactical_move_score(&pos2, moves2[j], gives_check_simple(&pos2, moves2[j]));
            }
            int sort_limit2 = (num_moves2 < 20) ? num_moves2 : 20;
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
            
            int worst2 = MATE_SCORE + 1;
            
            // Ply 2
            for (int j = 0; j < num_moves2 && j < 20; j++) {
                BoardState pos3 = pos2;
                make_move(&pos3, moves2[j]);
                Move moves3[MAX_MOVES];
                int num_moves3 = generate_legal_moves(&pos3, moves3);
                
                int s3;
                if (num_moves3 == 0) {
                    s3 = in_check(&pos3) ? -(MATE_SCORE - ply - 2) : 0;
                } else {
                    // Score ply 3 moves
                    int scores3[MAX_MOVES];
                    for (int k = 0; k < num_moves3; k++) {
                        scores3[k] = tactical_move_score(&pos3, moves3[k], gives_check_simple(&pos3, moves3[k]));
                    }
                    int sort_limit3 = (num_moves3 < 15) ? num_moves3 : 15;
                    for (int k = 0; k < sort_limit3; k++) {
                        int best_idx = k;
                        for (int m = k + 1; m < num_moves3; m++) {
                            if (scores3[m] > scores3[best_idx]) best_idx = m;
                        }
                        if (best_idx != k) {
                            Move tm = moves3[k]; moves3[k] = moves3[best_idx]; moves3[best_idx] = tm;
                            int ts = scores3[k]; scores3[k] = scores3[best_idx]; scores3[best_idx] = ts;
                        }
                    }
                    
                    int best3 = -(MATE_SCORE + 1);
                    
                    // Ply 3
                    for (int k = 0; k < num_moves3 && k < 15; k++) {
                        BoardState pos4 = pos3;
                        make_move(&pos4, moves3[k]);
                        Move moves4[MAX_MOVES];
                        int num_moves4 = generate_legal_moves(&pos4, moves4);
                        
                        int s4;
                        if (num_moves4 == 0) {
                            s4 = in_check(&pos4) ? (MATE_SCORE - ply - 3) : 0;
                        } else {
                            // Ply 4 - evaluate only
                            int worst4 = MATE_SCORE + 1;
                            for (int m = 0; m < num_moves4 && m < 10; m++) {
                                BoardState pos5 = pos4;
                                make_move(&pos5, moves4[m]);
                                int eval = gpu_evaluate(&pos5);
                                if (eval < worst4) worst4 = eval;
                            }
                            s4 = -worst4;
                        }
                        
                        if (s4 > best3) best3 = s4;
                        if (best3 >= -worst2 + 200) break; // Alpha-beta style cutoff
                    }
                    s3 = best3;
                }
                
                if (s3 < worst2) worst2 = s3;
                if (worst2 <= -beta) break;
            }
            score = -worst2;
        }
        
        if (score > best) best = score;
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    
    return best;
}

// ============================================================================
// TACTICAL DEPTH 6 - Iterative mate-in-6 solver (NO RECURSION)
// Very aggressive pruning for acceptable speed
// ============================================================================



// Simple mate-in-1 detection (called after making our move)
// Returns score from the side-to-move's perspective in pos
__device__ __forceinline__
int eval_position_simple(BoardState* pos, int ply) {
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(pos, moves);

    // Checkmate or stalemate
    if (num_moves == 0) {
        return in_check(pos) ? -(MATE_SCORE - ply) : 0;
    }

    // Not mate, just evaluate
    return gpu_evaluate(pos);
}

__global__ void TacticalSolver(
    const BoardState* __restrict__ positions,
    Move* __restrict__ best_moves,
    int* __restrict__ scores,
    int numPositions,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPositions) return;

    BoardState pos = positions[idx];

    // Generate legal moves
    Move moves[MAX_MOVES];
    int num_moves = generate_legal_moves(&pos, moves);

    if (num_moves == 0) {
        best_moves[idx] = 0;
        scores[idx] = 0;
        return;
    }

    // Score and sort moves
    int move_scores[MAX_MOVES];
    for (int i = 0; i < num_moves; i++) {
        bool gives_check = gives_check_simple(&pos, moves[i]);
        move_scores[i] = tactical_move_score(&pos, moves[i], gives_check);
    }

    // Sort top 20 moves
    int sort_limit = (num_moves < 20) ? num_moves : 20;
    for (int i = 0; i < sort_limit; i++) {
        int best_idx = i;
        for (int j = i + 1; j < num_moves; j++) {
            if (move_scores[j] > move_scores[best_idx]) best_idx = j;
        }
        if (best_idx != i) {
            Move tm = moves[i]; moves[i] = moves[best_idx]; moves[best_idx] = tm;
            int ts = move_scores[i]; move_scores[i] = move_scores[best_idx]; move_scores[best_idx] = ts;
        }
    }

    // Find best move
    Move best_move = moves[0];
    int best_score = -(MATE_SCORE + 1);

    // Search ALL moves (sorted moves first, then rest)
    for (int i = 0; i < num_moves; i++) {
        BoardState next_pos = pos;
        make_move(&next_pos, moves[i]);

        int score;
        int alpha = -(MATE_SCORE + 1);
        int beta = MATE_SCORE + 1;
        
        // Route to appropriate depth function (all iterative, no recursion)
        if (depth <= 1) {
            // Depth 1: immediate evaluation (mate-in-1 detection)
            score = -eval_position_simple(&next_pos, 1);
        } else if (depth <= 2) {
            // Depth 2: mate-in-2 (proven stable)
            score = -tactical_depth2(&next_pos, -beta, -alpha, 1);
        } else if (depth <= 4) {
            // Depth 4: mate-in-4 (iterative, no stack overflow)
            score = -tactical_depth4(&next_pos, -beta, -alpha, 1);
        } else {
            // Depth 6+: mate-in-5/6 (aggressive pruning for speed)
            score = -tactical_depth6(&next_pos, -beta, -alpha, 1);
        }

        // Sanity check for mate scores
        if (score >= MATE_SCORE - 10) {
            Move test_moves[MAX_MOVES];
            int test_count = generate_legal_moves(&next_pos, test_moves);
            bool is_check = in_check(&next_pos);
            if (test_count != 0 || !is_check) {
                // Not actually mate, use regular eval
                score = -gpu_evaluate(&next_pos);
            }
        }

        if (score > best_score) {
            best_score = score;
            best_move = moves[i];
        }
    }

    best_moves[idx] = best_move;
    scores[idx] = best_score;
}
