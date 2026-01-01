#define GPU_CONST_DEF
#include "monte_carlo_advanced_kernel.cuh"
#include <curand_kernel.h>
#include <cstdint>

// ============================================================================
// Device Functions - Evaluation and Heuristics
// ============================================================================

__device__ int mirror_index(int index, int color) {
    return (color == GPU_WHITE) ? index : (index ^ 56);
}

__device__ int piece_square_value(int piece, int square, int color) {
    int piece_type = (piece >= B_PAWN) ? (piece - 8) : piece;
    piece_type--; // Convert to 0-5 index
    
    if (piece_type < 0 || piece_type > 5) return 0;
    
    int idx = mirror_index(square, color);
    
    switch(piece_type) {
        case 0: return d_pawn_table[idx];
        case 1: return d_knight_table[idx];
        case 2: return d_bishop_table[idx];
        case 3: return d_rook_table[idx];
        case 4: return d_queen_table[idx];
        case 5: return d_king_table[idx];
        default: return 0;
    }
}

__device__ int get_piece_value(int piece) {
    if (piece == EMPTY) return 0;
    int piece_type = (piece >= B_PAWN) ? (piece - 8) : piece;
    piece_type--; // Convert to 0-5 index
    if (piece_type < 0 || piece_type > 5) return 0;
    return d_piece_values[piece_type];
}

// Evaluate position from perspective of side to move
__device__ int evaluate_position(const Position& pos) {
    int score = 0;
    
    // Material and piece-square tables
    for (int sq = 0; sq < 64; sq++) {
        int piece = pos.board[sq];
        if (piece == EMPTY) continue;
        
        int piece_value = get_piece_value(piece);
        int piece_color = (piece >= B_PAWN) ? GPU_BLACK : GPU_WHITE;
        int pst_value = piece_square_value(piece, sq, piece_color);
        
        int total_value = piece_value + pst_value;
        
        if (piece_color == GPU_WHITE) {
            score += total_value;
        } else {
            score -= total_value;
        }
    }
    
    // Tempo bonus
    score += (pos.side_to_move == GPU_WHITE) ? 10 : -10;
    // Return from perspective of side to move
    return (pos.side_to_move == GPU_WHITE) ? score : -score;
}

// Simple Static Exchange Evaluation for a capture
__device__ int simple_SEE(const Position& pos, const Move& move) {
    if (move.capture == EMPTY) return 0;
    
    int attacker_value = get_piece_value(move.piece);
    int victim_value = get_piece_value(move.capture);
    
    // Simpler heuristic: 
    // Good captures: victim_value >= attacker_value
    // Bad captures: victim_value < attacker_value (but might still be okay)
    // Return the material gain, penalized if we're using a more valuable piece
    
    int see_value = victim_value;
    
    // If attacker is more valuable than victim, assume we'll lose the attacker
    // (unless it's a pawn capturing anything, which is usually safe)
    if (attacker_value > victim_value) {
        // Penalty for using valuable piece to capture less valuable one
        see_value -= attacker_value / 2;  // Assume some risk of losing our piece
    }
    
    return see_value;
}

// ============================================================================
// Move Generation (Simplified for GPU)
// ============================================================================


// Simple version: checks if any enemy piece attacks the square
__device__ bool is_square_attacked(const Position& pos, int square, int attacking_color) {
    // Pawn attacks
    int pawn = (attacking_color == GPU_WHITE) ? W_PAWN : B_PAWN;
    int pawn_dir = (attacking_color == GPU_WHITE) ? -8 : 8;
    int pawn_left = square + pawn_dir - 1;
    int pawn_right = square + pawn_dir + 1;
    int sq_rank = square / 8, sq_file = square % 8;
    if (pawn_left >= 0 && pawn_left < 64 && abs((pawn_left % 8) - sq_file) == 1 && pos.board[pawn_left] == pawn) return true;
    if (pawn_right >= 0 && pawn_right < 64 && abs((pawn_right % 8) - sq_file) == 1 && pos.board[pawn_right] == pawn) return true;

    // Knight attacks
    int knight = (attacking_color == GPU_WHITE) ? W_KNIGHT : B_KNIGHT;
    const int knight_offsets[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    for (int i = 0; i < 8; i++) {
        int to = square + knight_offsets[i];
        if (to >= 0 && to < 64 && pos.board[to] == knight) return true;
    }

    // King attacks
    int king = (attacking_color == GPU_WHITE) ? W_KING : B_KING;
    const int king_offsets[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
    for (int i = 0; i < 8; i++) {
        int to = square + king_offsets[i];
        if (to >= 0 && to < 64 && pos.board[to] == king) return true;
    }

    // Sliding pieces: bishop/queen (diagonals)
    int bishop = (attacking_color == GPU_WHITE) ? W_BISHOP : B_BISHOP;
    int queen = (attacking_color == GPU_WHITE) ? W_QUEEN : B_QUEEN;
    const int bishop_dirs[4] = {-9, -7, 7, 9};
    for (int d = 0; d < 4; d++) {
        int to = square;
        while (true) {
            int next = to + bishop_dirs[d];
            if (next < 0 || next >= 64 || abs((next % 8) - (to % 8)) > 2) break;
            to = next;
            int p = pos.board[to];
            if (p == bishop || p == queen) return true;
            if (p != EMPTY) break;
        }
    }

    // Sliding pieces: rook/queen (straight)
    int rook = (attacking_color == GPU_WHITE) ? W_ROOK : B_ROOK;
    const int rook_dirs[4] = {-8, -1, 1, 8};
    for (int d = 0; d < 4; d++) {
        int to = square;
        while (true) {
            int next = to + rook_dirs[d];
            if (next < 0 || next >= 64 || (d == 1 && (to % 8) == 0) || (d == 2 && (to % 8) == 7)) break;
            to = next;
            int p = pos.board[to];
            if (p == rook || p == queen) return true;
            if (p != EMPTY) break;
        }
    }

    return false;
}

__device__ void generate_pawn_moves(const Position& pos, Move* moves, int& move_count) {
    int color = pos.side_to_move;
    int direction = (color == GPU_WHITE) ? -8 : 8;
    int start_rank = (color == GPU_WHITE) ? 6 : 1;
    int promotion_rank = (color == GPU_WHITE) ? 0 : 7;
    
    for (int from = 0; from < 64; from++) {
        int piece = pos.board[from];
        int expected_pawn = (color == GPU_WHITE) ? W_PAWN : B_PAWN;
        
        if (piece != expected_pawn) continue;
        
        int rank = from / 8;
        int file = from % 8;
        
        // Single push
        int to = from + direction;
        if (to >= 0 && to < 64 && pos.board[to] == EMPTY) {
            if (to / 8 == promotion_rank) {
                // Promotions
                moves[move_count++] = {from, to, 5, EMPTY, piece, 150.0f}; // Queen
                moves[move_count++] = {from, to, 2, EMPTY, piece, 50.0f};  // Knight
            } else {
                moves[move_count++] = {from, to, 0, EMPTY, piece, 10.0f};
            }
            
            // Double push
            if (rank == start_rank) {
                int double_to = from + 2 * direction;
                if (pos.board[double_to] == EMPTY) {
                    moves[move_count++] = {from, double_to, 0, EMPTY, piece, 10.0f};
                }
            }
        }
        
        // Captures
        int capture_offsets[2] = {direction - 1, direction + 1};
        for (int offset : capture_offsets) {
            int to = from + offset;
            if (to < 0 || to >= 64) continue;
            
            int to_file = to % 8;
            if (abs(to_file - file) != 1) continue; // Not on adjacent file
            
            int target = pos.board[to];
            if (target != EMPTY && ((target >= B_PAWN) != (color == GPU_WHITE))) {
                if (to / 8 == promotion_rank) {
                    moves[move_count++] = {from, to, 5, target, piece, 200.0f}; // Capture + promote
                } else {
                    moves[move_count++] = {from, to, 0, target, piece, 100.0f + get_piece_value(target) * 0.1f};
                }
            }
        }
    }
}

__device__ void generate_knight_moves(const Position& pos, Move* moves, int& move_count) {
    int color = pos.side_to_move;
    int knight = (color == GPU_WHITE) ? W_KNIGHT : B_KNIGHT;
    
    const int offsets[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    
    for (int from = 0; from < 64; from++) {
        if (pos.board[from] != knight) continue;
        
        int from_rank = from / 8;
        int from_file = from % 8;
        
        for (int offset : offsets) {
            int to = from + offset;
            if (to < 0 || to >= 64) continue;
            
            int to_rank = to / 8;
            int to_file = to % 8;
            
            // Check knight move validity (max 2 squares in any direction)
            if (abs(to_rank - from_rank) > 2 || abs(to_file - from_file) > 2) continue;
            
            int target = pos.board[to];
            bool is_enemy = (target != EMPTY && ((target >= B_PAWN) != (color == GPU_WHITE)));
            bool is_empty = (target == EMPTY);
            
            if (is_empty) {
                moves[move_count++] = {from, to, 0, EMPTY, knight, 5.0f};
            } else if (is_enemy) {
                moves[move_count++] = {from, to, 0, target, knight, 100.0f + get_piece_value(target) * 0.1f};
            }
        }
    }
}

__device__ void generate_sliding_moves(const Position& pos, Move* moves, int& move_count, 
                                       int piece_type, const int* directions, int num_dirs) {
    int color = pos.side_to_move;
    int piece = (color == GPU_WHITE) ? piece_type : (piece_type + 8);
    
    for (int from = 0; from < 64; from++) {
        if (pos.board[from] != piece) continue;
        
        int from_rank = from / 8;
        int from_file = from % 8;
        
        for (int d = 0; d < num_dirs; d++) {
            int dir = directions[d];
            int to = from + dir;
            
            while (to >= 0 && to < 64) {
                int to_rank = to / 8;
                int to_file = to % 8;
                
                // Check if move wraps around board
                if (abs(to_rank - from_rank) > 7 || abs(to_file - from_file) > 7) break;
                if ((dir == -1 || dir == 1) && to_rank != from_rank) break; // Horizontal
                if ((dir == -8 || dir == 8) && to_file != from_file) break; // Vertical
                
                int target = pos.board[to];
                bool is_enemy = (target != EMPTY && ((target >= B_PAWN) != (color == GPU_WHITE)));
                bool is_empty = (target == EMPTY);
                
                if (is_empty) {
                    moves[move_count++] = {from, to, 0, EMPTY, piece, 5.0f};
                } else if (is_enemy) {
                    moves[move_count++] = {from, to, 0, target, piece, 100.0f + get_piece_value(target) * 0.1f};
                    break; // Can't move further after capture
                } else {
                    break; // Blocked by own piece
                }
                
                to += dir;
                from_rank = to_rank;
                from_file = to_file;
            }
        }
    }
}

__device__ void generate_king_moves(const Position& pos, Move* moves, int& move_count) {
    int color = pos.side_to_move;
    int king = (color == GPU_WHITE) ? W_KING : B_KING;
    
    const int offsets[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
    
    for (int from = 0; from < 64; from++) {
        if (pos.board[from] != king) continue;
        
        int from_rank = from / 8;
        int from_file = from % 8;
        
        for (int offset : offsets) {
            int to = from + offset;
            if (to < 0 || to >= 64) continue;
            
            int to_rank = to / 8;
            int to_file = to % 8;
            
            // Check king move validity (max 1 square)
            if (abs(to_rank - from_rank) > 1 || abs(to_file - from_file) > 1) continue;
            
            int target = pos.board[to];
            bool is_enemy = (target != EMPTY && ((target >= B_PAWN) != (color == GPU_WHITE)));
            bool is_empty = (target == EMPTY);
            
            if (is_empty) {
                moves[move_count++] = {from, to, 0, EMPTY, king, 5.0f};
            } else if (is_enemy) {
                moves[move_count++] = {from, to, 0, target, king, 100.0f + get_piece_value(target) * 0.1f};
            }
        }
    }
}

__device__ int generate_all_moves(const Position& pos, Move* moves) {
    int move_count = 0;
    
    generate_pawn_moves(pos, moves, move_count);
    generate_knight_moves(pos, moves, move_count);
    
    // Bishop
    const int bishop_dirs[4] = {-9, -7, 7, 9};
    generate_sliding_moves(pos, moves, move_count, W_BISHOP, bishop_dirs, 4);
    
    // Rook
    const int rook_dirs[4] = {-8, -1, 1, 8};
    generate_sliding_moves(pos, moves, move_count, W_ROOK, rook_dirs, 4);
    
    // Queen
    const int queen_dirs[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
    generate_sliding_moves(pos, moves, move_count, W_QUEEN, queen_dirs, 8);
    
    generate_king_moves(pos, moves, move_count);
    
    return move_count;
}

// ============================================================================
// Make/Unmake Move
// ============================================================================

__device__ void make_move(Position& pos, const Move& move) {
    // Move piece
    pos.board[move.to] = (move.promotion > 0) ? 
        ((pos.side_to_move == GPU_WHITE) ? move.promotion : (move.promotion + 8)) : 
        move.piece;
    pos.board[move.from] = EMPTY;
    
    // Switch side
    pos.side_to_move = 1 - pos.side_to_move;
    pos.halfmove_clock++;
    
    // Reset en passant
    pos.en_passant = -1;
}

// ============================================================================
// Game Over Detection
// ============================================================================

__device__ bool is_king_captured(const Position& pos, int color) {
    int king = (color == GPU_WHITE) ? W_KING : B_KING;
    for (int sq = 0; sq < 64; sq++) {
        if (pos.board[sq] == king) return false;
    }
    return true;
}

__device__ int find_king(const Position& pos, int color) {
    int king = (color == GPU_WHITE) ? W_KING : B_KING;
    for (int sq = 0; sq < 64; sq++) {
        if (pos.board[sq] == king) return sq;
    }
    return -1;
}

__device__ bool is_square_attacked(const Position& pos, int square, int attacking_color);

__device__ bool is_in_check(const Position& pos, int color) {
    int king_square = find_king(pos, color);
    if (king_square < 0) return false; // King not found (shouldn't happen)
    
    int attacking_color = 1 - color;
    return is_square_attacked(pos, king_square, attacking_color);
}

__device__ int check_game_over(const Position& pos, int num_moves) {
    const int MATE_SCORE = 10000;
    
    // Check if current player's king was captured = current player already lost
    int current_king = (pos.side_to_move == GPU_WHITE) ? W_KING : B_KING;
    bool king_exists = false;
    for (int sq = 0; sq < 64; sq++) {
        if (pos.board[sq] == current_king) {
            king_exists = true;
            break;
        }
    }
    
    if (!king_exists) {
        // Current player's king is captured = current player lost
        return -MATE_SCORE;  // Current player loses
    }
    
    // No legal moves
    if (num_moves == 0) {
        // Check if in check -> checkmate, else stalemate
        if (is_in_check(pos, pos.side_to_move)) {
            return -MATE_SCORE;  // Checkmate - current player loses
        } else {
            return 0;  // Stalemate - draw
        }
    }
    
    // Draw by 50-move rule
    if (pos.halfmove_clock >= 100) return 0;
    
    return 999999; // Game continues
}

// ============================================================================
// Smart Move Selection with Heuristics
// ============================================================================

__device__ void score_moves(Position& pos, Move* moves, int num_moves) {
    for (int i = 0; i < num_moves; i++) {
        Move& move = moves[i];
        float score = 0.0f;
        
        // 1. CAPTURES - most important for tactical play
        if (move.capture != EMPTY) {
            int see_score = simple_SEE(pos, move);
            // Good captures get huge bonus
            if (see_score > 0) {
                score += 1000.0f + see_score;
            } else {
                // Even bad captures might be okay in some positions
                score += 200.0f + see_score;
            }
        }
        
        // 2. PROMOTIONS - almost always good
        if (move.promotion > 0) {
            score += 900.0f;  // Queen promotion is huge
        }
        
        // 3. CHECKS - giving check is often strong
        // We'll approximate this by seeing if we're moving towards opponent king
        int opp_king_sq = find_king(pos, 1 - pos.side_to_move);
        if (opp_king_sq >= 0) {
            int from_distance = abs((move.from / 8) - (opp_king_sq / 8)) + abs((move.from % 8) - (opp_king_sq % 8));
            int to_distance = abs((move.to / 8) - (opp_king_sq / 8)) + abs((move.to % 8) - (opp_king_sq % 8));
            
            if (to_distance < from_distance) {
                score += 50.0f;  // Bonus for moving closer to enemy king
            }
        }
        
        // 4. Piece-square table improvement
        int from_pst = piece_square_value(move.piece, move.from, pos.side_to_move);
        int to_pst = piece_square_value(move.piece, move.to, pos.side_to_move);
        score += (to_pst - from_pst);
        
        // 5. Central control bonus
        int to_rank = move.to / 8;
        int to_file = move.to % 8;
        if ((to_rank >= 3 && to_rank <= 4) && (to_file >= 3 && to_file <= 4)) {
            score += 20.0f;  // Center squares bonus
        }
        
        // 6. Small random component for variety
        score += 1.0f;
        
        move.score = score;
    }
}

__device__ int select_move_weighted(Move* moves, int num_moves, curandState* rand_state) {
    if (num_moves == 0) return -1;
    if (num_moves == 1) return 0;
    
    // Find max score for numerical stability
    float max_score = moves[0].score;
    for (int i = 1; i < num_moves; i++) {
        if (moves[i].score > max_score) max_score = moves[i].score;
    }
    
    // Calculate softmax probabilities with temperature
    // Lower temperature = more greedy (picks best moves more often)
    // Higher temperature = more random exploration
    const float temperature = 0.5f;  // Reduced from 2.0 - now much more greedy
    float total = 0.0f;
    float probs[MAX_MOVES];
    
    for (int i = 0; i < num_moves; i++) {
        probs[i] = expf((moves[i].score - max_score) / temperature);
        total += probs[i];
    }
    
    // Normalize
    for (int i = 0; i < num_moves; i++) {
        probs[i] /= total;
    }
    
    // Select move by weighted random
    float r = curand_uniform(rand_state);
    float cumulative = 0.0f;
    
    for (int i = 0; i < num_moves; i++) {
        cumulative += probs[i];
        if (r < cumulative) return i;
    }
    
    return num_moves - 1;
}

// ============================================================================
// Monte Carlo Playout
// ============================================================================

__device__ int monte_carlo_playout(Position pos, curandState* rand_state) {
    Move moves[MAX_MOVES];
    int initial_side = pos.side_to_move;  // Remember who is to move at start
    
    for (int ply = 0; ply < MAX_PLAYOUT_MOVES; ply++) {
        int num_moves = generate_all_moves(pos, moves);
        
        // Check if game is over
        int game_result = check_game_over(pos, num_moves);
        if (game_result != 999999) {
            // Game ended - return from perspective of initial side
            // game_result is from perspective of current player
            // If current player == initial player, return as-is
            // If current player != initial player, negate
            return (pos.side_to_move == initial_side) ? game_result : -game_result;
        }
        
        // Score moves using heuristics
        score_moves(pos, moves, num_moves);
        
        // Select move using weighted random selection
        int selected = select_move_weighted(moves, num_moves, rand_state);
        if (selected < 0) break;
        
        make_move(pos, moves[selected]);
    }
    
    // Game didn't end - evaluate position
    int eval = evaluate_position(pos);
    // Return from perspective of initial side
    return (pos.side_to_move == initial_side) ? eval : -eval;
}

// ============================================================================
// Main CUDA Kernel - Simulate games for a specific root move
// ============================================================================

__global__ void monte_carlo_simulate_kernel(
    const Position root_position,
    const Move root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize random state
    curandState rand_state;
    curand_init(seed, idx, 0, &rand_state);
    
    float total_score = 0.0f;
    
    for (int sim = 0; sim < num_simulations_per_thread; sim++) {
        // Make root move
        Position pos = root_position;
        make_move(pos, root_move);
        
        // Run playout
        int score = monte_carlo_playout(pos, &rand_state);
        
        // Negate because we made root move (from root player's perspective)
        total_score += -score;
    }
    
    // Store average score
    results[idx] = total_score / num_simulations_per_thread;
}

// ============================================================================
// Launch Function (extern "C" for C++ linking)
// ============================================================================

extern "C" void launch_monte_carlo_simulate_kernel(
    const Position* root_position,
    const Move* root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed,
    int blocks,
    int threads_per_block
) {
    monte_carlo_simulate_kernel<<<blocks, threads_per_block>>>(
        *root_position,
        *root_move,
        num_simulations_per_thread,
        results,
        seed
    );
    cudaDeviceSynchronize();
}

// ============================================================================
// Helper function to copy constant data (called from host)
// ============================================================================

void initialize_gpu_constants() {
    // Constants are already defined at compile time, no need to copy
    // This function is here for future extensibility
}
