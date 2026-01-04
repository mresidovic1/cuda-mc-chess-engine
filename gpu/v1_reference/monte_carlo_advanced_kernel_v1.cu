//N - v1: Complete implementation with P1 and P2 improvements
#define GPU_CONST_DEF
#include "monte_carlo_advanced_kernel_v1.cuh"
#include <curand_kernel.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
//N - v1: Host pointers for cleanup (device memory defined in header)
// ============================================================================
// Host pointers for cleanup
static GPUTTEntry* h_tt_ptr = nullptr;
static MCTSNode* h_mcts_ptr = nullptr;
static bool g_initialized = false;

// ============================================================================
//N - v1: Zobrist hashing functions
// ============================================================================
__device__ uint64_t compute_hash(const CompactPosition& pos) {
    uint64_t hash = 0;
    for (int sq = 0; sq < 64; sq++) {
        int piece = pos.board[sq];
        if (piece != EMPTY) {
            hash ^= d_zobrist_pieces[sq][piece];
        }
    }
    if (POS_SIDE_TO_MOVE(pos) == GPU_BLACK) {
        hash ^= d_zobrist_side;
    }
    // Add castling rights to hash
    hash ^= d_zobrist_castling[POS_CASTLING(pos)];
    // Add en passant
    if (pos.en_passant >= 0) {
        hash ^= d_zobrist_ep[pos.en_passant % 8];
    }
    return hash;
}

__device__ uint64_t compute_hash_legacy(const Position& pos) {
    uint64_t hash = 0;
    for (int sq = 0; sq < 64; sq++) {
        int piece = pos.board[sq];
        if (piece != EMPTY) {
            hash ^= d_zobrist_pieces[sq][piece];
        }
    }
    if (pos.side_to_move == GPU_BLACK) {
        hash ^= d_zobrist_side;
    }
    return hash;
}

//N - v1: Incremental hash update
__device__ uint64_t update_hash(uint64_t hash, int from, int to,
                                 int moving_piece, int captured_piece, int color) {
    // Remove piece from source
    hash ^= d_zobrist_pieces[from][moving_piece];
    // Add piece to destination
    hash ^= d_zobrist_pieces[to][moving_piece];
    // Remove captured piece if any
    if (captured_piece != EMPTY) {
        hash ^= d_zobrist_pieces[to][captured_piece];
    }
    // Flip side to move
    hash ^= d_zobrist_side;
    return hash;
}

// ============================================================================
//N - v1: Transposition table operations
// ============================================================================
__device__ GPUTTEntry* tt_probe(uint64_t key) {
    if (d_transposition_table == nullptr) return nullptr;
    uint32_t idx = (uint32_t)(key & GPU_TT_MASK);
    return &d_transposition_table[idx];
}

__device__ void tt_store(uint64_t key, int16_t score, uint8_t depth,
                         uint8_t flag, uint8_t from, uint8_t to) {
    if (d_transposition_table == nullptr) return;
    uint32_t idx = (uint32_t)(key & GPU_TT_MASK);
    GPUTTEntry* entry = &d_transposition_table[idx];

    // Simple replacement: always replace if depth >= existing or different key
    if (entry->key != key || depth >= entry->depth) {
        entry->key = key;
        entry->score = score;
        entry->depth = depth;
        entry->flag = flag;
        entry->best_from = from;
        entry->best_to = to;
    }
}

// ============================================================================
//N - v1: History table operations with atomic updates
// ============================================================================
__device__ void history_update(int color, int from, int to, int bonus) {
    // Bounds check to prevent illegal memory access
    if (color < 0 || color > 1 || from < 0 || from >= 64 || to < 0 || to >= 64) {
        return;  // Silently ignore invalid inputs
    }

    // Clamp bonus
    bonus = max(-2000, min(2000, bonus));

    int old_val = d_history_table[color][from][to];
    int new_val = old_val + bonus;
    new_val = max(HISTORY_MIN, min(HISTORY_MAX, new_val));

    // Atomic update
    atomicExch(&d_history_table[color][from][to], new_val);
}

__device__ int history_get(int color, int from, int to) {
    // Bounds check to prevent illegal memory access
    if (color < 0 || color > 1 || from < 0 || from >= 64 || to < 0 || to >= 64) {
        return 0;  // Return neutral value for invalid inputs
    }
    return d_history_table[color][from][to];
}

// ============================================================================
// Device Functions - Position conversion
// ============================================================================
//N - v1: Convert legacy Position to CompactPosition
__device__ void position_to_compact(const Position& src, CompactPosition& dst) {
    for (int i = 0; i < 64; i++) {
        dst.board[i] = (int8_t)src.board[i];
    }
    dst.flags = (uint8_t)src.side_to_move;
    if (src.castling_rights[0]) dst.flags |= 0x02;
    if (src.castling_rights[1]) dst.flags |= 0x04;
    if (src.castling_rights[2]) dst.flags |= 0x08;
    if (src.castling_rights[3]) dst.flags |= 0x10;
    dst.en_passant = (int8_t)src.en_passant;
    dst.halfmove_clock = (uint8_t)min(255, src.halfmove_clock);
    dst.fullmove_low = (uint8_t)(src.fullmove_number & 0xFF);
}

// ============================================================================
// Device Functions - Evaluation and Heuristics
// ============================================================================
__device__ int mirror_index(int index, int color) {
    return (color == GPU_WHITE) ? index : (index ^ 56);
}

__device__ int piece_square_value(int piece, int square, int color) {
    int piece_type = (piece >= B_PAWN) ? (piece - 8) : piece;
    piece_type--;

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
    piece_type--;
    if (piece_type < 0 || piece_type > 5) return 0;
    return d_piece_values[piece_type];
}

//N - v1: Evaluate compact position
__device__ int evaluate_compact(const CompactPosition& pos) {
    int score = 0;

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
    score += (POS_SIDE_TO_MOVE(pos) == GPU_WHITE) ? 10 : -10;
    return (POS_SIDE_TO_MOVE(pos) == GPU_WHITE) ? score : -score;
}

__device__ int evaluate_position(const Position& pos) {
    int score = 0;

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

    score += (pos.side_to_move == GPU_WHITE) ? 10 : -10;
    return (pos.side_to_move == GPU_WHITE) ? score : -score;
}

__device__ int simple_SEE(const Position& pos, const Move& move) {
    if (move.capture == EMPTY) return 0;

    int attacker_value = get_piece_value(move.piece);
    int victim_value = get_piece_value(move.capture);

    int see_value = victim_value;

    if (attacker_value > victim_value) {
        see_value -= attacker_value / 2;
    }

    return see_value;
}

// ============================================================================
// Move Generation (unchanged from original, but included for completeness)
// ============================================================================
__device__ bool is_square_attacked(const Position& pos, int square, int attacking_color) {
    // Pawn attacks
    int pawn = (attacking_color == GPU_WHITE) ? W_PAWN : B_PAWN;
    int pawn_dir = (attacking_color == GPU_WHITE) ? -8 : 8;
    int pawn_left = square + pawn_dir - 1;
    int pawn_right = square + pawn_dir + 1;
    int sq_file = square % 8;
    if (pawn_left >= 0 && pawn_left < 64 && abs((pawn_left % 8) - sq_file) == 1 && pos.board[pawn_left] == pawn) return true;
    if (pawn_right >= 0 && pawn_right < 64 && abs((pawn_right % 8) - sq_file) == 1 && pos.board[pawn_right] == pawn) return true;

    // Knight attacks
    int knight = (attacking_color == GPU_WHITE) ? W_KNIGHT : B_KNIGHT;
    const int knight_offsets[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    for (int i = 0; i < 8; i++) {
        int to = square + knight_offsets[i];
        if (to >= 0 && to < 64) {
            int to_file = to % 8;
            int from_file = square % 8;
            if (abs(to_file - from_file) <= 2 && pos.board[to] == knight) return true;
        }
    }

    // King attacks
    int king = (attacking_color == GPU_WHITE) ? W_KING : B_KING;
    const int king_offsets[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
    for (int i = 0; i < 8; i++) {
        int to = square + king_offsets[i];
        if (to >= 0 && to < 64) {
            int to_file = to % 8;
            int from_file = square % 8;
            if (abs(to_file - from_file) <= 1 && pos.board[to] == king) return true;
        }
    }

    // Sliding pieces: bishop/queen (diagonals)
    int bishop = (attacking_color == GPU_WHITE) ? W_BISHOP : B_BISHOP;
    int queen = (attacking_color == GPU_WHITE) ? W_QUEEN : B_QUEEN;
    const int bishop_dirs[4] = {-9, -7, 7, 9};
    for (int d = 0; d < 4; d++) {
        int to = square;
        while (true) {
            int prev_file = to % 8;
            int next = to + bishop_dirs[d];
            if (next < 0 || next >= 64) break;
            int next_file = next % 8;
            if (abs(next_file - prev_file) != 1) break;
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
            int prev_file = to % 8;
            int next = to + rook_dirs[d];
            if (next < 0 || next >= 64) break;
            int next_file = next % 8;
            // Check for wrap-around on horizontal moves
            if ((rook_dirs[d] == -1 || rook_dirs[d] == 1) && abs(next_file - prev_file) != 1) break;
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
                moves[move_count++] = {from, to, 5, EMPTY, piece, 150.0f};
                moves[move_count++] = {from, to, 2, EMPTY, piece, 50.0f};
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
            if (abs(to_file - file) != 1) continue;

            int target = pos.board[to];
            if (target != EMPTY && ((target >= B_PAWN) != (color == GPU_WHITE))) {
                if (to / 8 == promotion_rank) {
                    moves[move_count++] = {from, to, 5, target, piece, 200.0f};
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

        int from_file = from % 8;

        for (int d = 0; d < num_dirs; d++) {
            int dir = directions[d];
            int to = from + dir;
            int prev_file = from_file;

            while (to >= 0 && to < 64) {
                int to_file = to % 8;

                // Check for horizontal wrap
                if ((dir == -1 || dir == 1) && abs(to_file - prev_file) != 1) break;
                // Check for diagonal wrap
                if ((dir == -9 || dir == -7 || dir == 7 || dir == 9) && abs(to_file - prev_file) != 1) break;

                int target = pos.board[to];
                bool is_enemy = (target != EMPTY && ((target >= B_PAWN) != (color == GPU_WHITE)));
                bool is_empty = (target == EMPTY);

                if (is_empty) {
                    moves[move_count++] = {from, to, 0, EMPTY, piece, 5.0f};
                } else if (is_enemy) {
                    moves[move_count++] = {from, to, 0, target, piece, 100.0f + get_piece_value(target) * 0.1f};
                    break;
                } else {
                    break;
                }

                prev_file = to_file;
                to += dir;
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

        int from_file = from % 8;

        for (int offset : offsets) {
            int to = from + offset;
            if (to < 0 || to >= 64) continue;

            int to_file = to % 8;

            if (abs(to_file - from_file) > 1) continue;

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

    const int bishop_dirs[4] = {-9, -7, 7, 9};
    generate_sliding_moves(pos, moves, move_count, W_BISHOP, bishop_dirs, 4);

    const int rook_dirs[4] = {-8, -1, 1, 8};
    generate_sliding_moves(pos, moves, move_count, W_ROOK, rook_dirs, 4);

    const int queen_dirs[8] = {-9, -8, -7, -1, 1, 7, 8, 9};
    generate_sliding_moves(pos, moves, move_count, W_QUEEN, queen_dirs, 8);

    generate_king_moves(pos, moves, move_count);

    return move_count;
}

//N - v1: Generate only captures for quiescence
__device__ int generate_captures(const Position& pos, Move* moves) {
    int move_count = 0;
    int color = pos.side_to_move;

    // Pawn captures
    int pawn = (color == GPU_WHITE) ? W_PAWN : B_PAWN;
    int direction = (color == GPU_WHITE) ? -8 : 8;
    int promotion_rank = (color == GPU_WHITE) ? 0 : 7;

    for (int from = 0; from < 64; from++) {
        if (pos.board[from] != pawn) continue;
        int file = from % 8;

        int capture_offsets[2] = {direction - 1, direction + 1};
        for (int offset : capture_offsets) {
            int to = from + offset;
            if (to < 0 || to >= 64) continue;
            if (abs((to % 8) - file) != 1) continue;

            int target = pos.board[to];
            if (target != EMPTY && ((target >= B_PAWN) != (color == GPU_WHITE))) {
                if (to / 8 == promotion_rank) {
                    moves[move_count++] = {from, to, 5, target, pawn, 200.0f};
                } else {
                    moves[move_count++] = {from, to, 0, target, pawn, 100.0f + get_piece_value(target) * 0.1f};
                }
            }
        }
    }

    // Generate captures for other pieces (simplified - check if target is enemy)
    Move all_moves[MAX_MOVES];
    int all_count = generate_all_moves(pos, all_moves);

    for (int i = 0; i < all_count; i++) {
        if (all_moves[i].capture != EMPTY) {
            moves[move_count++] = all_moves[i];
        }
    }

    return move_count;
}

// ============================================================================
// Make Move
// ============================================================================
__device__ void make_move(Position& pos, const Move& move) {
    pos.board[move.to] = (move.promotion > 0) ?
        ((pos.side_to_move == GPU_WHITE) ? move.promotion : (move.promotion + 8)) :
        move.piece;
    pos.board[move.from] = EMPTY;

    pos.side_to_move = 1 - pos.side_to_move;
    pos.halfmove_clock++;
    pos.en_passant = -1;
}

// ============================================================================
// Game Over Detection
// ============================================================================
__device__ int find_king(const Position& pos, int color) {
    int king = (color == GPU_WHITE) ? W_KING : B_KING;
    for (int sq = 0; sq < 64; sq++) {
        if (pos.board[sq] == king) return sq;
    }
    return -1;
}

__device__ bool is_in_check(const Position& pos, int color) {
    int king_square = find_king(pos, color);
    if (king_square < 0) return false;
    return is_square_attacked(pos, king_square, 1 - color);
}

__device__ int check_game_over(const Position& pos, int num_moves) {
    const int MATE_SCORE = 100000;  //N - v1: Increased for better mate detection

    int current_king = (pos.side_to_move == GPU_WHITE) ? W_KING : B_KING;
    bool king_exists = false;
    for (int sq = 0; sq < 64; sq++) {
        if (pos.board[sq] == current_king) {
            king_exists = true;
            break;
        }
    }

    if (!king_exists) {
        return -MATE_SCORE;
    }

    if (num_moves == 0) {
        if (is_in_check(pos, pos.side_to_move)) {
            return -MATE_SCORE;
        } else {
            return 0;
        }
    }

    if (pos.halfmove_clock >= 100) return 0;

    return 999999;
}

// ============================================================================
//N - v1: Move scoring with history heuristic
// ============================================================================
__device__ void score_moves(Position& pos, Move* moves, int num_moves) {
    for (int i = 0; i < num_moves; i++) {
        Move& move = moves[i];
        float score = 0.0f;

        // Check detection
        Position test_pos = pos;
        make_move(test_pos, move);
        bool gives_check = is_in_check(test_pos, test_pos.side_to_move);

        if (gives_check) {
            score += 5000.0f;
        }

        // Captures with SEE
        if (move.capture != EMPTY) {
            int see_score = simple_SEE(pos, move);
            if (see_score > 0) {
                score += 1000.0f + see_score;
            } else {
                score += 200.0f + see_score;
            }
        }

        // Promotions
        if (move.promotion > 0) {
            score += 900.0f;
        }

        //N - v1: History heuristic
        int hist_score = history_get(pos.side_to_move, move.from, move.to);
        score += hist_score / 100.0f;

        // King proximity
        int opp_king_sq = find_king(pos, 1 - pos.side_to_move);
        if (opp_king_sq >= 0) {
            int from_distance = abs((move.from / 8) - (opp_king_sq / 8)) + abs((move.from % 8) - (opp_king_sq % 8));
            int to_distance = abs((move.to / 8) - (opp_king_sq / 8)) + abs((move.to % 8) - (opp_king_sq % 8));

            if (to_distance < from_distance) {
                score += 100.0f;
            }
        }

        // PST improvement
        int from_pst = piece_square_value(move.piece, move.from, pos.side_to_move);
        int to_pst = piece_square_value(move.piece, move.to, pos.side_to_move);
        score += (to_pst - from_pst);

        // Center control
        int to_rank = move.to / 8;
        int to_file = move.to % 8;
        if ((to_rank >= 3 && to_rank <= 4) && (to_file >= 3 && to_file <= 4)) {
            score += 20.0f;
        }

        move.score = score;
    }
}

__device__ int select_move_weighted(Move* moves, int num_moves, curandState* rand_state) {
    if (num_moves == 0) return -1;
    if (num_moves == 1) return 0;

    float max_score = moves[0].score;
    for (int i = 1; i < num_moves; i++) {
        if (moves[i].score > max_score) max_score = moves[i].score;
    }

    const float temperature = 0.1f;
    float total = 0.0f;
    float probs[MAX_MOVES];

    for (int i = 0; i < num_moves; i++) {
        probs[i] = expf((moves[i].score - max_score) / temperature);
        total += probs[i];
    }

    for (int i = 0; i < num_moves; i++) {
        probs[i] /= total;
    }

    float r = curand_uniform(rand_state);
    float cumulative = 0.0f;

    for (int i = 0; i < num_moves; i++) {
        cumulative += probs[i];
        if (r < cumulative) return i;
    }

    return num_moves - 1;
}

// ============================================================================
//N - v1: Quiescence extension
// ============================================================================
__device__ int quiescence_extension(Position pos, int alpha, int beta, int depth, curandState* rand_state) {
    if (depth <= 0) return evaluate_position(pos);

    int stand_pat = evaluate_position(pos);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    Move captures[64];
    int num_captures = generate_captures(pos, captures);

    if (num_captures == 0) return stand_pat;

    // Score and sort captures
    score_moves(pos, captures, num_captures);

    // Sort by score descending
    for (int i = 0; i < num_captures - 1; i++) {
        for (int j = i + 1; j < num_captures; j++) {
            if (captures[j].score > captures[i].score) {
                Move temp = captures[i];
                captures[i] = captures[j];
                captures[j] = temp;
            }
        }
    }

    // Evaluate top captures
    int max_captures = min(6, num_captures);
    for (int i = 0; i < max_captures; i++) {
        // Skip bad captures
        if (simple_SEE(pos, captures[i]) < -50) continue;

        Position new_pos = pos;
        make_move(new_pos, captures[i]);

        int score = -quiescence_extension(new_pos, -beta, -alpha, depth - 1, rand_state);

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

// ============================================================================
// Monte Carlo Playout with improvements
// ============================================================================
__device__ int monte_carlo_playout(Position pos, curandState* rand_state, uint64_t* position_hashes, int* hash_count) {
    Move moves[MAX_MOVES];
    int initial_side = pos.side_to_move;
    const int MATE_SCORE = 100000;

    //N - v1: Track position hash for repetition detection
    uint64_t current_hash = compute_hash_legacy(pos);
    position_hashes[0] = current_hash;
    *hash_count = 1;

    for (int ply = 0; ply < MAX_PLAYOUT_MOVES; ply++) {
        int num_moves = generate_all_moves(pos, moves);

        int game_result = check_game_over(pos, num_moves);
        if (game_result != 999999) {
            return (pos.side_to_move == initial_side) ? game_result : -game_result;
        }

        //N - v1: Check transposition table
        GPUTTEntry* tt_entry = tt_probe(current_hash);
        if (tt_entry && tt_entry->key == current_hash && tt_entry->depth >= 3) {
            // Use TT score with some probability
            if (curand_uniform(rand_state) < 0.5f) {
                int tt_score = tt_entry->score;
                return (pos.side_to_move == initial_side) ? tt_score : -tt_score;
            }
        }

        score_moves(pos, moves, num_moves);

        // Find best move
        int best_idx = 0;
        float best_score = moves[0].score;
        for (int i = 1; i < num_moves; i++) {
            if (moves[i].score > best_score) {
                best_score = moves[i].score;
                best_idx = i;
            }
        }

        // Check for immediate mate
        Position test_pos = pos;
        make_move(test_pos, moves[best_idx]);

        Move test_moves[MAX_MOVES];
        int test_num_moves = generate_all_moves(test_pos, test_moves);
        if (test_num_moves == 0 && is_in_check(test_pos, test_pos.side_to_move)) {
            int mate_bonus = MATE_SCORE - ply;
            return (pos.side_to_move == initial_side) ? mate_bonus : -mate_bonus;
        }

        // Select move
        int selected;
        if (ply < 2) {
            selected = best_idx;
        } else {
            int top_count = min(3, num_moves);
            // Partial sort
            for (int i = 0; i < top_count - 1; i++) {
                for (int j = i + 1; j < num_moves; j++) {
                    if (moves[j].score > moves[i].score) {
                        Move temp = moves[i];
                        moves[i] = moves[j];
                        moves[j] = temp;
                    }
                }
            }
            int r = curand(rand_state) % top_count;
            selected = r;
        }

        make_move(pos, moves[selected]);

        //N - v1: Update hash incrementally
        current_hash = update_hash(current_hash, moves[selected].from, moves[selected].to,
                                   moves[selected].piece, moves[selected].capture, initial_side);

        //N - v1: Check for repetition
        for (int h = 0; h < *hash_count; h++) {
            if (position_hashes[h] == current_hash) {
                return 0;  // Draw by repetition
            }
        }

        if (*hash_count < 100) {
            position_hashes[*hash_count] = current_hash;
            (*hash_count)++;
        }

        //N - v1: Early termination for clearly decided positions
        if (ply > 30) {
            int eval = evaluate_position(pos);
            if (abs(eval) > 1000) {
                // Run quiescence to verify
                int q_score = quiescence_extension(pos, -30000, 30000,
                                                   MCTS_QUIESCENCE_EXTENSION, rand_state);
                return (pos.side_to_move == initial_side) ? q_score : -q_score;
            }
        }
    }

    //N - v1: Use quiescence at end of playout
    int eval = quiescence_extension(pos, -30000, 30000, MCTS_QUIESCENCE_EXTENSION, rand_state);
    return (pos.side_to_move == initial_side) ? eval : -eval;
}

// ============================================================================
// Legacy Kernel (single move, for backward compatibility)
// ============================================================================
__global__ void monte_carlo_simulate_kernel(
    const Position root_position,
    const Move root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //N - v1: Thread-local hash storage for repetition detection
    uint64_t position_hashes[100];
    int hash_count = 0;

    // Check immediate mate
    Position immediate_test = root_position;
    make_move(immediate_test, root_move);

    Move test_moves[MAX_MOVES];
    int test_num_moves = generate_all_moves(immediate_test, test_moves);
    if (test_num_moves == 0 && is_in_check(immediate_test, immediate_test.side_to_move)) {
        results[idx] = 1000000.0f;
        return;
    }

    // Check bonus for moves that give check
    bool gives_check = is_in_check(immediate_test, immediate_test.side_to_move);
    float check_bonus = gives_check ? 50.0f : 0.0f;  // Reduced to be more reasonable

    curandState rand_state;
    curand_init(seed, idx, 0, &rand_state);

    float total_score = 0.0f;

    for (int sim = 0; sim < num_simulations_per_thread; sim++) {
        Position pos = root_position;
        make_move(pos, root_move);

        hash_count = 0;
        int score = monte_carlo_playout(pos, &rand_state, position_hashes, &hash_count);

        // Negate score (playout returns opponent's perspective) and normalize
        total_score += (float)(-score) / 1000.0f;
    }

    // Average score plus check bonus
    results[idx] = (total_score / num_simulations_per_thread) + check_bonus;

    //N - v1: Update history for successful moves
    if (total_score > 0) {
        int bonus = (int)(total_score / 10);
        history_update(root_position.side_to_move, root_move.from, root_move.to, bonus);
    }
}

// ============================================================================
//N - v1: Batched kernel for parallel move evaluation
// ============================================================================
__global__ void monte_carlo_simulate_batch_kernel(
    const Position root_position,
    const Move* all_moves,
    int num_moves,
    int num_simulations_per_move,
    float* results,
    unsigned long long seed
) {
    // Grid layout: blockIdx.y = move index, blockIdx.x * blockDim.x + threadIdx.x = simulation index
    int move_idx = blockIdx.y;
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sims_per_move = gridDim.x * blockDim.x;

    if (move_idx >= num_moves) return;
    if (sim_idx >= sims_per_move) return;

    // Thread-local hash storage
    uint64_t position_hashes[100];
    int hash_count = 0;

    Move root_move = all_moves[move_idx];

    // DEBUG: Bounds check for move data
    if (root_move.from < 0 || root_move.from >= 64 ||
        root_move.to < 0 || root_move.to >= 64) {
        // Invalid move data - corrupted
        if (move_idx == 0 && sim_idx == 0) {
            printf("[KERNEL ERROR] Invalid move data: from=%d, to=%d, piece=%d\\n",
                   root_move.from, root_move.to, root_move.piece);
        }
        return;
    }

    // Initialize random state with better seed differentiation
    curandState rand_state;
    curand_init(seed, move_idx * sims_per_move + sim_idx, 0, &rand_state);

    // Check immediate mate
    Position immediate_test = root_position;
    make_move(immediate_test, root_move);

    Move test_moves[MAX_MOVES];
    int test_num_moves = generate_all_moves(immediate_test, test_moves);
    if (test_num_moves == 0 && is_in_check(immediate_test, immediate_test.side_to_move)) {
        // Immediate checkmate - all threads contribute to the massive positive score
        atomicAdd(&results[move_idx], 1000000.0f / sims_per_move);
        return;
    }

    // Check bonus for moves that give check
    bool gives_check = is_in_check(immediate_test, immediate_test.side_to_move);
    float check_bonus = gives_check ? 50.0f : 0.0f;  // Reduced from 5000 to be more reasonable

    // Run playout
    Position pos = root_position;
    make_move(pos, root_move);

    int score = monte_carlo_playout(pos, &rand_state, position_hashes, &hash_count);

    // The playout returns score from opponent's perspective
    // Positive score = opponent is winning = bad for us
    // So we negate it to get score from our perspective
    float normalized_score = (float)(-score) / 1000.0f;  // Normalize to roughly -1 to +1 range
    float result = (check_bonus + normalized_score) / sims_per_move;

    // Atomic add to accumulate results
    atomicAdd(&results[move_idx], result);
}

// ============================================================================
//N - v1: Warp-level reduction for faster result aggregation
// ============================================================================
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//N - v1: Block-level reduction using shared memory
__global__ void reduce_simulation_results(
    float* block_results,
    float* final_results,
    int num_blocks_per_move,
    int num_moves
) {
    extern __shared__ float shared_data[];

    int move_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (move_idx >= num_moves) return;

    // Load data
    float sum = 0.0f;
    int start_block = move_idx * num_blocks_per_move;
    for (int i = tid; i < num_blocks_per_move; i += blockDim.x) {
        sum += block_results[start_block + i];
    }

    shared_data[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        final_results[move_idx] = shared_data[0];
    }
}

// ============================================================================
//N - v1: UCB score calculation for MCTS
// ============================================================================
__device__ float calculate_ucb(MCTSNode* node, int parent_visits) {
    if (node->visits == 0) {
        return 1e10f;  // Unexplored - high priority
    }

    float exploitation = node->total_value / (node->visits + node->virtual_loss);
    float exploration = MCTS_EXPLORATION_CONSTANT *
                        sqrtf(logf((float)parent_visits + 1) / (node->visits + node->virtual_loss + 1));

    return exploitation + exploration;
}

// ============================================================================
//N - v1: MCTS tree kernel with UCB selection
// ============================================================================
__global__ void mcts_tree_kernel(
    const Position root_position,
    MCTSNode* tree_nodes,
    int* node_count,
    GPUTTEntry* transposition_table,
    int num_iterations,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState rand_state;
    curand_init(seed, tid, 0, &rand_state);

    // Thread-local storage
    uint64_t position_hashes[100];
    int hash_count = 0;

    for (int iter = 0; iter < num_iterations; iter++) {
        // ========== SELECTION ==========
        Position current_pos = root_position;
        int current_node = 0;  // Root
        int path[64];
        int path_length = 0;

        // Walk down tree using UCB
        while (tree_nodes[current_node].num_children > 0) {
            path[path_length++] = current_node;

            //N - v1: Add virtual loss
            atomicAdd(&tree_nodes[current_node].virtual_loss, MCTS_VIRTUAL_LOSS_VALUE);

            // Find best child by UCB
            int best_child = tree_nodes[current_node].first_child;
            float best_ucb = calculate_ucb(&tree_nodes[best_child], tree_nodes[current_node].visits);

            for (int c = 1; c < tree_nodes[current_node].num_children; c++) {
                int child_idx = tree_nodes[current_node].first_child + c;
                float ucb = calculate_ucb(&tree_nodes[child_idx], tree_nodes[current_node].visits);
                if (ucb > best_ucb) {
                    best_ucb = ucb;
                    best_child = child_idx;
                }
            }

            // Make the move
            Move child_move;
            child_move.from = tree_nodes[best_child].move_from;
            child_move.to = tree_nodes[best_child].move_to;
            child_move.piece = tree_nodes[best_child].move_piece;
            child_move.promotion = tree_nodes[best_child].move_promotion;
            child_move.capture = current_pos.board[child_move.to];

            make_move(current_pos, child_move);
            current_node = best_child;

            if (path_length >= 60) break;  // Safety limit
        }

        path[path_length++] = current_node;

        // ========== EXPANSION ==========
        if (tree_nodes[current_node].visits > 0 && tree_nodes[current_node].num_children == 0) {
            Move moves[MAX_MOVES];
            int num_moves = generate_all_moves(current_pos, moves);

            if (num_moves > 0) {
                // Allocate children
                int first_child = atomicAdd(node_count, num_moves);

                if (first_child + num_moves < MCTS_MAX_NODES) {
                    tree_nodes[current_node].first_child = first_child;
                    tree_nodes[current_node].num_children = num_moves;

                    for (int i = 0; i < num_moves; i++) {
                        MCTSNode* child = &tree_nodes[first_child + i];
                        child->parent = current_node;
                        child->first_child = -1;
                        child->num_children = 0;
                        child->visits = 0;
                        child->total_value = 0.0f;
                        child->virtual_loss = 0;
                        child->move_from = moves[i].from;
                        child->move_to = moves[i].to;
                        child->move_piece = moves[i].piece;
                        child->move_promotion = moves[i].promotion;
                    }

                    // Select first child for simulation
                    current_node = first_child;
                    Move first_move;
                    first_move.from = moves[0].from;
                    first_move.to = moves[0].to;
                    first_move.piece = moves[0].piece;
                    first_move.promotion = moves[0].promotion;
                    first_move.capture = current_pos.board[first_move.to];
                    make_move(current_pos, first_move);
                }
            }
        }

        // ========== SIMULATION ==========
        hash_count = 0;
        int simulation_result = monte_carlo_playout(current_pos, &rand_state, position_hashes, &hash_count);

        // Normalize to 0-1 range
        float value = (simulation_result + 10000.0f) / 20000.0f;
        value = fminf(1.0f, fmaxf(0.0f, value));

        // ========== BACKPROPAGATION ==========
        for (int i = path_length - 1; i >= 0; i--) {
            int node_idx = path[i];

            // Remove virtual loss
            atomicSub(&tree_nodes[node_idx].virtual_loss, MCTS_VIRTUAL_LOSS_VALUE);

            // Update statistics
            atomicAdd(&tree_nodes[node_idx].visits, 1);
            atomicAdd(&tree_nodes[node_idx].total_value, value);

            // Flip value for opponent's perspective
            value = 1.0f - value;
        }
    }
}

// ============================================================================
// CUDA Error Checking
// ============================================================================
#define CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err)); \
    } \
} while(0)

// ============================================================================
// Launch Functions
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
    // DEBUG: Reset any previous CUDA errors
    cudaError_t prev_err = cudaGetLastError();
    if (prev_err != cudaSuccess) {
        printf("[DEBUG] Cleared previous CUDA error before simulate kernel: %s\n", cudaGetErrorString(prev_err));
    }

    // Validate inputs
    if (results == nullptr) {
        printf("[ERROR] launch_monte_carlo_simulate_kernel: results is null!\n");
        return;
    }

    // Clear results before launch
    cudaError_t err = cudaMemset(results, 0, blocks * threads_per_block * sizeof(float));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemset failed in simulate kernel: %s\n", cudaGetErrorString(err));
        return;
    }

    monte_carlo_simulate_kernel<<<blocks, threads_per_block>>>(
        *root_position,
        *root_move,
        num_simulations_per_thread,
        results,
        seed
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Simulate kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] Simulate kernel sync failed: %s\n", cudaGetErrorString(err));
    }
}


//N - v1: Batched launch with improved parallelism
extern "C" void launch_monte_carlo_batch_kernel(
    const Position* root_position,
    const Move* all_moves,
    int num_moves,
    int simulations_per_move,
    float* results,
    unsigned long long seed
) {
    // DEBUG: Reset any previous CUDA errors
    cudaError_t prev_err = cudaGetLastError();
    if (prev_err != cudaSuccess) {
        printf("[DEBUG] Cleared previous CUDA error before batch kernel: %s\n", cudaGetErrorString(prev_err));
    }

    // DEBUG: Validate input parameters
    printf("[DEBUG] launch_monte_carlo_batch_kernel: num_moves=%d, sims_per_move=%d, results=%p\n",
           num_moves, simulations_per_move, (void*)results);

    if (num_moves <= 0 || num_moves > 256) {
        printf("[ERROR] Invalid num_moves: %d\n", num_moves);
        return;
    }
    if (results == nullptr) {
        printf("[ERROR] results pointer is null!\n");
        return;
    }

    // Clear results with error checking
    cudaError_t err = cudaMemset(results, 0, num_moves * sizeof(float));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemset(results) failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Calculate grid dimensions
    int threads_per_block = 256;
    int sims_per_block = threads_per_block;
    int blocks_per_move = (simulations_per_move + sims_per_block - 1) / sims_per_block;

    dim3 grid(blocks_per_move, num_moves);
    dim3 block(threads_per_block);

    printf("[DEBUG] Grid: (%d, %d), Block: %d\n", blocks_per_move, num_moves, threads_per_block);

    // Copy moves to device with error checking
    Move* d_moves = nullptr;
    err = cudaMalloc(&d_moves, num_moves * sizeof(Move));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMalloc(d_moves) failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[DEBUG] Allocated d_moves at %p\n", (void*)d_moves);

    err = cudaMemcpy(d_moves, all_moves, num_moves * sizeof(Move), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpy(d_moves) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_moves);
        return;
    }

    // DEBUG: Validate position data
    printf("[DEBUG] root_position: side_to_move=%d\n", root_position->side_to_move);

    // Launch batched kernel
    printf("[DEBUG] Launching batch kernel...\n");
    monte_carlo_simulate_batch_kernel<<<grid, block>>>(
        *root_position,
        d_moves,
        num_moves,
        simulations_per_move,
        results,
        seed
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
    }

    cudaFree(d_moves);
    printf("[DEBUG] Batch kernel completed\n");
}

//N - v1: Initialize GPU resources
extern "C" void initialize_gpu_resources() {
    if (g_initialized) {
        printf("[DEBUG] GPU resources already initialized, skipping\n");
        return;
    }

    printf("[DEBUG] Initializing GPU resources...\n");
    cudaError_t err;

    // Increase GPU stack size to 32KB per thread (default is usually 1KB)
    err = cudaDeviceSetLimit(cudaLimitStackSize, 32768);
    if (err != cudaSuccess) {
        printf("[ERROR] Failed to set stack size: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[DEBUG] Stack size set to 32KB\n");

    // Check GPU device
    int device;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        printf("[ERROR] cudaGetDevice failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[DEBUG] Using CUDA device %d\n", device);

    // Allocate transposition table
    printf("[DEBUG] Allocating TT: %zu bytes\n", GPU_TT_SIZE * sizeof(GPUTTEntry));
    err = cudaMalloc(&h_tt_ptr, GPU_TT_SIZE * sizeof(GPUTTEntry));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMalloc(TT) failed: %s\n", cudaGetErrorString(err));
        return;
    }
    printf("[DEBUG] TT allocated at %p\n", (void*)h_tt_ptr);

    err = cudaMemset(h_tt_ptr, 0, GPU_TT_SIZE * sizeof(GPUTTEntry));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemset(TT) failed: %s\n", cudaGetErrorString(err));
        cudaFree(h_tt_ptr);
        h_tt_ptr = nullptr;
        return;
    }

    err = cudaMemcpyToSymbol(d_transposition_table, &h_tt_ptr, sizeof(GPUTTEntry*));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpyToSymbol(d_transposition_table) failed: %s\n", cudaGetErrorString(err));
    }
    printf("[DEBUG] TT symbol set\n");

    // Allocate MCTS nodes
    printf("[DEBUG] Allocating MCTS nodes: %zu bytes\n", MCTS_MAX_NODES * sizeof(MCTSNode));
    err = cudaMalloc(&h_mcts_ptr, MCTS_MAX_NODES * sizeof(MCTSNode));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMalloc(MCTS) failed: %s\n", cudaGetErrorString(err));
        cudaFree(h_tt_ptr);
        h_tt_ptr = nullptr;
        return;
    }
    printf("[DEBUG] MCTS allocated at %p\n", (void*)h_mcts_ptr);

    err = cudaMemset(h_mcts_ptr, 0, MCTS_MAX_NODES * sizeof(MCTSNode));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemset(MCTS) failed: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpyToSymbol(d_mcts_nodes, &h_mcts_ptr, sizeof(MCTSNode*));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpyToSymbol(d_mcts_nodes) failed: %s\n", cudaGetErrorString(err));
    }
    printf("[DEBUG] MCTS symbol set\n");

    // Initialize Zobrist keys with fixed seed for reproducibility
    uint64_t h_zobrist_pieces[64][16];
    uint64_t h_zobrist_side;
    uint64_t h_zobrist_castling[16];
    uint64_t h_zobrist_ep[8];

    // Simple PRNG for initialization
    uint64_t state = 0x12345678ABCDEF01ULL;
    auto next_random = [&state]() -> uint64_t {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    };

    for (int sq = 0; sq < 64; sq++) {
        for (int piece = 0; piece < 16; piece++) {
            h_zobrist_pieces[sq][piece] = next_random();
        }
    }
    h_zobrist_side = next_random();
    for (int i = 0; i < 16; i++) {
        h_zobrist_castling[i] = next_random();
    }
    for (int i = 0; i < 8; i++) {
        h_zobrist_ep[i] = next_random();
    }

    err = cudaMemcpyToSymbol(d_zobrist_pieces, h_zobrist_pieces, sizeof(h_zobrist_pieces));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpyToSymbol(d_zobrist_pieces) failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpyToSymbol(d_zobrist_side, &h_zobrist_side, sizeof(h_zobrist_side));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpyToSymbol(d_zobrist_side) failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpyToSymbol(d_zobrist_castling, h_zobrist_castling, sizeof(h_zobrist_castling));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpyToSymbol(d_zobrist_castling) failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpyToSymbol(d_zobrist_ep, h_zobrist_ep, sizeof(h_zobrist_ep));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpyToSymbol(d_zobrist_ep) failed: %s\n", cudaGetErrorString(err));
    }
    printf("[DEBUG] Zobrist keys initialized\n");

    // Clear history table
    int zero_history[2][64][64] = {{{0}}};
    err = cudaMemcpyToSymbol(d_history_table, zero_history, sizeof(zero_history));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpyToSymbol(d_history_table) failed: %s\n", cudaGetErrorString(err));
    }
    printf("[DEBUG] History table cleared\n");

    // Final sync and error check
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] Final sync failed: %s\n", cudaGetErrorString(err));
    }

    g_initialized = true;
    printf("[GPU] Resources initialized: TT=%zuMB, MCTS=%zuMB\n",
           GPU_TT_SIZE * sizeof(GPUTTEntry) / (1024*1024),
           MCTS_MAX_NODES * sizeof(MCTSNode) / (1024*1024));
}

extern "C" void cleanup_gpu_resources() {
    if (!g_initialized) return;

    if (h_tt_ptr) {
        cudaFree(h_tt_ptr);
        h_tt_ptr = nullptr;
    }
    if (h_mcts_ptr) {
        cudaFree(h_mcts_ptr);
        h_mcts_ptr = nullptr;
    }

    g_initialized = false;
}

extern "C" void clear_gpu_transposition_table() {
    if (h_tt_ptr) {
        cudaMemset(h_tt_ptr, 0, GPU_TT_SIZE * sizeof(GPUTTEntry));
    }
}

extern "C" void clear_gpu_history_table() {
    int zero_history[2][64][64] = {{{0}}};
    cudaMemcpyToSymbol(d_history_table, zero_history, sizeof(zero_history));
}

void initialize_gpu_constants() {
    // Constants are defined at compile time
    // This function initializes runtime resources
    initialize_gpu_resources();
}
