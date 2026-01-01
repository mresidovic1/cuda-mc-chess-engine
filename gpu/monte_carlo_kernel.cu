#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>

// Simple board representation for GPU (64 squares)
// Each square stores piece type and color encoded in single byte
// Format: [color(1bit)][piece_type(3bits)][unused(4bits)]
// 0 = empty, 1-6 = pieces (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)

#define EMPTY 0
#define PAWN 1
#define KNIGHT 2
#define BISHOP 3
#define ROOK 4
#define QUEEN 5
#define KING 6

#define WHITE_MASK 0x00
#define BLACK_MASK 0x80

#define GET_PIECE_TYPE(sq) ((sq) & 0x0F)
#define GET_COLOR(sq) ((sq) & 0x80)
#define MAKE_PIECE(type, color) ((type) | (color))

// Simple move structure
struct SimpleMove {
    uint8_t from;
    uint8_t to;
    uint8_t promotion; // 0 if no promotion
};

// Game result
enum GameResult {
    ONGOING = 0,
    WHITE_WIN = 1,
    BLACK_WIN = 2,
    DRAW = 3
};

// Board state for GPU
struct GPUBoard {
    uint8_t squares[64];
    uint8_t side_to_move; // 0 = white, 1 = black
    uint8_t castling_rights; // bits: KQkq
    uint8_t en_passant_file; // 0-7 or 255 for no ep
    uint16_t halfmove_clock;
    uint16_t fullmove_number;
};

// Device functions for move generation (simplified for random playouts)

__device__ bool is_valid_square(int sq) {
    return sq >= 0 && sq < 64;
}

__device__ int square_index(int rank, int file) {
    if (rank < 0 || rank > 7 || file < 0 || file > 7) return -1;
    return rank * 8 + file;
}

__device__ void get_rank_file(int sq, int& rank, int& file) {
    rank = sq / 8;
    file = sq % 8;
}

__device__ bool is_enemy(uint8_t piece, uint8_t my_color_mask) {
    if (GET_PIECE_TYPE(piece) == EMPTY) return false;
    return (GET_COLOR(piece) != my_color_mask);
}

__device__ bool is_friendly(uint8_t piece, uint8_t my_color_mask) {
    if (GET_PIECE_TYPE(piece) == EMPTY) return false;
    return (GET_COLOR(piece) == my_color_mask);
}

// Simplified move generation for random playouts
__device__ int generate_pseudo_legal_moves(GPUBoard* board, SimpleMove* moves, int max_moves) {
    int move_count = 0;
    uint8_t my_color = board->side_to_move == 0 ? WHITE_MASK : BLACK_MASK;
    int direction = (my_color == WHITE_MASK) ? 1 : -1;
    
    for (int sq = 0; sq < 64; sq++) {
        uint8_t piece = board->squares[sq];
        if (GET_PIECE_TYPE(piece) == EMPTY) continue;
        if (GET_COLOR(piece) != my_color) continue;
        
        int rank, file;
        get_rank_file(sq, rank, file);
        
        uint8_t piece_type = GET_PIECE_TYPE(piece);
        
        // Pawn moves (simplified - no en passant, no promotion for now)
        if (piece_type == PAWN) {
            int forward_sq = square_index(rank + direction, file);
            if (is_valid_square(forward_sq) && GET_PIECE_TYPE(board->squares[forward_sq]) == EMPTY) {
                if (move_count < max_moves) {
                    moves[move_count++] = {(uint8_t)sq, (uint8_t)forward_sq, 0};
                }
                
                // Double push from starting position
                int start_rank = (my_color == WHITE_MASK) ? 1 : 6;
                if (rank == start_rank) {
                    int double_sq = square_index(rank + 2 * direction, file);
                    if (is_valid_square(double_sq) && GET_PIECE_TYPE(board->squares[double_sq]) == EMPTY) {
                        if (move_count < max_moves) {
                            moves[move_count++] = {(uint8_t)sq, (uint8_t)double_sq, 0};
                        }
                    }
                }
            }
            
            // Pawn captures
            for (int df = -1; df <= 1; df += 2) {
                int capture_sq = square_index(rank + direction, file + df);
                if (is_valid_square(capture_sq) && is_enemy(board->squares[capture_sq], my_color)) {
                    if (move_count < max_moves) {
                        moves[move_count++] = {(uint8_t)sq, (uint8_t)capture_sq, 0};
                    }
                }
            }
        }
        
        // Knight moves
        else if (piece_type == KNIGHT) {
            int knight_offsets[8][2] = {{2,1}, {2,-1}, {-2,1}, {-2,-1}, {1,2}, {1,-2}, {-1,2}, {-1,-2}};
            for (int i = 0; i < 8; i++) {
                int to_sq = square_index(rank + knight_offsets[i][0], file + knight_offsets[i][1]);
                if (is_valid_square(to_sq) && !is_friendly(board->squares[to_sq], my_color)) {
                    if (move_count < max_moves) {
                        moves[move_count++] = {(uint8_t)sq, (uint8_t)to_sq, 0};
                    }
                }
            }
        }
        
        // King moves (no castling for now)
        else if (piece_type == KING) {
            for (int dr = -1; dr <= 1; dr++) {
                for (int df = -1; df <= 1; df++) {
                    if (dr == 0 && df == 0) continue;
                    int to_sq = square_index(rank + dr, file + df);
                    if (is_valid_square(to_sq) && !is_friendly(board->squares[to_sq], my_color)) {
                        if (move_count < max_moves) {
                            moves[move_count++] = {(uint8_t)sq, (uint8_t)to_sq, 0};
                        }
                    }
                }
            }
        }
        
        // Sliding pieces (Bishop, Rook, Queen)
        else if (piece_type == BISHOP || piece_type == ROOK || piece_type == QUEEN) {
            int directions[8][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}, {1,1}, {1,-1}, {-1,1}, {-1,-1}};
            int start_dir = (piece_type == BISHOP) ? 4 : 0;
            int end_dir = (piece_type == ROOK) ? 4 : 8;
            if (piece_type == QUEEN) { start_dir = 0; end_dir = 8; }
            
            for (int d = start_dir; d < end_dir; d++) {
                for (int dist = 1; dist <= 7; dist++) {
                    int to_sq = square_index(rank + dist * directions[d][0], file + dist * directions[d][1]);
                    if (!is_valid_square(to_sq)) break;
                    
                    uint8_t target = board->squares[to_sq];
                    if (is_friendly(target, my_color)) break;
                    
                    if (move_count < max_moves) {
                        moves[move_count++] = {(uint8_t)sq, (uint8_t)to_sq, 0};
                    }
                    
                    if (GET_PIECE_TYPE(target) != EMPTY) break; // Blocked by enemy
                }
            }
        }
    }
    
    return move_count;
}

__device__ void make_move(GPUBoard* board, SimpleMove move) {
    board->squares[move.to] = board->squares[move.from];
    board->squares[move.from] = EMPTY;
    board->side_to_move = 1 - board->side_to_move;
    board->halfmove_clock++;
    if (board->side_to_move == 0) board->fullmove_number++;
}

// Check if game is over (simplified - just check if there are legal moves)
__device__ GameResult check_game_over(GPUBoard* board, int move_count) {
    if (move_count == 0) {
        // No legal moves - either checkmate or stalemate
        // For simplicity, we'll count this as a draw in random playouts
        return DRAW;
    }
    
    // Check for 50-move rule
    if (board->halfmove_clock >= 100) {
        return DRAW;
    }
    
    // Check for insufficient material (simplified)
    int white_pieces = 0, black_pieces = 0;
    bool has_major_piece = false;
    
    for (int sq = 0; sq < 64; sq++) {
        uint8_t piece = board->squares[sq];
        uint8_t type = GET_PIECE_TYPE(piece);
        if (type == EMPTY) continue;
        
        if (GET_COLOR(piece) == WHITE_MASK) white_pieces++;
        else black_pieces++;
        
        if (type == PAWN || type == ROOK || type == QUEEN) has_major_piece = true;
    }
    
    // King vs King or King+minor vs King
    if (!has_major_piece && white_pieces <= 2 && black_pieces <= 2) {
        return DRAW;
    }
    
    return ONGOING;
}

// Main kernel: each thread simulates one game from the given position
__global__ void monte_carlo_playout_kernel(
    GPUBoard* initial_board,
    int* results,
    int num_simulations,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_simulations) return;
    
    // Initialize random state
    curandState state;
    curand_init(seed, tid, 0, &state);
    
    // Copy initial board to local memory
    GPUBoard board = *initial_board;
    
    // Maximum moves per game to avoid infinite loops
    const int MAX_MOVES_PER_GAME = 200;
    const int MAX_LEGAL_MOVES = 256;
    
    SimpleMove moves[MAX_LEGAL_MOVES];
    
    // Simulate game with random moves
    for (int ply = 0; ply < MAX_MOVES_PER_GAME; ply++) {
        int move_count = generate_pseudo_legal_moves(&board, moves, MAX_LEGAL_MOVES);
        
        GameResult result = check_game_over(&board, move_count);
        if (result != ONGOING) {
            results[tid] = result;
            return;
        }
        
        // Pick random move
        int move_idx = curand(&state) % move_count;
        make_move(&board, moves[move_idx]);
    }
    
    // If we reach max moves, call it a draw
    results[tid] = DRAW;
}

// Host function to launch kernel
extern "C" void launch_monte_carlo_kernel(
    GPUBoard* d_initial_board,
    int* d_results,
    int num_simulations,
    unsigned long long seed
) {
    int threads_per_block = 256;
    int num_blocks = (num_simulations + threads_per_block - 1) / threads_per_block;
    
    monte_carlo_playout_kernel<<<num_blocks, threads_per_block>>>(
        d_initial_board,
        d_results,
        num_simulations,
        seed
    );
    
    cudaDeviceSynchronize();
}
