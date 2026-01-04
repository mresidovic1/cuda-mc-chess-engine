// Source files are now in src/ directory - use local includes
#include "gpu_chess_types.cuh"
#include "gpu_chess_position.cuh"
#include "gpu_chess_movegen.cuh"
#include <cstdio>

// ============================================================================
// Forward Declarations
// ============================================================================

__device__ unsigned long long perft(Position* pos, int depth);

// ============================================================================
// Test Kernel: Perft (Performance Test)
// ============================================================================

__global__ void perft_kernel(Position* pos, int depth, unsigned long long* result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *result = perft(pos, depth);
    }
}

__device__ unsigned long long perft(Position* pos, int depth) {
    if (depth == 0) return 1;

    Move moves[256];
    int num_moves = generate_moves(pos, moves);

    if (depth == 1) return num_moves;

    unsigned long long nodes = 0;
    for (int i = 0; i < num_moves; i++) {
        // Make a copy of position
        Position new_pos = *pos;
        apply_move(&new_pos, moves[i]);
        nodes += perft(&new_pos, depth - 1);
    }

    return nodes;
}

// Divide perft: returns counts for each move individually
__device__ void perft_divide(Position* pos, int depth, unsigned long long* move_counts, Move* moves_out, int* num_moves_out) {
    Move moves[256];
    int num_moves = generate_moves(pos, moves);
    *num_moves_out = num_moves;

    for (int i = 0; i < num_moves; i++) {
        moves_out[i] = moves[i];
        if (depth == 1) {
            move_counts[i] = 1;
        } else {
            Position new_pos = *pos;
            apply_move(&new_pos, moves[i]);
            move_counts[i] = perft(&new_pos, depth - 1);
        }
    }
}

__global__ void perft_divide_kernel(Position* pos, int depth, unsigned long long* move_counts, Move* moves_out, int* num_moves_out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        perft_divide(pos, depth, move_counts, moves_out, num_moves_out);
    }
}

// Test bitboard flip
__global__ void test_flip_kernel(Bitboard* input, Bitboard* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = flip_bitboard(*input);
    }
}

// ============================================================================
// Test Kernel: Move Generation Count
// ============================================================================

__global__ void test_movegen_kernel(Position* d_pos, int* d_move_count, Move* d_moves) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_move_count = generate_moves(d_pos, d_moves);
    }
}

// ============================================================================
// Host Launch Functions
// ============================================================================

extern "C" unsigned long long run_perft(Position* h_pos, int depth) {
    Position* d_pos;
    unsigned long long* d_result;
    
    cudaMalloc(&d_pos, sizeof(Position));
    cudaMalloc(&d_result, sizeof(unsigned long long));
    
    cudaMemcpy(d_pos, h_pos, sizeof(Position), cudaMemcpyHostToDevice);
    
    perft_kernel<<<1, 1>>>(d_pos, depth, d_result);
    
    unsigned long long h_result;
    cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    
    cudaFree(d_pos);
    cudaFree(d_result);
    
    return h_result;
}

extern "C" int test_move_generation(Position* h_pos, Move* h_moves) {
    Position* d_pos;
    int* d_move_count;
    Move* d_moves;

    cudaMalloc(&d_pos, sizeof(Position));
    cudaMalloc(&d_move_count, sizeof(int));
    cudaMalloc(&d_moves, 256 * sizeof(Move));

    cudaMemcpy(d_pos, h_pos, sizeof(Position), cudaMemcpyHostToDevice);

    test_movegen_kernel<<<1, 1>>>(d_pos, d_move_count, d_moves);

    int h_move_count;
    cudaMemcpy(&h_move_count, d_move_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_moves, d_moves, h_move_count * sizeof(Move), cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_move_count);
    cudaFree(d_moves);

    return h_move_count;
}

extern "C" void run_perft_divide(Position* h_pos, int depth, unsigned long long* h_counts, Move* h_moves, int* h_num_moves) {
    Position* d_pos;
    unsigned long long* d_counts;
    Move* d_moves;
    int* d_num_moves;

    cudaMalloc(&d_pos, sizeof(Position));
    cudaMalloc(&d_counts, 256 * sizeof(unsigned long long));
    cudaMalloc(&d_moves, 256 * sizeof(Move));
    cudaMalloc(&d_num_moves, sizeof(int));

    cudaMemcpy(d_pos, h_pos, sizeof(Position), cudaMemcpyHostToDevice);

    perft_divide_kernel<<<1, 1>>>(d_pos, depth, d_counts, d_moves, d_num_moves);

    cudaMemcpy(h_num_moves, d_num_moves, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, d_counts, (*h_num_moves) * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_moves, d_moves, (*h_num_moves) * sizeof(Move), cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_counts);
    cudaFree(d_moves);
    cudaFree(d_num_moves);
}
