#include "../src/gpu_chess_types.cuh"
#include "../src/gpu_chess_position.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// Test kernel for flip_bitboard
__global__ void test_flip_squares_kernel(int* results) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Test individual square flips
        // A1 (0) should flip to A8 (56)
        Bitboard a1 = 1ULL << 0;
        Bitboard a1_flipped = flip_bitboard(a1);
        results[0] = __popcll(a1_flipped & (1ULL << 56)); // Should be 1
        
        // E1 (4) should flip to E8 (60)
        Bitboard e1 = 1ULL << 4;
        Bitboard e1_flipped = flip_bitboard(e1);
        results[1] = __popcll(e1_flipped & (1ULL << 60)); // Should be 1
        
        // H1 (7) should flip to H8 (63)
        Bitboard h1 = 1ULL << 7;
        Bitboard h1_flipped = flip_bitboard(h1);
        results[2] = __popcll(h1_flipped & (1ULL << 63)); // Should be 1
        
        // A8 (56) should flip to A1 (0)
        Bitboard a8 = 1ULL << 56;
        Bitboard a8_flipped = flip_bitboard(a8);
        results[3] = __popcll(a8_flipped & (1ULL << 0)); // Should be 1
        
        // Test rank 2 pawns (1ULL << 8 through 1ULL << 15) flip to rank 7
        Bitboard rank2 = 0xFF00ULL;  // Rank 2
        Bitboard rank2_flipped = flip_bitboard(rank2);
        results[4] = __popcll(rank2_flipped & 0x00FF000000000000ULL); // Should be 8
        
        // Print bitboards for visual verification
        printf("\\nFlip Bitboard Unit Tests:\\n");
        printf("A1 (0x%016llx) -> 0x%016llx (expect bit 56 set)\\n", a1, a1_flipped);
        printf("E1 (0x%016llx) -> 0x%016llx (expect bit 60 set)\\n", e1, e1_flipped);
        printf("H1 (0x%016llx) -> 0x%016llx (expect bit 63 set)\\n", h1, h1_flipped);
        printf("A8 (0x%016llx) -> 0x%016llx (expect bit 0 set)\\n", a8, a8_flipped);
        printf("Rank 2 (0x%016llx) -> 0x%016llx (expect rank 7)\\n", rank2, rank2_flipped);
    }
}

int main() {
    printf("GPU Chess Flip Unit Test\\n");
    printf("========================\\n\\n");
    
    int* d_results;
    cudaMalloc(&d_results, 5 * sizeof(int));
    
    test_flip_squares_kernel<<<1, 1>>>(d_results);
    
    int h_results[5];
    cudaMemcpy(h_results, d_results, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    printf("\nTest Results:\\n");
    printf("A1 -> A8: %s\\n", h_results[0] == 1 ? "PASS" : "FAIL");
    printf("E1 -> E8: %s\\n", h_results[1] == 1 ? "PASS" : "FAIL");
    printf("H1 -> H8: %s\\n", h_results[2] == 1 ? "PASS" : "FAIL");
    printf("A8 -> A1: %s\\n", h_results[3] == 1 ? "PASS" : "FAIL");
    printf("Rank 2 -> Rank 7: %s\\n", h_results[4] == 8 ? "PASS" : "FAIL");
    
    bool all_pass = (h_results[0] == 1 && h_results[1] == 1 && h_results[2] == 1 && 
                     h_results[3] == 1 && h_results[4] == 8);
    
    printf("\n%s\\n", all_pass ? "ALL TESTS PASSED!" : "SOME TESTS FAILED!");
    
    cudaFree(d_results);
    
    return all_pass ? 0 : 1;
}
