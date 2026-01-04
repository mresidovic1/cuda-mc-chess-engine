#include "../src/gpu_chess_types.cuh"
#include "../src/gpu_chess_position.cuh"
#include "../src/gpu_chess_movegen.cuh"
#include <cstdio>
#include <cuda_runtime.h>

extern "C" void init_startpos(Position* pos);

__global__ void test_phantom_castle_kernel(int* results) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        Position pos;
        
        // 1. Setup a position where White is to move, has Castling Rights, but NO Rook at H1
        pos.side_to_move = WHITE;
        pos.castling = CASTLE_WK; // White King Side only
        pos.halfmove = 0;
        pos.ep_square = -1;
        
        // Clear board
        for(int i=0; i<12; i++) pos.pieces[i] = 0;
        for(int i=0; i<3; i++) pos.occupied[i] = 0;
        
        // Place White King at E1 (4)
        pos.pieces[WHITE*6 + KING] = (1ULL << 4);
        pos.occupied[WHITE] |= (1ULL << 4);
        
        // Place Black King at E8 (60)
        pos.pieces[BLACK*6 + KING] = (1ULL << 60);
        pos.occupied[BLACK] |= (1ULL << 60);
        
        // NO WHITE ROOK at H1 (7).
        // Ensure path E1-H1 is empty (F1, G1 empty)
        // pos.occupied is handled above.
        
        pos.occupied[2] = pos.occupied[WHITE] | pos.occupied[BLACK];
        
        // Generate moves
        Move moves[256];
        int num_moves = generate_moves(&pos, moves);
        
        // Check if King Castle (E1->G1) is generated
        bool found_castle = false;
        for (int i = 0; i < num_moves; i++) {
            if (move_flags(moves[i]) == KING_CASTLE) {
                found_castle = true;
                printf("ERROR: Generated Phantom Castle! Move index %d\n", i);
            }
        }
        
        results[0] = found_castle ? 1 : 0; // 1 means BUG FOUND
        
        printf("Test 1 (Missing Rook): found_castle = %d (Expected 0)\n", found_castle);

        // ---------------------------------------------------------
        
        // 2. Test Capture Logic Bug
        // Setup a position where we capture a rook, and see if flags update properly
        // Note: apply_move flips the board, so we check the state *after* flip.
        
        Position pos2;
        // White to move. White Rook at A1. Black Rook at A8.
        // We will simulate a capture of Black Rook at A8 by a White piece (e.g. Rook at A1 captures A8)
        // Actually, let's keep it simple.
        
        pos2.side_to_move = WHITE;
        pos2.castling = CASTLE_BK | CASTLE_BQ; // Black has rights
        pos2.ep_square = -1;
        
        // White Rook at A7 (48), Black Rook at A8 (56)
        pos2.pieces[WHITE*6 + ROOK] = (1ULL << 48);
        pos2.pieces[BLACK*6 + ROOK] = (1ULL << 56);
        pos2.occupied[WHITE] = (1ULL << 48);
        pos2.occupied[BLACK] = (1ULL << 56); // And Kings...
        
        pos2.pieces[WHITE*6 + KING] = (1ULL << 4);
        pos2.occupied[WHITE] |= (1ULL << 4);
        pos2.pieces[BLACK*6 + KING] = (1ULL << 60);
        pos2.occupied[BLACK] |= (1ULL << 60);
        
        pos2.occupied[2] = pos2.occupied[WHITE] | pos2.occupied[BLACK];
        
        // Move: A7 captures A8
        Move capture = make_move(48, 56, CAPTURE); 
        
        printf("\nTest 2 (Capture Updates): Applying capture A7xRook(A8)...\n");
        printf("Before: Castling = 0x%02x\n", pos2.castling);
        
        apply_move(&pos2, capture);
        
        // After apply_move, the board is flipped.
        // Old Black becomes new White (us).
        // Old Black had CASTLE_BK | CASTLE_BQ (bits 2 and 3 -> 4, 8) => 0x0C (12)
        // Wait, constants: WK=1, WQ=2, BK=4, BQ=8.
        // Initial: 0x0C.
        // Capture at A8 (Queenside for Black). Should lose BQ (8). Result 0x04.
        
        // After flip:
        // BK (4) -> WK (1)
        // BQ (8) -> WQ (2)
        // So if correct, we expect new castling to be 0x01 (WK only).
        // If BUG, we stripped nothing, so we have BK|BQ -> WK|WQ = 0x03.
        
        printf("After (Flipped): Castling = 0x%02x\n", pos2.castling);
        
        // Check if we still have WQ (bit 2) set.
        // Remember: The capture was at A8 (Black's Queenside Rook). So BQ should be lost.
        // Flipped: BQ maps to WQ.
        // So if (pos2.castling & CASTLE_WQ) is set, it's a BUG.
        
        results[1] = (pos2.castling & CASTLE_WQ) ? 1 : 0;
        printf("Test 2 Result: Has WQ flag? %d (Expected 0)\n", results[1]);
    }
}

int main() {
    printf("GPU Chess Castling Unit Tests\n");
    printf("=============================\n\n");
    
    int* d_results;
    cudaMalloc(&d_results, 2 * sizeof(int));
    
    test_phantom_castle_kernel<<<1, 1>>>(d_results);
    
    int h_results[2];
    cudaMemcpy(h_results, d_results, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    printf("\nSummary:\n");
    printf("Test 1 (Generator check w/o Rook): %s\n", h_results[0] == 1 ? "FAILED (Generated Illegal Castle)" : "PASSED");
    printf("Test 2 (Capture Logic):            %s\n", h_results[1] == 1 ? "FAILED (Did not clear flag)" : "PASSED");
    
    bool all_pass = (h_results[0] == 0 && h_results[1] == 0);
    return all_pass ? 0 : 1;
}
