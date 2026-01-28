// chess_types.cuh - Core data structures for GPU chess engine
// Optimized for shared memory usage (< 256 bytes)

#ifndef CHESS_TYPES_CUH
#define CHESS_TYPES_CUH

#include <cstdint>
#include <cuda_runtime.h>

// Type definitions
typedef uint64_t Bitboard;
typedef uint16_t Move;
typedef uint8_t  Square;
typedef uint8_t  Piece;
typedef uint8_t  Color;

// Constants
#define WHITE 0
#define BLACK 1

#define PAWN   0
#define KNIGHT 1
#define BISHOP 2
#define ROOK   3
#define QUEEN  4
#define KING   5
#define NO_PIECE 6

// Castling rights flags
#define CASTLE_WK 0x01  // White kingside
#define CASTLE_WQ 0x02  // White queenside
#define CASTLE_BK 0x04  // Black kingside
#define CASTLE_BQ 0x08  // Black queenside

// Move flags (4 bits)
#define MOVE_QUIET       0x0
#define MOVE_DOUBLE_PUSH 0x1
#define MOVE_KING_CASTLE 0x2
#define MOVE_QUEEN_CASTLE 0x3
#define MOVE_CAPTURE     0x4
#define MOVE_EP_CAPTURE  0x5
#define MOVE_PROMO_N     0x8
#define MOVE_PROMO_B     0x9
#define MOVE_PROMO_R     0xA
#define MOVE_PROMO_Q     0xB
#define MOVE_PROMO_CAP_N 0xC
#define MOVE_PROMO_CAP_B 0xD
#define MOVE_PROMO_CAP_R 0xE
#define MOVE_PROMO_CAP_Q 0xF

// Game result
#define RESULT_ONGOING  0
#define RESULT_WHITE_WIN 1
#define RESULT_BLACK_WIN 2
#define RESULT_DRAW     3

// Bitboard constants
#define C64(x) x##ULL

#define RANK_1 C64(0x00000000000000FF)
#define RANK_2 C64(0x000000000000FF00)
#define RANK_3 C64(0x0000000000FF0000)
#define RANK_4 C64(0x00000000FF000000)
#define RANK_5 C64(0x000000FF00000000)
#define RANK_6 C64(0x0000FF0000000000)
#define RANK_7 C64(0x00FF000000000000)
#define RANK_8 C64(0xFF00000000000000)

#define FILE_A C64(0x0101010101010101)
#define FILE_B C64(0x0202020202020202)
#define FILE_C C64(0x0404040404040404)
#define FILE_D C64(0x0808080808080808)
#define FILE_E C64(0x1010101010101010)
#define FILE_F C64(0x2020202020202020)
#define FILE_G C64(0x4040404040404040)
#define FILE_H C64(0x8080808080808080)

#define EMPTY C64(0x0000000000000000)
#define ALL_SQUARES C64(0xFFFFFFFFFFFFFFFF)

// Square indices
enum SquareIndex {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8
};

struct BoardState {
    // Piece bitboards (48 bytes)
    Bitboard pieces[2][6];  // [color][piece_type]

    // Aggregate bitboards computed on demand or cached
    // Not stored to save space - computed as needed

    // Game state (8 bytes)
    uint8_t side_to_move;    // 0 = white, 1 = black
    uint8_t castling;        // Castling rights
    int8_t  ep_square;       // En passant target square (-1 if none)
    uint8_t halfmove;        // Halfmove clock for 50-move rule
    uint8_t result;          // Game result
    uint8_t padding[3];      // Pad to 8 bytes

    // Total: 96 + 8 = 104 bytes (fits well in shared memory, < 256 bytes)

    __host__ __device__ __forceinline__
    Bitboard occupied() const {
        return pieces[WHITE][PAWN] | pieces[WHITE][KNIGHT] | pieces[WHITE][BISHOP] |
               pieces[WHITE][ROOK] | pieces[WHITE][QUEEN]  | pieces[WHITE][KING]   |
               pieces[BLACK][PAWN] | pieces[BLACK][KNIGHT] | pieces[BLACK][BISHOP] |
               pieces[BLACK][ROOK] | pieces[BLACK][QUEEN]  | pieces[BLACK][KING];
    }

    __host__ __device__ __forceinline__
    Bitboard us() const {
        return pieces[side_to_move][PAWN]   | pieces[side_to_move][KNIGHT] |
               pieces[side_to_move][BISHOP] | pieces[side_to_move][ROOK]   |
               pieces[side_to_move][QUEEN]  | pieces[side_to_move][KING];
    }

    __host__ __device__ __forceinline__
    Bitboard them() const {
        int opp = side_to_move ^ 1;
        return pieces[opp][PAWN]   | pieces[opp][KNIGHT] |
               pieces[opp][BISHOP] | pieces[opp][ROOK]   |
               pieces[opp][QUEEN]  | pieces[opp][KING];
    }

    __host__ __device__ __forceinline__
    Bitboard color_pieces(int color) const {
        return pieces[color][PAWN]   | pieces[color][KNIGHT] |
               pieces[color][BISHOP] | pieces[color][ROOK]   |
               pieces[color][QUEEN]  | pieces[color][KING];
    }
};

static_assert(sizeof(BoardState) <= 256, "BoardState exceeds 256 bytes!");


__host__ __device__ __forceinline__
Move encode_move(Square from, Square to, uint8_t flags = MOVE_QUIET) {
    return (Move)((flags << 12) | (to << 6) | from);
}

__host__ __device__ __forceinline__
Square move_from(Move m) {
    return m & 0x3F;
}

__host__ __device__ __forceinline__
Square move_to(Move m) {
    return (m >> 6) & 0x3F;
}

__host__ __device__ __forceinline__
uint8_t move_flags(Move m) {
    return (m >> 12) & 0xF;
}

__host__ __device__ __forceinline__
bool is_capture(Move m) {
    uint8_t flags = move_flags(m);
    return (flags & MOVE_CAPTURE) || flags == MOVE_EP_CAPTURE;
}

__host__ __device__ __forceinline__
bool is_promotion(Move m) {
    return move_flags(m) >= MOVE_PROMO_N;
}

__host__ __device__ __forceinline__
Piece promotion_piece(Move m) {
    return (move_flags(m) & 0x3) + KNIGHT;  // N=0, B=1, R=2, Q=3 -> KNIGHT...QUEEN
}

#define MAX_MOVES 256


#ifdef _MSC_VER
#include <intrin.h>
#endif

__host__ __device__ __forceinline__
int popcount(Bitboard b) {
#ifdef __CUDA_ARCH__
    return __popcll(b);
#elif defined(_MSC_VER)
    return (int)__popcnt64(b);
#else
    return __builtin_popcountll(b);
#endif
}

__host__ __device__ __forceinline__
int lsb(Bitboard b) {
#ifdef __CUDA_ARCH__
    return __ffsll(b) - 1;
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return (int)idx;
#else
    return __builtin_ctzll(b);
#endif
}

__host__ __device__ __forceinline__
int msb(Bitboard b) {
#ifdef __CUDA_ARCH__
    return 63 - __clzll(b);
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse64(&idx, b);
    return (int)idx;
#else
    return 63 - __builtin_clzll(b);
#endif
}

__host__ __device__ __forceinline__
Bitboard pop_lsb(Bitboard& b) {
    Bitboard lsbit = b & (-b);
    b &= b - 1;
    return lsbit;
}

__host__ __device__ __forceinline__
int pop_lsb_index(Bitboard& b) {
    int idx = lsb(b);
    b &= b - 1;
    return idx;
}

#define ROOK_MAGIC_BITS   12
#define BISHOP_MAGIC_BITS 9

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#endif // CHESS_TYPES_CUH
