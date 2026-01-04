#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Type Aliases
// ============================================================================
typedef uint64_t Bitboard;
typedef uint16_t Move;

// ============================================================================
// Enums
// ============================================================================

enum Color : uint8_t {
    WHITE = 0,
    BLACK = 1
};

enum PieceType : uint8_t {
    PAWN = 0,
    KNIGHT = 1,
    BISHOP = 2,
    ROOK = 3,
    QUEEN = 4,
    KING = 5,
    NONE = 6
};

enum GameResult : uint8_t {
    ONGOING = 0,
    WHITE_WINS = 1,
    BLACK_WINS = 2,
    DRAW = 3
};

// ============================================================================
// Constants
// ============================================================================

// Files
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_B = FILE_A << 1;
constexpr Bitboard FILE_C = FILE_A << 2;
constexpr Bitboard FILE_D = FILE_A << 3;
constexpr Bitboard FILE_E = FILE_A << 4;
constexpr Bitboard FILE_F = FILE_A << 5;
constexpr Bitboard FILE_G = FILE_A << 6;
constexpr Bitboard FILE_H = FILE_A << 7;

// Ranks
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = RANK_1 << 8;
constexpr Bitboard RANK_3 = RANK_1 << 16;
constexpr Bitboard RANK_4 = RANK_1 << 24;
constexpr Bitboard RANK_5 = RANK_1 << 32;
constexpr Bitboard RANK_6 = RANK_1 << 40;
constexpr Bitboard RANK_7 = RANK_1 << 48;
constexpr Bitboard RANK_8 = RANK_1 << 56;

// Castling masks
constexpr uint8_t CASTLE_WK = 1;  // White kingside
constexpr uint8_t CASTLE_WQ = 2;  // White queenside
constexpr uint8_t CASTLE_BK = 4;  // Black kingside
constexpr uint8_t CASTLE_BQ = 8;  // Black queenside

// Move flags (bits 12-15 of Move)
constexpr Move QUIET_MOVE       = 0x0000;
constexpr Move DOUBLE_PUSH      = 0x1000;
constexpr Move KING_CASTLE      = 0x2000;
constexpr Move QUEEN_CASTLE     = 0x3000;
constexpr Move CAPTURE          = 0x4000;
constexpr Move EP_CAPTURE       = 0x5000;
constexpr Move KNIGHT_PROMO     = 0x8000;
constexpr Move BISHOP_PROMO     = 0x9000;
constexpr Move ROOK_PROMO       = 0xA000;
constexpr Move QUEEN_PROMO      = 0xB000;
constexpr Move KNIGHT_PROMO_CAP = 0xC000;
constexpr Move BISHOP_PROMO_CAP = 0xD000;
constexpr Move ROOK_PROMO_CAP   = 0xE000;
constexpr Move QUEEN_PROMO_CAP  = 0xF000;

// ============================================================================
// Position Structure
// ============================================================================

struct alignas(8) Position {
    // Piece bitboards: [Color][PieceType]
    // Index: color * 6 + piece_type
    Bitboard pieces[12];
    
    // Occupancy bitboards
    Bitboard occupied[3];  // [WHITE, BLACK, ALL]
    
    // Game state
    uint8_t side_to_move;  // WHITE or BLACK
    uint8_t castling;      // 4 bits for castling rights
    int8_t ep_square;      // -1 if none, else 0-63
    uint8_t halfmove;      // For 50-move rule
    uint8_t result;        // GameResult
    
    uint8_t _pad[3];       // Padding to 128 bytes for alignment
};

// ============================================================================
// Move Encoding/Decoding
// ============================================================================

// Move format: [0-5] from, [6-11] to, [12-15] flags
__host__ __device__ inline Move make_move(int from, int to, Move flags = 0) {
    return (Move)from | ((Move)to << 6) | flags;
}

__host__ __device__ inline int move_from(Move m) {
    return m & 0x3F;
}

__host__ __device__ inline int move_to(Move m) {
    return (m >> 6) & 0x3F;
}

__host__ __device__ inline Move move_flags(Move m) {
    return m & 0xF000;
}

__host__ __device__ inline bool is_capture(Move m) {
    return (m & 0x4000) != 0;
}

__host__ __device__ inline bool is_promotion(Move m) {
    return (m & 0x8000) != 0;
}

__host__ __device__ inline PieceType promotion_type(Move m) {
    Move flags = move_flags(m);
    if (flags == KNIGHT_PROMO || flags == KNIGHT_PROMO_CAP) return KNIGHT;
    if (flags == BISHOP_PROMO || flags == BISHOP_PROMO_CAP) return BISHOP;
    if (flags == ROOK_PROMO || flags == ROOK_PROMO_CAP) return ROOK;
    if (flags == QUEEN_PROMO || flags == QUEEN_PROMO_CAP) return QUEEN;
    return NONE;
}

// ============================================================================
// Utility Functions
// ============================================================================

__host__ __device__ inline int sq_file(int sq) {
    return sq & 7;
}

__host__ __device__ inline int sq_rank(int sq) {
    return sq >> 3;
}

__host__ __device__ inline int make_square(int file, int rank) {
    return rank * 8 + file;
}
