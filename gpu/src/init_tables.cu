#include "../include/chess_types.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Constants

#define ROOK_MAGIC_BITS   12
#define BISHOP_MAGIC_BITS 9

// Host-side tables (will be copied to GPU via copy functions)

static Bitboard h_knight_attacks[64];
static Bitboard h_king_attacks[64];
static Bitboard h_pawn_attacks[2][64];

static Bitboard h_rook_magics[64];
static Bitboard h_bishop_magics[64];
static Bitboard h_rook_masks[64];
static Bitboard h_bishop_masks[64];
static Bitboard h_rook_attacks[64][1 << ROOK_MAGIC_BITS];
static Bitboard h_bishop_attacks[64][1 << BISHOP_MAGIC_BITS];

// Copy functions declared in gpu_kernels.cu

extern cudaError_t copy_knight_attacks(const Bitboard* data);
extern cudaError_t copy_king_attacks(const Bitboard* data);
extern cudaError_t copy_pawn_attacks(const Bitboard* data);
extern cudaError_t copy_rook_magics(const Bitboard* data);
extern cudaError_t copy_bishop_magics(const Bitboard* data);
extern cudaError_t copy_rook_masks(const Bitboard* data);
extern cudaError_t copy_bishop_masks(const Bitboard* data);
extern cudaError_t copy_rook_attacks(const Bitboard* data);
extern cudaError_t copy_bishop_attacks(const Bitboard* data);

// Pre-calculated magic numbers (chess programming wiki)

static const Bitboard RookMagics[64] = {
    0x8a80104000800020ULL, 0x140002000100040ULL, 0x2801880a0017001ULL, 0x100081001000420ULL,
    0x200020010080420ULL, 0x3001c0002010008ULL, 0x8480008002000100ULL, 0x2080088004402900ULL,
    0x800098204000ULL, 0x2024401000200040ULL, 0x100802000801000ULL, 0x120800800801000ULL,
    0x208808088000400ULL, 0x2802200800400ULL, 0x2200800100020080ULL, 0x801000060821100ULL,
    0x80044006422000ULL, 0x100808020004000ULL, 0x12108a0010204200ULL, 0x140848010000802ULL,
    0x481828014002800ULL, 0x8094004002004100ULL, 0x4010040010010802ULL, 0x20008806104ULL,
    0x100400080208000ULL, 0x2040002120081000ULL, 0x21200680100081ULL, 0x20100080080080ULL,
    0x2000a00200410ULL, 0x20080800400ULL, 0x80088400100102ULL, 0x80004600042881ULL,
    0x4040008040800020ULL, 0x440003000200801ULL, 0x4200011004500ULL, 0x188020010100100ULL,
    0x14800401802800ULL, 0x2080040080800200ULL, 0x124080204001001ULL, 0x200046502000484ULL,
    0x480400080088020ULL, 0x1000422010034000ULL, 0x30200100110040ULL, 0x100021010009ULL,
    0x2002080100110004ULL, 0x202008004008002ULL, 0x20020004010100ULL, 0x2048440040820001ULL,
    0x101002200408200ULL, 0x40802000401080ULL, 0x4008142004410100ULL, 0x2060820c0120200ULL,
    0x1001004080100ULL, 0x20c020080040080ULL, 0x2935610830022400ULL, 0x44440041009200ULL,
    0x280001040802101ULL, 0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
    0x20030a0244872ULL, 0x12001008414402ULL, 0x2006104900a0804ULL, 0x1004081002402ULL
};

static const Bitboard BishopMagics[64] = {
    0x40040844404084ULL, 0x2004208a004208ULL, 0x10190041080202ULL, 0x108060845042010ULL,
    0x581104180800210ULL, 0x2112080446200010ULL, 0x1080820820060210ULL, 0x3c0808410220200ULL,
    0x4050404440404ULL, 0x21001420088ULL, 0x24d0080801082102ULL, 0x1020a0a020400ULL,
    0x40308200402ULL, 0x4011002100800ULL, 0x401484104104005ULL, 0x801010402020200ULL,
    0x400210c3880100ULL, 0x404022024108200ULL, 0x810018200204102ULL, 0x4002801a02003ULL,
    0x85040820080400ULL, 0x810102c808880400ULL, 0xe900410884800ULL, 0x8002020480840102ULL,
    0x220200865090201ULL, 0x2010100a02021202ULL, 0x152048408022401ULL, 0x20080002081110ULL,
    0x4001001021004000ULL, 0x800040400a011002ULL, 0xe4004081011002ULL, 0x1c004001012080ULL,
    0x8004200962a00220ULL, 0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
    0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL, 0x42008c0340209202ULL,
    0x209188240001000ULL, 0x400408a884001800ULL, 0x110400a6080400ULL, 0x1840060a44020800ULL,
    0x90080104000041ULL, 0x201011000808101ULL, 0x1a2208080504f080ULL, 0x8012020600211212ULL,
    0x500861011240000ULL, 0x180806108200800ULL, 0x4000020e01040044ULL, 0x300000261044000aULL,
    0x802241102020002ULL, 0x20906061210001ULL, 0x5a84841004010310ULL, 0x4010801011c04ULL,
    0xa010109502200ULL, 0x4a02012000ULL, 0x500201010098b028ULL, 0x8040002811040900ULL,
    0x28000010020204ULL, 0x6000020202d0240ULL, 0x8918844842082200ULL, 0x4010011029020020ULL
};

// Utility functions

static inline int rank_of(int sq) { return sq >> 3; }
static inline int file_of(int sq) { return sq & 7; }

static Bitboard sliding_attack(int sq, Bitboard occ, int dx, int dy) {
    Bitboard attacks = 0;
    int r = rank_of(sq);
    int f = file_of(sq);

    for (;;) {
        r += dy;
        f += dx;
        if (r < 0 || r > 7 || f < 0 || f > 7) break;

        int target = r * 8 + f;
        attacks |= (C64(1) << target);

        if (occ & (C64(1) << target)) break;
    }
    return attacks;
}

static Bitboard rook_attacks_slow(int sq, Bitboard occ) {
    return sliding_attack(sq, occ, 1, 0) |
           sliding_attack(sq, occ, -1, 0) |
           sliding_attack(sq, occ, 0, 1) |
           sliding_attack(sq, occ, 0, -1);
}

static Bitboard bishop_attacks_slow(int sq, Bitboard occ) {
    return sliding_attack(sq, occ, 1, 1) |
           sliding_attack(sq, occ, 1, -1) |
           sliding_attack(sq, occ, -1, 1) |
           sliding_attack(sq, occ, -1, -1);
}

// Initialize attack tables

static void init_knight_attacks() {
    const int knight_offsets[8][2] = {
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
        {1, -2}, {1, 2}, {2, -1}, {2, 1}
    };

    for (int sq = 0; sq < 64; sq++) {
        h_knight_attacks[sq] = 0;
        int r = rank_of(sq);
        int f = file_of(sq);

        for (int i = 0; i < 8; i++) {
            int nr = r + knight_offsets[i][0];
            int nf = f + knight_offsets[i][1];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                h_knight_attacks[sq] |= (C64(1) << (nr * 8 + nf));
            }
        }
    }
}

static void init_king_attacks() {
    const int king_offsets[8][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
        {0, 1}, {1, -1}, {1, 0}, {1, 1}
    };

    for (int sq = 0; sq < 64; sq++) {
        h_king_attacks[sq] = 0;
        int r = rank_of(sq);
        int f = file_of(sq);

        for (int i = 0; i < 8; i++) {
            int nr = r + king_offsets[i][0];
            int nf = f + king_offsets[i][1];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
                h_king_attacks[sq] |= (C64(1) << (nr * 8 + nf));
            }
        }
    }
}

static void init_pawn_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        int r = rank_of(sq);
        int f = file_of(sq);

        // White pawn attacks
        h_pawn_attacks[WHITE][sq] = 0;
        if (r < 7) {
            if (f > 0) h_pawn_attacks[WHITE][sq] |= (C64(1) << (sq + 7));
            if (f < 7) h_pawn_attacks[WHITE][sq] |= (C64(1) << (sq + 9));
        }

        // Black pawn attacks
        h_pawn_attacks[BLACK][sq] = 0;
        if (r > 0) {
            if (f > 0) h_pawn_attacks[BLACK][sq] |= (C64(1) << (sq - 9));
            if (f < 7) h_pawn_attacks[BLACK][sq] |= (C64(1) << (sq - 7));
        }
    }
}

static void init_slider_masks() {
    for (int sq = 0; sq < 64; sq++) {
        int r = rank_of(sq);
        int f = file_of(sq);

        // Rook mask (excludes edges)
        h_rook_masks[sq] = 0;
        for (int i = r + 1; i < 7; i++) h_rook_masks[sq] |= (C64(1) << (i * 8 + f));
        for (int i = r - 1; i > 0; i--) h_rook_masks[sq] |= (C64(1) << (i * 8 + f));
        for (int i = f + 1; i < 7; i++) h_rook_masks[sq] |= (C64(1) << (r * 8 + i));
        for (int i = f - 1; i > 0; i--) h_rook_masks[sq] |= (C64(1) << (r * 8 + i));

        // Bishop mask (excludes edges)
        h_bishop_masks[sq] = 0;
        for (int i = 1; r + i < 7 && f + i < 7; i++)
            h_bishop_masks[sq] |= (C64(1) << ((r + i) * 8 + f + i));
        for (int i = 1; r + i < 7 && f - i > 0; i++)
            h_bishop_masks[sq] |= (C64(1) << ((r + i) * 8 + f - i));
        for (int i = 1; r - i > 0 && f + i < 7; i++)
            h_bishop_masks[sq] |= (C64(1) << ((r - i) * 8 + f + i));
        for (int i = 1; r - i > 0 && f - i > 0; i++)
            h_bishop_masks[sq] |= (C64(1) << ((r - i) * 8 + f - i));
    }
}

static Bitboard set_occupancy(int index, int bits_in_mask, Bitboard attack_mask) {
    Bitboard occupancy = 0;
    for (int count = 0; count < bits_in_mask; count++) {
        int sq = lsb(attack_mask);
        attack_mask &= attack_mask - 1;
        if (index & (1 << count)) {
            occupancy |= (C64(1) << sq);
        }
    }
    return occupancy;
}

static void init_magic_tables() {
    // Copy pre-calculated magics
    for (int sq = 0; sq < 64; sq++) {
        h_rook_magics[sq] = RookMagics[sq];
        h_bishop_magics[sq] = BishopMagics[sq];
    }

    // Initialize rook attack tables
    for (int sq = 0; sq < 64; sq++) {
        Bitboard mask = h_rook_masks[sq];
        int bit_count = popcount(mask);

        for (int i = 0; i < (1 << bit_count); i++) {
            Bitboard occ = set_occupancy(i, bit_count, mask);
            int idx = (int)((occ * h_rook_magics[sq]) >> (64 - ROOK_MAGIC_BITS));
            h_rook_attacks[sq][idx] = rook_attacks_slow(sq, occ);
        }
    }

    // Initialize bishop attack tables
    for (int sq = 0; sq < 64; sq++) {
        Bitboard mask = h_bishop_masks[sq];
        int bit_count = popcount(mask);

        for (int i = 0; i < (1 << bit_count); i++) {
            Bitboard occ = set_occupancy(i, bit_count, mask);
            int idx = (int)((occ * h_bishop_magics[sq]) >> (64 - BISHOP_MAGIC_BITS));
            h_bishop_attacks[sq][idx] = bishop_attacks_slow(sq, occ);
        }
    }
}

// Public initialization function

void init_attack_tables() {
    printf("Initializing attack tables...\n");

    // Initialize host tables
    init_knight_attacks();
    init_king_attacks();
    init_pawn_attacks();
    init_slider_masks();
    init_magic_tables();

    CUDA_CHECK(copy_knight_attacks(h_knight_attacks));
    CUDA_CHECK(copy_king_attacks(h_king_attacks));
    CUDA_CHECK(copy_pawn_attacks(&h_pawn_attacks[0][0]));
    CUDA_CHECK(copy_rook_magics(h_rook_magics));
    CUDA_CHECK(copy_bishop_magics(h_bishop_magics));
    CUDA_CHECK(copy_rook_masks(h_rook_masks));
    CUDA_CHECK(copy_bishop_masks(h_bishop_masks));
    CUDA_CHECK(copy_rook_attacks(&h_rook_attacks[0][0]));
    CUDA_CHECK(copy_bishop_attacks(&h_bishop_attacks[0][0]));

    printf("Attack tables initialized successfully!\n");
}

// Board initialization helpers 
void init_startpos(BoardState* pos) {
    memset(pos, 0, sizeof(BoardState));

    // White pieces
    pos->pieces[WHITE][PAWN]   = RANK_2;
    pos->pieces[WHITE][KNIGHT] = C64(0x42);           // b1, g1
    pos->pieces[WHITE][BISHOP] = C64(0x24);           // c1, f1
    pos->pieces[WHITE][ROOK]   = C64(0x81);           // a1, h1
    pos->pieces[WHITE][QUEEN]  = C64(0x08);           // d1
    pos->pieces[WHITE][KING]   = C64(0x10);           // e1

    // Black pieces
    pos->pieces[BLACK][PAWN]   = RANK_7;
    pos->pieces[BLACK][KNIGHT] = C64(0x4200000000000000);  // b8, g8
    pos->pieces[BLACK][BISHOP] = C64(0x2400000000000000);  // c8, f8
    pos->pieces[BLACK][ROOK]   = C64(0x8100000000000000);  // a8, h8
    pos->pieces[BLACK][QUEEN]  = C64(0x0800000000000000);  // d8
    pos->pieces[BLACK][KING]   = C64(0x1000000000000000);  // e8

    pos->side_to_move = WHITE;
    pos->castling = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ;
    pos->ep_square = -1;
    pos->halfmove = 0;
    pos->result = RESULT_ONGOING;
}
