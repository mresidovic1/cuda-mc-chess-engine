#include "../../include/chess_types.cuh"

// Evaluation is done in centipawns https://www.chessprogramming.org/Centipawns

// Middlegame values
#define EVAL_PAWN_MG    136
#define EVAL_KNIGHT_MG  782
#define EVAL_BISHOP_MG  830
#define EVAL_ROOK_MG    1289
#define EVAL_QUEEN_MG   2529

// Endgame values
#define EVAL_PAWN_EG    208
#define EVAL_KNIGHT_EG  865
#define EVAL_BISHOP_EG  918
#define EVAL_ROOK_EG    1378
#define EVAL_QUEEN_EG   2687

// Positional Bonuses

#define EVAL_TEMPO                10
#define EVAL_BISHOP_PAIR_MG       57
#define EVAL_BISHOP_PAIR_EG       93
#define EVAL_BISHOP_PAIR_WINGS    25
#define EVAL_ROOK_ON_7TH          20
#define EVAL_ROOK_OPEN_FILE_MG    20
#define EVAL_ROOK_OPEN_FILE_EG    30
#define EVAL_ROOK_SEMI_OPEN_MG    10
#define EVAL_ROOK_SEMI_OPEN_EG    15
#define EVAL_DOUBLED_ROOKS_MG     15
#define EVAL_DOUBLED_ROOKS_EG     20
#define EVAL_KNIGHT_CENTER        10
#define EVAL_BISHOP_CENTER        10
#define EVAL_KNIGHT_EXTENDED_CENTER 5
#define EVAL_BISHOP_EXTENDED_CENTER 5

// King safety - pawn shield
#define EVAL_KING_PAWN_SHIELD     30
#define EVAL_KING_MISSING_SHIELD  -20

// Game Phase Detection

#define PHASE_KNIGHT  1
#define PHASE_BISHOP  1
#define PHASE_ROOK    2
#define PHASE_QUEEN   4
#define TOTAL_PHASE   (4 * PHASE_KNIGHT + 4 * PHASE_BISHOP + 4 * PHASE_ROOK + 2 * PHASE_QUEEN) // 24

// Material Imbalance Bonuses (Stockfish 10 based)

// Bonus for having pieces that work well together
#define IMBALANCE_BISHOP_PAIR_MG      57   
#define IMBALANCE_KNIGHT_VS_BISHOP    -10  

// Bonus for multiple minor pieces vs rook
#define IMBALANCE_THREE_MINORS_MG     30
#define IMBALANCE_THREE_MINORS_EG     50

// Penalty for having rook but no minor pieces
#define IMBALANCE_ROOK_NO_MINORS_MG   -20
#define IMBALANCE_ROOK_NO_MINORS_EG   -30

// Center Control Masks

// Original 4-square center (d4, d5, e4, e5)
const uint64_t CENTER_MASK = 0x0000001818000000ULL;

// Extended center (16 squares: c3-f3, c6-f6, c3-c6, f3-f6)
// Includes squares: c3, d3, e3, f3, c4, d4, e4, f4, c5, d5, e5, f5, c6, d6, e6, f6
const uint64_t EXTENDED_CENTER_MASK = 0x00003C3C3C3C0000ULL;

// Wing masks for bishop pair enhancement (queen side vs king side)
const uint64_t QUEEN_SIDE_MASK = 0x0F0F0F0F0F0F0F0FULL;  // Files a-d
const uint64_t KING_SIDE_MASK = 0xF0F0F0F0F0F0F0F0ULL;   // Files e-h

// Piece-Square Tables (MG/EG separated for tapered evaluation)

// PAWN PST (same for MG and EG)
__constant__ int8_t g_PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0
};

// KNIGHT PST - Middlegame
__constant__ int8_t g_PST_KNIGHT_MG[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};

// KNIGHT PST - Endgame (slightly different values)
__constant__ int8_t g_PST_KNIGHT_EG[64] = {
   -40,-30,-20,-20,-20,-20,-30,-40,
   -30,-10, 10, 15, 15, 10,-10,-30,
   -20, 10, 20, 25, 25, 20, 10,-20,
   -20,  5, 25, 30, 30, 25,  5,-20,
   -20, 10, 25, 30, 30, 25, 10,-20,
   -20,  5, 20, 25, 25, 20,  5,-20,
   -30,-10, 10, 10, 10, 10,-10,-30,
   -40,-30,-20,-20,-20,-20,-30,-40
};

// BISHOP PST - Middlegame
__constant__ int8_t g_PST_BISHOP_MG[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

// BISHOP PST - Endgame (more active)
__constant__ int8_t g_PST_BISHOP_EG[64] = {
   -15,-10,-10,-10,-10,-10,-10,-15,
   -10, 10,  5,  5,  5,  5, 10,-10,
   -10, 15, 15, 15, 15, 15, 15,-10,
   -10,  5, 15, 20, 20, 15,  5,-10,
   -10, 10, 15, 20, 20, 15, 10,-10,
   -10,  5, 15, 15, 15, 15,  5,-10,
   -10,  5,  5,  5,  5,  5,  5,-10,
   -15,-10,-10,-10,-10,-10,-10,-15
};

// ROOK PST (same for MG and EG)
__constant__ int8_t g_PST_ROOK[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

// KING PST - Middlegame (stay safe)
__constant__ int8_t g_PST_KING_MG[64] = {
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30
};

// KING PST - Endgame (centralize)
__constant__ int8_t g_PST_KING_EG[64] = {
   -50,-30,-30,-30,-30,-30,-30,-50,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -50,-40,-30,-20,-20,-30,-40,-50
};


// Calculate game phase (0 = endgame, 256 = opening/middlegame)
__device__ __forceinline__
int calculate_phase(const BoardState* pos) {
    int phase = TOTAL_PHASE;

    phase -= popcount(pos->pieces[WHITE][KNIGHT] | pos->pieces[BLACK][KNIGHT]) * PHASE_KNIGHT;
    phase -= popcount(pos->pieces[WHITE][BISHOP] | pos->pieces[BLACK][BISHOP]) * PHASE_BISHOP;
    phase -= popcount(pos->pieces[WHITE][ROOK]   | pos->pieces[BLACK][ROOK])   * PHASE_ROOK;
    phase -= popcount(pos->pieces[WHITE][QUEEN]  | pos->pieces[BLACK][QUEEN])  * PHASE_QUEEN;

    // Scale to 0-256 range
    return (phase * 256 + TOTAL_PHASE / 2) / TOTAL_PHASE;
}

// Tapered evaluation: interpolate between MG and EG scores
__device__ __forceinline__
int tapered_eval(int mg_score, int eg_score, int phase) {
    // phase: 256 = all MG, 0 = all EG
    return (mg_score * phase + eg_score * (256 - phase)) / 256;
}


__device__ __forceinline__
void evaluate_material(const BoardState* pos, int phase, int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    // White pieces
    int w_pawns   = popcount(pos->pieces[WHITE][PAWN]);
    int w_knights = popcount(pos->pieces[WHITE][KNIGHT]);
    int w_bishops = popcount(pos->pieces[WHITE][BISHOP]);
    int w_rooks   = popcount(pos->pieces[WHITE][ROOK]);
    int w_queens  = popcount(pos->pieces[WHITE][QUEEN]);

    // Black pieces
    int b_pawns   = popcount(pos->pieces[BLACK][PAWN]);
    int b_knights = popcount(pos->pieces[BLACK][KNIGHT]);
    int b_bishops = popcount(pos->pieces[BLACK][BISHOP]);
    int b_rooks   = popcount(pos->pieces[BLACK][ROOK]);
    int b_queens  = popcount(pos->pieces[BLACK][QUEEN]);

    // Base material values
    mg_score = (w_pawns * EVAL_PAWN_MG + w_knights * EVAL_KNIGHT_MG + w_bishops * EVAL_BISHOP_MG +
               w_rooks * EVAL_ROOK_MG + w_queens * EVAL_QUEEN_MG) -
              (b_pawns * EVAL_PAWN_MG + b_knights * EVAL_KNIGHT_MG + b_bishops * EVAL_BISHOP_MG +
               b_rooks * EVAL_ROOK_MG + b_queens * EVAL_QUEEN_MG);

    eg_score = (w_pawns * EVAL_PAWN_EG + w_knights * EVAL_KNIGHT_EG + w_bishops * EVAL_BISHOP_EG +
               w_rooks * EVAL_ROOK_EG + w_queens * EVAL_QUEEN_EG) -
              (b_pawns * EVAL_PAWN_EG + b_knights * EVAL_KNIGHT_EG + b_bishops * EVAL_BISHOP_EG +
               b_rooks * EVAL_ROOK_EG + b_queens * EVAL_QUEEN_EG);

    // Bishop pair bonus
    if (w_bishops >= 2) {
        int bishop_bonus_mg = EVAL_BISHOP_PAIR_MG;
        int bishop_bonus_eg = EVAL_BISHOP_PAIR_EG;

        // Additional bonus if pawns are on both wings (enhanced bishop pair)
        Bitboard white_pawns = pos->pieces[WHITE][PAWN];
        bool pawns_queenside = (white_pawns & QUEEN_SIDE_MASK) != 0;
        bool pawns_kingside = (white_pawns & KING_SIDE_MASK) != 0;
        if (pawns_queenside && pawns_kingside) {
            bishop_bonus_mg += EVAL_BISHOP_PAIR_WINGS;
            bishop_bonus_eg += EVAL_BISHOP_PAIR_WINGS;
        }

        mg_score += bishop_bonus_mg;
        eg_score += bishop_bonus_eg;
    }
    if (b_bishops >= 2) {
        int bishop_bonus_mg = EVAL_BISHOP_PAIR_MG;
        int bishop_bonus_eg = EVAL_BISHOP_PAIR_EG;

        // Additional bonus if pawns are on both wings
        Bitboard black_pawns = pos->pieces[BLACK][PAWN];
        bool pawns_queenside = (black_pawns & QUEEN_SIDE_MASK) != 0;
        bool pawns_kingside = (black_pawns & KING_SIDE_MASK) != 0;
        if (pawns_queenside && pawns_kingside) {
            bishop_bonus_mg += EVAL_BISHOP_PAIR_WINGS;
            bishop_bonus_eg += EVAL_BISHOP_PAIR_WINGS;
        }

        mg_score -= bishop_bonus_mg;
        eg_score -= bishop_bonus_eg;
    }

    // Material imbalance bonuses
    // Three minors vs two minors (or similar)
    int w_minors = w_knights + w_bishops;
    int b_minors = b_knights + b_bishops;

    if (w_minors >= 3 && b_minors >= 3) {
        mg_score += IMBALANCE_THREE_MINORS_MG;
        eg_score += IMBALANCE_THREE_MINORS_EG;
    }
    if (b_minors >= 3 && w_minors >= 3) {
        mg_score -= IMBALANCE_THREE_MINORS_MG;
        eg_score -= IMBALANCE_THREE_MINORS_EG;
    }

    // Rook with no minors (disadvantage)
    if (w_rooks > 0 && w_minors == 0) {
        mg_score += IMBALANCE_ROOK_NO_MINORS_MG * w_rooks;
        eg_score += IMBALANCE_ROOK_NO_MINORS_EG * w_rooks;
    }
    if (b_rooks > 0 && b_minors == 0) {
        mg_score -= IMBALANCE_ROOK_NO_MINORS_MG * b_rooks;
        eg_score -= IMBALANCE_ROOK_NO_MINORS_EG * b_rooks;
    }

    *score_mg = mg_score;
    *score_eg = eg_score;
}


__device__ __forceinline__
void evaluate_pst(const BoardState* pos, int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    Bitboard bb;

    // White pieces
    // Pawns
    bb = pos->pieces[WHITE][PAWN];
    while (bb) {
        int sq = pop_lsb_index(bb);
        int val = g_PST_PAWN[sq];
        mg_score += val;
        eg_score += val;
    }

    // Knights
    bb = pos->pieces[WHITE][KNIGHT];
    while (bb) {
        int sq = pop_lsb_index(bb);
        mg_score += g_PST_KNIGHT_MG[sq];
        eg_score += g_PST_KNIGHT_EG[sq];
    }

    // Bishops
    bb = pos->pieces[WHITE][BISHOP];
    while (bb) {
        int sq = pop_lsb_index(bb);
        mg_score += g_PST_BISHOP_MG[sq];
        eg_score += g_PST_BISHOP_EG[sq];
    }

    // Rooks
    bb = pos->pieces[WHITE][ROOK];
    while (bb) {
        int sq = pop_lsb_index(bb);
        int val = g_PST_ROOK[sq];
        mg_score += val;
        eg_score += val;
    }

    // Kings
    int wk = lsb(pos->pieces[WHITE][KING]);
    mg_score += g_PST_KING_MG[wk];
    eg_score += g_PST_KING_EG[wk];

    // Black pieces (mirrored)
    // Pawns
    bb = pos->pieces[BLACK][PAWN];
    while (bb) {
        int sq = pop_lsb_index(bb);
        int val = g_PST_PAWN[sq ^ 56];
        mg_score -= val;
        eg_score -= val;
    }

    // Knights
    bb = pos->pieces[BLACK][KNIGHT];
    while (bb) {
        int sq = pop_lsb_index(bb);
        mg_score -= g_PST_KNIGHT_MG[sq ^ 56];
        eg_score -= g_PST_KNIGHT_EG[sq ^ 56];
    }

    // Bishops
    bb = pos->pieces[BLACK][BISHOP];
    while (bb) {
        int sq = pop_lsb_index(bb);
        mg_score -= g_PST_BISHOP_MG[sq ^ 56];
        eg_score -= g_PST_BISHOP_EG[sq ^ 56];
    }

    // Rooks
    bb = pos->pieces[BLACK][ROOK];
    while (bb) {
        int sq = pop_lsb_index(bb);
        int val = g_PST_ROOK[sq ^ 56];
        mg_score -= val;
        eg_score -= val;
    }

    // Kings
    int bk = lsb(pos->pieces[BLACK][KING]);
    mg_score -= g_PST_KING_MG[bk ^ 56];
    eg_score -= g_PST_KING_EG[bk ^ 56];

    *score_mg = mg_score;
    *score_eg = eg_score;
}


// File masks for pawn structure analysis
const uint64_t FILE_A_MASK = 0x0101010101010101ULL;
const uint64_t FILE_B_MASK = 0x0202020202020202ULL;
const uint64_t FILE_C_MASK = 0x0404040404040404ULL;
const uint64_t FILE_D_MASK = 0x0808080808080808ULL;
const uint64_t FILE_E_MASK = 0x1010101010101010ULL;
const uint64_t FILE_F_MASK = 0x2020202020202020ULL;
const uint64_t FILE_G_MASK = 0x4040404040404040ULL;
const uint64_t FILE_H_MASK = 0x8080808080808080ULL;

// Rank masks for advancement
const uint64_t RANK_2_MASK = 0xFF00ULL;
const uint64_t RANK_3_MASK = 0xFF0000ULL;
const uint64_t RANK_4_MASK = 0xFF000000ULL;
const uint64_t RANK_5_MASK = 0xFF00000000ULL;
const uint64_t RANK_6_MASK = 0xFF0000000000ULL;
const uint64_t RANK_7_MASK = 0xFF000000000000ULL;

__device__ __forceinline__
void evaluate_pawn_structure(const BoardState* pos, int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    Bitboard white_pawns = pos->pieces[WHITE][PAWN];
    Bitboard black_pawns = pos->pieces[BLACK][PAWN];

    // Evaluate white pawns
    Bitboard wp = white_pawns;
    while (wp) {
        int sq = pop_lsb_index(wp);
        int file = sq % 8;
        int rank = sq / 8;

        // Check for passed pawn
        bool is_passed = true;
        Bitboard enemy_pawns_in_front = 0;

        // Check adjacent files for enemy pawns that could block
        Bitboard blocking_files = 0;
        if (file > 0) blocking_files |= FILE_A_MASK << (file - 1);
        if (file < 7) blocking_files |= FILE_A_MASK << (file + 1);
        blocking_files |= (FILE_A_MASK << file);

        // Enemy pawns on or ahead of this rank in adjacent files
        Bitboard rank_mask = 0;
        if (rank >= 1) rank_mask |= RANK_2_MASK;
        if (rank >= 2) rank_mask |= RANK_3_MASK;
        if (rank >= 3) rank_mask |= RANK_4_MASK;
        if (rank >= 4) rank_mask |= RANK_5_MASK;
        if (rank >= 5) rank_mask |= RANK_6_MASK;
        if (rank >= 6) rank_mask |= RANK_7_MASK;

        Bitboard enemy_pawns_ahead = black_pawns & blocking_files & rank_mask;
        if (enemy_pawns_ahead) {
            is_passed = false;
        }

        // Passed pawn bonus (rank-based)
        if (is_passed) {
            int rank_bonus = rank * 20;  // More bonus as pawn advances
            mg_score += rank_bonus;
            eg_score += rank_bonus * 2;  // Even more valuable in endgame
        }

        // Check for isolated pawn (no friendly pawns on adjacent files)
        bool is_isolated = true;
        Bitboard adjacent_files = 0;
        if (file > 0) {
            Bitboard left_file = FILE_A_MASK << (file - 1);
            if (white_pawns & left_file) is_isolated = false;
        }
        if (file < 7) {
            Bitboard right_file = FILE_A_MASK << (file + 1);
            if (white_pawns & right_file) is_isolated = false;
        }

        if (is_isolated) {
            mg_score -= 20;
            eg_score -= 30;  // Worse in endgame
        }

        // Check for doubled pawns (two pawns on same file)
        Bitboard file_mask = FILE_A_MASK << file;
        if (popcount(white_pawns & file_mask) >= 2) {
            // Only count penalty once per file, so we skip if this is not the first pawn
            // For simplicity, we apply a small penalty per pawn on the file
            mg_score -= 10;
            eg_score -= 15;
        }

        // Check for backward pawn (pawn that can't advance due to enemy pawns)
        // A backward pawn is one that cannot be protected by another pawn
        // and has enemy pawns on adjacent files controlling the square in front
        bool is_backward = false;
        if (rank < 6) {  // Not on 7th rank
            int sq_ahead = sq + 8;
            Bitboard square_ahead = 1ULL << sq_ahead;

            // Check if enemy pawns control adjacent squares in front
            Bitboard enemy_control = 0;
            if (file > 0) {
                Bitboard diag_left = (FILE_A_MASK << (file - 1)) & (RANK_2_MASK << rank);
                enemy_control |= black_pawns & diag_left;
            }
            if (file < 7) {
                Bitboard diag_right = (FILE_A_MASK << (file + 1)) & (RANK_2_MASK << rank);
                enemy_control |= black_pawns & diag_right;
            }

            // Check if this pawn can be defended by another pawn
            bool can_be_defended = false;
            Bitboard defending_squares = 0;
            if (sq >= 8) {  // Has rank behind
                int sq_behind = sq - 8;
                Bitboard diag_left_def = 0;
                Bitboard diag_right_def = 0;
                if (file > 0) diag_left_def = 1ULL << (sq_behind - 1);
                if (file < 7) diag_right_def = 1ULL << (sq_behind + 1);
                if (white_pawns & (diag_left_def | diag_right_def)) {
                    can_be_defended = true;
                }
            }

            if (enemy_control && !can_be_defended && !is_passed) {
                is_backward = true;
            }
        }

        if (is_backward) {
            mg_score -= 10;
            eg_score -= 15;
        }
    }

    // Evaluate black pawns (mirror logic)
    Bitboard bp = black_pawns;
    while (bp) {
        int sq = pop_lsb_index(bp);
        int file = sq % 8;
        int rank = sq / 8;

        // Check for passed pawn (from black's perspective)
        bool is_passed = true;
        Bitboard blocking_files = 0;
        if (file > 0) blocking_files |= FILE_A_MASK << (file - 1);
        if (file < 7) blocking_files |= FILE_A_MASK << (file + 1);
        blocking_files |= (FILE_A_MASK << file);

        // White pawns on or ahead of this rank (from black's perspective = lower ranks)
        Bitboard rank_mask = 0;
        if (rank <= 5) rank_mask |= RANK_7_MASK;
        if (rank <= 4) rank_mask |= RANK_6_MASK;
        if (rank <= 3) rank_mask |= RANK_5_MASK;
        if (rank <= 2) rank_mask |= RANK_4_MASK;
        if (rank <= 1) rank_mask |= RANK_3_MASK;
        if (rank <= 0) rank_mask |= RANK_2_MASK;

        Bitboard white_pawns_ahead = white_pawns & blocking_files & rank_mask;
        if (white_pawns_ahead) {
            is_passed = false;
        }

        // Passed pawn bonus
        if (is_passed) {
            int rank_bonus = (7 - rank) * 20;  // Distance from 8th rank
            mg_score -= rank_bonus;
            eg_score -= rank_bonus * 2;
        }

        // Isolated pawn
        bool is_isolated = true;
        if (file > 0) {
            Bitboard left_file = FILE_A_MASK << (file - 1);
            if (black_pawns & left_file) is_isolated = false;
        }
        if (file < 7) {
            Bitboard right_file = FILE_A_MASK << (file + 1);
            if (black_pawns & right_file) is_isolated = false;
        }

        if (is_isolated) {
            mg_score += 20;
            eg_score += 30;
        }

        // Doubled pawns
        Bitboard file_mask = FILE_A_MASK << file;
        if (popcount(black_pawns & file_mask) >= 2) {
            mg_score += 10;
            eg_score += 15;
        }

        // Backward pawn
        bool is_backward = false;
        if (rank > 1) {  // Not on 2nd rank
            int sq_ahead = sq - 8;
            Bitboard enemy_control = 0;
            if (file > 0) {
                Bitboard diag_left = (FILE_A_MASK << (file - 1)) & (RANK_7_MASK >> (6 - rank));
                enemy_control |= white_pawns & diag_left;
            }
            if (file < 7) {
                Bitboard diag_right = (FILE_A_MASK << (file + 1)) & (RANK_7_MASK >> (6 - rank));
                enemy_control |= white_pawns & diag_right;
            }

            bool can_be_defended = false;
            if (sq <= 55) {  // Has rank behind
                int sq_behind = sq + 8;
                Bitboard diag_left_def = 0;
                Bitboard diag_right_def = 0;
                if (file > 0) diag_left_def = 1ULL << (sq_behind - 1);
                if (file < 7) diag_right_def = 1ULL << (sq_behind + 1);
                if (black_pawns & (diag_left_def | diag_right_def)) {
                    can_be_defended = true;
                }
            }

            if (enemy_control && !can_be_defended && !is_passed) {
                is_backward = true;
            }
        }

        if (is_backward) {
            mg_score += 10;
            eg_score += 15;
        }
    }

    *score_mg = mg_score;
    *score_eg = eg_score;
}


__device__ __forceinline__
void evaluate_positional(const BoardState* pos, int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    // Rook on 7th rank bonus
    const uint64_t RANK_7_WHITE = 0xFF00ULL;
    const uint64_t RANK_7_BLACK = 0xFF000000000000ULL;

    if (pos->pieces[WHITE][ROOK] & RANK_7_WHITE) {
        mg_score += EVAL_ROOK_ON_7TH;
        eg_score += EVAL_ROOK_ON_7TH;
    }
    if (pos->pieces[BLACK][ROOK] & RANK_7_BLACK) {
        mg_score -= EVAL_ROOK_ON_7TH;
        eg_score -= EVAL_ROOK_ON_7TH;
    }

    // Center control - original 4 squares
    int w_knights_center = popcount(pos->pieces[WHITE][KNIGHT] & CENTER_MASK);
    int b_knights_center = popcount(pos->pieces[BLACK][KNIGHT] & CENTER_MASK);
    int w_bishops_center = popcount(pos->pieces[WHITE][BISHOP] & CENTER_MASK);
    int b_bishops_center = popcount(pos->pieces[BLACK][BISHOP] & CENTER_MASK);

    mg_score += EVAL_KNIGHT_CENTER * w_knights_center;
    mg_score -= EVAL_KNIGHT_CENTER * b_knights_center;
    mg_score += EVAL_BISHOP_CENTER * w_bishops_center;
    mg_score -= EVAL_BISHOP_CENTER * b_bishops_center;

    // Extended center control (16 squares)
    int w_knights_ext = popcount(pos->pieces[WHITE][KNIGHT] & EXTENDED_CENTER_MASK);
    int b_knights_ext = popcount(pos->pieces[BLACK][KNIGHT] & EXTENDED_CENTER_MASK);
    int w_bishops_ext = popcount(pos->pieces[WHITE][BISHOP] & EXTENDED_CENTER_MASK);
    int b_bishops_ext = popcount(pos->pieces[BLACK][BISHOP] & EXTENDED_CENTER_MASK);

    mg_score += EVAL_KNIGHT_EXTENDED_CENTER * w_knights_ext;
    mg_score -= EVAL_KNIGHT_EXTENDED_CENTER * b_knights_ext;
    mg_score += EVAL_BISHOP_EXTENDED_CENTER * w_bishops_ext;
    mg_score -= EVAL_BISHOP_EXTENDED_CENTER * b_bishops_ext;

    *score_mg = mg_score;
    *score_eg = eg_score;
}


__device__ __forceinline__
void evaluate_rooks(const BoardState* pos, int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    Bitboard white_pawns = pos->pieces[WHITE][PAWN];
    Bitboard black_pawns = pos->pieces[BLACK][PAWN];
    Bitboard white_rooks = pos->pieces[WHITE][ROOK];
    Bitboard black_rooks = pos->pieces[BLACK][ROOK];

    // Evaluate white rooks
    Bitboard wr = white_rooks;
    while (wr) {
        int sq = pop_lsb_index(wr);
        int file = sq % 8;

        // Check if file is open (no pawns)
        Bitboard file_mask = FILE_A_MASK << file;
        bool is_open = ((white_pawns & file_mask) == 0) && ((black_pawns & file_mask) == 0);
        bool is_semi_open = ((white_pawns & file_mask) == 0) || ((black_pawns & file_mask) == 0);

        if (is_open) {
            mg_score += EVAL_ROOK_OPEN_FILE_MG;
            eg_score += EVAL_ROOK_OPEN_FILE_EG;
        } else if (is_semi_open) {
            mg_score += EVAL_ROOK_SEMI_OPEN_MG;
            eg_score += EVAL_ROOK_SEMI_OPEN_EG;
        }
    }

    // Evaluate black rooks
    Bitboard br = black_rooks;
    while (br) {
        int sq = pop_lsb_index(br);
        int file = sq % 8;

        Bitboard file_mask = FILE_A_MASK << file;
        bool is_open = ((white_pawns & file_mask) == 0) && ((black_pawns & file_mask) == 0);
        bool is_semi_open = ((white_pawns & file_mask) == 0) || ((black_pawns & file_mask) == 0);

        if (is_open) {
            mg_score -= EVAL_ROOK_OPEN_FILE_MG;
            eg_score -= EVAL_ROOK_OPEN_FILE_EG;
        } else if (is_semi_open) {
            mg_score -= EVAL_ROOK_SEMI_OPEN_MG;
            eg_score -= EVAL_ROOK_SEMI_OPEN_EG;
        }
    }

    // Doubled rooks bonus (two rooks on same file or rank)
    int w_rook_count = popcount(white_rooks);
    int b_rook_count = popcount(black_rooks);

    if (w_rook_count >= 2) {
        mg_score += EVAL_DOUBLED_ROOKS_MG;
        eg_score += EVAL_DOUBLED_ROOKS_EG;
    }
    if (b_rook_count >= 2) {
        mg_score -= EVAL_DOUBLED_ROOKS_MG;
        eg_score -= EVAL_DOUBLED_ROOKS_EG;
    }

    *score_mg = mg_score;
    *score_eg = eg_score;
}


__device__ __forceinline__
void evaluate_king_safety(const BoardState* pos, int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    Bitboard white_pawns = pos->pieces[WHITE][PAWN];
    Bitboard black_pawns = pos->pieces[BLACK][PAWN];

    // White king safety - check pawn shield in front of king
    int wk = lsb(pos->pieces[WHITE][KING]);
    int wk_file = wk % 8;
    int wk_rank = wk / 8;

    // Only evaluate in middlegame (king not in center)
    if (wk_rank <= 1) {  // King on back rank
        int shield_count = 0;

        // Check three squares in front of king
        for (int df = -1; df <= 1; df++) {
            int file = wk_file + df;
            if (file >= 0 && file <= 7) {
                Bitboard file_mask = FILE_A_MASK << file;
                // Check if pawn is on rank 2 or 3 in front of king
                Bitboard shield_squares = file_mask & (RANK_2_MASK | RANK_3_MASK);
                if (white_pawns & shield_squares) {
                    shield_count++;
                }
            }
        }

        if (shield_count >= 2) {
            mg_score += EVAL_KING_PAWN_SHIELD;
        } else if (shield_count == 0) {
            mg_score += EVAL_KING_MISSING_SHIELD;
        }
    }

    // Black king safety
    int bk = lsb(pos->pieces[BLACK][KING]);
    int bk_file = bk % 8;
    int bk_rank = bk / 8;

    if (bk_rank >= 6) {  // King on back rank
        int shield_count = 0;

        for (int df = -1; df <= 1; df++) {
            int file = bk_file + df;
            if (file >= 0 && file <= 7) {
                Bitboard file_mask = FILE_A_MASK << file;
                Bitboard shield_squares = file_mask & (RANK_7_MASK | RANK_6_MASK);
                if (black_pawns & shield_squares) {
                    shield_count++;
                }
            }
        }

        if (shield_count >= 2) {
            mg_score -= EVAL_KING_PAWN_SHIELD;
        } else if (shield_count == 0) {
            mg_score -= EVAL_KING_MISSING_SHIELD;
        }
    }

    *score_mg = mg_score;
    *score_eg = eg_score;
}


__device__ int gpu_evaluate(const BoardState* pos) {
    int score_mg = 0;
    int score_eg = 0;
    int mat_mg = 0, mat_eg = 0;
    int pst_mg = 0, pst_eg = 0;
    int pos_mg = 0, pos_eg = 0;
    int pawn_mg = 0, pawn_eg = 0;
    int rook_mg = 0, rook_eg = 0;
    int king_mg = 0, king_eg = 0;

    // Material evaluation with imbalance (includes enhanced bishop pair)
    evaluate_material(pos, 0, &mat_mg, &mat_eg);

    // Piece-square tables
    evaluate_pst(pos, &pst_mg, &pst_eg);

    // Pawn structure evaluation (Phase 2)
    evaluate_pawn_structure(pos, &pawn_mg, &pawn_eg);

    // Positional heuristics
    evaluate_positional(pos, &pos_mg, &pos_eg);

    // Rook evaluation (Phase 3)
    evaluate_rooks(pos, &rook_mg, &rook_eg);

    // King safety evaluation (Phase 3)
    evaluate_king_safety(pos, &king_mg, &king_eg);

    // Combine MG and EG scores
    score_mg = mat_mg + pst_mg + pos_mg + pawn_mg + rook_mg + king_mg;
    score_eg = mat_eg + pst_eg + pos_eg + pawn_eg + rook_eg + king_eg;

    // Calculate game phase
    int phase = calculate_phase(pos);

    // Tapered evaluation
    int score = tapered_eval(score_mg, score_eg, phase);

    // Tempo bonus
    if (pos->side_to_move == WHITE) {
        score += EVAL_TEMPO;
    } else {
        score -= EVAL_TEMPO;
    }

    return score;
}

// Convert centipawn score to win probability using sigmoid
__device__
float score_to_winprob(int score, int side_to_move) {
    if (side_to_move == BLACK) score = -score;

    // Sigmoid: 1 / (1 + exp(-score/400))
    float x = (float)score / 400.0f;
    float ex = expf(-x);
    return 1.0f / (1.0f + ex);
}
