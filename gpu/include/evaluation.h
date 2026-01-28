#ifndef EVALUATION_H
#define EVALUATION_H

#include "chess_types.cuh"


struct EvalWeights {
    // Middlegame values
    int pawn_mg   = 136;
    int knight_mg = 782;
    int bishop_mg = 830;
    int rook_mg   = 1289;
    int queen_mg  = 2529;

    // Endgame values
    int pawn_eg   = 208;
    int knight_eg = 865;
    int bishop_eg = 918;
    int rook_eg   = 1378;
    int queen_eg  = 2687;

    int king   = 20000;
};

// Default weights
static const EvalWeights DEFAULT_WEIGHTS;


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


#define IMBALANCE_THREE_MINORS_MG     30
#define IMBALANCE_THREE_MINORS_EG     50
#define IMBALANCE_ROOK_NO_MINORS_MG   -20
#define IMBALANCE_ROOK_NO_MINORS_EG   -30


static const int PHASE_KNIGHT = 1;
static const int PHASE_BISHOP = 1;
static const int PHASE_ROOK   = 2;
static const int PHASE_QUEEN  = 4;
static const int TOTAL_PHASE  = 4 * PHASE_KNIGHT + 4 * PHASE_BISHOP +
                                4 * PHASE_ROOK + 2 * PHASE_QUEEN;  // = 24

// Calculate game phase (0 = endgame, 256 = opening/middlegame)
inline int calculate_phase(const BoardState& board) {
    int phase = TOTAL_PHASE;

    phase -= popcount(board.pieces[WHITE][KNIGHT] | board.pieces[BLACK][KNIGHT]) * PHASE_KNIGHT;
    phase -= popcount(board.pieces[WHITE][BISHOP] | board.pieces[BLACK][BISHOP]) * PHASE_BISHOP;
    phase -= popcount(board.pieces[WHITE][ROOK]   | board.pieces[BLACK][ROOK])   * PHASE_ROOK;
    phase -= popcount(board.pieces[WHITE][QUEEN]  | board.pieces[BLACK][QUEEN])  * PHASE_QUEEN;

    // Scale to 0-256 range (256 = full material, 0 = endgame)
    return (phase * 256 + TOTAL_PHASE / 2) / TOTAL_PHASE;
}

// Tapered evaluation: interpolate between MG and EG scores
inline int tapered_eval(int mg_score, int eg_score, int phase) {
    // phase: 256 = all MG, 0 = all EG
    return (mg_score * phase + eg_score * (256 - phase)) / 256;
}


// PAWN PST (same for MG and EG)
static const int PAWN_PST[64] = {
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
static const int KNIGHT_MG_PST[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};

// KNIGHT PST - Endgame
static const int KNIGHT_EG_PST[64] = {
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
static const int BISHOP_MG_PST[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

// BISHOP PST - Endgame
static const int BISHOP_EG_PST[64] = {
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
static const int ROOK_PST[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

// KING PST (Middlegame)
static const int KING_MG_PST[64] = {
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30
};

// KING PST (Endgame)
static const int KING_EG_PST[64] = {
   -50,-30,-30,-30,-30,-30,-30,-50,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -50,-40,-30,-20,-20,-30,-40,-50
};

// Center control masks
static const uint64_t CENTER_MASK = 0x0000001818000000ULL;  // d4, d5, e4, e5
static const uint64_t EXTENDED_CENTER_MASK = 0x00003C3C3C3C0000ULL;  // c3-f6 area

// Wing masks for bishop pair enhancement (Phase 3)
static const uint64_t QUEEN_SIDE_MASK = 0x0F0F0F0F0F0F0F0FULL;  // Files a-d
static const uint64_t KING_SIDE_MASK = 0xF0F0F0F0F0F0F0F0ULL;   // Files e-h

// File masks for pawn structure
static const uint64_t FILE_A_MASK = 0x0101010101010101ULL;
static const uint64_t FILE_B_MASK = 0x0202020202020202ULL;
static const uint64_t FILE_C_MASK = 0x0404040404040404ULL;
static const uint64_t FILE_D_MASK = 0x0808080808080808ULL;
static const uint64_t FILE_E_MASK = 0x1010101010101010ULL;
static const uint64_t FILE_F_MASK = 0x2020202020202020ULL;
static const uint64_t FILE_G_MASK = 0x4040404040404040ULL;
static const uint64_t FILE_H_MASK = 0x8080808080808080ULL;


// Flip square for black's perspective
inline int flip_square(int sq) {
    return sq ^ 56;  // Flip rank (0-7 becomes 7-0)
}

// Evaluate material with imbalance (Phase 1)
inline void evaluate_material(const BoardState& board, const EvalWeights& weights,
                              int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    // White pieces
    int w_pawns   = popcount(board.pieces[WHITE][PAWN]);
    int w_knights = popcount(board.pieces[WHITE][KNIGHT]);
    int w_bishops = popcount(board.pieces[WHITE][BISHOP]);
    int w_rooks   = popcount(board.pieces[WHITE][ROOK]);
    int w_queens  = popcount(board.pieces[WHITE][QUEEN]);

    // Black pieces
    int b_pawns   = popcount(board.pieces[BLACK][PAWN]);
    int b_knights = popcount(board.pieces[BLACK][KNIGHT]);
    int b_bishops = popcount(board.pieces[BLACK][BISHOP]);
    int b_rooks   = popcount(board.pieces[BLACK][ROOK]);
    int b_queens  = popcount(board.pieces[BLACK][QUEEN]);

    // Base material values
    mg_score = (w_pawns * weights.pawn_mg + w_knights * weights.knight_mg + w_bishops * weights.bishop_mg +
               w_rooks * weights.rook_mg + w_queens * weights.queen_mg) -
              (b_pawns * weights.pawn_mg + b_knights * weights.knight_mg + b_bishops * weights.bishop_mg +
               b_rooks * weights.rook_mg + b_queens * weights.queen_mg);

    eg_score = (w_pawns * weights.pawn_eg + w_knights * weights.knight_eg + w_bishops * weights.bishop_eg +
               w_rooks * weights.rook_eg + w_queens * weights.queen_eg) -
              (b_pawns * weights.pawn_eg + b_knights * weights.knight_eg + b_bishops * weights.bishop_eg +
               b_rooks * weights.rook_eg + b_queens * weights.queen_eg);

    // Bishop pair bonus with wing pawn enhancement (Phase 3)
    if (w_bishops >= 2) {
        int bishop_bonus_mg = EVAL_BISHOP_PAIR_MG;
        int bishop_bonus_eg = EVAL_BISHOP_PAIR_EG;

        // Additional bonus if pawns are on both wings
        Bitboard white_pawns = board.pieces[WHITE][PAWN];
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
        Bitboard black_pawns = board.pieces[BLACK][PAWN];
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
    int w_minors = w_knights + w_bishops;
    int b_minors = b_knights + b_bishops;

    // Three minors bonus
    if (w_minors >= 3 && b_minors >= 3) {
        mg_score += IMBALANCE_THREE_MINORS_MG;
        eg_score += IMBALANCE_THREE_MINORS_EG;
    }
    if (b_minors >= 3 && w_minors >= 3) {
        mg_score -= IMBALANCE_THREE_MINORS_MG;
        eg_score -= IMBALANCE_THREE_MINORS_EG;
    }

    // Rook with no minors penalty
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

// Evaluate piece-square table contribution for one side (Phase 1 - with MG/EG)
inline void evaluate_pst_side(const BoardState& board, int color,
                              int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;
    int sign = (color == WHITE) ? 1 : -1;

    // Pawns
    Bitboard pawns = board.pieces[color][PAWN];
    while (pawns) {
        int sq = pop_lsb_index(pawns);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        int val = PAWN_PST[psq] * sign;
        mg_score += val;
        eg_score += val;
    }

    // Knights
    Bitboard knights = board.pieces[color][KNIGHT];
    while (knights) {
        int sq = pop_lsb_index(knights);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        mg_score += KNIGHT_MG_PST[psq] * sign;
        eg_score += KNIGHT_EG_PST[psq] * sign;
    }

    // Bishops
    Bitboard bishops = board.pieces[color][BISHOP];
    while (bishops) {
        int sq = pop_lsb_index(bishops);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        mg_score += BISHOP_MG_PST[psq] * sign;
        eg_score += BISHOP_EG_PST[psq] * sign;
    }

    // Rooks
    Bitboard rooks = board.pieces[color][ROOK];
    while (rooks) {
        int sq = pop_lsb_index(rooks);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        int val = ROOK_PST[psq] * sign;
        mg_score += val;
        eg_score += val;
    }

    *score_mg = mg_score;
    *score_eg = eg_score;
}

// Evaluate king position with game phase interpolation (already existed)
inline int evaluate_king(const BoardState& board, int phase) {
    int mg_score = 0;
    int eg_score = 0;

    // White king
    int wk_sq = lsb(board.pieces[WHITE][KING]);
    mg_score += KING_MG_PST[wk_sq];
    eg_score += KING_EG_PST[wk_sq];

    // Black king (flip perspective)
    int bk_sq = lsb(board.pieces[BLACK][KING]);
    int bk_psq = flip_square(bk_sq);
    mg_score -= KING_MG_PST[bk_psq];
    eg_score -= KING_EG_PST[bk_psq];

    // Interpolate between middlegame and endgame scores
    return tapered_eval(mg_score, eg_score, phase);
}

// Evaluate pawn structure (Phase 2)
inline void evaluate_pawn_structure(const BoardState& board,
                                    int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    Bitboard white_pawns = board.pieces[WHITE][PAWN];
    Bitboard black_pawns = board.pieces[BLACK][PAWN];

    // Evaluate white pawns
    Bitboard wp = white_pawns;
    while (wp) {
        int sq = pop_lsb_index(wp);
        int file = sq % 8;
        int rank = sq / 8;

        // Check for passed pawn
        bool is_passed = true;
        Bitboard blocking_files = 0;
        if (file > 0) blocking_files |= FILE_A_MASK << (file - 1);
        if (file < 7) blocking_files |= FILE_A_MASK << (file + 1);
        blocking_files |= (FILE_A_MASK << file);

        // Black pawns ahead of this pawn
        Bitboard rank_mask = 0;
        if (rank >= 1) rank_mask |= 0xFF00ULL;
        if (rank >= 2) rank_mask |= 0xFF0000ULL;
        if (rank >= 3) rank_mask |= 0xFF000000ULL;
        if (rank >= 4) rank_mask |= 0xFF00000000ULL;
        if (rank >= 5) rank_mask |= 0xFF0000000000ULL;
        if (rank >= 6) rank_mask |= 0xFF000000000000ULL;

        Bitboard black_pawns_ahead = black_pawns & blocking_files & rank_mask;
        if (black_pawns_ahead) is_passed = false;

        // Passed pawn bonus
        if (is_passed) {
            int rank_bonus = rank * 20;
            mg_score += rank_bonus;
            eg_score += rank_bonus * 2;
        }

        // Isolated pawn
        bool is_isolated = true;
        if (file > 0) {
            if (white_pawns & (FILE_A_MASK << (file - 1))) is_isolated = false;
        }
        if (file < 7) {
            if (white_pawns & (FILE_A_MASK << (file + 1))) is_isolated = false;
        }

        if (is_isolated) {
            mg_score -= 20;
            eg_score -= 30;
        }

        // Doubled pawns
        Bitboard file_mask = FILE_A_MASK << file;
        if (popcount(white_pawns & file_mask) >= 2) {
            mg_score -= 10;
            eg_score -= 15;
        }

        // Backward pawn (simplified check)
        bool is_backward = false;
        if (rank < 6 && rank > 0) {
            // Check if enemy pawns control square ahead
            bool enemy_control = false;
            if (file > 0 && (black_pawns & ((FILE_A_MASK << (file - 1)) & (0xFFULL << (sq + 8))))) enemy_control = true;
            if (file < 7 && (black_pawns & ((FILE_A_MASK << (file + 1)) & (0xFFULL << (sq + 8))))) enemy_control = true;

            // Check if can be defended
            bool can_be_defended = false;
            if (sq >= 8) {
                if (file > 0 && (white_pawns & (1ULL << (sq - 9)))) can_be_defended = true;
                if (file < 7 && (white_pawns & (1ULL << (sq - 7)))) can_be_defended = true;
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

    // Evaluate black pawns (mirrored)
    Bitboard bp = black_pawns;
    while (bp) {
        int sq = pop_lsb_index(bp);
        int file = sq % 8;
        int rank = sq / 8;

        // Check for passed pawn
        bool is_passed = true;
        Bitboard blocking_files = 0;
        if (file > 0) blocking_files |= FILE_A_MASK << (file - 1);
        if (file < 7) blocking_files |= FILE_A_MASK << (file + 1);
        blocking_files |= (FILE_A_MASK << file);

        // White pawns ahead
        Bitboard rank_mask = 0;
        if (rank <= 5) rank_mask |= 0xFF000000000000ULL;
        if (rank <= 4) rank_mask |= 0xFF0000000000ULL;
        if (rank <= 3) rank_mask |= 0xFF00000000ULL;
        if (rank <= 2) rank_mask |= 0xFF000000ULL;
        if (rank <= 1) rank_mask |= 0xFF0000ULL;
        if (rank <= 0) rank_mask |= 0xFF00ULL;

        Bitboard white_pawns_ahead = white_pawns & blocking_files & rank_mask;
        if (white_pawns_ahead) is_passed = false;

        // Passed pawn bonus
        if (is_passed) {
            int rank_bonus = (7 - rank) * 20;
            mg_score -= rank_bonus;
            eg_score -= rank_bonus * 2;
        }

        // Isolated pawn
        bool is_isolated = true;
        if (file > 0) {
            if (black_pawns & (FILE_A_MASK << (file - 1))) is_isolated = false;
        }
        if (file < 7) {
            if (black_pawns & (FILE_A_MASK << (file + 1))) is_isolated = false;
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
        if (rank > 1 && rank < 7) {
            bool enemy_control = false;
            if (file > 0 && (white_pawns & ((FILE_A_MASK << (file - 1)) & (0xFFULL >> (8 - rank))))) enemy_control = true;
            if (file < 7 && (white_pawns & ((FILE_A_MASK << (file + 1)) & (0xFFULL >> (8 - rank))))) enemy_control = true;

            bool can_be_defended = false;
            if (sq <= 55) {
                if (file > 0 && (black_pawns & (1ULL << (sq + 7)))) can_be_defended = true;
                if (file < 7 && (black_pawns & (1ULL << (sq + 9)))) can_be_defended = true;
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

// Evaluate positional features (Phase 1)
inline void evaluate_positional(const BoardState& board,
                                int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    // Rook on 7th rank bonus
    const uint64_t RANK_7_WHITE = 0xFF00ULL;
    const uint64_t RANK_7_BLACK = 0xFF000000000000ULL;

    if (board.pieces[WHITE][ROOK] & RANK_7_WHITE) {
        mg_score += EVAL_ROOK_ON_7TH;
        eg_score += EVAL_ROOK_ON_7TH;
    }
    if (board.pieces[BLACK][ROOK] & RANK_7_BLACK) {
        mg_score -= EVAL_ROOK_ON_7TH;
        eg_score -= EVAL_ROOK_ON_7TH;
    }

    // Center control - original 4 squares
    int w_knights_center = popcount(board.pieces[WHITE][KNIGHT] & CENTER_MASK);
    int b_knights_center = popcount(board.pieces[BLACK][KNIGHT] & CENTER_MASK);
    int w_bishops_center = popcount(board.pieces[WHITE][BISHOP] & CENTER_MASK);
    int b_bishops_center = popcount(board.pieces[BLACK][BISHOP] & CENTER_MASK);

    mg_score += EVAL_KNIGHT_CENTER * w_knights_center;
    mg_score -= EVAL_KNIGHT_CENTER * b_knights_center;
    mg_score += EVAL_BISHOP_CENTER * w_bishops_center;
    mg_score -= EVAL_BISHOP_CENTER * b_bishops_center;

    // Extended center control (16 squares)
    int w_knights_ext = popcount(board.pieces[WHITE][KNIGHT] & EXTENDED_CENTER_MASK);
    int b_knights_ext = popcount(board.pieces[BLACK][KNIGHT] & EXTENDED_CENTER_MASK);
    int w_bishops_ext = popcount(board.pieces[WHITE][BISHOP] & EXTENDED_CENTER_MASK);
    int b_bishops_ext = popcount(board.pieces[BLACK][BISHOP] & EXTENDED_CENTER_MASK);

    mg_score += EVAL_KNIGHT_EXTENDED_CENTER * w_knights_ext;
    mg_score -= EVAL_KNIGHT_EXTENDED_CENTER * b_knights_ext;
    mg_score += EVAL_BISHOP_EXTENDED_CENTER * w_bishops_ext;
    mg_score -= EVAL_BISHOP_EXTENDED_CENTER * b_bishops_ext;

    *score_mg = mg_score;
    *score_eg = eg_score;
}


inline void evaluate_rooks(const BoardState& board,
                          int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    Bitboard white_pawns = board.pieces[WHITE][PAWN];
    Bitboard black_pawns = board.pieces[BLACK][PAWN];
    Bitboard white_rooks = board.pieces[WHITE][ROOK];
    Bitboard black_rooks = board.pieces[BLACK][ROOK];

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

    // Doubled rooks bonus
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


inline void evaluate_king_safety(const BoardState& board,
                                int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    Bitboard white_pawns = board.pieces[WHITE][PAWN];
    Bitboard black_pawns = board.pieces[BLACK][PAWN];

    // Rank masks for pawn shield evaluation
    const uint64_t RANK_2_MASK = 0xFF00ULL;
    const uint64_t RANK_3_MASK = 0xFF0000ULL;
    const uint64_t RANK_6_MASK = 0xFF0000000000ULL;
    const uint64_t RANK_7_MASK = 0xFF000000000000ULL;

    // White king safety - check pawn shield in front of king
    int wk = lsb(board.pieces[WHITE][KING]);
    int wk_file = wk % 8;
    int wk_rank = wk / 8;

    // Only evaluate in middlegame (king on back rank)
    if (wk_rank <= 1) {
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
    int bk = lsb(board.pieces[BLACK][KING]);
    int bk_file = bk % 8;
    int bk_rank = bk / 8;

    if (bk_rank >= 6) {
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

// Full static evaluation with Phase 1+2+3 features
inline int evaluate(const BoardState& board, const EvalWeights& weights = DEFAULT_WEIGHTS) {
    int mg_score = 0;
    int eg_score = 0;
    int mat_mg = 0, mat_eg = 0;
    int pst_mg = 0, pst_eg = 0;
    int pos_mg = 0, pos_eg = 0;
    int pawn_mg = 0, pawn_eg = 0;
    int rook_mg = 0, rook_eg = 0;
    int king_mg = 0, king_eg = 0;

    // Material evaluation with imbalance (includes enhanced bishop pair)
    evaluate_material(board, weights, &mat_mg, &mat_eg);

    // Piece-square tables
    evaluate_pst_side(board, WHITE, &pst_mg, &pst_eg);
    int black_pst_mg = 0, black_pst_eg = 0;
    evaluate_pst_side(board, BLACK, &black_pst_mg, &black_pst_eg);
    pst_mg += black_pst_mg;
    pst_eg += black_pst_eg;

    // Pawn structure evaluation (Phase 2)
    evaluate_pawn_structure(board, &pawn_mg, &pawn_eg);

    // Positional heuristics
    evaluate_positional(board, &pos_mg, &pos_eg);

    // Rook evaluation (Phase 3)
    evaluate_rooks(board, &rook_mg, &rook_eg);

    // King safety evaluation (Phase 3)
    evaluate_king_safety(board, &king_mg, &king_eg);

    // Combine all scores
    mg_score = mat_mg + pst_mg + pos_mg + pawn_mg + rook_mg + king_mg;
    eg_score = mat_eg + pst_eg + pos_eg + pawn_eg + rook_eg + king_eg;

    // Calculate game phase for tapered evaluation
    int phase = calculate_phase(board);

    // Tapered evaluation
    int score = tapered_eval(mg_score, eg_score, phase);

    // Tempo
    if (board.side_to_move == WHITE) {
        score += EVAL_TEMPO;
    } else {
        score -= EVAL_TEMPO;
    }

    return score;
}

// Quick evaluation (material only) for use in playouts
inline int evaluate_quick(const BoardState& board) {
    int score = 0;

    score += popcount(board.pieces[WHITE][PAWN])   * 100;
    score += popcount(board.pieces[WHITE][KNIGHT]) * 320;
    score += popcount(board.pieces[WHITE][BISHOP]) * 330;
    score += popcount(board.pieces[WHITE][ROOK])   * 500;
    score += popcount(board.pieces[WHITE][QUEEN])  * 900;

    score -= popcount(board.pieces[BLACK][PAWN])   * 100;
    score -= popcount(board.pieces[BLACK][KNIGHT]) * 320;
    score -= popcount(board.pieces[BLACK][BISHOP]) * 330;
    score -= popcount(board.pieces[BLACK][ROOK])   * 500;
    score -= popcount(board.pieces[BLACK][QUEEN])  * 900;

    return (board.side_to_move == WHITE) ? score : -score;
}


#ifdef __CUDACC__

__constant__ int d_PAWN_PST[64];
__constant__ int d_KNIGHT_MG_PST[64];
__constant__ int d_KNIGHT_EG_PST[64];
__constant__ int d_BISHOP_MG_PST[64];
__constant__ int d_BISHOP_EG_PST[64];
__constant__ int d_ROOK_PST[64];
__constant__ int d_KING_MG_PST[64];
__constant__ int d_KING_EG_PST[64];

__device__ __forceinline__
int d_flip_square(int sq) {
    return sq ^ 56;
}

__device__ __forceinline__
int d_tapered_eval(int mg_score, int eg_score, int phase) {
    return (mg_score * phase + eg_score * (256 - phase)) / 256;
}

__device__ __forceinline__
int d_calculate_phase(const BoardState* board) {
    int phase = TOTAL_PHASE;

    phase -= popcount(board->pieces[WHITE][KNIGHT] | board->pieces[BLACK][KNIGHT]) * PHASE_KNIGHT;
    phase -= popcount(board->pieces[WHITE][BISHOP] | board->pieces[BLACK][BISHOP]) * PHASE_BISHOP;
    phase -= popcount(board->pieces[WHITE][ROOK]   | board->pieces[BLACK][ROOK])   * PHASE_ROOK;
    phase -= popcount(board->pieces[WHITE][QUEEN]  | board->pieces[BLACK][QUEEN])  * PHASE_QUEEN;

    return (phase * 256 + TOTAL_PHASE / 2) / TOTAL_PHASE;
}

__device__ __forceinline__
int d_evaluate_material(const BoardState* board, int* score_mg, int* score_eg) {
    int mg_score = 0;
    int eg_score = 0;

    mg_score = (popcount(board->pieces[WHITE][PAWN])   * 136 +
                popcount(board->pieces[WHITE][KNIGHT]) * 782 +
                popcount(board->pieces[WHITE][BISHOP]) * 830 +
                popcount(board->pieces[WHITE][ROOK])   * 1289 +
                popcount(board->pieces[WHITE][QUEEN])  * 2529) -
               (popcount(board->pieces[BLACK][PAWN])   * 136 +
                popcount(board->pieces[BLACK][KNIGHT]) * 782 +
                popcount(board->pieces[BLACK][BISHOP]) * 830 +
                popcount(board->pieces[BLACK][ROOK])   * 1289 +
                popcount(board->pieces[BLACK][QUEEN])  * 2529);

    eg_score = (popcount(board->pieces[WHITE][PAWN])   * 208 +
                popcount(board->pieces[WHITE][KNIGHT]) * 865 +
                popcount(board->pieces[WHITE][BISHOP]) * 918 +
                popcount(board->pieces[WHITE][ROOK])   * 1378 +
                popcount(board->pieces[WHITE][QUEEN])  * 2687) -
               (popcount(board->pieces[BLACK][PAWN])   * 208 +
                popcount(board->pieces[BLACK][KNIGHT]) * 865 +
                popcount(board->pieces[BLACK][BISHOP]) * 918 +
                popcount(board->pieces[BLACK][ROOK])   * 1378 +
                popcount(board->pieces[BLACK][QUEEN])  * 2687);

    // Bishop pair
    if (popcount(board->pieces[WHITE][BISHOP]) >= 2) {
        mg_score += 57;
        eg_score += 93;
    }
    if (popcount(board->pieces[BLACK][BISHOP]) >= 2) {
        mg_score -= 57;
        eg_score -= 93;
    }

    *score_mg = mg_score;
    *score_eg = eg_score;
    return 0;
}

__device__ __forceinline__
int d_evaluate(const BoardState* board) {
    int score_mg = 0, score_eg = 0;
    int mat_mg = 0, mat_eg = 0;

    d_evaluate_material(board, &mat_mg, &mat_eg);
    score_mg = mat_mg;
    score_eg = mat_eg;

    int phase = d_calculate_phase(board);
    int score = d_tapered_eval(score_mg, score_eg, phase);

    return (board->side_to_move == WHITE) ? score : -score;
}

#endif

#endif
