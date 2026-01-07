#ifndef EVALUATION_H
#define EVALUATION_H

#include "chess_types.cuh"

// Material Values (in centipawns)

struct EvalWeights {
    int pawn   = 100;
    int knight = 320;
    int bishop = 330;
    int rook   = 500;
    int queen  = 900;
    int king   = 20000;  
};

// Default weights
static const EvalWeights DEFAULT_WEIGHTS;

// Piece-Square Tables 
// Values in centipawns, added to base piece value

// Pawn PST - encourage central pawns and advancement
static const int PAWN_PST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,   // Rank 1 
     5, 10, 10,-20,-20, 10, 10,  5,   // Rank 2
     5, -5,-10,  0,  0,-10, -5,  5,   // Rank 3
     0,  0,  0, 20, 20,  0,  0,  0,   // Rank 4
     5,  5, 10, 25, 25, 10,  5,  5,   // Rank 5
    10, 10, 20, 30, 30, 20, 10, 10,   // Rank 6
    50, 50, 50, 50, 50, 50, 50, 50,   // Rank 7
     0,  0,  0,  0,  0,  0,  0,  0    // Rank 8 
};

// Knight PST - prefer center, avoid edges
static const int KNIGHT_PST[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,   // Rank 1
   -40,-20,  0,  5,  5,  0,-20,-40,   // Rank 2
   -30,  5, 10, 15, 15, 10,  5,-30,   // Rank 3
   -30,  0, 15, 20, 20, 15,  0,-30,   // Rank 4
   -30,  5, 15, 20, 20, 15,  5,-30,   // Rank 5
   -30,  0, 10, 15, 15, 10,  0,-30,   // Rank 6
   -40,-20,  0,  0,  0,  0,-20,-40,   // Rank 7
   -50,-40,-30,-30,-30,-30,-40,-50    // Rank 8
};

// Bishop PST - prefer diagonals and center
static const int BISHOP_PST[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,   // Rank 1
   -10,  5,  0,  0,  0,  0,  5,-10,   // Rank 2
   -10, 10, 10, 10, 10, 10, 10,-10,   // Rank 3
   -10,  0, 10, 10, 10, 10,  0,-10,   // Rank 4
   -10,  5,  5, 10, 10,  5,  5,-10,   // Rank 5
   -10,  0,  5, 10, 10,  5,  0,-10,   // Rank 6
   -10,  0,  0,  0,  0,  0,  0,-10,   // Rank 7
   -20,-10,-10,-10,-10,-10,-10,-20    // Rank 8
};

// Rook PST - prefer 7th rank and open files
static const int ROOK_PST[64] = {
     0,  0,  0,  5,  5,  0,  0,  0,   // Rank 1
    -5,  0,  0,  0,  0,  0,  0, -5,   // Rank 2
    -5,  0,  0,  0,  0,  0,  0, -5,   // Rank 3
    -5,  0,  0,  0,  0,  0,  0, -5,   // Rank 4
    -5,  0,  0,  0,  0,  0,  0, -5,   // Rank 5
    -5,  0,  0,  0,  0,  0,  0, -5,   // Rank 6
     5, 10, 10, 10, 10, 10, 10,  5,   // Rank 7 
     0,  0,  0,  0,  0,  0,  0,  0    // Rank 8
};

// Queen PST - slight center preference, avoid early development
static const int QUEEN_PST[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,   // Rank 1
   -10,  0,  5,  0,  0,  0,  0,-10,   // Rank 2
   -10,  5,  5,  5,  5,  5,  0,-10,   // Rank 3
     0,  0,  5,  5,  5,  5,  0, -5,   // Rank 4
    -5,  0,  5,  5,  5,  5,  0, -5,   // Rank 5
   -10,  0,  5,  5,  5,  5,  0,-10,   // Rank 6
   -10,  0,  0,  0,  0,  0,  0,-10,   // Rank 7
   -20,-10,-10, -5, -5,-10,-10,-20    // Rank 8
};

// King PST (Middlegame) - encourage castling, stay safe
static const int KING_MG_PST[64] = {
    20, 30, 10,  0,  0, 10, 30, 20,   // Rank 1 
    20, 20,  0,  0,  0,  0, 20, 20,   // Rank 2
   -10,-20,-20,-20,-20,-20,-20,-10,   // Rank 3
   -20,-30,-30,-40,-40,-30,-30,-20,   // Rank 4
   -30,-40,-40,-50,-50,-40,-40,-30,   // Rank 5
   -30,-40,-40,-50,-50,-40,-40,-30,   // Rank 6
   -30,-40,-40,-50,-50,-40,-40,-30,   // Rank 7
   -30,-40,-40,-50,-50,-40,-40,-30    // Rank 8
};

// King PST (Endgame) - centralize the king
static const int KING_EG_PST[64] = {
   -50,-30,-30,-30,-30,-30,-30,-50,   // Rank 1
   -30,-30,  0,  0,  0,  0,-30,-30,   // Rank 2
   -30,-10, 20, 30, 30, 20,-10,-30,   // Rank 3
   -30,-10, 30, 40, 40, 30,-10,-30,   // Rank 4
   -30,-10, 30, 40, 40, 30,-10,-30,   // Rank 5
   -30,-10, 20, 30, 30, 20,-10,-30,   // Rank 6
   -30,-20,-10,  0,  0,-10,-20,-30,   // Rank 7
   -50,-40,-30,-20,-20,-30,-40,-50    // Rank 8
};

// Game Phase Detection

// Phase values for each piece type (used to detect endgame)
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

// Evaluation Functions

// Flip square for black's perspective
inline int flip_square(int sq) {
    return sq ^ 56;  // Flip rank (0-7 becomes 7-0)
}

// Evaluate material only
inline int evaluate_material(const BoardState& board, const EvalWeights& weights = DEFAULT_WEIGHTS) {
    int score = 0;

    // White material
    score += popcount(board.pieces[WHITE][PAWN])   * weights.pawn;
    score += popcount(board.pieces[WHITE][KNIGHT]) * weights.knight;
    score += popcount(board.pieces[WHITE][BISHOP]) * weights.bishop;
    score += popcount(board.pieces[WHITE][ROOK])   * weights.rook;
    score += popcount(board.pieces[WHITE][QUEEN])  * weights.queen;

    // Black material (subtract)
    score -= popcount(board.pieces[BLACK][PAWN])   * weights.pawn;
    score -= popcount(board.pieces[BLACK][KNIGHT]) * weights.knight;
    score -= popcount(board.pieces[BLACK][BISHOP]) * weights.bishop;
    score -= popcount(board.pieces[BLACK][ROOK])   * weights.rook;
    score -= popcount(board.pieces[BLACK][QUEEN])  * weights.queen;

    return score;
}

// Evaluate piece-square table contribution for one side
inline int evaluate_pst_side(const BoardState& board, int color) {
    int score = 0;
    int sign = (color == WHITE) ? 1 : -1;

    // Pawns
    Bitboard pawns = board.pieces[color][PAWN];
    while (pawns) {
        int sq = pop_lsb_index(pawns);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        score += PAWN_PST[psq] * sign;
    }

    // Knights
    Bitboard knights = board.pieces[color][KNIGHT];
    while (knights) {
        int sq = pop_lsb_index(knights);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        score += KNIGHT_PST[psq] * sign;
    }

    // Bishops
    Bitboard bishops = board.pieces[color][BISHOP];
    while (bishops) {
        int sq = pop_lsb_index(bishops);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        score += BISHOP_PST[psq] * sign;
    }

    // Rooks
    Bitboard rooks = board.pieces[color][ROOK];
    while (rooks) {
        int sq = pop_lsb_index(rooks);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        score += ROOK_PST[psq] * sign;
    }

    // Queens
    Bitboard queens = board.pieces[color][QUEEN];
    while (queens) {
        int sq = pop_lsb_index(queens);
        int psq = (color == WHITE) ? sq : flip_square(sq);
        score += QUEEN_PST[psq] * sign;
    }

    return score;
}

// Evaluate king position with game phase interpolation
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
    return (mg_score * phase + eg_score * (256 - phase)) / 256;
}

// Full static evaluation
// Returns score in centipawns from WHITE's perspective
inline int evaluate(const BoardState& board, const EvalWeights& weights = DEFAULT_WEIGHTS) {
    int score = 0;

    // Material
    score += evaluate_material(board, weights);

    // Piece-square tables
    score += evaluate_pst_side(board, WHITE);
    score += evaluate_pst_side(board, BLACK);

    // King safety with phase interpolation
    int phase = calculate_phase(board);
    score += evaluate_king(board, phase);

    // Return from perspective of side to move
    return (board.side_to_move == WHITE) ? score : -score;
}

// Quick evaluation (material only) for use in playouts
inline int evaluate_quick(const BoardState& board) {
    int score = evaluate_material(board);
    return (board.side_to_move == WHITE) ? score : -score;
}

// GPU-compatible evaluation (device functions)

#ifdef __CUDACC__

__constant__ int d_PAWN_PST[64];
__constant__ int d_KNIGHT_PST[64];
__constant__ int d_BISHOP_PST[64];
__constant__ int d_ROOK_PST[64];
__constant__ int d_QUEEN_PST[64];
__constant__ int d_KING_MG_PST[64];
__constant__ int d_KING_EG_PST[64];

__device__ __forceinline__
int d_flip_square(int sq) {
    return sq ^ 56;
}

__device__ __forceinline__
int d_evaluate_material(const BoardState* board) {
    int score = 0;

    score += popcount(board->pieces[WHITE][PAWN])   * 100;
    score += popcount(board->pieces[WHITE][KNIGHT]) * 320;
    score += popcount(board->pieces[WHITE][BISHOP]) * 330;
    score += popcount(board->pieces[WHITE][ROOK])   * 500;
    score += popcount(board->pieces[WHITE][QUEEN])  * 900;

    score -= popcount(board->pieces[BLACK][PAWN])   * 100;
    score -= popcount(board->pieces[BLACK][KNIGHT]) * 320;
    score -= popcount(board->pieces[BLACK][BISHOP]) * 330;
    score -= popcount(board->pieces[BLACK][ROOK])   * 500;
    score -= popcount(board->pieces[BLACK][QUEEN])  * 900;

    return score;
}

__device__ __forceinline__
int d_evaluate(const BoardState* board) {
    int score = d_evaluate_material(board);

    return (board->side_to_move == WHITE) ? score : -score;
}

#endif 

#endif 
