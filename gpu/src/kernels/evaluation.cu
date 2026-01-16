#include "../../include/chess_types.cuh"

// Evaluation is done in centipawns https://www.chessprogramming.org/Centipawns
// Next steps include Tapered eval so the engine can distinguish better between the phases of the game https://www.chessprogramming.org/Tapered_Eval

// Material Values
#define EVAL_PAWN   100
#define EVAL_KNIGHT 320
#define EVAL_BISHOP 330
#define EVAL_ROOK   500
#define EVAL_QUEEN  900

// Positional bonuses
#define EVAL_BISHOP_PAIR          50
#define EVAL_ROOK_ON_7TH          20
#define EVAL_TEMPO                10
#define EVAL_KNIGHT_CENTER        10
#define EVAL_BISHOP_CENTER        10
#define EVAL_QUEEN_NEAR_KING      10  

// Piece-square tables (PST)

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

__constant__ int8_t g_PST_KNIGHT[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};

__constant__ int8_t g_PST_BISHOP[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

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

// Mask for the 4 central squares (d4, d5, e4, e5)
// 00000000
// 00000000
// 00000000
// 00011000
// 00011000
// 00000000
// 00000000
// 00000000

const uint64_t CENTER_MASK = 0x0000001818000000ULL;

__device__ int gpu_evaluate(const BoardState* pos) {
    int score = 0;

    // Simple material eval
    // Popcount to get number of bits set to 1 * eval of the piece to get the value
    score += popcount(pos->pieces[WHITE][PAWN])   * EVAL_PAWN;
    score += popcount(pos->pieces[WHITE][KNIGHT]) * EVAL_KNIGHT;
    score += popcount(pos->pieces[WHITE][BISHOP]) * EVAL_BISHOP;
    score += popcount(pos->pieces[WHITE][ROOK])   * EVAL_ROOK;
    score += popcount(pos->pieces[WHITE][QUEEN])  * EVAL_QUEEN;

    score -= popcount(pos->pieces[BLACK][PAWN])   * EVAL_PAWN;
    score -= popcount(pos->pieces[BLACK][KNIGHT]) * EVAL_KNIGHT;
    score -= popcount(pos->pieces[BLACK][BISHOP]) * EVAL_BISHOP;
    score -= popcount(pos->pieces[BLACK][ROOK])   * EVAL_ROOK;
    score -= popcount(pos->pieces[BLACK][QUEEN])  * EVAL_QUEEN;


    // PST
    Bitboard bb;

    // White Pieces
    // Pop peace -> get lsb -> look up value at position in pst -> add to overall value
    bb = pos->pieces[WHITE][PAWN];
    while (bb) { int sq = pop_lsb_index(bb); score += g_PST_PAWN[sq]; }

    bb = pos->pieces[WHITE][KNIGHT];
    while (bb) { int sq = pop_lsb_index(bb); score += g_PST_KNIGHT[sq]; }

    bb = pos->pieces[WHITE][BISHOP];
    while (bb) { int sq = pop_lsb_index(bb); score += g_PST_BISHOP[sq]; }

    bb = pos->pieces[WHITE][ROOK];
    while (bb) { int sq = pop_lsb_index(bb); score += g_PST_ROOK[sq]; }

    // Black Pieces (Mirrored)
    bb = pos->pieces[BLACK][PAWN];
    while (bb) { int sq = pop_lsb_index(bb); score -= g_PST_PAWN[sq ^ 56]; }

    bb = pos->pieces[BLACK][KNIGHT];
    while (bb) { int sq = pop_lsb_index(bb); score -= g_PST_KNIGHT[sq ^ 56]; }

    bb = pos->pieces[BLACK][BISHOP];
    while (bb) { int sq = pop_lsb_index(bb); score -= g_PST_BISHOP[sq ^ 56]; }

    bb = pos->pieces[BLACK][ROOK];
    while (bb) { int sq = pop_lsb_index(bb); score -= g_PST_ROOK[sq ^ 56]; }

    // King safety - same as other pieces
    int wk = lsb(pos->pieces[WHITE][KING]);
    int bk = lsb(pos->pieces[BLACK][KING]);
    score += g_PST_KING_MG[wk];
    score -= g_PST_KING_MG[bk ^ 56];


    // Positional heuristics 

    // Bishop pair bonus - usually an advantage
    if (popcount(pos->pieces[WHITE][BISHOP]) >= 2) score += EVAL_BISHOP_PAIR;
    if (popcount(pos->pieces[BLACK][BISHOP]) >= 2) score -= EVAL_BISHOP_PAIR;

    // Rook on 7th rank bonus

    // 11111111
    // 00000000
    const uint64_t RANK_7_WHITE = 0xFF00ULL;

    // 11111111
    // 00000000
    // 00000000
    // 00000000
    // 00000000
    // 00000000
    // 00000000
    const uint64_t RANK_7_BLACK = 0xFF000000000000ULL;
    
    if (pos->pieces[WHITE][ROOK] & RANK_7_WHITE) score += EVAL_ROOK_ON_7TH;
    if (pos->pieces[BLACK][ROOK] & RANK_7_BLACK) score -= EVAL_ROOK_ON_7TH;

    // Centralization bonuse - knights/bishops on center squares
    if (pos->pieces[WHITE][KNIGHT] & CENTER_MASK) score += EVAL_KNIGHT_CENTER * popcount(pos->pieces[WHITE][KNIGHT] & CENTER_MASK);
    if (pos->pieces[BLACK][KNIGHT] & CENTER_MASK) score -= EVAL_KNIGHT_CENTER * popcount(pos->pieces[BLACK][KNIGHT] & CENTER_MASK);
    if (pos->pieces[WHITE][BISHOP] & CENTER_MASK) score += EVAL_BISHOP_CENTER * popcount(pos->pieces[WHITE][BISHOP] & CENTER_MASK);
    if (pos->pieces[BLACK][BISHOP] & CENTER_MASK) score -= EVAL_BISHOP_CENTER * popcount(pos->pieces[BLACK][BISHOP] & CENTER_MASK);

    // Tempo - small bonus for the moving side
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