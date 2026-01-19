#include "chess.hpp"

using namespace chess;
// Games usually do not go over 20-30 plies
const int MAX_DEPTH = 64;
struct KillerMoves {
    Move killers[MAX_DEPTH][2];
    
    KillerMoves() {
        for (int i = 0; i < MAX_DEPTH; i++) {
            killers[i][0] = Move::NO_MOVE;
            killers[i][1] = Move::NO_MOVE;
        }
    }
    
    void addKiller(int depth, Move move) {
        if (move != killers[depth][0]) {
            killers[depth][1] = killers[depth][0];
            killers[depth][0] = move;
        }
    }
    
    bool isKiller(int depth, Move move) const {
        return move == killers[depth][0] || move == killers[depth][1];
    }
};

