#include <vector>
#include "include/chess.hpp"

// 2^23 (23 bytes = uint64(8) + int(4) + int(4) + uint8(1) + Move - uint16(2) + int(4))
const int TT_SIZE = 8388608;

struct TTEntry {
    // Zobrist hashing produces 64 bit values, also leads to less collisions
    uint64_t key;
    int depth;
    int score;
    uint8_t flag; // 0 (exact score), 1 (score >= beta), 2 (score <= alpha) possible values
    chess::Move bestMove;  
    int staticEval;        
};

struct TranspositionTable {
    std::vector<TTEntry> table;
    
    TranspositionTable() : table(TT_SIZE) {
        for (auto& entry : table) {
            entry.bestMove = chess::Move::NO_MOVE;
            // 30001 greater than const for infinity
            entry.staticEval = 30001;  
        }
    }
    
    void clear() {
        for (auto& entry : table) {
            entry.key = 0;
            entry.depth = 0;
            entry.score = 0;
            entry.flag = 0;
            entry.bestMove = chess::Move::NO_MOVE;
            entry.staticEval = 30001;  
        }
    }
    
    TTEntry* probe(uint64_t key) {
        return &table[key % TT_SIZE];
    }
    
    void store(uint64_t key, int depth, int score, uint8_t flag, chess::Move bestMove = chess::Move::NO_MOVE, int staticEval = 30001) {
        TTEntry* entry = &table[key % TT_SIZE];
        entry->key = key;
        entry->depth = depth;
        entry->score = score;
        entry->flag = flag;
        entry->bestMove = bestMove;
        entry->staticEval = staticEval;
    }
};
