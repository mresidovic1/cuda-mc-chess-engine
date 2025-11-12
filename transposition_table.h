#include <vector>
#include <cstring>

const int TT_SIZE = 8388608;

struct TTEntry {
    uint64_t key;
    int depth;
    int score;
    uint8_t flag;
};

struct TranspositionTable {
    std::vector<TTEntry> table;
    
    TranspositionTable() : table(TT_SIZE) {
        std::memset(table.data(), 0, TT_SIZE * sizeof(TTEntry));
    }
    
    void clear() {
        std::memset(table.data(), 0, TT_SIZE * sizeof(TTEntry));
    }
    
    TTEntry* probe(uint64_t key) {
        return &table[key % TT_SIZE];
    }
    
    void store(uint64_t key, int depth, int score, uint8_t flag) {
        TTEntry* entry = &table[key % TT_SIZE];
        entry->key = key;
        entry->depth = depth;
        entry->score = score;
        entry->flag = flag;
    }
};
