#ifndef TT_PARALLEL_H
#define TT_PARALLEL_H

#include <vector>
#include <cstring>

// Simple transposition table (same interface as before, will add locking later if needed)
constexpr size_t TT_SIZE_MB = 256;

struct TTEntrySimple {
    uint64_t key;
    int32_t score;
    uint16_t depth;
    uint8_t flag;
    uint8_t age;
    
    TTEntrySimple() : key(0), score(0), depth(0), flag(0), age(0) {}
};

class TTParallel {
private:
    std::vector<TTEntrySimple> table;
    uint8_t generation;
    
public:
    TTParallel(size_t mb_size = 256) : generation(0) {
        size_t entries = (mb_size * 1024 * 1024) / sizeof(TTEntrySimple);
        table.resize(entries);
    }
    
    void clear() {
        for (auto& entry : table) {
            entry = TTEntrySimple();
        }
    }
    
    void new_search() {
        generation++;
    }
    
    bool probe(uint64_t hash, int& score, int& depth, uint8_t& flag, int ply) const {
        size_t idx = hash % table.size();
        const TTEntrySimple& entry = table[idx];
        
        if (entry.key == hash) {
            score = entry.score;
            depth = entry.depth;
            flag = entry.flag;
            
            // Adjust mate scores
            if (score > 9000) score -= ply;
            if (score < -9000) score += ply;
            
            return true;
        }
        
        return false;
    }
    
    void store(uint64_t hash, int depth, int score, uint8_t flag, int ply) {
        size_t idx = hash % table.size();
        TTEntrySimple& entry = table[idx];
        
        // Adjust mate scores
        if (score > 9000) score += ply;
        if (score < -9000) score -= ply;
        
        // Replace if empty, same position, or lower depth
        if (entry.key == 0 || entry.key == hash || entry.depth <= depth) {
            entry.key = hash;
            entry.score = score;
            entry.depth = depth;
            entry.flag = flag;
            entry.age = generation;
        }
    }
};

#endif // TT_PARALLEL_H
