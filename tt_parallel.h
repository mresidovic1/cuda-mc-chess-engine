#ifndef TT_PARALLEL_H
#define TT_PARALLEL_H

#include <vector>
#include <cstring>
#include "include/chess.hpp"

// Size in MB
constexpr size_t TT_SIZE_MB = 256;

struct TTEntryParallel {
    uint64_t key;
    chess::Move bestMove;
    int16_t score;
    int16_t staticEval;
    int16_t depth;
    uint8_t flag; // 0: exact, 1: >= beta (lowerbound), 2: <= alpha (upperbound)
    uint8_t generation; // For aging (replacement strategy)

    TTEntryParallel() 
        : key(0), bestMove(chess::Move::NO_MOVE), score(0), 
          staticEval(30001), depth(0), flag(0), generation(0) {}
};

class TTParallel {
private:
    std::vector<TTEntryParallel> table;
    uint8_t current_generation;

public:
    TTParallel(size_t mb_size = TT_SIZE_MB) : current_generation(0) {
        // Calculate number of entries based on size
        size_t entry_count = (mb_size * 1024 * 1024) / sizeof(TTEntryParallel);
        table.resize(entry_count);
    }

    void clear() {
        std::fill(table.begin(), table.end(), TTEntryParallel());
    }

    void new_search() {
        current_generation++; // Increment age to prefer new search results
    }

    // Lockless probe - we accept rare data races for speed
    TTEntryParallel* probe(uint64_t key) {
        size_t idx = key % table.size();
        TTEntryParallel* entry = &table[idx];
        
        // Return entry if keys match, otherwise standard logic handles mismatches
        return entry; 
    }

    void store(uint64_t key, int depth, int score, uint8_t flag, chess::Move bestMove, int staticEval, int ply) {
        size_t idx = key % table.size();
        TTEntryParallel* entry = &table[idx];

        // Mate score adjustment for TT storage (independent of ply)
        int score_to_store = score;
        if (score > 9000) score_to_store += ply;
        if (score < -9000) score_to_store -= ply;

        // Replacement Strategy:
        // 1. If slot is empty or key matches (update)
        // 2. If existing entry is from an old generation (always replace old data)
        // 3. If new depth is greater (deeper search is more valuable)
        bool replace = (entry->key == 0) || 
                       (entry->key == key) || 
                       (entry->generation != current_generation) ||
                       (depth >= entry->depth);

        if (replace) {
            // Note: In a strictly safe environment, we would use a spinlock here.
            // In chess engines, we accept the "torn read/write" risk for performance.
            // The key is written last or validated, but here we do a direct write.
            
            entry->key = key;
            entry->bestMove = bestMove;
            entry->depth = (int16_t)depth;
            entry->flag = flag;
            entry->staticEval = (int16_t)staticEval;
            entry->score = (int16_t)score_to_store;
            entry->generation = current_generation;
        }
    }
    
    // Helper to retrieve score corrected for current ply
    int retrieve_score(int tt_score, int ply) const {
        if (tt_score > 9000) return tt_score - ply;
        if (tt_score < -9000) return tt_score + ply;
        return tt_score;
    }
};

#endif // TT_PARALLEL_H