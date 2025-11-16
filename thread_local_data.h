#ifndef THREAD_LOCAL_DATA_H
#define THREAD_LOCAL_DATA_H

#include "killer_move.h"
#include <atomic>
#include <cstdint>
#include <cstring>

// Thread-local data to avoid false sharing
struct ThreadLocalData {
    KillerMoves killer_moves;
    uint64_t nodes_searched;
    int thread_id;
    
    ThreadLocalData(int id = 0) : nodes_searched(0), thread_id(id) {}
    
    void clear() {
        killer_moves = KillerMoves();
        nodes_searched = 0;
    }
    
    void increment_nodes() {
        nodes_searched++;
    }
};

#endif // THREAD_LOCAL_DATA_H
