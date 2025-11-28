#ifndef THREAD_LOCAL_DATA_H
#define THREAD_LOCAL_DATA_H

#include "killer_move.h"
#include <cstdint>

struct ThreadLocalData {
    KillerMoves killer_moves;
    uint64_t nodes_searched;
    int thread_id;

    ThreadLocalData(int id = 0) : nodes_searched(0), thread_id(id) {}

    void clear() {
        // We re-initialize killer moves every search or keep them? 
        // Usually keeping them is fine, but clearing is safer for new games.
        killer_moves = KillerMoves(); 
        nodes_searched = 0;
    }
};

#endif // THREAD_LOCAL_DATA_H