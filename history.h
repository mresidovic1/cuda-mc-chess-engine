#pragma once

#include "include/chess.hpp"
#include <array>
#include <cstring>

using namespace chess;

// History heuristic: tracks how successful moves have been

constexpr int HISTORY_SIZE = 64;  
constexpr int HISTORY_MAX = 32767;
constexpr int HISTORY_MIN = -32768;

struct HistoryTable {
    // [color][from][to] = history score
    // int16_t - Stockfish uses it to save memory
    std::array<std::array<std::array<int16_t, HISTORY_SIZE>, HISTORY_SIZE>, 2> table;
    
    HistoryTable() {
        for (auto& color_table : table) {
            for (auto& from_table : color_table) {
                std::memset(from_table.data(), 0, HISTORY_SIZE * sizeof(int16_t));
            }
        }
    }
    
    void clear() {
        for (auto& color_table : table) {
            for (auto& from_table : color_table) {
                std::memset(from_table.data(), 0, HISTORY_SIZE * sizeof(int16_t));
            }
        }
    }
    
    int get(Color color, Square from, Square to) const {
        return table[static_cast<int>(color)][from.index()][to.index()];
    }
    
    // Update history with bonus (positive for good moves, negative for bad)
    // Formula: new_value = old_value + bonus (with clamping - goes out of bounds, assign max or min value)
    void update(Color color, Square from, Square to, int bonus) {
        int16_t& entry = table[static_cast<int>(color)][from.index()][to.index()];
        
        int new_value = static_cast<int>(entry) + bonus;
        
        if (new_value > HISTORY_MAX) new_value = HISTORY_MAX;
        if (new_value < HISTORY_MIN) new_value = HISTORY_MIN;
        
        entry = static_cast<int16_t>(new_value);
    }
    
    int get(Move move, Color side_to_move) const {
        Square from = move.from();
        Square to = move.to();
        return get(side_to_move, from, to);
    }
    
    void update(Move move, Color side_to_move, int bonus) {
        Square from = move.from();
        Square to = move.to();
        update(side_to_move, from, to, bonus);
    }
};

