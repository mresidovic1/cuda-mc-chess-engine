// cpu_engine_wrapper.cpp - Adapter for CPU Negamax engine
// Wraps the parallelized CPU engine for benchmark interface

#include "../include/engine_interface.h"
#include "../../cpu/include/chess.hpp"
#include <chrono>
#include <iostream>

// External functions from CPU engine
extern chess::Move find_best_move(chess::Board& board, int max_depth, int time_limit_ms);
extern void attacks_init();  // May need to initialize attack tables

// ============================================================================
// CPU Engine Wrapper Implementation
// ============================================================================

class CPUEngineImpl : public EngineInterface {
public:
    CPUEngineImpl() {
        // Initialize attack tables if needed
        static bool initialized = false;
        if (!initialized) {
            chess::attacks::initAttacks();
            initialized = true;
        }
    }
    
    SearchResult search(const std::string& fen, SearchParams params) override {
        SearchResult result;
        
        try {
            // Parse FEN into chess::Board
            chess::Board board(fen);
            
            // Start timer
            auto start = std::chrono::high_resolution_clock::now();
            
            // Perform search
            chess::Move best_move;
            if (params.use_time_limit && params.time_limit_ms > 0) {
                best_move = find_best_move(board, params.max_depth, params.time_limit_ms);
            } else {
                best_move = find_best_move(board, params.max_depth, 0);
            }
            
            // Calculate elapsed time
            auto end = std::chrono::high_resolution_clock::now();
            result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Convert move to UCI
            result.move_uci = chess::uci::moveToUci(best_move);
            
            // Make the move and get evaluation (approximate)
            board.makeMove(best_move);
            // CPU engine doesn't directly expose eval, so we estimate
            result.eval_cp = 0;  // Would need to expose eval from search
            
            // Note: CPU engine doesn't easily expose nodes count
            // We'd need to modify it or track separately
            result.nodes = 0;  // TODO: expose from engine
            result.depth_reached = params.max_depth;
            
        } catch (const std::exception& e) {
            std::cerr << "CPU search error: " << e.what() << "\n";
            result.move_uci = "(none)";
        }
        
        return result;
    }
    
    void reset() override {
        // CPU engine resets transposition table internally
        // No persistent state to clear in wrapper
    }
    
    std::string get_name() const override {
        return "CPU-Negamax";
    }
    
    std::string get_type() const override {
        return "CPU";
    }
};

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<EngineInterface> create_cpu_engine() {
    return std::make_unique<CPUEngineImpl>();
}
