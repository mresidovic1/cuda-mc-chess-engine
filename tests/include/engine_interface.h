// engine_interface.h - Abstract interface for chess engine testing
// Provides black-box wrappers for CPU and GPU engines

#ifndef ENGINE_INTERFACE_H
#define ENGINE_INTERFACE_H

#include <string>
#include <memory>
#include <chrono>

// Forward declarations - avoid pulling in full engine headers
namespace chess {
    class Board;
    class Move;
}

struct BoardState;
typedef uint16_t Move;

// ============================================================================
// Search Parameters
// ============================================================================

struct SearchParams {
    int max_depth = 20;          // CPU: max search depth
    int max_simulations = 10000; // GPU: max MCTS playouts
    int time_limit_ms = 0;       // Time limit (0 = no limit)
    
    // Additional parameters for fine control
    bool use_time_limit = false;
    bool verbose = false;
};

// ============================================================================
// Benchmark Search Result (renamed to avoid conflict with GPU SearchResult)
// ============================================================================

struct BenchmarkSearchResult {
    std::string move_uci;        // Move in UCI notation (e.g., "e2e4")
    int eval_cp = 0;             // Evaluation in centipawns
    uint64_t nodes = 0;          // CPU: nodes searched
    uint64_t simulations = 0;    // GPU: MCTS simulations
    int depth_reached = 0;       // CPU: final depth
    double time_ms = 0;          // Actual search time
    
    // Helper
    double throughput() const {
        if (time_ms <= 0) return 0;
        return (nodes + simulations) * 1000.0 / time_ms;
    }
};

// ============================================================================
// Abstract Engine Interface
// ============================================================================

class EngineInterface {
public:
    virtual ~EngineInterface() = default;
    
    // Search for best move
    virtual BenchmarkSearchResult search(const std::string& fen, SearchParams params) = 0;
    
    // Reset engine state (clear hash tables, etc.)
    virtual void reset() = 0;
    
    // Get engine identifier
    virtual std::string get_name() const = 0;
    
    // Get engine type
    virtual std::string get_type() const = 0; // "CPU" or "GPU"
    
    // Initialize (if needed)
    virtual void initialize() {}
    
    // Check if engine is available
    virtual bool is_available() const { return true; }
};

// ============================================================================
// Factory Functions
// ============================================================================

// Create CPU engine wrapper
std::unique_ptr<EngineInterface> create_cpu_engine();

// Create GPU engine wrapper (may return nullptr if CUDA unavailable)
std::unique_ptr<EngineInterface> create_gpu_engine();

#endif // ENGINE_INTERFACE_H
