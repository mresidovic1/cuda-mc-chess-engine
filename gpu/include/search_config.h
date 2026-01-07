#ifndef SEARCH_CONFIG_H
#define SEARCH_CONFIG_H

#include <cstdint>
#include <limits>

// Playout Mode

enum class PlayoutMode {
    RANDOM,         // Pure random playouts 
    EVAL_HYBRID,    // Short random + static evaluation 
    STATIC_EVAL,    // Static evaluation only 
    QUIESCENCE      // Tactical search - extend captures/checks 
};

// Search Configuration

struct SearchConfig {
    // Tree search parameters
    int max_depth;              // Maximum tree depth to explore (0 = unlimited)
    int max_iterations;         // Maximum MCTS iterations (simulations)
    int simulations_per_batch;  // GPU batch size for playouts

    // Time control
    float time_limit_ms;        // Time limit in milliseconds 

    // Playout parameters
    int max_playout_moves;      // Maximum moves per random playout
    PlayoutMode playout_mode;   // Type of playout to use
    int quiescence_depth;       // Max depth for quiescence search 

    float exploration_constant; // UCB1 exploration constant (default: sqrt(2))

    bool use_progressive_widening; // Enable progressive widening
    int progressive_widening_base;  // Base number of children 
    float progressive_widening_alpha; // Growth rate (default: 0.5 for sqrt)

    // Output control
    bool verbose;               // Print search progress
    int info_interval;          // Print info every N iterations 

    // Default constructor with sensible defaults
    SearchConfig()
        : max_depth(0)              
        , max_iterations(10000)     
        , simulations_per_batch(512) 
        , time_limit_ms(0)          
        , max_playout_moves(500)    
        , playout_mode(PlayoutMode::EVAL_HYBRID)  
        , quiescence_depth(3)      
        , exploration_constant(1.414f) 
        , use_progressive_widening(true)  
        , progressive_widening_base(3)    
        , progressive_widening_alpha(0.5f) 
        , verbose(false)
        , info_interval(0)
    {}

    SearchConfig& setMaxDepth(int depth) { max_depth = depth; return *this; }
    SearchConfig& setIterations(int iters) { max_iterations = iters; return *this; }
    SearchConfig& setBatchSize(int size) { simulations_per_batch = size; return *this; }
    SearchConfig& setTimeLimit(float ms) { time_limit_ms = ms; return *this; }
    SearchConfig& setPlayoutMoves(int moves) { max_playout_moves = moves; return *this; }
    SearchConfig& setPlayoutMode(PlayoutMode mode) { playout_mode = mode; return *this; }
    SearchConfig& setQuiescenceDepth(int depth) { quiescence_depth = depth; return *this; }
    SearchConfig& setExploration(float c) { exploration_constant = c; return *this; }
    SearchConfig& setProgressiveWidening(bool enable, int base = 3, float alpha = 0.5f) {
        use_progressive_widening = enable;
        progressive_widening_base = base;
        progressive_widening_alpha = alpha;
        return *this;
    }
    SearchConfig& setVerbose(bool v) { verbose = v; return *this; }
    SearchConfig& setInfoInterval(int interval) { info_interval = interval; return *this; }

    // Preset configurations
    static SearchConfig Quick() {
        return SearchConfig()
            .setIterations(1000)
            .setBatchSize(256);
    }

    static SearchConfig Normal() {
        return SearchConfig()
            .setIterations(10000)
            .setBatchSize(512);
    }

    static SearchConfig Deep() {
        return SearchConfig()
            .setIterations(50000)
            .setBatchSize(1024);
    }

    static SearchConfig ForDepth(int depth) {
        int iters = 2000 * depth;
        return SearchConfig()
            .setMaxDepth(depth)
            .setIterations(iters)
            .setBatchSize(512);
    }

    static SearchConfig TimeLimited(float ms) {
        return SearchConfig()
            .setTimeLimit(ms)
            .setIterations(std::numeric_limits<int>::max());
    }
};


struct SearchResult {
    uint16_t best_move;         // Best move found 
    float score;                // Win rate of best move (0.0 - 1.0)
    int nodes_searched;         // Total nodes in tree
    int simulations_run;        // Total playouts performed
    float time_ms;              // Time taken in milliseconds
    int depth_reached;          // Maximum depth reached in tree

    static const int MAX_PV_LENGTH = 10;
    uint16_t pv[MAX_PV_LENGTH];
    int pv_length;

    SearchResult()
        : best_move(0)
        , score(0.5f)
        , nodes_searched(0)
        , simulations_run(0)
        , time_ms(0)
        , depth_reached(0)
        , pv_length(0)
    {
        for (int i = 0; i < MAX_PV_LENGTH; i++) pv[i] = 0;
    }

    float nps() const {
        if (time_ms <= 0) return 0;
        return (simulations_run * 1000.0f) / time_ms;
    }
};

#endif 
