// gpu_engine_wrapper.cpp - Adapter for GPU PUCT/MCTS engine
// Wraps the GPU MCTS engine for benchmark interface

#include "../include/engine_interface.h"

#ifdef BUILD_GPU_ENGINE
#include "../../gpu/include/puct_mcts.h"
#include "../../gpu/include/fen.h"
#include "../../gpu/include/chess_types.cuh"
#include <chrono>
#include <iostream>

// External initialization functions
extern void init_attack_tables();


class GPUEngineImpl : public EngineInterface {
public:
    GPUEngineImpl() : current_simulations_(10000) {
        // Initialize attack tables if needed
        static bool initialized = false;
        if (!initialized) {
            init_attack_tables();
            initialized = true;
        }
        
        // Create PUCT engine with default config
        PUCTConfig config;
        config.num_simulations = 10000;
        config.batch_size = 256;
        config.use_virtual_loss = true;
        config.add_dirichlet_noise = false;  // Deterministic for benchmarking
        config.verbose = false;
        
        engine_ = std::make_unique<PUCTEngine>(config);
        
        // Check GPU availability
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        gpu_available_ = (err == cudaSuccess && device_count > 0);
        
        if (!gpu_available_) {
            std::cerr << "Warning: GPU not available for PUCT engine\n";
        }
    }
    
    BenchmarkSearchResult search(const std::string& fen, SearchParams params) override {
        BenchmarkSearchResult result;
        
        if (!gpu_available_) {
            std::cerr << "GPU not available\n";
            result.move_uci = "(none)";
            return result;
        }
        
        try {
            // Parse FEN into BoardState
            BoardState board;
            FENError error = FENParser::parse(fen, board);
            
            if (error != FENError::OK) {
                std::cerr << "FEN parse error: " << static_cast<int>(error) << "\n";
                result.move_uci = "(none)";
                return result;
            }
            
            // Update simulation count if specified
            if (params.max_simulations > 0 && params.max_simulations != current_simulations_) {
                current_simulations_ = params.max_simulations;
                // Recreate engine with new config
                PUCTConfig config;
                config.num_simulations = current_simulations_;
                config.batch_size = 256;
                config.use_virtual_loss = true;
                config.add_dirichlet_noise = false;
                config.verbose = false;
                engine_ = std::make_unique<PUCTEngine>(config);
                engine_->init();
            }
            
            // Start timer
            auto start = std::chrono::high_resolution_clock::now();
            
            // Perform search
            Move best_move = engine_->search(board);
            
            // Calculate elapsed time
            auto end = std::chrono::high_resolution_clock::now();
            result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Convert move to UCI
            if (best_move == 0) {
                result.move_uci = "(none)";
            } else {
                int from = best_move & 0x3F;
                int to = (best_move >> 6) & 0x3F;
                int flags = (best_move >> 12) & 0xF;
                
                char from_sq[3], to_sq[3];
                from_sq[0] = 'a' + (from % 8);
                from_sq[1] = '1' + (from / 8);
                from_sq[2] = '\0';
                to_sq[0] = 'a' + (to % 8);
                to_sq[1] = '1' + (to / 8);
                to_sq[2] = '\0';
                
                result.move_uci = std::string(from_sq) + std::string(to_sq);
                
                // Add promotion piece if applicable (flags >= 8 means promotion)
                if (flags >= 8) {
                    const char promo[] = "nbrq";
                    result.move_uci += promo[(flags - 8) & 0x3];
                }
            }
            
            // Get statistics
            result.simulations = engine_->get_total_visits();
            result.eval_cp = static_cast<int>(engine_->get_root_value() * 100);  // Convert to centipawns
            
        } catch (const std::exception& e) {
            std::cerr << "GPU search error: " << e.what() << "\n";
            result.move_uci = "(none)";
        }
        
        return result;
    }
    
    void reset() override {
        // PUCT engine resets between searches
        // No persistent state to clear
    }
    
    std::string get_name() const override {
        return "GPU-PUCT-MCTS";
    }
    
    std::string get_type() const override {
        return "GPU";
    }
    
    bool is_available() const override {
        return gpu_available_;
    }
    
private:
    std::unique_ptr<PUCTEngine> engine_;
    bool gpu_available_;
    int current_simulations_;
};


std::unique_ptr<EngineInterface> create_gpu_engine() {
    return std::make_unique<GPUEngineImpl>();
}

#else // !BUILD_GPU_ENGINE

// Stub implementation when GPU support is not compiled
std::unique_ptr<EngineInterface> create_gpu_engine() {
    return nullptr;
}

#endif // BUILD_GPU_ENGINE
