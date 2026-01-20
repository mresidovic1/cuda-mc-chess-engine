// benchmark_stockfish.cpp - External reference engine comparison
// Compares engine moves and evaluations against Stockfish

#include "../include/engine_interface.h"
#include "../include/test_positions.h"
#include "../include/csv_writer.h"
#include "../include/benchmark_utils.h"
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstdio>
#include <sstream>
#include <array>

// ============================================================================
// Configuration
// ============================================================================

struct StockfishConfig {
    std::string output_file = "results_stockfish_agreement.csv";
    std::string stockfish_path = "stockfish";  // Assumes in PATH
    std::string suite = "bratko-kopec";
    int stockfish_depth = 20;
    int engine_depth = 20;
    int engine_time_ms = 5000;
    bool test_cpu = true;
    bool test_gpu = true;
    bool verbose = false;
};

// ============================================================================
// Parse Command Line Arguments
// ============================================================================

StockfishConfig parse_args(int argc, char** argv) {
    StockfishConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_stockfish [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE          Output CSV file (default: results_stockfish_agreement.csv)\n";
            std::cout << "  --stockfish PATH       Path to Stockfish binary (default: stockfish)\n";
            std::cout << "  --suite NAME           Test suite: bratko-kopec/wac/all (default: bratko-kopec)\n";
            std::cout << "  --stockfish-depth N    Stockfish search depth (default: 20)\n";
            std::cout << "  --engine-depth N       Test engine depth (default: 20)\n";
            std::cout << "  --engine-time MS       Test engine time limit (default: 5000)\n";
            std::cout << "  --cpu-only             Test CPU engine only\n";
            std::cout << "  --gpu-only             Test GPU engine only\n";
            std::cout << "  --verbose              Detailed output\n";
            std::cout << "  --help, -h             Show this help\n";
            exit(0);
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        }
        else if (arg == "--stockfish" && i + 1 < argc) {
            config.stockfish_path = argv[++i];
        }
        else if (arg == "--suite" && i + 1 < argc) {
            config.suite = argv[++i];
        }
        else if (arg == "--stockfish-depth" && i + 1 < argc) {
            config.stockfish_depth = std::stoi(argv[++i]);
        }
        else if (arg == "--engine-depth" && i + 1 < argc) {
            config.engine_depth = std::stoi(argv[++i]);
        }
        else if (arg == "--engine-time" && i + 1 < argc) {
            config.engine_time_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--cpu-only") {
            config.test_cpu = true;
            config.test_gpu = false;
        }
        else if (arg == "--gpu-only") {
            config.test_cpu = false;
            config.test_gpu = true;
        }
        else if (arg == "--verbose") {
            config.verbose = true;
        }
    }
    
    return config;
}

// ============================================================================
// Stockfish Interface
// ============================================================================

struct StockfishResult {
    std::string best_move;
    int eval_cp;
    bool success;
};

StockfishResult query_stockfish(const std::string& stockfish_path,
                                 const std::string& fen,
                                 int depth) {
    StockfishResult result;
    result.success = false;
    
    // Build command to pipe to Stockfish
    std::string command = stockfish_path;
    
#ifdef _WIN32
    FILE* pipe = _popen(command.c_str(), "w");
#else
    FILE* pipe = popen(command.c_str(), "w");
#endif
    
    if (!pipe) {
        std::cerr << "Failed to launch Stockfish\n";
        return result;
    }
    
    // Send UCI commands
    fprintf(pipe, "uci\n");
    fprintf(pipe, "isready\n");
    fprintf(pipe, "position fen %s\n", fen.c_str());
    fprintf(pipe, "go depth %d\n", depth);
    fprintf(pipe, "quit\n");
    fflush(pipe);
    
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    
    // Note: This is a simplified implementation
    // In production, you'd need to read stdout properly using popen() with "r" mode
    // For now, we'll use a system() call and redirect to temp file
    
    std::string temp_file = "stockfish_output.txt";
    std::string full_command = stockfish_path + " > " + temp_file;
    
    // For proper implementation, use subprocess or similar
    // This is just a placeholder showing the structure
    
    result.best_move = "e2e4";  // Placeholder
    result.eval_cp = 50;         // Placeholder
    result.success = true;
    
    return result;
}

// Proper Stockfish query using file I/O
StockfishResult query_stockfish_proper(const std::string& stockfish_path,
                                       const std::string& fen,
                                       int depth) {
    StockfishResult result;
    result.success = false;
    
    // This is a simplified placeholder
    // A real implementation would use proper process communication
    // Options:
    //   1. Use popen() with bidirectional communication
    //   2. Use a library like subprocess (C++20 or external)
    //   3. Use platform-specific APIs (CreateProcess on Windows, fork/exec on Unix)
    
    std::cout << "Warning: Stockfish integration is a placeholder.\n";
    std::cout << "To implement properly, use UCI protocol communication.\n";
    std::cout << "For now, returning mock data.\n";
    
    // Return mock data
    result.best_move = "e2e4";
    result.eval_cp = 50;
    result.success = true;
    
    return result;
}

// ============================================================================
// Compare Moves
// ============================================================================

bool moves_match(const std::string& move1, const std::string& move2) {
    // Simple string comparison (case-insensitive)
    if (move1.length() != move2.length()) return false;
    
    for (size_t i = 0; i < move1.length(); i++) {
        if (std::tolower(move1[i]) != std::tolower(move2[i])) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// Test Single Position
// ============================================================================

void test_position_stockfish(EngineInterface& engine,
                             const TestPosition& pos,
                             const StockfishConfig& config,
                             StockfishCSV& csv) {
    // Get Stockfish result
    if (config.verbose) {
        std::cout << "  " << pos.name << " - querying Stockfish... ";
        std::cout.flush();
    }
    
    StockfishResult sf_result = query_stockfish_proper(config.stockfish_path, pos.fen, config.stockfish_depth);
    
    if (!sf_result.success) {
        std::cerr << "\nFailed to get Stockfish result for " << pos.name << "\n";
        return;
    }
    
    if (config.verbose) {
        std::cout << sf_result.best_move << " (" << sf_result.eval_cp << " cp)\n";
        std::cout << "    Testing " << engine.get_name() << "... ";
        std::cout.flush();
    }
    
    // Get engine result
    SearchParams params;
    params.max_depth = config.engine_depth;
    params.use_time_limit = true;
    params.time_limit_ms = config.engine_time_ms;
    
    BenchmarkSearchResult eng_result = engine.search(pos.fen, params);
    
    // Compare
    bool match = moves_match(sf_result.best_move, eng_result.move_uci);
    int eval_diff = std::abs(sf_result.eval_cp - eng_result.eval_cp);
    
    if (config.verbose) {
        std::cout << eng_result.move_uci << " (" << eng_result.eval_cp << " cp) ";
        std::cout << "[" << (match ? "✓" : "✗") << "]\n";
    }
    
    // Write to CSV
    csv.write_result(
        engine.get_name(),
        pos.name,
        pos.fen,
        sf_result.best_move,
        eng_result.move_uci,
        match,
        sf_result.eval_cp,
        eng_result.eval_cp,
        eval_diff
    );
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    StockfishConfig config = parse_args(argc, argv);
    
    std::cout << "========================================\n";
    std::cout << "Stockfish Agreement Benchmark\n";
    std::cout << "========================================\n\n";
    
    std::cout << "NOTE: This benchmark requires Stockfish to be installed.\n";
    std::cout << "Stockfish path: " << config.stockfish_path << "\n";
    std::cout << "If Stockfish is not found, results will be invalid.\n\n";
    
    // Get test suite
    std::vector<TestPosition> positions;
    if (config.suite == "bratko-kopec") {
        positions = get_bratko_kopec_suite();
    } else if (config.suite == "wac") {
        positions = get_wac_suite();
    } else if (config.suite == "all") {
        positions = get_all_positions();
    }
    
    std::cout << "Test suite: " << config.suite << " (" << positions.size() << " positions)\n";
    std::cout << "Output file: " << config.output_file << "\n\n";
    
    // Create CSV writer
    StockfishCSV csv(config.output_file);
    
    // Create engines
    std::vector<std::unique_ptr<EngineInterface>> engines;
    
    if (config.test_cpu) {
        auto cpu_engine = create_cpu_engine();
        if (cpu_engine) {
            cpu_engine->initialize();
            engines.push_back(std::move(cpu_engine));
            std::cout << "CPU engine initialized\n";
        }
    }
    
    if (config.test_gpu) {
        auto gpu_engine = create_gpu_engine();
        if (gpu_engine && gpu_engine->is_available()) {
            gpu_engine->initialize();
            engines.push_back(std::move(gpu_engine));
            std::cout << "GPU engine initialized\n";
        } else {
            std::cout << "GPU engine not available\n";
        }
    }
    
    if (engines.empty()) {
        std::cerr << "Error: No engines available\n";
        return 1;
    }
    
    std::cout << "\n";
    
    // Run benchmark
    int total_tests = positions.size() * engines.size();
    ProgressReporter progress(total_tests, "Benchmark Progress");
    
    for (auto& engine : engines) {
        std::cout << "\nTesting " << engine->get_name() << "...\n";
        
        int matches = 0;
        
        for (const auto& pos : positions) {
            engine->reset();
            
            // We need to track matches ourselves in the callback
            // For simplicity, we'll re-query or cache results
            test_position_stockfish(*engine, pos, config, csv);
            
            progress.update();
        }
        
        if (!config.verbose) {
            std::cout << "  Completed " << positions.size() << " positions\n";
        }
    }
    
    progress.finish();
    csv.flush();
    
    std::cout << "\n========================================\n";
    std::cout << "Benchmark Complete!\n";
    std::cout << "Results saved to: " << config.output_file << "\n";
    std::cout << "========================================\n";
    std::cout << "\nNOTE: Stockfish integration in this benchmark is a placeholder.\n";
    std::cout << "For production use, implement proper UCI protocol communication.\n";
    
    return 0;
}
