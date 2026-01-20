// benchmark_fixed_time.cpp - Fixed-time quality comparison
// Evaluates move quality under time constraints (50ms, 100ms, 500ms, 1s, 5s)

#include "../include/engine_interface.h"
#include "../include/test_positions.h"
#include "../include/csv_writer.h"
#include "../include/benchmark_utils.h"
#include <iostream>
#include <memory>
#include <vector>
#include <string>

// ============================================================================
// Configuration
// ============================================================================

struct FixedTimeConfig {
    std::string output_file = "results_fixed_time.csv";
    std::vector<int> time_budgets = {50, 100, 500, 1000, 5000}; // milliseconds
    std::string suite = "bratko-kopec"; // bratko-kopec, wac, performance, all
    bool test_cpu = true;
    bool test_gpu = true;
    bool verbose = false;
};

// ============================================================================
// Parse Command Line Arguments
// ============================================================================

FixedTimeConfig parse_args(int argc, char** argv) {
    FixedTimeConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_fixed_time [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE       Output CSV file (default: results_fixed_time.csv)\n";
            std::cout << "  --times MS,MS,...   Comma-separated time budgets in ms\n";
            std::cout << "                      (default: 50,100,500,1000,5000)\n";
            std::cout << "  --suite NAME        Test suite: bratko-kopec/wac/performance/all\n";
            std::cout << "                      (default: bratko-kopec)\n";
            std::cout << "  --cpu-only          Test CPU engine only\n";
            std::cout << "  --gpu-only          Test GPU engine only\n";
            std::cout << "  --verbose           Detailed output\n";
            std::cout << "  --help, -h          Show this help\n";
            exit(0);
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        }
        else if (arg == "--times" && i + 1 < argc) {
            config.time_budgets.clear();
            std::string times_str = argv[++i];
            size_t pos = 0;
            while ((pos = times_str.find(',')) != std::string::npos) {
                config.time_budgets.push_back(std::stoi(times_str.substr(0, pos)));
                times_str.erase(0, pos + 1);
            }
            config.time_budgets.push_back(std::stoi(times_str));
        }
        else if (arg == "--suite" && i + 1 < argc) {
            config.suite = argv[++i];
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
// Get Test Suite
// ============================================================================

std::vector<TestPosition> get_test_suite(const std::string& suite_name) {
    if (suite_name == "bratko-kopec") {
        return get_bratko_kopec_suite();
    } else if (suite_name == "wac") {
        return get_wac_suite();
    } else if (suite_name == "performance") {
        return get_performance_suite();
    } else if (suite_name == "all") {
        return get_all_positions();
    } else {
        std::cerr << "Unknown suite: " << suite_name << "\n";
        return get_bratko_kopec_suite();
    }
}

// ============================================================================
// Test Single Position with Time Budget
// ============================================================================

struct QualityResult {
    std::string move;
    int eval;
    double actual_time;
    uint64_t nodes;
    int depth;
    bool correct;  // If best_move is known
};

QualityResult test_position_timed(EngineInterface& engine, const TestPosition& pos,
                                   int time_budget_ms, bool verbose) {
    SearchParams params;
    params.use_time_limit = true;
    params.time_limit_ms = time_budget_ms;
    params.max_depth = 100;  // Effectively unlimited, time controls
    params.max_simulations = 1000000;  // Effectively unlimited
    
    if (verbose) {
        std::cout << "    " << pos.name << " @ " << time_budget_ms << "ms... ";
        std::cout.flush();
    }
    
    BenchmarkSearchResult result = engine.search(pos.fen, params);
    
    QualityResult quality;
    quality.move = result.move_uci;
    quality.eval = result.eval_cp;
    quality.actual_time = result.time_ms;
    quality.nodes = result.nodes + result.simulations;
    quality.depth = result.depth_reached;
    
    // Check if move matches known best move
    quality.correct = pos.best_move.empty() || (result.move_uci == pos.best_move);
    
    if (verbose) {
        std::cout << result.move_uci;
        if (!pos.best_move.empty()) {
            std::cout << " (expected: " << pos.best_move << ")";
            std::cout << " [" << (quality.correct ? "✓" : "✗") << "]";
        }
        std::cout << "\n";
    }
    
    return quality;
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    FixedTimeConfig config = parse_args(argc, argv);
    
    std::cout << "========================================\n";
    std::cout << "Fixed-Time Quality Benchmark\n";
    std::cout << "========================================\n\n";
    
    // Get test positions
    std::vector<TestPosition> positions = get_test_suite(config.suite);
    
    std::cout << "Test suite: " << config.suite << " (" << positions.size() << " positions)\n";
    std::cout << "Time budgets: ";
    for (size_t i = 0; i < config.time_budgets.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << config.time_budgets[i] << "ms";
    }
    std::cout << "\n";
    std::cout << "Output file: " << config.output_file << "\n\n";
    
    // Create CSV writer
    FixedTimeCSV csv(config.output_file);
    
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
    int total_tests = positions.size() * engines.size() * config.time_budgets.size();
    ProgressReporter progress(total_tests, "Benchmark Progress");
    
    for (auto& engine : engines) {
        std::cout << "\nTesting " << engine->get_name() << "...\n";
        
        for (int time_budget : config.time_budgets) {
            if (config.verbose) {
                std::cout << "\n  Time budget: " << time_budget << "ms\n";
            }
            
            // Track correctness for this time budget
            int correct_moves = 0;
            int positions_with_answer = 0;
            
            for (const auto& pos : positions) {
                engine->reset();
                
                QualityResult quality = test_position_timed(*engine, pos, time_budget, config.verbose);
                
                csv.write_result(
                    engine->get_name(),
                    pos.name,
                    pos.fen,
                    time_budget,
                    quality.actual_time,
                    quality.move,
                    quality.eval,
                    quality.depth,
                    quality.nodes
                );
                
                if (!pos.best_move.empty()) {
                    positions_with_answer++;
                    if (quality.correct) {
                        correct_moves++;
                    }
                }
                
                progress.update();
            }
            
            // Print summary for this time budget
            if (positions_with_answer > 0 && !config.verbose) {
                double accuracy = 100.0 * correct_moves / positions_with_answer;
                std::cout << "  " << time_budget << "ms: "
                          << correct_moves << "/" << positions_with_answer
                          << " (" << std::fixed << std::setprecision(1) << accuracy << "%)\n";
            }
        }
    }
    
    progress.finish();
    csv.flush();
    
    std::cout << "\n========================================\n";
    std::cout << "Benchmark Complete!\n";
    std::cout << "Results saved to: " << config.output_file << "\n";
    std::cout << "========================================\n";
    
    return 0;
}
