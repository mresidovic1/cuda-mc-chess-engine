// benchmark_throughput.cpp - Raw performance measurement
// Measures nodes/sec (CPU) and playouts/sec (GPU) across various positions

#include "../include/engine_interface.h"
#include "../include/test_positions.h"
#include "../include/csv_writer.h"
#include "../include/benchmark_utils.h"
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <cstring>


struct BenchmarkConfig {
    std::string output_file = "results_throughput.csv";
    std::string difficulty = "all";  // easy, medium, hard, all
    int cpu_depth = 15;              // Depth for CPU engine
    int gpu_simulations = 5000;      // Simulations for GPU engine
    int time_per_position_ms = 5000; // Max time per position
    bool test_cpu = true;
    bool test_gpu = true;
    bool verbose = false;
};


BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_throughput [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE       Output CSV file (default: results_throughput.csv)\n";
            std::cout << "  --difficulty LEVEL  Test difficulty: easy/medium/hard/all (default: all)\n";
            std::cout << "  --cpu-depth N       CPU search depth (default: 15)\n";
            std::cout << "  --gpu-sims N        GPU simulations (default: 5000)\n";
            std::cout << "  --time N            Max time per position in ms (default: 5000)\n";
            std::cout << "  --cpu-only          Test CPU engine only\n";
            std::cout << "  --gpu-only          Test GPU engine only\n";
            std::cout << "  --verbose           Detailed output\n";
            std::cout << "  --help, -h          Show this help\n";
            exit(0);
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        }
        else if (arg == "--difficulty" && i + 1 < argc) {
            config.difficulty = argv[++i];
        }
        else if (arg == "--cpu-depth" && i + 1 < argc) {
            config.cpu_depth = std::stoi(argv[++i]);
        }
        else if (arg == "--gpu-sims" && i + 1 < argc) {
            config.gpu_simulations = std::stoi(argv[++i]);
        }
        else if (arg == "--time" && i + 1 < argc) {
            config.time_per_position_ms = std::stoi(argv[++i]);
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


void test_position(EngineInterface& engine, const TestPosition& pos,
                  const BenchmarkConfig& config, ThroughputCSV& csv) {
    SearchParams params;
    
    if (engine.get_type() == "CPU") {
        params.max_depth = config.cpu_depth;
        params.use_time_limit = true;
        params.time_limit_ms = config.time_per_position_ms;
    } else {
        params.max_simulations = config.gpu_simulations;
        params.use_time_limit = false;
    }
    
    if (config.verbose) {
        std::cout << "  Testing " << pos.name << " with " << engine.get_name() << "...\n";
    }
    
    BenchmarkSearchResult result = engine.search(pos.fen, params);
    
    uint64_t operations = result.nodes + result.simulations;
    double throughput = (result.time_ms > 0) ? (operations * 1000.0 / result.time_ms) : 0;
    
    csv.write_result(
        engine.get_name(),
        pos.name,
        pos.fen,
        result.time_ms,
        operations,
        throughput,
        result.depth_reached
    );
    
    if (config.verbose) {
        std::cout << "    Time: " << result.time_ms << " ms, ";
        std::cout << "Throughput: " << format_throughput(throughput) << "/s\n";
    }
}


int main(int argc, char** argv) {
    BenchmarkConfig config = parse_args(argc, argv);
    
    std::cout << "========================================\n";
    std::cout << "Chess Engine Throughput Benchmark\n";
    std::cout << "========================================\n\n";
    
    // Get test positions
    std::vector<TestPosition> positions;
    if (config.difficulty == "all") {
        positions = get_performance_suite();
    } else {
        auto all_pos = get_performance_suite();
        positions = filter_by_difficulty(all_pos, config.difficulty);
    }
    
    std::cout << "Testing " << positions.size() << " positions\n";
    std::cout << "Output file: " << config.output_file << "\n\n";
    
    // Create CSV writer
    ThroughputCSV csv(config.output_file);
    
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
        
        for (const auto& pos : positions) {
            engine->reset();
            test_position(*engine, pos, config, csv);
            progress.update();
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
