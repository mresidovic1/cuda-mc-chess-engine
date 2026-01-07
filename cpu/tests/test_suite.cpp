#include "../include/chess.hpp"
#include "../include/test_positions.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>

extern chess::Move find_best_move(chess::Board& board, int max_depth, int time_limit_ms);
extern std::string run_engine(chess::Board& board, int depth);

using namespace chess;
using namespace std::chrono;

struct TestConfig {
    enum Mode { FIXED_DEPTH, TIME_LIMITED };
    Mode mode = FIXED_DEPTH;
    int depth = 20;
    int time_limit_seconds = 30;
    int difficulty_filter = 0;
    bool json_output = false;
    bool quiet = false;
    std::string engine_name = "default";
};

struct TestResult {
    std::string fen;
    std::string expected_move;
    std::string got_move;
    std::string description;
    int difficulty;
    bool passed;
    int64_t time_ms;
    int depth_reached;
    uint64_t nodes_searched;
};

class TestRunner {
public:
    TestRunner(const TestConfig& config) : config_(config) {}
    
    std::vector<TestResult> run_all_tests() {
        std::vector<TestResult> results;
        std::vector<TestPosition> positions;
        
        if (config_.difficulty_filter == 0) {
            positions = get_all_positions();
        } else {
            positions = get_positions_by_difficulty(config_.difficulty_filter);
        }
        
        if (!config_.quiet) {
            print_header();
        }
        
        auto suite_start = high_resolution_clock::now();
        
        for (size_t i = 0; i < positions.size(); i++) {
            const auto& pos = positions[i];
            
            if (!config_.quiet) {
                std::cout << "\n[" << (i + 1) << "/" << positions.size() << "] "
                          << difficulty_name(pos.difficulty) << ": " 
                          << pos.description << std::endl;
                std::cout << "FEN: " << pos.fen << std::endl;
            }
            
            TestResult result = run_single_test(pos);
            results.push_back(result);
            
            if (!config_.quiet) {
                print_result(result);
            }
        }
        
        auto suite_end = high_resolution_clock::now();
        int64_t total_time = duration_cast<milliseconds>(suite_end - suite_start).count();
        
        if (!config_.quiet) {
            print_summary(results, total_time);
        }
        
        if (config_.json_output) {
            output_json(results, total_time);
        }
        
        return results;
    }
    
private:
    TestConfig config_;
    
    TestResult run_single_test(const TestPosition& pos) {
        TestResult result;
        result.fen = pos.fen;
        result.expected_move = pos.expected_move;
        result.description = pos.description;
        result.difficulty = pos.difficulty;
        result.depth_reached = 0;
        result.nodes_searched = 0;
        
        Board board(pos.fen);
        
        auto start = high_resolution_clock::now();
        
        Move best_move;
        constexpr int MAX_TIME_MS = 60000;
        if (config_.mode == TestConfig::FIXED_DEPTH) {
            best_move = find_best_move(board, config_.depth, MAX_TIME_MS);
        } else {
            int time_ms = std::min(config_.time_limit_seconds * 1000, MAX_TIME_MS);
            best_move = find_best_move(board, 100, time_ms);
        }
        
        result.depth_reached = 0;
        result.nodes_searched = 0;
        
        auto end = high_resolution_clock::now();
        result.time_ms = duration_cast<milliseconds>(end - start).count();
        
        result.got_move = uci::moveToUci(best_move);
        result.passed = (result.got_move == result.expected_move);
        
        return result;
    }
    
    void print_header() {
        std::cout << std::string(80, '=') << "\n";
        std::cout << "                    CHESS ENGINE TEST SUITE\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << "Engine: " << config_.engine_name << "\n";
        std::cout << "Mode:   " << (config_.mode == TestConfig::FIXED_DEPTH ? "Fixed Depth" : "Time Limited") << "\n";
        if (config_.mode == TestConfig::FIXED_DEPTH) {
            std::cout << "Depth:  " << config_.depth << "\n";
        } else {
            std::cout << "Time:   " << config_.time_limit_seconds << " seconds per position\n";
        }
        std::cout << "Level:  " << (config_.difficulty_filter == 0 ? "ALL" : 
                     difficulty_name(config_.difficulty_filter)) << "\n";
        std::cout << std::string(80, '-') << "\n";
    }
    
    void print_result(const TestResult& result) {
        std::cout << "Expected: " << result.expected_move 
                  << " | Got: " << result.got_move
                  << " | Time: " << result.time_ms << "ms"
                  << " | Depth: " << result.depth_reached
                  << " | Nodes: " << result.nodes_searched
                  << " | " << (result.passed ? "[PASS]" : "[FAIL]") << std::endl;
    }
    
    void print_summary(const std::vector<TestResult>& results, int64_t total_time) {
        int passed_easy = 0, total_easy = 0;
        int passed_medium = 0, total_medium = 0;
        int passed_hard = 0, total_hard = 0;
        
        for (const auto& r : results) {
            switch (r.difficulty) {
                case EASY:
                    total_easy++;
                    if (r.passed) passed_easy++;
                    break;
                case MEDIUM:
                    total_medium++;
                    if (r.passed) passed_medium++;
                    break;
                case HARD:
                    total_hard++;
                    if (r.passed) passed_hard++;
                    break;
            }
        }
        
        int total_passed = passed_easy + passed_medium + passed_hard;
        int total_tests = total_easy + total_medium + total_hard;
        
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "                              TEST SUMMARY\n";
        std::cout << std::string(80, '=') << "\n";
        std::cout << std::left;
        std::cout << std::setw(15) << "Difficulty" << std::setw(15) << "Passed" 
                  << std::setw(15) << "Total" << "Rate\n";
        std::cout << std::string(80, '-') << "\n";
        
        if (total_easy > 0) {
            std::cout << std::setw(15) << "EASY" 
                      << std::setw(15) << passed_easy 
                      << std::setw(15) << total_easy 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * passed_easy / total_easy) << "%\n";
        }
        if (total_medium > 0) {
            std::cout << std::setw(15) << "MEDIUM" 
                      << std::setw(15) << passed_medium 
                      << std::setw(15) << total_medium 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * passed_medium / total_medium) << "%\n";
        }
        if (total_hard > 0) {
            std::cout << std::setw(15) << "HARD" 
                      << std::setw(15) << passed_hard 
                      << std::setw(15) << total_hard 
                      << std::fixed << std::setprecision(1) 
                      << (100.0 * passed_hard / total_hard) << "%\n";
        }
        
        std::cout << std::string(80, '-') << "\n";
        std::cout << std::setw(15) << "TOTAL" 
                  << std::setw(15) << total_passed 
                  << std::setw(15) << total_tests 
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * total_passed / total_tests) << "%\n";
        std::cout << std::string(80, '-') << "\n";
        std::cout << "Total execution time: " << (total_time / 1000.0) << " seconds\n";
        std::cout << std::string(80, '=') << "\n";
    }
    
    void output_json(const std::vector<TestResult>& results, int64_t total_time) {
        std::cout << "\n{\"engine\": \"" << config_.engine_name << "\", \"results\": [\n";
        for (size_t i = 0; i < results.size(); i++) {
            const auto& r = results[i];
            std::cout << "  {";
            std::cout << "\"fen\": \"" << r.fen << "\", ";
            std::cout << "\"expected\": \"" << r.expected_move << "\", ";
            std::cout << "\"got\": \"" << r.got_move << "\", ";
            std::cout << "\"passed\": " << (r.passed ? "true" : "false") << ", ";
            std::cout << "\"time_ms\": " << r.time_ms << ", ";
            std::cout << "\"depth_reached\": " << r.depth_reached << ", ";
            std::cout << "\"nodes_searched\": " << r.nodes_searched << ", ";
            std::cout << "\"difficulty\": \"" << difficulty_name(r.difficulty) << "\"";
            std::cout << "}" << (i < results.size() - 1 ? "," : "") << "\n";
        }
        std::cout << "], \"total_time_ms\": " << total_time << "}\n";
    }
};

TestConfig parse_args(int argc, char* argv[]) {
    TestConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("--mode=") == 0) {
            std::string mode = arg.substr(7);
            if (mode == "time") {
                config.mode = TestConfig::TIME_LIMITED;
            } else {
                config.mode = TestConfig::FIXED_DEPTH;
            }
        } else if (arg.find("--level=") == 0) {
            std::string level = arg.substr(8);
            if (level == "easy") config.difficulty_filter = EASY;
            else if (level == "medium") config.difficulty_filter = MEDIUM;
            else if (level == "hard") config.difficulty_filter = HARD;
            else config.difficulty_filter = 0;
        } else if (arg.find("--depth=") == 0) {
            config.depth = std::stoi(arg.substr(8));
        } else if (arg.find("--time=") == 0) {
            config.time_limit_seconds = std::stoi(arg.substr(7));
        } else if (arg.find("--engine=") == 0) {
            config.engine_name = arg.substr(9);
        } else if (arg == "--json") {
            config.json_output = true;
        } else if (arg == "--quiet") {
            config.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ./test_suite [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --mode=depth    Fixed depth mode (default)\n";
            std::cout << "  --mode=time     Time-limited mode\n";
            std::cout << "  --level=easy    Run only easy positions\n";
            std::cout << "  --level=medium  Run only medium positions\n";
            std::cout << "  --level=hard    Run only hard positions\n";
            std::cout << "  --level=all     Run all positions (default)\n";
            std::cout << "  --depth=N       Set search depth (default: 20)\n";
            std::cout << "  --time=N        Time per position in seconds (default: 30)\n";
            std::cout << "  --engine=NAME   Engine name for reports\n";
            std::cout << "  --json          Output JSON format\n";
            std::cout << "  --quiet         Minimal output\n";
            exit(0);
        }
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    attacks::initAttacks();
    
    TestConfig config = parse_args(argc, argv);
    TestRunner runner(config);
    
    auto results = runner.run_all_tests();
    
    int failed = 0;
    for (const auto& r : results) {
        if (!r.passed) failed++;
    }
    
    return failed > 0 ? 1 : 0;
}

