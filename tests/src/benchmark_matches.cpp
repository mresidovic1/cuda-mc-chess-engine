// benchmark_matches.cpp - Head-to-head engine matches
// Play games between CPU and GPU engines, compute Elo difference

#include "../include/engine_interface.h"
#include "../include/test_positions.h"
#include "../include/csv_writer.h"
#include "../include/benchmark_utils.h"
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <random>

// ============================================================================
// Configuration
// ============================================================================

struct MatchConfig {
    std::string output_file = "results_matches.csv";
    int num_games = 100;
    int moves_per_game = 200;  // Max moves before draw
    int time_per_move_ms = 1000;
    bool alternate_colors = true;
    std::string opening_fen = "";  // Empty = standard start position
    bool verbose = false;
};

// ============================================================================
// Parse Command Line Arguments
// ============================================================================

MatchConfig parse_args(int argc, char** argv) {
    MatchConfig config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: benchmark_matches [options]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --output FILE       Output CSV file (default: results_matches.csv)\n";
            std::cout << "  --games N           Number of games to play (default: 100)\n";
            std::cout << "  --time MS           Time per move in milliseconds (default: 1000)\n";
            std::cout << "  --max-moves N       Max moves before draw (default: 200)\n";
            std::cout << "  --no-alternate      Don't alternate colors (CPU always white)\n";
            std::cout << "  --opening FEN       Starting position (default: standard)\n";
            std::cout << "  --verbose           Detailed output\n";
            std::cout << "  --help, -h          Show this help\n";
            exit(0);
        }
        else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        }
        else if (arg == "--games" && i + 1 < argc) {
            config.num_games = std::stoi(argv[++i]);
        }
        else if (arg == "--time" && i + 1 < argc) {
            config.time_per_move_ms = std::stoi(argv[++i]);
        }
        else if (arg == "--max-moves" && i + 1 < argc) {
            config.moves_per_game = std::stoi(argv[++i]);
        }
        else if (arg == "--no-alternate") {
            config.alternate_colors = false;
        }
        else if (arg == "--opening" && i + 1 < argc) {
            config.opening_fen = argv[++i];
        }
        else if (arg == "--verbose") {
            config.verbose = true;
        }
    }
    
    return config;
}

// ============================================================================
// Game State
// ============================================================================

enum class GameResult {
    ONGOING,
    WHITE_WIN,
    BLACK_WIN,
    DRAW_STALEMATE,
    DRAW_REPETITION,
    DRAW_FIFTY_MOVE,
    DRAW_INSUFFICIENT,
    DRAW_MOVE_LIMIT
};

std::string result_to_string(GameResult result) {
    switch (result) {
        case GameResult::WHITE_WIN: return "1-0";
        case GameResult::BLACK_WIN: return "0-1";
        case GameResult::DRAW_STALEMATE: return "1/2-1/2 (stalemate)";
        case GameResult::DRAW_REPETITION: return "1/2-1/2 (repetition)";
        case GameResult::DRAW_FIFTY_MOVE: return "1/2-1/2 (fifty-move)";
        case GameResult::DRAW_INSUFFICIENT: return "1/2-1/2 (insufficient material)";
        case GameResult::DRAW_MOVE_LIMIT: return "1/2-1/2 (move limit)";
        default: return "ongoing";
    }
}

std::string termination_reason(GameResult result) {
    switch (result) {
        case GameResult::WHITE_WIN: return "checkmate";
        case GameResult::BLACK_WIN: return "checkmate";
        case GameResult::DRAW_STALEMATE: return "stalemate";
        case GameResult::DRAW_REPETITION: return "repetition";
        case GameResult::DRAW_FIFTY_MOVE: return "fifty-move rule";
        case GameResult::DRAW_INSUFFICIENT: return "insufficient material";
        case GameResult::DRAW_MOVE_LIMIT: return "move limit";
        default: return "ongoing";
    }
}

// ============================================================================
// Simplified Game Manager
// ============================================================================

class GameManager {
public:
    GameManager(const std::string& fen) : current_fen_(fen), move_count_(0) {
        if (fen.empty()) {
            current_fen_ = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        }
        position_history_.push_back(current_fen_);
    }
    
    std::string get_fen() const { return current_fen_; }
    
    int get_move_count() const { return move_count_; }
    
    void make_move(const std::string& move_uci) {
        // In a real implementation, we'd:
        // 1. Parse current FEN
        // 2. Apply move
        // 3. Generate new FEN
        // 4. Update position history
        
        // For simplicity, this is a placeholder
        move_count_++;
        moves_.push_back(move_uci);
        
        // Update FEN (placeholder - would need real chess logic)
        // current_fen_ = apply_move_to_fen(current_fen_, move_uci);
        
        position_history_.push_back(current_fen_);
    }
    
    GameResult check_game_over(int max_moves) {
        // Simplified game termination checks
        // In real implementation, parse FEN and check:
        // - Checkmate/stalemate
        // - Repetition
        // - Fifty-move rule
        // - Insufficient material
        
        if (move_count_ >= max_moves) {
            return GameResult::DRAW_MOVE_LIMIT;
        }
        
        // Check for repetition (simplified - check if position appears 3 times)
        int count = 0;
        for (const auto& pos : position_history_) {
            if (pos == current_fen_) count++;
        }
        if (count >= 3) {
            return GameResult::DRAW_REPETITION;
        }
        
        // Placeholder: would need real chess logic to detect checkmate, etc.
        
        return GameResult::ONGOING;
    }
    
    const std::vector<std::string>& get_moves() const { return moves_; }
    
private:
    std::string current_fen_;
    int move_count_;
    std::vector<std::string> moves_;
    std::vector<std::string> position_history_;
};

// ============================================================================
// Play Single Game
// ============================================================================

struct GameStats {
    GameResult result;
    int moves;
    std::string final_fen;
};

GameStats play_game(EngineInterface& white_engine,
                    EngineInterface& black_engine,
                    const MatchConfig& config,
                    int game_id) {
    GameManager game(config.opening_fen);
    GameStats stats;
    
    if (config.verbose) {
        std::cout << "\nGame " << game_id << ": "
                  << white_engine.get_name() << " (W) vs "
                  << black_engine.get_name() << " (B)\n";
    }
    
    SearchParams params;
    params.use_time_limit = true;
    params.time_limit_ms = config.time_per_move_ms;
    params.max_depth = 100;
    params.max_simulations = 1000000;
    
    while (true) {
        // Check game over
        GameResult result = game.check_game_over(config.moves_per_game);
        if (result != GameResult::ONGOING) {
            stats.result = result;
            break;
        }
        
        // Determine whose turn it is
        bool white_to_move = (game.get_move_count() % 2 == 0);
        EngineInterface& current_engine = white_to_move ? white_engine : black_engine;
        
        // Get move
        SearchResult search_result = current_engine.search(game.get_fen(), params);
        
        if (search_result.move_uci == "(none)" || search_result.move_uci.empty()) {
            // No legal moves - checkmate or stalemate
            // In real implementation, we'd check if in check
            stats.result = white_to_move ? GameResult::BLACK_WIN : GameResult::WHITE_WIN;
            break;
        }
        
        if (config.verbose) {
            std::cout << "  Move " << (game.get_move_count() + 1) << ": "
                      << search_result.move_uci << " ("
                      << current_engine.get_name() << ")\n";
        }
        
        // Make move
        game.make_move(search_result.move_uci);
    }
    
    stats.moves = game.get_move_count();
    stats.final_fen = game.get_fen();
    
    if (config.verbose) {
        std::cout << "  Result: " << result_to_string(stats.result)
                  << " (" << stats.moves << " moves)\n";
    }
    
    return stats;
}

// ============================================================================
// Main Benchmark
// ============================================================================

int main(int argc, char** argv) {
    MatchConfig config = parse_args(argc, argv);
    
    std::cout << "========================================\n";
    std::cout << "Head-to-Head Match Benchmark\n";
    std::cout << "========================================\n\n";
    
    std::cout << "NOTE: This benchmark plays games between engines.\n";
    std::cout << "For full functionality, integration with a chess library is needed.\n";
    std::cout << "Current implementation is a simplified demonstration.\n\n";
    
    std::cout << "Games to play: " << config.num_games << "\n";
    std::cout << "Time per move: " << config.time_per_move_ms << "ms\n";
    std::cout << "Max moves: " << config.moves_per_game << "\n";
    std::cout << "Alternate colors: " << (config.alternate_colors ? "yes" : "no") << "\n";
    std::cout << "Output file: " << config.output_file << "\n\n";
    
    // Create engines
    auto cpu_engine = create_cpu_engine();
    auto gpu_engine = create_gpu_engine();
    
    if (!cpu_engine) {
        std::cerr << "Error: CPU engine not available\n";
        return 1;
    }
    
    if (!gpu_engine || !gpu_engine->is_available()) {
        std::cerr << "Error: GPU engine not available\n";
        std::cerr << "Head-to-head matches require both engines.\n";
        return 1;
    }
    
    cpu_engine->initialize();
    gpu_engine->initialize();
    
    std::cout << "Engines initialized:\n";
    std::cout << "  " << cpu_engine->get_name() << "\n";
    std::cout << "  " << gpu_engine->get_name() << "\n\n";
    
    // Create CSV writer
    MatchCSV csv(config.output_file);
    
    // Track match statistics
    int cpu_wins = 0, gpu_wins = 0, draws = 0;
    
    // Run matches
    ProgressReporter progress(config.num_games, "Match Progress");
    
    for (int i = 0; i < config.num_games; i++) {
        bool cpu_white = !config.alternate_colors || (i % 2 == 0);
        
        cpu_engine->reset();
        gpu_engine->reset();
        
        GameStats stats;
        std::string white_name, black_name;
        
        if (cpu_white) {
            white_name = cpu_engine->get_name();
            black_name = gpu_engine->get_name();
            stats = play_game(*cpu_engine, *gpu_engine, config, i + 1);
        } else {
            white_name = gpu_engine->get_name();
            black_name = cpu_engine->get_name();
            stats = play_game(*gpu_engine, *cpu_engine, config, i + 1);
        }
        
        // Determine result string
        std::string result_str;
        if (stats.result == GameResult::WHITE_WIN) {
            result_str = "1-0";
            if (cpu_white) cpu_wins++; else gpu_wins++;
        } else if (stats.result == GameResult::BLACK_WIN) {
            result_str = "0-1";
            if (cpu_white) gpu_wins++; else cpu_wins++;
        } else {
            result_str = "1/2-1/2";
            draws++;
        }
        
        // Write to CSV
        csv.write_result(
            i + 1,
            white_name,
            black_name,
            result_str,
            stats.moves,
            termination_reason(stats.result),
            stats.final_fen
        );
        
        progress.update();
    }
    
    progress.finish();
    csv.flush();
    
    // Print final statistics
    EloCalculator::print_match_summary(
        cpu_engine->get_name(),
        gpu_engine->get_name(),
        cpu_wins,
        draws,
        gpu_wins
    );
    
    std::cout << "Results saved to: " << config.output_file << "\n";
    std::cout << "========================================\n";
    
    return 0;
}
