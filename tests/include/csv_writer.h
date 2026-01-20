// csv_writer.h - Type-safe CSV output for benchmarking results
// Handles escaping, buffering, and formatting

#ifndef CSV_WRITER_H
#define CSV_WRITER_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

// ============================================================================
// CSV Writer Class
// ============================================================================

class CSVWriter {
public:
    explicit CSVWriter(const std::string& filename, const std::vector<std::string>& headers)
        : file_(filename), first_row_(true) {
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open CSV file: " + filename);
        }
        write_row(headers);
    }
    
    ~CSVWriter() {
        if (file_.is_open()) {
            file_.close();
        }
    }
    
    // Write a row (variadic template for type safety)
    template<typename... Args>
    void write_row(Args&&... args) {
        std::vector<std::string> row;
        append_values(row, std::forward<Args>(args)...);
        write_row_internal(row);
    }
    
    // Write vector of strings
    void write_row(const std::vector<std::string>& row) {
        write_row_internal(row);
    }
    
    // Flush buffer
    void flush() {
        file_.flush();
    }
    
private:
    std::ofstream file_;
    bool first_row_;
    
    // Recursive template to append values
    template<typename T>
    void append_values(std::vector<std::string>& row, T&& value) {
        row.push_back(to_csv_string(std::forward<T>(value)));
    }
    
    template<typename T, typename... Args>
    void append_values(std::vector<std::string>& row, T&& first, Args&&... rest) {
        row.push_back(to_csv_string(std::forward<T>(first)));
        append_values(row, std::forward<Args>(rest)...);
    }
    
    // Convert value to CSV string
    template<typename T>
    std::string to_csv_string(const T& value) {
        std::ostringstream oss;
        oss << value;
        return escape_csv_field(oss.str());
    }
    
    std::string to_csv_string(const std::string& value) {
        return escape_csv_field(value);
    }
    
    std::string to_csv_string(const char* value) {
        return escape_csv_field(std::string(value));
    }
    
    std::string to_csv_string(bool value) {
        return value ? "true" : "false";
    }
    
    // Escape CSV field (handle quotes, commas, newlines)
    std::string escape_csv_field(const std::string& field) {
        if (field.find(',') == std::string::npos &&
            field.find('"') == std::string::npos &&
            field.find('\n') == std::string::npos &&
            field.find('\r') == std::string::npos) {
            return field;
        }
        
        std::string escaped = "\"";
        for (char c : field) {
            if (c == '"') {
                escaped += "\"\"";  // Escape quotes by doubling
            } else {
                escaped += c;
            }
        }
        escaped += "\"";
        return escaped;
    }
    
    // Write row to file
    void write_row_internal(const std::vector<std::string>& row) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) file_ << ",";
            file_ << row[i];
        }
        file_ << "\n";
    }
};

// ============================================================================
// Specialized CSV Writers for Benchmark Results
// ============================================================================

// Throughput benchmark CSV
class ThroughputCSV : public CSVWriter {
public:
    explicit ThroughputCSV(const std::string& filename)
        : CSVWriter(filename, {
            "engine", "position_name", "fen", "time_ms", 
            "nodes_or_playouts", "throughput", "depth"
        }) {}
    
    void write_result(const std::string& engine, const std::string& pos_name,
                     const std::string& fen, double time_ms, uint64_t nodes,
                     double throughput, int depth) {
        write_row(engine, pos_name, fen, time_ms, nodes, throughput, depth);
    }
};

// Fixed-time quality CSV
class FixedTimeCSV : public CSVWriter {
public:
    explicit FixedTimeCSV(const std::string& filename)
        : CSVWriter(filename, {
            "engine", "position_name", "fen", "time_budget_ms", 
            "actual_time_ms", "move_uci", "eval_cp", "depth", "nodes"
        }) {}
    
    void write_result(const std::string& engine, const std::string& pos_name,
                     const std::string& fen, int time_budget, double actual_time,
                     const std::string& move, int eval, int depth, uint64_t nodes) {
        write_row(engine, pos_name, fen, time_budget, actual_time, move, eval, depth, nodes);
    }
};

// Stockfish agreement CSV
class StockfishCSV : public CSVWriter {
public:
    explicit StockfishCSV(const std::string& filename)
        : CSVWriter(filename, {
            "engine", "position_name", "fen", "stockfish_move", 
            "engine_move", "top1_match", "stockfish_eval", "engine_eval", "eval_diff"
        }) {}
    
    void write_result(const std::string& engine, const std::string& pos_name,
                     const std::string& fen, const std::string& sf_move,
                     const std::string& eng_move, bool match, int sf_eval,
                     int eng_eval, int eval_diff) {
        write_row(engine, pos_name, fen, sf_move, eng_move, match, sf_eval, eng_eval, eval_diff);
    }
};

// Head-to-head match CSV
class MatchCSV : public CSVWriter {
public:
    explicit MatchCSV(const std::string& filename)
        : CSVWriter(filename, {
            "game_id", "white_engine", "black_engine", "result", 
            "moves", "termination_reason", "final_fen"
        }) {}
    
    void write_result(int game_id, const std::string& white, const std::string& black,
                     const std::string& result, int moves, const std::string& termination,
                     const std::string& final_fen) {
        write_row(game_id, white, black, result, moves, termination, final_fen);
    }
};

#endif // CSV_WRITER_H
