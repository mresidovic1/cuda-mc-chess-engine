// benchmark_utils.h - Utility functions for benchmarking
// Timing, progress reporting, statistics

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================================
// High-Precision Timer
// ============================================================================

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    double elapsed_sec() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Progress Reporter
// ============================================================================

class ProgressReporter {
public:
    ProgressReporter(int total, const std::string& task_name = "Progress")
        : total_(total), current_(0), task_name_(task_name) {
        timer_.reset();
    }
    
    void update(int increment = 1) {
        current_ += increment;
        print_progress();
    }
    
    void set_current(int current) {
        current_ = current;
        print_progress();
    }
    
    void finish() {
        current_ = total_;
        print_progress();
        std::cout << "\n";
    }
    
private:
    int total_;
    int current_;
    std::string task_name_;
    Timer timer_;
    
    void print_progress() {
        if (total_ == 0) return;
        
        double percent = 100.0 * current_ / total_;
        double elapsed = timer_.elapsed_sec();
        double eta = (current_ > 0) ? elapsed * (total_ - current_) / current_ : 0;
        
        std::cout << "\r" << task_name_ << ": "
                  << current_ << "/" << total_ << " ("
                  << std::fixed << std::setprecision(1) << percent << "%) "
                  << "Elapsed: " << std::setprecision(1) << elapsed << "s "
                  << "ETA: " << std::setprecision(1) << eta << "s   "
                  << std::flush;
    }
};

// ============================================================================
// Statistics Calculator
// ============================================================================

class Statistics {
public:
    Statistics() = default;
    
    void add(double value) {
        values_.push_back(value);
    }
    
    double mean() const {
        if (values_.empty()) return 0;
        double sum = 0;
        for (double v : values_) sum += v;
        return sum / values_.size();
    }
    
    double median() const {
        if (values_.empty()) return 0;
        auto sorted = values_;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        if (n % 2 == 0) {
            return (sorted[n/2-1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
    }
    
    double stddev() const {
        if (values_.size() < 2) return 0;
        double m = mean();
        double sum_sq_diff = 0;
        for (double v : values_) {
            double diff = v - m;
            sum_sq_diff += diff * diff;
        }
        return std::sqrt(sum_sq_diff / (values_.size() - 1));
    }
    
    double min() const {
        if (values_.empty()) return 0;
        return *std::min_element(values_.begin(), values_.end());
    }
    
    double max() const {
        if (values_.empty()) return 0;
        return *std::max_element(values_.begin(), values_.end());
    }
    
    size_t count() const {
        return values_.size();
    }
    
    void print_summary(const std::string& name = "Statistics") const {
        std::cout << "\n" << name << ":\n";
        std::cout << "  Count:  " << count() << "\n";
        std::cout << "  Mean:   " << std::fixed << std::setprecision(2) << mean() << "\n";
        std::cout << "  Median: " << std::fixed << std::setprecision(2) << median() << "\n";
        std::cout << "  StdDev: " << std::fixed << std::setprecision(2) << stddev() << "\n";
        std::cout << "  Min:    " << std::fixed << std::setprecision(2) << min() << "\n";
        std::cout << "  Max:    " << std::fixed << std::setprecision(2) << max() << "\n";
    }
    
private:
    std::vector<double> values_;
};

// ============================================================================
// Elo Rating Calculator
// ============================================================================

class EloCalculator {
public:
    // Calculate expected score for player A
    static double expected_score(double rating_a, double rating_b) {
        return 1.0 / (1.0 + std::pow(10.0, (rating_b - rating_a) / 400.0));
    }
    
    // Calculate Elo difference from win/draw/loss record
    // Returns (elo_diff, confidence_interval_95)
    static std::pair<double, double> elo_difference(int wins, int draws, int losses) {
        int total = wins + draws + losses;
        if (total == 0) return {0, 0};
        
        double score = (wins + 0.5 * draws) / total;
        
        // Clamp score to avoid log(0)
        score = std::max(0.001, std::min(0.999, score));
        
        // Elo difference from score
        double elo_diff = -400.0 * std::log10(1.0 / score - 1.0);
        
        // 95% confidence interval (normal approximation)
        double p = score;
        double stderr = std::sqrt(p * (1 - p) / total);
        double margin = 1.96 * stderr;  // 95% CI
        
        double lower_score = std::max(0.001, p - margin);
        double upper_score = std::min(0.999, p + margin);
        
        double lower_elo = -400.0 * std::log10(1.0 / lower_score - 1.0);
        double upper_elo = -400.0 * std::log10(1.0 / upper_score - 1.0);
        
        double confidence = (upper_elo - lower_elo) / 2.0;
        
        return {elo_diff, confidence};
    }
    
    // Print match summary with Elo estimate
    static void print_match_summary(const std::string& engine_a, const std::string& engine_b,
                                   int wins_a, int draws, int wins_b) {
        int total = wins_a + draws + wins_b;
        double score_a = (wins_a + 0.5 * draws) / total;
        
        auto [elo_diff, conf] = elo_difference(wins_a, draws, wins_b);
        
        std::cout << "\n========================================\n";
        std::cout << "Match Summary: " << engine_a << " vs " << engine_b << "\n";
        std::cout << "========================================\n";
        std::cout << "Games:  " << total << "\n";
        std::cout << "  " << std::setw(12) << std::left << engine_a 
                  << ": " << wins_a << " wins\n";
        std::cout << "  " << std::setw(12) << std::left << "Draws" 
                  << ": " << draws << "\n";
        std::cout << "  " << std::setw(12) << std::left << engine_b 
                  << ": " << wins_b << " wins\n";
        std::cout << "\nScore:  " << std::fixed << std::setprecision(1) 
                  << (score_a * 100) << "% - " 
                  << ((1 - score_a) * 100) << "%\n";
        std::cout << "Elo:    " << engine_a << " is " 
                  << std::setprecision(0) << std::showpos << elo_diff << std::noshowpos
                  << " ± " << std::setprecision(0) << conf << " Elo\n";
        std::cout << "========================================\n\n";
    }
};

// ============================================================================
// Formatting Helpers
// ============================================================================

inline std::string format_throughput(double throughput) {
    if (throughput >= 1e6) {
        return std::to_string(static_cast<int>(throughput / 1e6)) + "M";
    } else if (throughput >= 1e3) {
        return std::to_string(static_cast<int>(throughput / 1e3)) + "K";
    } else {
        return std::to_string(static_cast<int>(throughput));
    }
}

inline std::string format_time(double ms) {
    if (ms >= 1000) {
        return std::to_string(static_cast<int>(ms / 1000)) + "s";
    } else {
        return std::to_string(static_cast<int>(ms)) + "ms";
    }
}

#endif // BENCHMARK_UTILS_H
