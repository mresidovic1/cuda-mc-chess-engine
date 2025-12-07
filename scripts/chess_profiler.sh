#!/bin/bash

# Configuration
CHESS_ENGINE="./build/chess-tests"
OUTPUT_DIR="profile_results_$(date +%Y%m%d_%H%M%S)"
PROFILE_DURATION=30  # seconds

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "=========================================="
echo "Chess Engine Performance Profiler"
echo "=========================================="
echo ""

# 1. Time profiling with sample
echo "[1/3] Collecting time profile data..."
../build/chess-tests &
CHESS_PID=$!
sample $CHESS_PID $PROFILE_DURATION -file time_profile.txt 2>/dev/null
wait $CHESS_PID
echo "✓ Time profile saved to time_profile.txt"

# 2. Memory profiling
echo "[2/3] Collecting memory allocation data..."
leaks --atExit -- ../build/chess-tests > memory_leaks.txt 2>&1
echo "✓ Memory analysis saved to memory_leaks.txt"

# 3. Generate summary report
echo "[3/3] Generating summary report..."

cat > summary_report.txt << 'EOF'
================================================================================
                    CHESS ENGINE PERFORMANCE REPORT
================================================================================

EOF

echo "Generated: $(date)" >> summary_report.txt
echo "" >> summary_report.txt

# Parse time profile for top functions
echo "TOP 20 FUNCTIONS BY TIME:" >> summary_report.txt
echo "----------------------------------------" >> summary_report.txt
grep -E "^\s+[0-9]+" time_profile.txt | head -20 >> summary_report.txt
echo "" >> summary_report.txt

# Memory summary
echo "MEMORY ANALYSIS:" >> summary_report.txt
echo "----------------------------------------" >> summary_report.txt
grep -E "(Process|leaks for|total leaked bytes)" memory_leaks.txt >> summary_report.txt
echo "" >> summary_report.txt

echo "✓ Summary report saved to summary_report.txt"
echo ""
echo "All results saved in: $OUTPUT_DIR"
echo ""
echo "View summary: cat $OUTPUT_DIR/summary_report.txt"
echo "View detailed time profile: open $OUTPUT_DIR/time_profile.txt"

# Return to original directory
cd ..
