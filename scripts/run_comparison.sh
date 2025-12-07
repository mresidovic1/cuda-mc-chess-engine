#!/bin/bash
#
# Chess Engine Comparison Test Runner
#
# Runs the test suite against multiple engine versions and generates
# a comparison report.
#
# Usage:
#   ./run_comparison.sh [options]
#
# Options:
#   --engines="v1,v2,v3"    Comma-separated list of engine binaries
#   --mode=depth|time       Test mode (default: depth)
#   --output=DIR            Output directory for results
#

set -e

# Default configuration
OUTPUT_DIR="comparison_results_$(date +%Y%m%d_%H%M%S)"
TEST_MODE="depth"
DEPTH=20
TIME_LIMIT=30

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --engines=*)
            ENGINES="${1#*=}"
            shift
            ;;
        --mode=*)
            TEST_MODE="${1#*=}"
            shift
            ;;
        --output=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --depth=*)
            DEPTH="${1#*=}"
            shift
            ;;
        --time=*)
            TIME_LIMIT="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Chess Engine Comparison Test Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --engines=v1,v2,v3   Comma-separated engine binaries (default: build/test_suite)"
            echo "  --mode=depth|time    Test mode (default: depth)"
            echo "  --depth=N            Search depth for depth mode (default: 20)"
            echo "  --time=N             Time limit in seconds for time mode (default: 30)"
            echo "  --output=DIR         Output directory"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default engine if not specified
if [ -z "$ENGINES" ]; then
    ENGINES="build/test_suite"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================================================="
echo "                    CHESS ENGINE COMPARISON TESTS"
echo "=============================================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Test mode: $TEST_MODE"
if [ "$TEST_MODE" = "depth" ]; then
    echo "Search depth: $DEPTH"
else
    echo "Time limit: ${TIME_LIMIT}s per position"
fi
echo "------------------------------------------------------------------------------"
echo ""

# Convert comma-separated engines to array
IFS=',' read -ra ENGINE_LIST <<< "$ENGINES"

# Run tests for each engine
for engine in "${ENGINE_LIST[@]}"; do
    engine_name=$(basename "$engine")
    echo ""
    echo "=============================================================================="
    echo "Testing: $engine_name"
    echo "=============================================================================="
    
    if [ ! -f "$engine" ]; then
        echo "ERROR: Engine binary not found: $engine"
        continue
    fi
    
    output_file="$OUTPUT_DIR/${engine_name}_results.txt"
    json_file="$OUTPUT_DIR/${engine_name}_results.json"
    
    # Build test command
    if [ "$TEST_MODE" = "depth" ]; then
        test_args="--mode=depth --depth=$DEPTH --engine=$engine_name"
    else
        test_args="--mode=time --time=$TIME_LIMIT --engine=$engine_name"
    fi
    
    # Run tests and capture output
    echo "Running tests..."
    $engine $test_args > "$output_file" 2>&1 || true
    
    # Also generate JSON output
    $engine $test_args --json --quiet > "$json_file" 2>&1 || true
    
    echo "Results saved to: $output_file"
done

# Generate comparison summary
echo ""
echo "=============================================================================="
echo "                         GENERATING COMPARISON REPORT"
echo "=============================================================================="

# Create summary report
summary_file="$OUTPUT_DIR/comparison_summary.txt"

cat > "$summary_file" << 'HEADER'
================================================================================
                    CHESS ENGINE COMPARISON REPORT
================================================================================

HEADER

echo "Generated: $(date)" >> "$summary_file"
echo "Test Mode: $TEST_MODE" >> "$summary_file"
if [ "$TEST_MODE" = "depth" ]; then
    echo "Search Depth: $DEPTH" >> "$summary_file"
else
    echo "Time Limit: ${TIME_LIMIT}s" >> "$summary_file"
fi
echo "" >> "$summary_file"

echo "INDIVIDUAL RESULTS:" >> "$summary_file"
echo "--------------------------------------------------------------------------------" >> "$summary_file"

for engine in "${ENGINE_LIST[@]}"; do
    engine_name=$(basename "$engine")
    result_file="$OUTPUT_DIR/${engine_name}_results.txt"
    
    if [ -f "$result_file" ]; then
        echo "" >> "$summary_file"
        echo "=== $engine_name ===" >> "$summary_file"
        # Extract summary section from results
        grep -A 20 "TEST SUMMARY" "$result_file" >> "$summary_file" 2>/dev/null || echo "No summary found" >> "$summary_file"
    fi
done

echo "" >> "$summary_file"
echo "=================================================================================" >> "$summary_file"
echo "                              END OF REPORT" >> "$summary_file"
echo "=================================================================================" >> "$summary_file"

echo ""
echo "Comparison complete!"
echo "Summary report: $summary_file"
echo ""
echo "Individual results:"
for engine in "${ENGINE_LIST[@]}"; do
    engine_name=$(basename "$engine")
    echo "  - $OUTPUT_DIR/${engine_name}_results.txt"
done

