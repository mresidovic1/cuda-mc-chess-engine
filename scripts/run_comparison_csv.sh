#!/bin/bash

# Chess Engine Comparison Script
# Compares Sequential v1/v3 and Parallel v1/v3
# Exports results to CSV

cd "$(dirname "$0")/.."

OUTPUT_DIR="comparison_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="${OUTPUT_DIR}/comparison_${TIMESTAMP}.csv"
REPORT_FILE="${OUTPUT_DIR}/report_${TIMESTAMP}.txt"

mkdir -p "$OUTPUT_DIR"

# Time limit per position (seconds)
TIME_LIMIT=60

echo ""
echo "========================================"
echo "  CHESS ENGINE COMPARISON"
echo "  $(date)"
echo "========================================"
echo ""

# CSV Header
echo "Engine,Version,Level,Passed,Total,Percentage,Time_sec,Nodes,NPS" > "$CSV_FILE"

# Initialize report
{
    echo "========================================"
    echo "  CHESS ENGINE COMPARISON REPORT"
    echo "  Generated: $(date)"
    echo "  Time limit: ${TIME_LIMIT}s per position"
    echo "========================================"
    echo ""
} > "$REPORT_FILE"

# Function to run a single test
run_single_test() {
    local ENGINE_PATH=$1
    local ENGINE_NAME=$2
    local VERSION=$3
    local LEVEL=$4
    
    echo "Testing: $ENGINE_NAME $VERSION - $LEVEL"
    
    # Run test
    OUTPUT=$($ENGINE_PATH --mode=time --time=$TIME_LIMIT --level=$LEVEL --engine="$ENGINE_NAME-$VERSION" 2>&1)
    
    # Parse results - look for the level line or TOTAL
    if [ "$LEVEL" = "all" ]; then
        SEARCH_PATTERN="^TOTAL"
    else
        LEVEL_UPPER=$(echo "$LEVEL" | tr '[:lower:]' '[:upper:]')
        SEARCH_PATTERN="^${LEVEL_UPPER}"
    fi
    
    RESULT_LINE=$(echo "$OUTPUT" | grep -E "$SEARCH_PATTERN" | head -1)
    
    PASSED=$(echo "$RESULT_LINE" | awk '{print $2}')
    TOTAL=$(echo "$RESULT_LINE" | awk '{print $3}')
    PCT=$(echo "$RESULT_LINE" | awk '{print $4}' | tr -d '%')
    
    # Get time
    TIME_SEC=$(echo "$OUTPUT" | grep "Total execution time" | awk '{print $4}')
    
    # Get nodes and NPS from output (look for "nodes" and "nps" in info lines)
    # Sum up all nodes from info depth lines
    NODES=$(echo "$OUTPUT" | grep -E "info depth.*nodes" | awk '{sum+=$5} END {print sum+0}')
    
    # Get average NPS (nodes per second) - take last NPS value or calculate from total
    NPS=$(echo "$OUTPUT" | grep -E "info depth.*nps" | tail -1 | awk '{print $7}')
    
    # If NPS not found, calculate from nodes and time
    if [ -z "$NPS" ] || [ "$NPS" = "0" ]; then
        if [ -n "$NODES" ] && [ "$NODES" != "0" ] && [ -n "$TIME_SEC" ] && [ "$TIME_SEC" != "0" ]; then
            NPS=$(echo "scale=0; $NODES / $TIME_SEC" | bc 2>/dev/null || echo "0")
        else
            NPS="0"
        fi
    fi
    
    # Default values if parsing failed
    [ -z "$PASSED" ] && PASSED="0"
    [ -z "$TOTAL" ] && TOTAL="0"
    [ -z "$PCT" ] && PCT="0"
    [ -z "$TIME_SEC" ] && TIME_SEC="0"
    [ -z "$NODES" ] && NODES="0"
    [ -z "$NPS" ] && NPS="0"
    
    echo "  -> $PASSED/$TOTAL ($PCT%) in ${TIME_SEC}s (${NODES} nodes, ${NPS} nps)"
    
    # Write to CSV
    echo "$ENGINE_NAME,$VERSION,$LEVEL,$PASSED,$TOTAL,$PCT,$TIME_SEC,$NODES,$NPS" >> "$CSV_FILE"
    
    # Append to report
    {
        echo "--- $ENGINE_NAME $VERSION - $LEVEL ---"
        echo "$OUTPUT"
        echo ""
    } >> "$REPORT_FILE"
}

# Function to run first hard test only
run_first_hard_test() {
    local ENGINE_PATH=$1
    local ENGINE_NAME=$2
    local VERSION=$3
    
    echo "Testing: $ENGINE_NAME $VERSION - hard (first position only)"
    
    OUTPUT=$($ENGINE_PATH --mode=time --time=$TIME_LIMIT --level=hard --engine="$ENGINE_NAME-$VERSION" 2>&1)
    
    # Check if first test passed
    FIRST_RESULT=$(echo "$OUTPUT" | grep -E "^\[1/" | head -1)
    TIME_LINE=$(echo "$OUTPUT" | grep -E "^\[1/" -A 20 | grep -E "Time:" | head -1)
    
    if echo "$OUTPUT" | grep -E "^\[1/.*\[PASS\]" > /dev/null; then
        PASSED=1
        PCT="100"
    else
        PASSED=0
        PCT="0"
    fi
    
    # Get time for first test
    TIME_MS=$(echo "$TIME_LINE" | grep -oE "Time: [0-9]+" | grep -oE "[0-9]+")
    TIME_SEC=$(echo "scale=1; ${TIME_MS:-0} / 1000" | bc 2>/dev/null || echo "0")
    
    # Get nodes and NPS for first test
    NODES=$(echo "$OUTPUT" | grep -E "info depth.*nodes" | head -1 | awk '{print $5}')
    NPS=$(echo "$OUTPUT" | grep -E "info depth.*nps" | head -1 | awk '{print $7}')
    
    [ -z "$NODES" ] && NODES="0"
    [ -z "$NPS" ] && NPS="0"
    
    echo "  -> $PASSED/1 ($PCT%) in ${TIME_SEC}s (${NODES} nodes, ${NPS} nps)"
    
    # Write to CSV
    echo "$ENGINE_NAME,$VERSION,hard_first,$PASSED,1,$PCT,$TIME_SEC,$NODES,$NPS" >> "$CSV_FILE"
    
    # Append to report
    {
        echo "--- $ENGINE_NAME $VERSION - hard (first only) ---"
        echo "$OUTPUT"
        echo ""
    } >> "$REPORT_FILE"
}

echo ""
echo "=== SEQUENTIAL TESTS ==="
echo ""

# Sequential v3: easy, medium, 1 hard
if [ -x "./versions/sequential_v3" ]; then
    run_single_test "./versions/sequential_v3" "Sequential" "v3" "easy"
    run_single_test "./versions/sequential_v3" "Sequential" "v3" "medium"
    run_first_hard_test "./versions/sequential_v3" "Sequential" "v3"
else
    echo "WARNING: versions/sequential_v3 not found"
fi

echo ""
echo "=== PARALLEL TESTS ==="
echo ""

# Parallel v1: all tests
if [ -x "./versions/parallel_v1" ]; then
    run_single_test "./versions/parallel_v1" "Parallel" "v1" "easy"
    run_single_test "./versions/parallel_v1" "Parallel" "v1" "medium"
    run_single_test "./versions/parallel_v1" "Parallel" "v1" "hard"
else
    echo "WARNING: versions/parallel_v1 not found"
fi

echo ""

# Parallel v3: all tests
if [ -x "./versions/parallel_v3" ]; then
    run_single_test "./versions/parallel_v3" "Parallel" "v3" "easy"
    run_single_test "./versions/parallel_v3" "Parallel" "v3" "medium"
    run_single_test "./versions/parallel_v3" "Parallel" "v3" "hard"
else
    echo "WARNING: versions/parallel_v3 not found"
fi

echo ""
echo "========================================"
echo "  COMPARISON COMPLETE"
echo "========================================"
echo ""
echo "Files saved:"
echo "  CSV:    $CSV_FILE"
echo "  Report: $REPORT_FILE"
echo ""
echo "CSV Contents:"
echo "----------------------------------------"
cat "$CSV_FILE"
echo "----------------------------------------"
