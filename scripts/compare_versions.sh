#!/bin/bash

# Chess Engine Version Comparison Script
# This script builds and compares different versions of the chess engine

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
RESULTS_DIR="$PROJECT_DIR/comparison_results"
DEPTH=${DEPTH:-20}

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Chess Engine Version Comparison"
echo "=========================================="
echo "Depth: $DEPTH"
echo ""

# Function to build and test a branch
build_and_test() {
    local branch=$1
    local name=$2
    local source_file=$3
    
    echo "----------------------------------------"
    echo "Testing: $name (branch: $branch)"
    echo "----------------------------------------"
    
    # Save current branch
    current_branch=$(git branch --show-current)
    
    # Checkout target branch
    git checkout "$branch" --quiet
    
    # Determine structure and build
    if [ -f "src/chess_engine_parallelized.cpp" ]; then
        # New structure
        cd "$BUILD_DIR"
        meson setup --reconfigure .. >/dev/null 2>&1 || true
        ninja >/dev/null 2>&1
        
        if [ -f "test_suite_parallel" ]; then
            ./test_suite_parallel --depth=$DEPTH --json --quiet > "$RESULTS_DIR/${name}.json" 2>/dev/null || true
        elif [ -f "test_suite" ]; then
            ./test_suite --depth=$DEPTH --json --quiet > "$RESULTS_DIR/${name}.json" 2>/dev/null || true
        fi
    elif [ -f "chess_engine_parallelized.cpp" ] || [ -f "chess_parallelization.cpp" ]; then
        # Old structure - need custom build
        cd "$BUILD_DIR"
        meson setup --reconfigure .. >/dev/null 2>&1 || true
        ninja >/dev/null 2>&1
        
        if [ -f "test_suite" ]; then
            ./test_suite --depth=$DEPTH --json --quiet > "$RESULTS_DIR/${name}.json" 2>/dev/null || true
        fi
    fi
    
    cd "$PROJECT_DIR"
    
    # Checkout back to original branch
    git checkout "$current_branch" --quiet
    
    echo "Results saved to: $RESULTS_DIR/${name}.json"
}

# Test current parallel version
echo ""
echo "Building current version..."
cd "$BUILD_DIR"
ninja >/dev/null 2>&1

echo ""
echo "Running Parallel v3..."
./test_suite_parallel --depth=$DEPTH --json --quiet > "$RESULTS_DIR/parallel_v3.json" 2>/dev/null || true

echo "Running Sequential v3..."
./test_suite_sequential --depth=$DEPTH --json --quiet > "$RESULTS_DIR/sequential_v3.json" 2>/dev/null || true

cd "$PROJECT_DIR"

echo ""
echo "=========================================="
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "To compare results, run:"
echo "  python3 scripts/run_benchmarks.py \\"
echo "    --engines \"build/test_suite_sequential:Sequential,build/test_suite_parallel:Parallel\" \\"
echo "    --depth $DEPTH"
echo "=========================================="

