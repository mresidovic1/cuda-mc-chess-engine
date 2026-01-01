#!/bin/bash
# Build script for Monte Carlo Chess GPU version on Linux

echo "Building Monte Carlo Chess Engine (GPU Version)..."

# Check if nvcc exists
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found! Please install CUDA Toolkit."
    exit 1
fi

# Detect CUDA architecture (adjust as needed)
CUDA_ARCH="sm_70"  # Change to sm_75, sm_80, sm_86, etc. based on your GPU

echo "Step 1: Compiling CUDA kernel..."
nvcc -O3 -arch=$CUDA_ARCH -c monte_carlo_kernel.cu -o monte_carlo_kernel.o
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile CUDA kernel"
    exit 1
fi

echo "Step 2: Compiling C++ wrapper..."
g++ -O3 -c monte_carlo_gpu.cpp -o monte_carlo_gpu.o -I.. -I/usr/local/cuda/include
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile C++ wrapper"
    exit 1
fi

echo "Step 3: Compiling main program..."
g++ -O3 -c main_gpu.cpp -o main_gpu.o -I.. -I/usr/local/cuda/include
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile main program"
    exit 1
fi

echo "Step 4: Linking executable..."
g++ -O3 main_gpu.o monte_carlo_gpu.o monte_carlo_kernel.o -o monte_carlo_chess_gpu \
    -L/usr/local/cuda/lib64 -lcudart -lcurand
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to link executable"
    exit 1
fi

echo ""
echo "Build successful! Executable: monte_carlo_chess_gpu"
echo ""
echo "Usage:"
echo "  ./monte_carlo_chess_gpu [num_simulations] [optional_fen]"
echo ""
echo "Example:"
echo "  ./monte_carlo_chess_gpu 50000"
echo "  ./monte_carlo_chess_gpu 100000 \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\""
echo ""
