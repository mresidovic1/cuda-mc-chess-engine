#!/bin/bash
# Advanced Monte Carlo Chess Engine - Build script

echo "Building Advanced Monte Carlo Chess Engine..."

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found! Install CUDA Toolkit."
    exit 1
fi

CUDA_ARCH="sm_75"  # T4=sm_75, V100=sm_70, A100=sm_80, RTX3090=sm_86

echo "Compiling CUDA kernel..."
nvcc -O3 -arch=$CUDA_ARCH -c monte_carlo_advanced_kernel.cu -o monte_carlo_advanced_kernel.o || exit 1
nvcc -O3 -arch=$CUDA_ARCH -c monte_carlo_advanced_launcher.cu -o monte_carlo_advanced_launcher.o || exit 1

echo "Compiling C++ wrapper..."
g++ -O3 -std=c++17 -c monte_carlo_advanced.cpp -o monte_carlo_advanced.o -I.. -I/usr/local/cuda/include || exit 1

echo "Compiling main..."
g++ -O3 -std=c++17 -c main_advanced.cpp -o main_advanced.o -I.. -I/usr/local/cuda/include || exit 1

echo "Linking..."
g++ -O3 main_advanced.o monte_carlo_advanced.o monte_carlo_advanced_kernel.o monte_carlo_advanced_launcher.o -o monte_carlo_advanced \
    -L/usr/local/cuda/lib64 -lcudart -lcurand || exit 1

echo ""
echo "Build OK! Run: ./monte_carlo_advanced [simulations] [fen]"
echo "Example: ./monte_carlo_advanced 50000"
echo "  ./monte_carlo_chess_gpu 50000"
echo "  ./monte_carlo_chess_gpu 100000 \"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\""
echo ""
