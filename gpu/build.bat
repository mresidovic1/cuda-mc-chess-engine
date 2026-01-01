@echo off
REM Build script for Monte Carlo Chess GPU version on Windows

echo Building Monte Carlo Chess Engine (GPU Version)...

REM Set CUDA path (adjust to your CUDA installation)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6

REM Check if nvcc exists
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nvcc not found! Please install CUDA Toolkit or add it to PATH.
    echo Expected path: "%CUDA_PATH%\bin"
    exit /b 1
)

echo Step 1: Compiling CUDA kernel...
nvcc -O3 -arch=sm_70 -c monte_carlo_kernel.cu -o monte_carlo_kernel.o
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile CUDA kernel
    exit /b 1
)

echo Step 2: Compiling C++ wrapper...
g++ -O3 -c monte_carlo_gpu.cpp -o monte_carlo_gpu.o -I.. -I"%CUDA_PATH%\include"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile C++ wrapper
    exit /b 1
)

echo Step 3: Compiling main program...
g++ -O3 -c main_gpu.cpp -o main_gpu.o -I.. -I"%CUDA_PATH%\include"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to compile main program
    exit /b 1
)

echo Step 4: Linking executable...
g++ -O3 main_gpu.o monte_carlo_gpu.o monte_carlo_kernel.o -o monte_carlo_chess_gpu.exe -L"%CUDA_PATH%\lib\x64" -lcudart -lcurand
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to link executable
    exit /b 1
)

echo.
echo Build successful! Executable: monte_carlo_chess_gpu.exe
echo.
echo Usage:
echo   monte_carlo_chess_gpu.exe [num_simulations] [optional_fen]
echo.
echo Example:
echo   monte_carlo_chess_gpu.exe 50000
echo   monte_carlo_chess_gpu.exe 100000 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
echo.
