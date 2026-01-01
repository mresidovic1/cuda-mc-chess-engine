@echo off
REM Advanced Monte Carlo Chess Engine - Build for Windows

echo Building Advanced Monte Carlo...

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set CUDA_ARCH=sm_75

where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: nvcc not found!
    exit /b 1
)

echo Compiling CUDA kernel...
nvcc -O3 -arch=%CUDA_ARCH% -c monte_carlo_advanced_kernel.cu -o monte_carlo_advanced_kernel.o
nvcc -O3 -arch=%CUDA_ARCH% -c monte_carlo_advanced_launcher.cu -o monte_carlo_advanced_launcher.o
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Compiling C++ wrapper...
g++ -O3 -std=c++17 -c monte_carlo_advanced.cpp -o monte_carlo_advanced.o -I.. -I"%CUDA_PATH%\include"
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Compiling main...
g++ -O3 -std=c++17 -c main_advanced.cpp -o main_advanced.o -I.. -I"%CUDA_PATH%\include"
if %ERRORLEVEL% NEQ 0 exit /b 1

echo Linking...
g++ -O3 main_advanced.o monte_carlo_advanced.o monte_carlo_advanced_kernel.o monte_carlo_advanced_launcher.o -o monte_carlo_advanced.exe -L"%CUDA_PATH%\lib\x64" -lcudart -lcurand
if %ERRORLEVEL% NEQ 0 exit /b 1

echo.
echo Build OK! Run: monte_carlo_advanced.exe [simulations] [fen]
