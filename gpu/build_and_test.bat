@echo off
REM Build and Test Script for PUCT MCTS Chess Engine
REM Builds main executable and runs all test suites

setlocal enabledelayedexpansion

REM ========================================
REM Check if VS environment is initialized
REM ========================================

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Visual Studio environment not initialized.
    echo.
    echo Please run this script from one of:
    echo   - Developer Command Prompt for VS 2022
    echo   - Developer Command Prompt for VS 2019
    echo   - x64 Native Tools Command Prompt for VS
    echo.
    echo Or run this first:
    echo   "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    echo.
    exit /b 1
)

echo ========================================
echo PUCT MCTS Chess Engine - Build and Test
echo ========================================
echo.

REM Parse arguments
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help

set MODE=tests
if "%1"=="--clean" set MODE=clean
if "%1"=="-c" set MODE=clean
if "%1"=="--main" set MODE=main
if "%1"=="--puct" set MODE=puct
if "%1"=="--bratko" set MODE=bratko
if "%1"=="--easy" set MODE=easy
if "%1"=="--medium" set MODE=medium
if "%1"=="--hard" set MODE=hard
if "%1"=="--runner" set MODE=runner
if "%1"=="--all" set MODE=all

REM ========================================
REM Configuration
REM ========================================

set BUILD_DIR=build
set CUDA_ARCH=sm_75

REM Find CUDA installation
if defined CUDA_PATH (
    set CUDA_ROOT=%CUDA_PATH%
) else (
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" (
        set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" (
        set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5" (
        set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" (
        set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0
    ) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" (
        set CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
    ) else (
        echo ERROR: CUDA not found. Please install CUDA or set CUDA_PATH environment variable.
        exit /b 1
    )
)

echo Using CUDA: %CUDA_ROOT%
echo.

set NVCC=%CUDA_ROOT%\bin\nvcc.exe
if not exist "%NVCC%" (
    echo ERROR: nvcc.exe not found at %NVCC%
    exit /b 1
)

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM ========================================
REM Clean
REM ========================================

if "%MODE%"=="clean" (
    echo Cleaning build artifacts...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    if exist *.exe del /q *.exe
    if exist *.obj del /q *.obj
    echo Clean complete!
    goto :eof
)

REM ========================================
REM Compile CUDA kernels
REM ========================================

echo ========================================
echo Compiling CUDA kernels...
echo ========================================

set CUDA_SOURCES=src\init_tables.cu src\kernels\movegen.cu src\kernels\evaluation.cu src\kernels\playouts.cu src\kernels\tactical_search.cu src\kernels\kernel_launchers.cu

set CUDA_OBJS=
for %%f in (%CUDA_SOURCES%) do (
    set SRC=%%f
    for %%n in (%%~nf) do set NAME=%%n
    set OBJ=%BUILD_DIR%\!NAME!.obj

    echo   Compiling: %%f
    "%NVCC%" -O2 -arch=%CUDA_ARCH% -std=c++20 -Iinclude -Xcompiler=/wd4819 -Xcompiler=-std:c++20 --use_fast_math --disable-warnings -rdc=true -c %%f -o !OBJ!
    if errorlevel 1 (
        echo ERROR: Failed to compile %%f
        exit /b 1
    )

    set CUDA_OBJS=!CUDA_OBJS! !OBJ!
)

echo.

REM ========================================
REM Compile C++ sources
REM ========================================

echo ========================================
echo Compiling C++ sources...
echo ========================================

set CPP_SOURCES=src\cpu_movegen.cpp src\puct_mcts.cpp

set CPP_OBJS=
for %%f in (%CPP_SOURCES%) do (
    set SRC=%%f
    for %%n in (%%~nf) do set NAME=%%n
    set OBJ=%BUILD_DIR%\!NAME!.obj

    echo   Compiling: %%f
    cl.exe /std:c++20 /O2 /EHsc -Iinclude -I"%CUDA_ROOT%\include" /c %%f /Fo!OBJ!
    if errorlevel 1 (
        echo ERROR: Failed to compile %%f
        exit /b 1
    )

    set CPP_OBJS=!CPP_OBJS! !OBJ!
)

echo.

REM ========================================
REM Device link CUDA objects
REM ========================================

echo ========================================
echo Device linking CUDA objects...
echo ========================================

set DEVICE_LINK_OBJ=%BUILD_DIR%\device_link.obj

"%NVCC%" -dlink %CUDA_OBJS% -o %DEVICE_LINK_OBJ% -arch=%CUDA_ARCH%
if errorlevel 1 (
    echo ERROR: Failed to device link CUDA objects
    exit /b 1
)

echo Device link complete!
echo.

REM All object files for linking
set ALL_OBJS=%CUDA_OBJS% %CPP_OBJS% %DEVICE_LINK_OBJ%

REM ========================================
REM Build main executable
REM ========================================

if "%MODE%"=="main" goto :build_main
if "%MODE%"=="all" goto :build_main
goto :skip_main

:build_main
echo ========================================
echo Building main executable...
echo ========================================

set MAIN_OBJ=%BUILD_DIR%\main.obj
echo   Compiling: src\main.cpp
cl.exe /std:c++20 /O2 /EHsc -Iinclude -I"%CUDA_ROOT%\include" /c src\main.cpp /Fo%MAIN_OBJ%
if errorlevel 1 (
    echo ERROR: Failed to compile main.cpp
    exit /b 1
)

echo   Linking: puct_chess.exe
cl.exe %ALL_OBJS% %MAIN_OBJ% /Fe:puct_chess.exe /link /LIBPATH:"%CUDA_ROOT%\lib\x64" cudart.lib curand.lib
if errorlevel 1 (
    echo ERROR: Failed to link puct_chess.exe
    exit /b 1
)

echo Built puct_chess.exe successfully!
echo.

if "%MODE%"=="main" goto :done

:skip_main

REM ========================================
REM Build and run PUCT MCTS tests
REM ========================================

if "%MODE%"=="puct" goto :build_puct
if "%MODE%"=="tests" goto :build_puct
if "%MODE%"=="all" goto :build_puct
goto :skip_puct

:build_puct
echo ========================================
echo Building PUCT MCTS tests...
echo ========================================

set TEST_PUCT_OBJ=%BUILD_DIR%\test_puct_mcts.obj
echo   Compiling: tests\test_puct_mcts.cpp
cl.exe /std:c++20 /O2 /EHsc -Iinclude -I"%CUDA_ROOT%\include" /c tests\test_puct_mcts.cpp /Fo%TEST_PUCT_OBJ%
if errorlevel 1 (
    echo ERROR: Failed to compile test_puct_mcts.cpp
    exit /b 1
)

echo   Linking: test_puct_mcts.exe
cl.exe %ALL_OBJS% %TEST_PUCT_OBJ% /Fe:test_puct_mcts.exe /link /LIBPATH:"%CUDA_ROOT%\lib\x64" cudart.lib curand.lib
if errorlevel 1 (
    echo ERROR: Failed to link test_puct_mcts.exe
    exit /b 1
)

echo Built test_puct_mcts.exe successfully!
echo.

echo ========================================
echo Running PUCT MCTS tests...
echo ========================================
test_puct_mcts.exe
if errorlevel 1 (
    echo.
    echo ERROR: PUCT tests failed
    exit /b 1
)
echo.

if "%MODE%"=="puct" goto :done

:skip_puct

REM ========================================
REM Build and run Bratko-Kopec tests
REM ========================================

if "%MODE%"=="bratko" goto :build_bratko
if "%MODE%"=="tests" goto :build_bratko
if "%MODE%"=="all" goto :build_bratko
goto :skip_bratko

:build_bratko
echo ========================================
echo Building Bratko-Kopec tests...
echo ========================================

set TEST_BRATKO_OBJ=%BUILD_DIR%\test_bratko_kopec.obj
echo   Compiling: tests\test_bratko_kopec.cpp
cl.exe /std:c++20 /O2 /EHsc -Iinclude -I"%CUDA_ROOT%\include" /c tests\test_bratko_kopec.cpp /Fo%TEST_BRATKO_OBJ%
if errorlevel 1 (
    echo ERROR: Failed to compile test_bratko_kopec.cpp
    exit /b 1
)

echo   Linking: test_bratko_kopec.exe
cl.exe %ALL_OBJS% %TEST_BRATKO_OBJ% /Fe:test_bratko_kopec.exe /link /LIBPATH:"%CUDA_ROOT%\lib\x64" cudart.lib curand.lib
if errorlevel 1 (
    echo ERROR: Failed to link test_bratko_kopec.exe
    exit /b 1
)

echo Built test_bratko_kopec.exe successfully!
echo.

echo ========================================
echo Running Bratko-Kopec tests...
echo ========================================
test_bratko_kopec.exe --sims 350000
if errorlevel 1 (
    echo.
    echo ERROR: Bratko-Kopec tests failed
    exit /b 1
)
echo.

:skip_bratko

REM ========================================
REM Build test_runner
REM ========================================

if "%MODE%"=="easy" goto :build_runner
if "%MODE%"=="medium" goto :build_runner
if "%MODE%"=="hard" goto :build_runner
if "%MODE%"=="runner" goto :build_runner
goto :skip_runner

:build_runner
echo ========================================
echo Building test_runner...
echo ========================================

set RUNNER_OBJ=%BUILD_DIR%\test_runner.obj
echo   Compiling: tests\test_runner.cpp
cl.exe /std:c++20 /O2 /EHsc -Iinclude -I"%CUDA_ROOT%\include" /c tests\test_runner.cpp /Fo%RUNNER_OBJ%
if errorlevel 1 (
    echo ERROR: Failed to compile test_runner.cpp
    exit /b 1
)

echo   Linking: test_runner.exe
cl.exe %ALL_OBJS% %RUNNER_OBJ% /Fe:test_runner.exe /link /LIBPATH:"%CUDA_ROOT%\lib\x64" cudart.lib curand.lib
if errorlevel 1 (
    echo ERROR: Failed to link test_runner.exe
    exit /b 1
)

echo Built test_runner.exe successfully!
echo.

REM ========================================
REM Run appropriate tests
REM ========================================

if "%MODE%"=="easy" (
    echo ========================================
    echo Running EASY tests ^(Mate in 1-2^)...
    echo ========================================
    test_runner.exe --easy
    if errorlevel 1 (
        echo.
        echo ERROR: Easy tests failed
        exit /b 1
    )
    echo.
    goto :done
)

if "%MODE%"=="medium" (
    echo ========================================
    echo Running MEDIUM tests ^(Mate in 4-5^)...
    echo ========================================
    test_runner.exe --medium
    if errorlevel 1 (
        echo.
        echo ERROR: Medium tests failed
        exit /b 1
    )
    echo.
    goto :done
)

if "%MODE%"=="hard" (
    echo ========================================
    echo Running HARD tests ^(Mate in 8-12^)...
    echo ========================================
    test_runner.exe --hard
    if errorlevel 1 (
        echo.
        echo ERROR: Hard tests failed
        exit /b 1
    )
    echo.
    goto :done
)

if "%MODE%"=="runner" goto :show_runner_usage
goto :skip_runner

:show_runner_usage
echo test_runner.exe built successfully!
echo.
echo Usage:
echo   test_runner.exe --easy     Run easy tests (mate in 1-2)
echo   test_runner.exe --medium   Run medium tests (mate in 4-5)
echo   test_runner.exe --hard     Run hard tests (mate in 8-12)
echo   test_runner.exe --all      Run all tactical tests
echo   test_runner.exe --fen      Run FEN parser tests
echo   test_runner.exe --perft    Run perft tests
echo.
goto :done

:skip_runner

REM ========================================
REM Done
REM ========================================

:done
echo ========================================
echo All builds and tests completed successfully!
echo ========================================
goto :eof

REM ========================================
REM Help
REM ========================================

:help
echo Usage: build_and_test.bat [OPTIONS]
echo.
echo Options:
echo   --help, -h      Show this help message
echo   --clean, -c     Clean build artifacts
echo   --main          Build only main executable
echo   --puct          Build and run PUCT MCTS tests
echo   --bratko        Build and run Bratko-Kopec tests
echo   --easy          Build and run easy tactical tests (mate in 1-2)
echo   --medium        Build and run medium tactical tests (mate in 4-5)
echo   --hard          Build and run hard tactical tests (mate in 8-12)
echo   --runner        Build test_runner.exe (doesn't run tests)
echo   --tests         Build and run all tests (default)
echo   --all           Build everything (main + all tests)
echo.
echo Examples:
echo   build_and_test.bat              Run all tests (default)
echo   build_and_test.bat --main       Build main executable only
echo   build_and_test.bat --all        Build main and run all tests
echo   build_and_test.bat --clean      Clean build artifacts
echo   build_and_test.bat --puct       Run only PUCT tests
echo   build_and_test.bat --bratko     Run only Bratko-Kopec tests
echo   build_and_test.bat --easy       Run easy tactical tests
echo   build_and_test.bat --medium     Run medium tactical tests
echo   build_and_test.bat --hard       Run hard tactical tests
echo.
echo NOTE: Must be run from Visual Studio Developer Command Prompt
echo.
goto :eof
