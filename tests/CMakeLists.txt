# CMakeLists.txt for chess engine benchmarking framework
# Builds CPU and optionally GPU benchmarks

cmake_minimum_required(VERSION 3.18)
project(ChessEngineBenchmarks VERSION 1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_GPU_BENCHMARKS "Build GPU benchmarks (requires CUDA)" ON)
option(BUILD_CPU_BENCHMARKS "Build CPU benchmarks" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cpu/include)

# Find OpenMP (for CPU engine)
find_package(OpenMP REQUIRED)

# Common source files
set(COMMON_SOURCES
    src/cpu_engine_wrapper.cpp
)

# ============================================================================
# CPU-only benchmarks
# ============================================================================

if(BUILD_CPU_BENCHMARKS)
    message(STATUS "Building CPU benchmarks")
    
    # Link against CPU engine
    set(CPU_ENGINE_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../cpu/src/chess_engine_parallelized.cpp)
    
    # Throughput benchmark
    add_executable(benchmark_throughput
        src/benchmark_throughput.cpp
        ${COMMON_SOURCES}
        ${CPU_ENGINE_SOURCE}
    )
    target_link_libraries(benchmark_throughput PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(benchmark_throughput PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>)
    target_compile_definitions(benchmark_throughput PRIVATE UNIT_TESTS)
    
    # Fixed-time benchmark
    add_executable(benchmark_fixed_time
        src/benchmark_fixed_time.cpp
        ${COMMON_SOURCES}
        ${CPU_ENGINE_SOURCE}
    )
    target_link_libraries(benchmark_fixed_time PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(benchmark_fixed_time PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>)
    target_compile_definitions(benchmark_fixed_time PRIVATE UNIT_TESTS)
    
    # Stockfish comparison benchmark
    add_executable(benchmark_stockfish
        src/benchmark_stockfish.cpp
        ${COMMON_SOURCES}
        ${CPU_ENGINE_SOURCE}
    )
    target_link_libraries(benchmark_stockfish PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(benchmark_stockfish PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>)
    target_compile_definitions(benchmark_stockfish PRIVATE UNIT_TESTS)
    
    # Head-to-head matches
    add_executable(benchmark_matches
        src/benchmark_matches.cpp
        ${COMMON_SOURCES}
        ${CPU_ENGINE_SOURCE}
    )
    target_link_libraries(benchmark_matches PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(benchmark_matches PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/O2> $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>)
    target_compile_definitions(benchmark_matches PRIVATE UNIT_TESTS)
    
    message(STATUS "CPU benchmarks configured")
endif()

# ============================================================================
# GPU benchmarks (if CUDA available)
# ============================================================================

if(BUILD_GPU_BENCHMARKS)
    # Check for CUDA
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA found, building GPU benchmarks")
        
        # CUDA settings
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        
        # Include GPU headers and CUDA toolkit
        include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../gpu/include)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        
        # GPU engine sources
        set(GPU_ENGINE_SOURCES
            ../gpu/src/puct_mcts.cpp
            ../gpu/src/cpu_movegen.cpp
            ../gpu/src/init_tables.cu
            ../gpu/src/kernels/evaluation.cu
            ../gpu/src/kernels/kernel_launchers.cu
            ../gpu/src/kernels/movegen.cu
            ../gpu/src/kernels/playouts.cu
            ../gpu/src/kernels/tactical_search.cu
        )
        
        # Full benchmarks with GPU support
        add_executable(benchmark_throughput_gpu
            src/benchmark_throughput.cpp
            src/cpu_engine_wrapper.cpp
            src/gpu_engine_wrapper.cpp
            ${CPU_ENGINE_SOURCE}
            ${GPU_ENGINE_SOURCES}
        )
        target_link_libraries(benchmark_throughput_gpu PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(benchmark_throughput_gpu PRIVATE BUILD_GPU_ENGINE UNIT_TESTS)
        target_compile_options(benchmark_throughput_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/O2>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
        set_target_properties(benchmark_throughput_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "75;80;86;89"
        )
        
        add_executable(benchmark_fixed_time_gpu
            src/benchmark_fixed_time.cpp
            src/cpu_engine_wrapper.cpp
            src/gpu_engine_wrapper.cpp
            ${CPU_ENGINE_SOURCE}
            ${GPU_ENGINE_SOURCES}
        )
        target_link_libraries(benchmark_fixed_time_gpu PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(benchmark_fixed_time_gpu PRIVATE BUILD_GPU_ENGINE UNIT_TESTS)
        target_compile_options(benchmark_fixed_time_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/O2>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
        set_target_properties(benchmark_fixed_time_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "75;80;86;89"
        )
        
        add_executable(benchmark_stockfish_gpu
            src/benchmark_stockfish.cpp
            src/cpu_engine_wrapper.cpp
            src/gpu_engine_wrapper.cpp
            ${CPU_ENGINE_SOURCE}
            ${GPU_ENGINE_SOURCES}
        )
        target_link_libraries(benchmark_stockfish_gpu PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(benchmark_stockfish_gpu PRIVATE BUILD_GPU_ENGINE UNIT_TESTS)
        target_compile_options(benchmark_stockfish_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/O2>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
        set_target_properties(benchmark_stockfish_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "75;80;86;89"
        )
        
        add_executable(benchmark_matches_gpu
            src/benchmark_matches.cpp
            src/cpu_engine_wrapper.cpp
            src/gpu_engine_wrapper.cpp
            ${CPU_ENGINE_SOURCE}
            ${GPU_ENGINE_SOURCES}
        )
        target_link_libraries(benchmark_matches_gpu PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(benchmark_matches_gpu PRIVATE BUILD_GPU_ENGINE UNIT_TESTS)
        target_compile_options(benchmark_matches_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:$<$<CXX_COMPILER_ID:MSVC>:/O2>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O3 -march=native>>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
        set_target_properties(benchmark_matches_gpu PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "75;80;86;89"
        )
        
        message(STATUS "GPU benchmarks configured")
    else()
        message(WARNING "CUDA not found, GPU benchmarks disabled")
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS
    benchmark_throughput
    benchmark_fixed_time
    benchmark_stockfish
    benchmark_matches
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "")
message(STATUS "===============================================")
message(STATUS "Chess Engine Benchmark Configuration Summary")
message(STATUS "===============================================")
message(STATUS "  C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  CPU Benchmarks:    ${BUILD_CPU_BENCHMARKS}")
message(STATUS "  GPU Benchmarks:    ${BUILD_GPU_BENCHMARKS}")
if(CMAKE_CUDA_COMPILER)
    message(STATUS "  CUDA Compiler:     ${CMAKE_CUDA_COMPILER}")
    message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "===============================================")
message(STATUS "")
