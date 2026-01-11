# Makefile for PUCT MCTS Chess Engine (Heuristic AlphaZero-style)
# NO neural networks - pure heuristic search with AlphaZero concepts

NVCC = nvcc
CXX = g++
CUDA_ARCH = sm_75
NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH) -Iinclude
CXX_FLAGS = -O3 -std=c++17 -pthread -Iinclude

ifeq ($(OS),Windows_NT)
    CUDA_PATH ?= C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6
    CUDA_INC = -I"$(CUDA_PATH)/include"
    CUDA_LIB = -L"$(CUDA_PATH)/lib/x64"
    EXE_SUFFIX = .exe
    RM = del /Q
else
    CUDA_PATH ?= /usr/local/cuda
    CUDA_INC = -I$(CUDA_PATH)/include
    CUDA_LIB = -L$(CUDA_PATH)/lib64
    EXE_SUFFIX =
    RM = rm -f
endif

CUDA_LIBS = -lcudart -lcurand

# PUCT MCTS Engine (Heuristic-based)
TARGET = puct_chess$(EXE_SUFFIX)
OBJS = gpu_kernels.o init_tables.o mcts.o puct_mcts.o main.o

# Default: build PUCT engine
all: $(TARGET)

# PUCT MCTS Engine
$(TARGET): $(OBJS)
	@echo "Linking PUCT Chess Engine..."
	$(CXX) $(CXX_FLAGS) $(OBJS) -o $(TARGET) $(CUDA_LIB) $(CUDA_LIBS)
	@echo "PUCT Engine built successfully!"

gpu_kernels.o: src/gpu_kernels.cu include/chess_types.cuh include/move_gen.cuh
	@echo "Compiling GPU kernels..."
	$(NVCC) $(NVCC_FLAGS) -c src/gpu_kernels.cu -o gpu_kernels.o

init_tables.o: src/init_tables.cu include/chess_types.cuh
	@echo "Compiling attack table initialization..."
	$(NVCC) $(NVCC_FLAGS) -c src/init_tables.cu -o init_tables.o

mcts.o: src/mcts.cpp include/mcts.h include/chess_types.cuh include/search_config.h
	@echo "Compiling original MCTS..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c src/mcts.cpp -o mcts.o

puct_mcts.o: src/puct_mcts.cpp include/puct_mcts.h include/chess_types.cuh include/search_config.h
	@echo "Compiling PUCT MCTS (heuristic AlphaZero)..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c src/puct_mcts.cpp -o puct_mcts.o

main.o: src/main.cpp include/mcts.h include/puct_mcts.h include/chess_types.cuh
	@echo "Compiling main..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c src/main.cpp -o main.o

# Test Suite
TEST_TARGET = test_puct_mcts$(EXE_SUFFIX)
TEST_OBJS = test_puct_mcts.o gpu_kernels.o init_tables.o mcts.o puct_mcts.o

test: $(TEST_TARGET)
	@echo "Running PUCT MCTS tests..."
	./$(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJS)
	@echo "Linking test suite..."
	$(CXX) $(CXX_FLAGS) $(TEST_OBJS) -o $(TEST_TARGET) $(CUDA_LIB) $(CUDA_LIBS)
	@echo "Test suite built successfully!"

test_puct_mcts.o: tests/test_puct_mcts.cpp include/puct_mcts.h include/mcts.h tests/test_positions.h
	@echo "Compiling PUCT tests..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c tests/test_puct_mcts.cpp -o test_puct_mcts.o

# Clean build artifacts
clean:
	@echo "Cleaning..."
ifeq ($(OS),Windows_NT)
	-$(RM) *.o $(TARGET) $(TEST_TARGET) 2>nul
else
	$(RM) *.o $(TARGET) $(TEST_TARGET)
endif
	@echo "Clean complete!"

# Run engine
run: $(TARGET)
	./$(TARGET)

# Rebuild from scratch
rebuild: clean all

# Build and run tests
test-build: $(TEST_TARGET)

test-run: $(TEST_TARGET)
	./$(TEST_TARGET)

# Build everything (engine + tests)
all-with-tests: $(TARGET) $(TEST_TARGET)

help:
	@echo "PUCT MCTS Chess Engine - Build Targets:"
	@echo "  make             - Build main engine"
	@echo "  make test        - Build and run test suite"
	@echo "  make test-build  - Build test suite only"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make rebuild     - Clean and rebuild"
	@echo "  make run         - Run main engine"
ifeq ($(OS),Windows_NT)
	-$(RM) *.o $(TARGET) 2>nul
else
	$(RM) *.o $(TARGET)
endif
	@echo "Clean complete!"

# Run engine
run: $(TARGET)
	./$(TARGET)

# Rebuild from scratch
rebuild: clean all

# Test engine
test: $(TARGET)
	@echo "Running PUCT MCTS test..."
	./$(TARGET) --test