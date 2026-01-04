# Makefile for Advanced Monte Carlo Chess GPU
# Supports both original and v1 (improved) versions

NVCC = nvcc
CXX = g++
CUDA_ARCH = sm_75
NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH)
CXX_FLAGS = -O3 -std=c++17 -I..

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

# Original version
TARGET = monte_carlo_advanced$(EXE_SUFFIX)
OBJS = monte_carlo_advanced_kernel.o monte_carlo_advanced.o main_advanced.o

# N - v1: Improved version targets
TARGET_V1 = monte_carlo_v1$(EXE_SUFFIX)
OBJS_V1 = monte_carlo_advanced_kernel_v1.o monte_carlo_advanced_v1.o main_advanced_v1.o

# Default: build original version
all: $(TARGET)

# Build both versions
both: $(TARGET) $(TARGET_V1)

# Original version
$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXX_FLAGS) $(OBJS) -o $(TARGET) $(CUDA_LIB) $(CUDA_LIBS)
	@echo "Build OK!"

monte_carlo_advanced_kernel.o: monte_carlo_advanced_kernel.cu monte_carlo_advanced_kernel.cuh
	@echo "Compiling CUDA kernel (original)..."
	$(NVCC) $(NVCC_FLAGS) -c monte_carlo_advanced_kernel.cu -o monte_carlo_advanced_kernel.o

monte_carlo_advanced.o: monte_carlo_advanced.cpp monte_carlo_advanced.hpp
	@echo "Compiling C++ wrapper (original)..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c monte_carlo_advanced.cpp -o monte_carlo_advanced.o

main_advanced.o: main_advanced.cpp monte_carlo_advanced.hpp
	@echo "Compiling main (original)..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c main_advanced.cpp -o main_advanced.o

# N - v1: Improved version
v1: $(TARGET_V1)

$(TARGET_V1): $(OBJS_V1)
	@echo "Linking $(TARGET_V1)..."
	$(CXX) $(CXX_FLAGS) $(OBJS_V1) -o $(TARGET_V1) $(CUDA_LIB) $(CUDA_LIBS)
	@echo "Build v1 OK!"

monte_carlo_advanced_kernel_v1.o: monte_carlo_advanced_kernel_v1.cu monte_carlo_advanced_kernel_v1.cuh
	@echo "Compiling CUDA kernel (v1)..."
	$(NVCC) $(NVCC_FLAGS) -c monte_carlo_advanced_kernel_v1.cu -o monte_carlo_advanced_kernel_v1.o

monte_carlo_advanced_v1.o: monte_carlo_advanced_v1.cpp monte_carlo_advanced_v1.hpp
	@echo "Compiling C++ wrapper (v1)..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c monte_carlo_advanced_v1.cpp -o monte_carlo_advanced_v1.o

main_advanced_v1.o: main_advanced_v1.cpp monte_carlo_advanced_v1.hpp
	@echo "Compiling main (v1)..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c main_advanced_v1.cpp -o main_advanced_v1.o

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
ifeq ($(OS),Windows_NT)
	-$(RM) *.o $(TARGET) $(TARGET_V1) 2>nul
else
	$(RM) *.o $(TARGET) $(TARGET_V1)
endif
	@echo "Clean complete!"

# Run versions
run: $(TARGET)
	./$(TARGET) 10000

run-v1: $(TARGET_V1)
	./$(TARGET_V1) --sims 10000

benchmark: $(TARGET_V1)
	./$(TARGET_V1) --benchmark --sims 25000

rebuild: clean all

help:
	@echo "Targets: all, v1, both, run, run-v1, benchmark, clean, help"

.PHONY: all both v1 clean run run-v1 benchmark rebuild help
