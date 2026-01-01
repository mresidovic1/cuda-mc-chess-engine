# Makefile for Advanced Monte Carlo Chess GPU

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
TARGET = monte_carlo_advanced$(EXE_SUFFIX)
OBJS = monte_carlo_advanced_kernel.o monte_carlo_advanced.o main_advanced.o

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	$(CXX) $(CXX_FLAGS) $(OBJS) -o $(TARGET) $(CUDA_LIB) $(CUDA_LIBS)
	@echo "Build OK!"

monte_carlo_advanced_kernel.o: monte_carlo_advanced_kernel.cu
	@echo "Compiling CUDA kernel..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

monte_carlo_advanced.o: monte_carlo_advanced.cpp monte_carlo_advanced.hpp
	@echo "Compiling C++ wrapper..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c $< -o $@

main_advanced.o: main_advanced.cpp monte_carlo_advanced.hpp
	@echo "Compiling main..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c $< -o $@

clean:
	$(RM) $(OBJS) $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

# Compile CUDA kernel
monte_carlo_kernel.o: monte_carlo_kernel.cu monte_carlo_gpu.hpp
	@echo "Compiling CUDA kernel..."
	$(NVCC) $(NVCC_FLAGS) -c monte_carlo_kernel.cu -o monte_carlo_kernel.o

# Compile C++ wrapper
monte_carlo_gpu.o: monte_carlo_gpu.cpp monte_carlo_gpu.hpp
	@echo "Compiling C++ wrapper..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c monte_carlo_gpu.cpp -o monte_carlo_gpu.o

# Compile main program
main_gpu.o: main_gpu.cpp monte_carlo_gpu.hpp
	@echo "Compiling main program..."
	$(CXX) $(CXX_FLAGS) $(CUDA_INC) -c main_gpu.cpp -o main_gpu.o

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
ifeq ($(OS),Windows_NT)
	-$(RM) *.o $(TARGET) 2>nul
else
	$(RM) *.o $(TARGET)
endif
	@echo "Clean complete!"

# Run with default parameters
run: $(TARGET)
	@echo "Running Monte Carlo Chess GPU..."
	./$(TARGET) 10000

# Run with custom simulation count
run-benchmark: $(TARGET)
	@echo "Running benchmark with 100,000 simulations..."
	./$(TARGET) 100000

# Rebuild from scratch
rebuild: clean all

# Help target
help:
	@echo "Monte Carlo Chess GPU - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  all (default) - Build the executable"
	@echo "  clean         - Remove build artifacts"
	@echo "  run           - Build and run with 10,000 simulations"
	@echo "  run-benchmark - Build and run with 100,000 simulations"
	@echo "  rebuild       - Clean and rebuild from scratch"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH     - CUDA compute capability (default: sm_70)"
	@echo "  CUDA_PATH     - Path to CUDA toolkit"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Build with default settings"
	@echo "  make CUDA_ARCH=sm_80          # Build for RTX 30xx series"
	@echo "  make run                      # Build and run"
	@echo "  make clean                    # Clean build files"

.PHONY: all clean run run-benchmark rebuild help
