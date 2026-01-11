# TESTS BUILD AND RUN

```bash
# checkout to branch
!git checkout monte-carlo-v4-alpha-zero

# build
!nvcc -std=c++17 -arch=sm_75 -O3 -Iinclude \
  tests/test_puct_mcts.cpp src/gpu_kernels.cu src/init_tables.cu \
  src/mcts.cpp src/puct_mcts.cpp -o test_puct_mcts -lcudart -lcurand

# run
!./test_puct_mcts
```