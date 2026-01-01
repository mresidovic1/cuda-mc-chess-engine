#include "monte_carlo_advanced_kernel.cuh"
#include <cuda_runtime.h>

extern "C" void launch_monte_carlo_simulate_kernel(
    const Position* root_position,
    const Move* root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed,
    int blocks,
    int threads_per_block
) {
    monte_carlo_simulate_kernel<<<blocks, threads_per_block>>>(
        *root_position,
        *root_move,
        num_simulations_per_thread,
        results,
        seed
    );
    cudaDeviceSynchronize();
}
