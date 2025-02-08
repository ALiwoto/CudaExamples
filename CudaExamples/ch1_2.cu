#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printThreadInfo() {
    printf("Block: %d, Thread: %d, Global Thread ID: %d\n",
        blockIdx.x,
        threadIdx.x,
        blockIdx.x * blockDim.x + threadIdx.x);
}

int ch1_2_main() {
    // Launch with 2 blocks, 4 threads each
    printThreadInfo << <2, 4 >> > ();
    cudaDeviceSynchronize();

    return 0;
}