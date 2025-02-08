#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cassert>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", \
                   cudaGetErrorString(error), \
                   __FILE__, __LINE__); \
            exit(1); \
        } \
    } while(0)

__global__ void debugExample(int* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Print debugging info
    printf("Thread [%d,%d] (global %d) starting...\n",
        blockIdx.x, threadIdx.x, idx);

    // Assert to catch out-of-bounds access
    assert(idx < size);

    // Do some work
    if (idx < size) {
        data[idx] *= 2;
        printf("Thread %d: processed %d -> %d\n",
            idx, data[idx] / 2, data[idx]);
    }
}

int ch1_5_main() {
    const int SIZE = 10;
    int* hostData, * deviceData;

    // Allocate host memory
    hostData = new int[SIZE];
    for (int i = 0; i < SIZE; i++) {
        hostData[i] = i;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&deviceData, SIZE * sizeof(int)));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(deviceData, hostData,
        SIZE * sizeof(int),
        cudaMemcpyHostToDevice));

    // Launch kernel with error checking
    debugExample << <2, 5 >> > (deviceData, SIZE);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());  // Check for runtime errors

    // Copy back results
    CUDA_CHECK(cudaMemcpy(hostData, deviceData,
        SIZE * sizeof(int),
        cudaMemcpyDeviceToHost));

    // Verify results
    printf("\nFinal results:\n");
    for (int i = 0; i < SIZE; i++) {
        printf("hostData[%d] = %d\n", i, hostData[i]);
    }

    // Cleanup
    delete[] hostData;
    CUDA_CHECK(cudaFree(deviceData));

    return 0;
}