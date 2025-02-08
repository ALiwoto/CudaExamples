# Chapter 1.4

## Shared memory and thread cooperation - a parallel reduction example

```cuda
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

// Maximum threads per block for this example
const int THREADS_PER_BLOCK = 256;

__global__ void sumArray(
    int* inputArray,
    int* blockSums,
    int arraySize
) {
    // Shared memory for this thread block
    __shared__ int sharedData[THREADS_PER_BLOCK];
    
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int localThreadId = threadIdx.x;
    
    // Load data into shared memory
    if (threadId < arraySize) {
        sharedData[localThreadId] = inputArray[threadId];
    } else {
        sharedData[localThreadId] = 0;
    }
    
    // Wait for all threads to load their data
    __syncthreads();
    
    // Perform parallel reduction in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (localThreadId < stride) {
            sharedData[localThreadId] += sharedData[localThreadId + stride];
        }
        __syncthreads();
    }
    
    // First thread in block writes result
    if (localThreadId == 0) {
        blockSums[blockIdx.x] = sharedData[0];
        printf("Block %d sum: %d\n", blockIdx.x, sharedData[0]);
    }
}

int main() {
    const int ARRAY_SIZE = 1000;
    const int BYTES_NEEDED = ARRAY_SIZE * sizeof(int);
    
    // Host arrays
    int* hostInput = new int[ARRAY_SIZE];
    
    // Initialize input array with numbers 1 to ARRAY_SIZE
    for (int i = 0; i < ARRAY_SIZE; i++) {
        hostInput[i] = 1;  // All ones for easy verification
    }
    
    // Calculate grid dimensions
    const int NUM_BLOCKS = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int* hostBlockSums = new int[NUM_BLOCKS];
    
    // Device arrays
    int *deviceInput, *deviceBlockSums;
    
    // Allocate GPU memory
    cudaMalloc(&deviceInput, BYTES_NEEDED);
    cudaMalloc(&deviceBlockSums, NUM_BLOCKS * sizeof(int));
    
    // Copy data to GPU
    cudaMemcpy(deviceInput, hostInput, BYTES_NEEDED, cudaMemcpyHostToDevice);
    
    // Launch kernel
    sumArray<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
        deviceInput,
        deviceBlockSums,
        ARRAY_SIZE
    );
    
    // Get results back
    cudaMemcpy(hostBlockSums, deviceBlockSums, NUM_BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Sum up block sums on CPU
    int totalSum = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
        totalSum += hostBlockSums[i];
    }
    
    printf("\nTotal sum: %d\n", totalSum);
    printf("Expected sum: %d\n", ARRAY_SIZE);  // Should match since all inputs are 1
    
    // Cleanup
    delete[] hostInput;
    delete[] hostBlockSums;
    cudaFree(deviceInput);
    cudaFree(deviceBlockSums);
    
    return 0;
}
```

This example introduces several new concepts:

1. `__shared__` memory - Fast memory shared between threads in a block
2. `__syncthreads()` - Synchronization barrier for threads in a block
3. Parallel reduction algorithm - A common parallel processing pattern
4. Block-level cooperation between threads
5. Dynamic calculation of grid dimensions

Key points to note:

- Shared memory is much faster than global memory
- Threads in a block can cooperate using shared memory
- `__syncthreads()` ensures all threads reach a point before continuing
- The reduction pattern halves the working set each step

This is a more advanced example showing how threads can work together efficiently.
