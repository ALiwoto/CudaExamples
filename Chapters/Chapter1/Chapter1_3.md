
# Chapter 1.3

## Parallel Array Addition - a classic CUDA example

```cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void addArrays(
    int* inputArray1,
    int* inputArray2,
    int* resultArray,
    int arraySize
) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < arraySize) {
        resultArray[threadId] = inputArray1[threadId] + inputArray2[threadId];
        printf("Thread %d adding: %d + %d = %d\n",
            threadId,
            inputArray1[threadId],
            inputArray2[threadId],
            resultArray[threadId]);
    }
}

int main() {
    const int ARRAY_SIZE = 10;
    const int BYTES_NEEDED = ARRAY_SIZE * sizeof(int);

    // Host (CPU) arrays
    int hostInput1[ARRAY_SIZE];
    int hostInput2[ARRAY_SIZE];
    int hostResult[ARRAY_SIZE];

    // Device (GPU) arrays
    int* deviceInput1;
    int* deviceInput2;
    int* deviceResult;

    // Initialize host arrays
    for (int i = 0; i < ARRAY_SIZE; i++) {
        hostInput1[i] = i;      // [0,1,2,3,...]
        hostInput2[i] = 2 * i;  // [0,2,4,6,...]
    }

    // Allocate GPU memory
    cudaMalloc(&deviceInput1, BYTES_NEEDED);
    cudaMalloc(&deviceInput2, BYTES_NEEDED);
    cudaMalloc(&deviceResult, BYTES_NEEDED);

    // Copy data from CPU to GPU
    cudaMemcpy(deviceInput1, hostInput1, BYTES_NEEDED, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, BYTES_NEEDED, cudaMemcpyHostToDevice);

    // Configure parallel processing
    const int THREADS_PER_BLOCK = 5;
    const int NUMBER_OF_BLOCKS = 2;

    // Launch parallel processing
    addArrays<<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (
        deviceInput1,
        deviceInput2,
        deviceResult,
        ARRAY_SIZE
        );
    cudaDeviceSynchronize();

    // Copy results back from GPU to CPU
    cudaMemcpy(hostResult, deviceResult, BYTES_NEEDED, cudaMemcpyDeviceToHost);

    // Print results
    printf("\nFinal Results:\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%d + %d = %d\n",
            hostInput1[i],
            hostInput2[i],
            hostResult[i]);
    }

    // Clean up GPU memory
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceResult);

    return 0;
}
```
