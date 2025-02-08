
# Chapter 1.5

## CPU and GPU debugging

One of the limitations when debugging CUDA code is that you cannot put normal breakpoints in functions which are going to get executed on GPU.

However, there are several ways to debug GPU code:

1. **printf debugging** (Basic but effective):

    ```cuda
    __global__ void myKernel() {
        printf("Thread %d: value = %d\n", threadIdx.x, someValue);
    }
    ```

2. **CUDA error checking** (Recommended practice):

    ```cuda
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

    // Usage:
    CUDA_CHECK(cudaMalloc(&deviceArray, size));
    CUDA_CHECK(cudaMemcpy(deviceArray, hostArray, size, cudaMemcpyHostToDevice));
    ```

3. **Nsight Compute** (NVIDIA's profiler and debugger):

    - Can set conditional breakpoints in kernel code
    - Can inspect memory
    - Can profile performance

4. **Device-side assertions**:

    ```cuda
    __global__ void myKernel() {
        assert(threadIdx.x < 256);  // Will break if condition is false
    }

    ```

5. **Memory checking** with cudaMemcheck:
bash
compute-sanitizer ./your_program

-------------------------------

Here's a practical example combining several debugging techniques:

```cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

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
               idx, data[idx]/2, data[idx]);
    }
}

int main() {
    const int SIZE = 10;
    int *hostData, *deviceData;
    
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
    debugExample<<<2, 5>>>(deviceData, SIZE);
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
```

While not as convenient as CPU debugging, these methods can help track down issues in your CUDA code.
