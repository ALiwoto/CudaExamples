
# Chapter 1.3

## Parallel Array Addition - a classic CUDA example

```cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Kernel function to add two arrays
__global__ void addArrays(int* a, int* b, int* c, int n) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't access beyond array bounds
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        printf("Thread %d adding: %d + %d = %d\n", 
               idx, a[idx], b[idx], c[idx]);
    }
}

int main() {
    const int N = 10;  // Array size
    const int bytes = N * sizeof(int);
    
    // Host arrays (CPU)
    int ha[N], hb[N], hc[N];
    
    // Device arrays (GPU)
    int *da, *db, *dc;
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        ha[i] = i;      // First array: 0,1,2,3,...
        hb[i] = 2 * i;  // Second array: 0,2,4,6,...
    }
    
    // Allocate GPU memory
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);
    
    // Copy from host to device
    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel with 2 blocks, 5 threads each
    addArrays<<<2, 5>>>(da, db, dc, N);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);
    
    // Print final results
    printf("\nFinal Results:\n");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", ha[i], hb[i], hc[i]);
    }
    
    // Free GPU memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    
    return 0;
}
```
