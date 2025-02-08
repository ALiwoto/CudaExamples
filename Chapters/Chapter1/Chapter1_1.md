
# Chapter 1.1

The additional headers (`cuda_runtime.h` and `device_launch_parameters.h`) are good practice - they provide better IDE support and more explicit declarations.

```cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU!\n");
}

int main() {
    printf("Hello World from CPU!\n");
    
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
```

## Basic CUDA thread launching

```cuda
// Launch 1000 threads in blocks of 256
myKernel<<<4, 256>>>(args);  // 4 blocks × 256 threads = 1024 threads
```

Think of CUDA threads like this:

```cuda
Grid (all threads)
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   └── ... (up to blockDim)
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   └── ... (up to blockDim)
└── ... (more blocks)
```

## Qualifiers

In CUDA, there are three important function qualifiers:

01. `__global__` - Runs on GPU, called from CPU

    ```cuda
    __global__ void gpuFunc() { ... }
    // Called with: gpuFunc<<<blocks, threads>>>()
    ```

2. `__device__` - Runs on GPU, can only be called from GPU

    ```cuda
    __device__ void deviceFunc() { ... }
    // Can only be called from within a __global__ or another __device__ function
    ```

3. Regular functions (no qualifier) - Run on CPU

    ```cuda
    void cpuFunc() { ... }
    // Called normally: cpuFunc()
    ```

Here's an example showing all three:

```cuda
# include "cuda_runtime.h"
# include "device_launch_parameters.h"
# include <stdio.h>

// CPU function
void cpuFunction() {
    printf("Running on CPU\n");
}

// GPU device function
__device__ void deviceFunction() {
    printf("Running on GPU (called from another GPU function)\n");
}

// GPU kernel function
__global__ void gpuFunction() {
    printf("Running on GPU\n");
    deviceFunction(); // Can call device function from here
}

int main() {
    cpuFunction();     // Normal CPU call
    gpuFunction<<<1, 1>>>();  // GPU kernel launch
    cudaDeviceSynchronize();
    return 0;
}
```

The `<<<...>>>` syntax is called the "**kernel launch syntax**" and is required for `__global__` functions. It tells CUDA:

- How many blocks to create
- How many threads per block to use
- (optionally) How much shared memory to allocate
- (optionally) Which CUDA stream to use

Without this syntax, the GPU won't be involved at all!
