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

int ch1_1_main() {
    cpuFunction();     // Normal CPU call
    gpuFunction << <1, 1 >> > ();  // GPU kernel launch
    cudaDeviceSynchronize();
    return 0;
}