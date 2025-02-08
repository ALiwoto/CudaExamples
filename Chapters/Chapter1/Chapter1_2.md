# Chapter 1.2

## Parallel execution

Let's try something more interesting that shows actual parallel execution. Here's a modified version:

```cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printThreadInfo() {
    printf("Block: %d, Thread: %d, Global Thread ID: %d\n", 
           blockIdx.x, 
           threadIdx.x,
           blockIdx.x * blockDim.x + threadIdx.x);
}

int main() {
    // Launch with 2 blocks, 4 threads each
    printThreadInfo<<<2, 4>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
```

This will show you:

- How blocks and threads are organized
- How global thread IDs are calculated
- Parallel execution in action

The output should show 8 lines (2 blocks × 4 threads) in some order (the order might not be sequential because of parallel execution).

```bash
Block: 0, Thread: 0, Global Thread ID: 0
Block: 0, Thread: 1, Global Thread ID: 1
Block: 0, Thread: 2, Global Thread ID: 2
Block: 0, Thread: 3, Global Thread ID: 3
Block: 1, Thread: 0, Global Thread ID: 4
Block: 1, Thread: 1, Global Thread ID: 5
Block: 1, Thread: 2, Global Thread ID: 6
Block: 1, Thread: 3, Global Thread ID: 7
```

## Output explanation

- You launched with <<<2, 4>>> meaning 2 blocks, 4 threads each
- Block 0 has threads 0-3
- Block 1 has threads 0-3
- Global Thread ID = (blockIdx.x * blockDim.x + threadIdx.x)
  - For example: Block 1, Thread 2 → (1 * 4 + 2) = 6

## What is cudaDeviceSynchronize()

- GPU operations are asynchronous
- Main function would end before GPU finishes without synchronization
- cudaDeviceSynchronize() makes CPU wait for GPU to complete

```cuda
// Without synchronization
kernel<<<2,4>>>(); 
return 0;  // Might return before GPU prints!

// With synchronization
kernel<<<2,4>>>();
cudaDeviceSynchronize();  // Wait for GPU
return 0;  // Only returns after GPU is done
```

## blockIdx and threadIdx

- They're not global variables
- They're built-in variables provided by CUDA hardware
- Each thread has its own private copy
- Hardware ensures each thread sees correct values
- No race conditions because each thread reads its own unique values

```bash
Block 0                  Block 1
[T0][T1][T2][T3]        [T0][T1][T2][T3]
 0   1   2   3           4   5   6   7   <- Global Thread IDs
```

Each thread executes the same code but with different blockIdx and threadIdx values, making it SPMD (Single Program, Multiple Data).
