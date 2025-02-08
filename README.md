# CudaExamples

A repository containing multiple examples of Cuda programming language for beginners.

At the time of writing this repository, I am also a beginner, so don't expect too much from it.

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
