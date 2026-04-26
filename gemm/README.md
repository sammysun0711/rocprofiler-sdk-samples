# FP8 GEMM Optimization with AMD MFMA

A comprehensive demonstration of thread tracing on GPU matrix multiplication optimization using AMD's FP8 MFMA (Matrix Fused Multiply-Add) instructions on MI300 series GPUs.

## Overview

This project demonstrates the dramatic performance improvements achievable and thread tracing through GPU kernel optimization. It implements two versions of FP8 (8-bit floating point) General Matrix Multiply (GEMM) operations:

1. **Naive Implementation** - Straightforward, unoptimized approach
2. **Optimized Implementation** - Highly optimized using AMD MFMA instructions

The optimized version achieves **20-100x speedup** over the naive implementation by leveraging:
- Hardware matrix multiplication instructions (MFMA)
- Shared memory tiling for data reuse
- Dual-wave compute-memory overlap
- Optimized thread and memory access patterns

### Operation

Both implementations compute: **C = A × B** where:
- Matrices A and B are in FP8 format (8-bit floating point)
- Output matrix C is in BF16 format (Brain Float 16)
- Per-block scaling factors are applied (quantized arithmetic)
- Formula: `C[i,j] = Σ(A[i,k] × B[k,j] × scale_a[i,k/128] × scale_b[k/128,j/128])`

---

## File Structure

### Source Files

| File | Description |
|------|-------------|
| `naive_gemm.cpp` | Unoptimized FP8 GEMM implementation. Each thread computes one output element using simple loops. No hardware acceleration or memory optimizations. |
| `optimized_gemm.cpp` | Highly optimized FP8 GEMM using MFMA instructions, shared memory tiling, and dual-wave execution. Fixed WIDTH=1, HEIGHT=4 configuration. |

### Build and Profiling Scripts

| File | Description |
|------|-------------|
| `build_run.sh` | Automated build and benchmark script. Compiles both naive and optimized versions in release and debug modes, then runs performance benchmarks. |
| `thread_tracing.sh` | GPU profiling script using rocprofv3 with ATT (Async Trace Tool). Generates detailed thread-level execution traces for performance analysis. |

### Generated Files

| File/Directory | Description |
|----------------|-------------|
| `naive_gemm_release` | Release build of naive implementation (optimized compilation, -O3) |
| `naive_gemm_debug` | Debug build of naive implementation (with symbols, -g) |
| `optimized_gemm_release` | Release build of optimized implementation |
| `optimized_gemm_debug` | Debug build of optimized implementation |
| `naive_gemm_trace/` | Profiling results for naive implementation (ATT traces, JSON metadata, CSV stats) |
| `optimized_gemm_trace/` | Profiling results for optimized implementation |

---

## Key Differences: Naive vs Optimized

### Naive Implementation

**Architecture:**
```
- Thread block: 16×16 (256 threads)
- Each thread: Computes 1 output element
- Memory access: Direct global memory reads
- Computation: Simple FP8→float conversion + accumulation loop
```

**Characteristics:**
- ❌ No hardware MFMA instructions
- ❌ No shared memory usage
- ❌ No data tiling or reuse
- ❌ Poor memory coalescing
- ❌ Each thread reads O(K) elements from global memory
- ❌ No compute-memory overlap

**Kernel Structure:**
```cpp
__global__ void naive_fp8_gemm_kernel(...) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // Direct global memory access every iteration
        float a_val = static_cast<float>(a[row * K + k]);
        float b_val = static_cast<float>(b[k * N + col]);
        sum += a_val * b_val * scale_a[...] * scale_b[...];
    }
    c[row * N + col] = sum;
}
```

### Optimized Implementation

**Architecture:**
```
- Thread block: 16×32 (512 threads, 2 waves)
- Output tile: 64×64 (WIDTH=1, HEIGHT=4)
- Shared memory: ~12KB per block
- Wave 0: Computation using MFMA
- Wave 1: Data loading from global memory
```

**Characteristics:**
- ✅ AMD MFMA instructions (16×16×32 FP8 matrix multiply)
- ✅ Shared memory tiling (64×64 output, 64×64 input tiles)
- ✅ Data reuse through shared memory
- ✅ Optimized memory coalescing
- ✅ Dual-wave execution (compute-memory overlap)
- ✅ Complex thread mapping for MFMA layout
- ✅ Wavefront-level parallelism

**Kernel Structure:**
```cpp
__global__ void fp8_gemm_kernel(...) {
    // Shared memory for tiled data
    __shared__ uint8_t a_shared[WIDTH][4][4][4][TBLOCK][4];
    __shared__ uint8_t b_shared[4][HEIGHT][4][TBLOCK][4];
    __shared__ float   scalar[4][TBLOCK][WIDTH];
    
    if (TW != 0) {
        // Wave 1: Load data into shared memory
        // ... cooperative loading ...
        return;
    }
    
    // Wave 0: Compute using MFMA
    Vec16 reg_res[WIDTH] = {};
    
    for (k-tiles) {
        __syncthreads();
        // Load from shared memory
        float8x8 a_load, b_load;
        // ...
        
        // MFMA: 16×16×32 matrix multiply in single instruction
        Vec4 res = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(...);
        
        // Accumulate with scaling
        reg_res[...] += scale * res;
    }
}
```

### Performance Comparison

| Metric | Naive | Optimized | Improvement |
|--------|-------|-----------|-------------|
| **Time (512³)** | 0.342 ms | 0.011 ms | **31x faster** |
| **Throughput (512³)** | 0.784 TFLOPS | 24.826 TFLOPS | **31.7x** |
| **Bandwidth (512³)** | 3.1 GB/s | 97.0 GB/s | **31.3x** |
| **Global Memory Reads** | O(M×N×K) per kernel | O(M×K + K×N) per kernel | **~K/64x less** |
| **Thread Efficiency** | Low (scalar) | High (vectorized) | **16-32x** |

### Memory Access Patterns

**Naive:**
```
Thread(i,j) reads: A[i,0], A[i,1], ..., A[i,K-1]  (K reads)
                   B[0,j], B[1,j], ..., B[K-1,j]  (K reads)
Total per block: ~256×2K global memory transactions
```

**Optimized:**
```
Block loads: 64×64 tile of A, 64×64 tile of B into shared memory
Each thread: Reuses shared memory data for MFMA operations
Total per block: ~2×64×64/4 = 2048 global memory transactions
Reuse factor: ~32x (64 elements reused across 64 output computations)
```

---

## Build and Usage

### Prerequisites

- AMD ROCm 5.0 or later
- hipcc compiler
- AMD MI300, MI250, or MI200 series GPU (gfx90a, gfx942)
- rocprofv3 (for profiling)

### Building

#### Automatic Build (Recommended)

```bash
# Build all versions and run benchmarks with default size (1024×1536×7168)
./build_run.sh

# Build and run with custom matrix dimensions
./build_run.sh <M> <N> <K>
./build_run.sh 2048 2048 2048
```

The script will:
1. Auto-detect GPU architecture (gfx942, gfx90a, etc.)
2. Build 4 binaries: naive and optimized, each in release and debug
3. Run both release versions with specified dimensions
4. Display performance comparison

#### Manual Build

```bash
# Naive GEMM (Release)
hipcc --offload-arch=gfx942 -std=c++17 -O3 -DNDEBUG -ffast-math \
      naive_gemm.cpp -o naive_gemm_release

# Optimized GEMM (Release)
hipcc --offload-arch=gfx942 -std=c++17 -O3 -DNDEBUG -ffast-math \
      optimized_gemm.cpp -o optimized_gemm_release

# Debug builds (add -g -O0)
hipcc --offload-arch=gfx942 -std=c++17 -g -O0 -DDEBUG \
      naive_gemm.cpp -o naive_gemm_debug
```

### Running Benchmarks

```bash
# Run individual benchmarks
./naive_gemm_release 1024 1536 7168
./optimized_gemm_release 1024 1536 7168

# Run debug versions
./naive_gemm_debug 512 512 512
./optimized_gemm_debug 512 512 512
```

### GPU Architecture

To override auto-detection:
```bash
export GPU_ARCH=gfx942   # For MI300 series
./build_run.sh
```

---

## Profiling and Tracing

### Thread-Level Profiling

```bash
# Profile both implementations with default size
./thread_tracing.sh

# Profile with custom dimensions
./thread_tracing.sh 2048 2048 2048
./thread_tracing.sh <M> <N> <K>
```


---

## References

- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [AMD MFMA Instructions](https://rocm.docs.amd.com/en/latest/understand/gpu_arch/mi300.html)
- [rocprof Profiler](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)

---

## License

This demonstration code is provided for educational purposes.

## Author

Created for demonstrating GPU kernel optimization techniques on AMD MI300 series GPUs.
