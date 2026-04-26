# Matrix Transpose Occupancy Optimization

This demo shows how GPU kernel optimization (tiling + shared memory + improved memory access patterns + launch bounds) increases effective memory bandwidth and occupancy for a matrix transpose on an AMD accelerator.

## Contents

- `naive_transpose.cpp` – Baseline (unoptimized) transpose
- `optimized_transpose.cpp` – Tiled shared-memory transpose

## Key Differences (Naive vs Optimized)

| Aspect | Naive | Optimized |
|--------|-------|-----------|
| Global reads | Coalesced | Coalesced |
| Global writes | Strided (non-coalesced) | Coalesced (after tile transpose) |
| Shared memory | None | Tiled (32 x 32 + padding) |
| Bank conflict avoidance | No | Yes (+1 column padding) |

## Build

```bash
hipcc -std=c++17 -O3 -g --offload-arch=gfx942 naive_transpose.cpp -o naive_transpose
hipcc -std=c++17 -O3 -g --offload-arch=gfx942 optimized_transpose.cpp -o optimized_transpose
```
## Thread tracing with rocprofv3

```bash
rocprofv3 --att --kernel-include-regex transposeNaive --att-activity 10 -d ./unoptimized -- ./naive_transpose 
rocprofv3 --att --kernel-include-regex transposeTiled --att-activity 10 -d ./optimized -- ./optimized_transpose 
```

## Example output

```bash
# ./naive_transpose 
Device:  (CUs=80)
Kernel: transposeNaive, regsPerThread=6, localSizeBytes=0, sharedSizeBytes=0
Block 16x16 (256 threads); grid 512x512; Max active blocks/CU (runtime est): 8
Avg time: 1.219 ms; Effective bandwidth: 440.24 GB/s
Validation: PASS

# ./optimized_transpose 
Device:  (CUs=80)
Kernel: transposeTiled, regsPerThread=8, localSizeBytes=0, sharedSizeBytes=4224
Block 32x8 (256 threads); grid 256x256; Max active blocks/CU (runtime est): 7
Avg time: 0.421 ms; Effective bandwidth: 1275.66 GB/s
Validation: PASS

```

