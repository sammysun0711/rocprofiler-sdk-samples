# ROCm profiling samples

This repository collects small, self-contained **AMD GPU** workloads used to learn **profiling** with **ROCm** rocprofiler-sdk (including **ROCm Compute Profiler**, **ROCm System Profiler**, **rocprofv3**, ATT thread traces, and PMC counters). Each sample lives in its own directory with a detailed `README.md`.

| Sample | Directory | What it demonstrates |
|--------|-----------|----------------------|
| **FP8 GEMM** | [`gemm/`](gemm/) | Naive vs MFMA-optimized FP8 GEMM; build scripts and `thread_tracing.sh` for rocprofv3 ATT traces. |
| **Matrix transpose** | [`matrix_transpose/`](matrix_transpose/) | Naive vs tiled shared-memory transpose; occupancy and memory behavior; rocprofv3 examples in the local README. |
| **Depthwise Conv3D** | [`depthwise_conv3d/`](depthwise_conv3d/) | HIP-only depthwise 3D conv (BF16): original vs `sched_group_barrier` kernel; `hipcc` host in `bench_conv3d.cpp`. |

---

## Prerequisites 

- **ROCm 7+** stack (HIP, `hipcc`, `rocprof` / **rocprofv3**) matched to your GPU (`gfx942`, `gfx950`, etc.).

## Rocprofling-sdk usage 

### ROCm Compute Profiler
```bash
pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
```
```bash
rocprof-compute profile -n mytest --no-roof -- target_app
rocprof-compute analyze -p workloads/mytest/MI308X
rocprof-compute analyze -p workloads/mytest/MI308X --gui
```
### ROCProfv3 Basic Performance Analysis
```bash
rocprofv3 --hip-trace --kernel-trace --output-format pftrace --summary -- target_app
```

### ROCProfv3 PMC Counter Analysis
```bash
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,GRBM_COUNT,SQ_WAVES,TCC_HIT_sum,TCC_MISS_sum,TCP_TOTAL_ACCESSES --output-format csv  -- target_app
```
### ROCProfv3 Advanced Thread Trace (ATT)
```bash
wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
bash ./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --skip-license --prefix=./
cp ./opt/rocm/lib/librocprof-trace-decoder.so /opt/rocm/lib/
```
```bash
rocprofv3 --att --kernel-include-regex kernel-name --att-activity 10 -d ./trace_results -- target_app
``` 

### ROCm Systems Profiler
```bash
rocprof-sys-instrument -- target_app
```
## Examples
### 1. FP8 GEMM (`gemm/`)

```bash
cd gemm
./build_run.sh
./thread_tracing.sh          # optional: rocprofv3 ATT traces
```

See [`gemm/README.md`](gemm/README.md)

### 2. Matrix transpose (`matrix_transpose/`)

```bash
cd matrix_transpose
# build (adjust gfx for your GPU)
hipcc -std=c++17 -O3 -g --offload-arch=gfx942 naive_transpose.cpp -o naive_transpose
hipcc -std=c++17 -O3 -g --offload-arch=gfx942 optimized_transpose.cpp -o optimized_transpose
./naive_transpose
./optimized_transpose
```

```bash
rocprofv3 --att --kernel-include-regex transposeNaive  --att-activity 10 -d ./unoptimized -- ./naive_transpose
rocprofv3 --att --kernel-include-regex transposeTiled --att-activity 10 -d ./optimized  -- ./optimized_transpose
```
See [`matrix_transpose/README.md`](matrix_transpose/README.md)

### 3. Depthwise Conv3D (`depthwise_conv3d/`)

```bash
cd depthwise_conv3d
export HIP_ARCH=gfx942   # match your GPU
./build.sh
./conv3d_depthwise_sgb 200 10
```

```bash
rocprofv3 --att --kernel-include-regex '.*conv_depthwise3d_hip.*' --att-activity 10 -d ./trace_sgb -- ./conv3d_depthwise_sgb 5 2
rocprofv3 --att --kernel-include-regex '.*conv_depthwise3d_hip.*' --att-activity 10 -d ./trace_orig -- ./conv3d_depthwise_original 5 2
```
See [`depthwise_conv3d/README.md`](depthwise_conv3d/README.md).

## References

- [ROCm documentation](https://rocm.docs.amd.com/)
- [rocprofiler / rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/)
- [HIP programming guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)

Credit: **FP8 GEMM** and **Matrix transpose** based on
- https://github.com/huanrwan-amd/GEMM_thread_tracing 
- https://github.com/huanrwan-amd/matrix_transpose_optimization
