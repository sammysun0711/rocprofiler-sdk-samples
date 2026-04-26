# Depthwise Conv3D (HIP)

This demo compares two BF16 depthwise 3D convolution implementations for rocprofling samples 

## Contents

- `bench_conv3d.cpp` – HIP host: allocations, `hipLaunchKernelGGL`, timing, optional CPU reference check  
- `build.sh` – Builds both binaries with `HIP_ARCH` (default `gfx942`)  
- `kernels/conv_depthwise3d_hip.cpp` – “Original” kernel: batched `ds_read` then MACs  
- `kernels/conv_depthwise3d_hip_sgb.cpp` – **SGB** variant: row-wise read/MAC interleave + `sched_group_barrier` hints  
- `trace.yaml` – Example **rocprofv3** job (ATT + PMC); adjust paths and GPU as needed  

| Binary (from `./build.sh`) | Kernel source |
|----------------------------|-----------------|
| `conv3d_depthwise_original` | `kernels/conv_depthwise3d_hip.cpp` |
| `conv3d_depthwise_sgb` | `kernels/conv_depthwise3d_hip_sgb.cpp` |

Both device translation units export the same HIP entry name: **`conv_depthwise3d_hip`** (filter rocprof with a regex on that substring).

## Key differences (original vs SGB)

| Aspect | Original (`conv_depthwise3d_hip.cpp`) | SGB (`conv_depthwise3d_hip_sgb.cpp`) |
|--------|----------------------------------------|--------------------------------------|
| Read / MAC scheduling | Batch many `ds_read`, then many `v_fmac` | One filter row: few `ds_read`, then MACs for that row |
| Scheduler hints | Default LLVM ordering | `__builtin_amdgcn_sched_group_barrier` to overlap `ds_read` latency with VALU |
| Typical effect | Fewer live values in one pattern | Lower VGPR pressure in that pattern; may improve occupancy on some GPU arch |

Problem size: 
- input **NCHW** `[1, 512, 61, 45, 80]` BF16, 
- weights `[512, 1, 3, 5, 5]`, 
- padding `(0, 2, 2)`, 
- stride/dilation 1
- output spatial **45×80**, depth **59**. 
- Launch: grid `(B, C_out, D_out)`, block **256** threads, `__launch_bounds__(256, 1)`.

## Build

```bash
export HIP_ARCH=gfx942   # or gfx950, etc.
./build.sh
```

Equivalent manual compile (from this directory):

```bash
hipcc -std=c++17 -O3 -g --offload-arch=gfx942 -DKERNEL_SGB=0 bench_conv3d.cpp -o conv3d_depthwise_original
hipcc -std=c++17 -O3 -g --offload-arch=gfx942 -DKERNEL_SGB=1 bench_conv3d.cpp -o conv3d_depthwise_sgb
```

## Run

```bash
./conv3d_depthwise_sgb                 # defaults: niter=10, warmup=10
./conv3d_depthwise_sgb 200 10          # 200 timed launches, 10 warmup
./conv3d_depthwise_original --no-check # timing only, skip CPU reference
```

Arguments: **`[niter] [nwarmup] [--no-check]`**. `niter` = number of timed kernel launches (default **10**). `--no-check` skips the CPU float32 reference (useful under heavy profilers).

The binary prints device properties, tensor sizes, buffer sizes, grid/block, warmup/benchmark banners, TFLOPS and a rough HBM bandwidth line, then correctness (unless `--no-check`).

## Correctness

After timing, the host optionally runs a **CPU float32** depthwise conv3d with the same indexing and **zero padding outside the tensor** (aligned with `torch.nn.functional.conv3d`). GPU BF16 outputs are compared with **atol = rtol = 0.02**. Failure prints the first mismatch and exits **1**.

## Thread tracing with rocprofv3

```bash
rocprofv3 --att --kernel-include-regex '.*conv_depthwise3d_hip.*' --att-activity 10 -d ./trace_sgb -- ./conv3d_depthwise_sgb 5 2
rocprofv3 --att --kernel-include-regex '.*conv_depthwise3d_hip.*' --att-activity 10 -d ./trace_orig -- ./conv3d_depthwise_original 5 2
```

Optional: drive the same style of session from **`trace.yaml`** (edit `output_directory` and binary path to match your install). Regex in the YAML matches both builds because the symbol name is shared.

## Example output

```text
=== Depthwise Conv3D BF16 (sgb (sched_group_barrier variant)) ===
Device id 0: <GPU name>
  multiProcessorCount=...  maxThreadsPerBlock=1024  warpSize=64
  gcnArchName: gfx942

--- Problem (case3-style depthwise) ---
  Input NCHW:   [1, 512, 61, 45, 80]  BF16
  ...

Running warmup (10 iterations)...
Running benchmark (niter=10)...

--- Performance (same style as gemm samples) ---
  niter (timed):        10
  Average kernel time:  ... ms  (total ... ms)
  Throughput:           ... TFLOPS
  ...

=== Correctness ===
PASS: all ... outputs within atol=0.02 rtol=0.02 (CPU float32 reference, zero pad like conv3d)
```