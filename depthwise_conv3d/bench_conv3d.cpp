/*
 * Standalone host for depthwise Conv3D case3 (BF16).
 * Compile twice with -DKERNEL_SGB=1 (sgb kernel) or -DKERNEL_SGB=0 (original kernel).
 *
 * Args: [niter] [nwarmup] [--no-check]
 *   niter    — timed kernel launches (default 10)
 *   nwarmup  — warmup launches (default 10)
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#ifndef KERNEL_SGB
#define KERNEL_SGB 1
#endif

#define KD 3
#define KH 5
#define KW 5
#define PaddingD 0
#define PaddingH 2
#define PaddingW 2
#define BLOCK_H 45
#define BLOCK_W 80
#define IO_DTYPE __hip_bfloat16

#if KERNEL_SGB
#include "kernels/conv_depthwise3d_hip_sgb.cpp"
#else
#include "kernels/conv_depthwise3d_hip.cpp"
#endif

#define HIP_CHECK(call)                                                                            \
    do {                                                                                           \
        hipError_t err = (call);                                                                   \
        if (err != hipSuccess) {                                                                   \
            fprintf(stderr, "%s:%d hip error: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

static void bf16_fill_pattern(__hip_bfloat16* dst, size_t n, unsigned seed) {
    for (size_t i = 0; i < n; ++i) {
        uint16_t u = static_cast<uint16_t>((seed * 1103515245u + static_cast<unsigned>(i)) & 0x7fffu);
        u |= 0x3f00u;
        std::memcpy(dst + i, &u, sizeof(u));
    }
}

static float bf16_to_float(__hip_bfloat16 x) {
    uint16_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t u32 = static_cast<uint32_t>(bits) << 16u;
    float f = 0.f;
    std::memcpy(&f, &u32, sizeof(f));
    return f;
}

// NCHW linear: in[b,c,d,h,w]
static size_t lin5(int B, int C, int D, int H, int W, int b, int c, int d, int h, int w) {
    return (((static_cast<size_t>(b) * static_cast<size_t>(C) + static_cast<size_t>(c)) * static_cast<size_t>(D) +
             static_cast<size_t>(d)) *
                static_cast<size_t>(H) +
            static_cast<size_t>(h)) *
               static_cast<size_t>(W) +
           static_cast<size_t>(w);
}

// weight [oC, 1, KD, KH, KW] contiguous like PyTorch (in_channels/groups == 1)
static size_t lin_w(int oC, int kd, int kh, int kw) {
    const size_t per_oc = static_cast<size_t>(KD) * KH * KW;
    return static_cast<size_t>(oC) * per_oc + static_cast<size_t>(kd) * static_cast<size_t>(KH * KW) +
           static_cast<size_t>(kh) * static_cast<size_t>(KW) + static_cast<size_t>(kw);
}

/*
 * CPU reference: depthwise conv3d, stride=1, dilation=1, same padding as kernel.
 * Matches torch.nn.functional.conv3d with groups=C (depthwise).
 */
static void cpu_depthwise_conv3d_ref(const std::vector<__hip_bfloat16>& in_bf16,
                                     const std::vector<__hip_bfloat16>& w_bf16,
                                     const std::vector<__hip_bfloat16>& bias_bf16, std::vector<float>& out_f32, int B,
                                     int C, int D, int H, int W, int C_out, int D_out, int H_out, int W_out) {
    out_f32.assign(static_cast<size_t>(B) * C_out * D_out * H_out * W_out, 0.f);
    const int stride_d = 1, stride_h = 1, stride_w = 1;
    const int dil_d = 1, dil_h = 1, dil_w = 1;

    for (int b = 0; b < B; ++b) {
        for (int oc = 0; oc < C_out; ++oc) {
            const int ic = oc;
            const float bias = bf16_to_float(bias_bf16[static_cast<size_t>(oc)]);
            for (int od = 0; od < D_out; ++od) {
                for (int oh = 0; oh < H_out; ++oh) {
                    for (int ow = 0; ow < W_out; ++ow) {
                        float acc = bias;
                        for (int kd = 0; kd < KD; ++kd) {
                            for (int kh = 0; kh < KH; ++kh) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    const int id = od * stride_d - PaddingD + kd * dil_d;
                                    const int ih = oh * stride_h - PaddingH + kh * dil_h;
                                    const int iw = ow * stride_w - PaddingW + kw * dil_w;
                                    if (id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 || iw >= W) {
                                        continue;
                                    }
                                    const float iv =
                                        bf16_to_float(in_bf16[lin5(B, C, D, H, W, b, ic, id, ih, iw)]);
                                    const float wv = bf16_to_float(w_bf16[lin_w(oc, kd, kh, kw)]);
                                    acc += iv * wv;
                                }
                            }
                        }
                        out_f32[lin5(B, C_out, D_out, H_out, W_out, b, oc, od, oh, ow)] = acc;
                    }
                }
            }
        }
    }
}

static void print_bytes_line(const char* label, size_t bytes) {
    const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
    printf("  %-14s %10zu B  (%8.2f MiB)\n", label, bytes, mib);
}

static void print_run_header(int device_id) {
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    printf("\n");
    printf("=== Depthwise Conv3D BF16 (%s) ===\n", KERNEL_SGB ? "sgb (sched_group_barrier variant)" : "original");
    printf("Device id %d: %s\n", device_id, prop.name);
    printf("  multiProcessorCount=%d  maxThreadsPerBlock=%d  warpSize=%d\n", prop.multiProcessorCount,
           prop.maxThreadsPerBlock, prop.warpSize);
#ifdef __HIP_PLATFORM_AMD__
    if (prop.gcnArchName[0] != '\0') {
        printf("  gcnArchName: %s\n", prop.gcnArchName);
    }
#endif
}

static void print_problem_and_memory(int B, int C, int D, int H, int W, int C_out, int D_out, int H_out, int W_out,
                                     size_t n_in, size_t n_out, size_t n_w, size_t n_bias) {
    const size_t bsz = sizeof(__hip_bfloat16);
    const size_t bytes_in = n_in * bsz;
    const size_t bytes_out = n_out * bsz;
    const size_t bytes_w = n_w * bsz;
    const size_t bytes_bias = n_bias * bsz;
    const size_t bytes_dev = bytes_in + bytes_out + bytes_w + bytes_bias;

    printf("\n--- Problem (case3-style depthwise) ---\n");
    printf("  Input NCHW:   [%d, %d, %d, %d, %d]  BF16\n", B, C, D, H, W);
    printf("  Weight:       [%d, 1, %d, %d, %d]  BF16  (groups=%d depthwise)\n", C_out, KD, KH, KW, C);
    printf("  Output NCHW:  [%d, %d, %d, %d, %d]  BF16\n", B, C_out, D_out, H_out, W_out);
    printf("  stride=(1,1,1)  dilation=(1,1,1)  padding=(%d,%d,%d)\n", PaddingD, PaddingH, PaddingW);

    printf("\n--- Device buffers ---\n");
    print_bytes_line("input", bytes_in);
    print_bytes_line("output", bytes_out);
    print_bytes_line("weights", bytes_w);
    print_bytes_line("bias", bytes_bias);
    print_bytes_line("total alloc", bytes_dev);
}

static void print_kernel_config(dim3 grid, dim3 block) {
    const long long total_blocks = static_cast<long long>(grid.x) * grid.y * grid.z;
    const int threads_per_block = block.x * block.y * block.z;
    const long long total_threads = total_blocks * threads_per_block;

    printf("\n--- Kernel configuration ---\n");
    printf("  Kernel symbol: conv_depthwise3d_hip   (__launch_bounds__(256, 1) in HIP source)\n");
    printf("  Compile-time: KD=%d KH=%d KW=%d  PaddingD/H/W=%d/%d/%d  BLOCK_H=%d BLOCK_W=%d  IO_DTYPE=BF16\n", KD, KH,
           KW, PaddingD, PaddingH, PaddingW, BLOCK_H, BLOCK_W);
    printf("  Block:  (%d, %d, %d) -> %d threads / block\n", block.x, block.y, block.z, threads_per_block);
    printf("  Grid:   (%d, %d, %d) -> %lld blocks  (~%lld total thread slots launched per iter)\n", grid.x, grid.y,
           grid.z, total_blocks, total_threads);
}

static void print_performance(double avg_ms, int niter, int B, int C_out, int D_out, int H_out, int W_out, size_t n_in,
                              size_t n_out, size_t n_w, size_t n_bias) {
    const double sec = static_cast<double>(avg_ms) / 1000.0;
    const double gflops =
        (2.0 * static_cast<double>(B) * C_out * D_out * H_out * W_out * 1.0 * KD * KH * KW) / 1e9;
    const double tflops = (gflops / 1000.0) / sec;
    const size_t bsz = sizeof(__hip_bfloat16);
    const double bytes_hbm = static_cast<double>((n_in + n_w + n_bias + n_out) * bsz);
    const double gbps = bytes_hbm / sec / 1e9;

    printf("\n--- Performance (same style as gemm samples) ---\n");
    printf("  niter (timed):        %d\n", niter);
    printf("  Average kernel time:  %.4f ms  (total %.4f ms)\n", avg_ms, avg_ms * niter);
    printf("  Workload:             %.4f GFLOP / iter\n", gflops);
    printf("  Throughput:           %.4f TFLOPS\n", tflops);
    printf("  Efficive Bandwidth:   %.1f GB/s \n", gbps);
}

static bool check_output_bf16_vs_ref(const std::vector<__hip_bfloat16>& gpu_out,
                                     const std::vector<float>& ref_f32, float atol, float rtol, int B, int C_out,
                                     int D_out, int H_out, int W_out) {
    size_t bad = 0;
    float max_abs = 0.f;
    size_t first_i = 0;
    for (size_t i = 0; i < ref_f32.size(); ++i) {
        const float g = bf16_to_float(gpu_out[i]);
        const float r = ref_f32[i];
        const float diff = std::fabs(g - r);
        const float tol = atol + rtol * std::fabs(r);
        if (diff > tol) {
            if (bad == 0) {
                first_i = i;
            }
            ++bad;
            max_abs = std::fmax(max_abs, diff);
        }
    }
    if (bad > 0) {
        fprintf(stderr,
                "\n=== Correctness ===\n"
                "FAIL: %zu / %zu elements outside tol (atol=%g rtol=%g)\n"
                "  max_abs_err=%g  first_idx=%zu  gpu=%g  ref=%g\n",
                bad, ref_f32.size(), atol, rtol, max_abs, first_i, bf16_to_float(gpu_out[first_i]),
                ref_f32[first_i]);
        return false;
    }
    printf("\n=== Correctness ===\n");
    printf("PASS: all %zu outputs within atol=%g rtol=%g (CPU float32 reference, zero pad like conv3d)\n",
           ref_f32.size(), atol, rtol);
    return true;
}

int main(int argc, char** argv) {
    int niter = 10;
    int warmup = 10;
    bool do_check = true;
    int pos = 1;
    if (pos < argc && argv[pos][0] != '-') {
        niter = std::atoi(argv[pos++]);
    }
    if (pos < argc && argv[pos][0] != '-') {
        warmup = std::atoi(argv[pos++]);
    }
    for (; pos < argc; ++pos) {
        if (std::strcmp(argv[pos], "--no-check") == 0) {
            do_check = false;
        }
    }
    if (niter < 1) {
        niter = 1;
    }
    if (warmup < 0) {
        warmup = 0;
    }

    const int B = 1;
    const int C = 512;
    const int D = 61;
    const int H = 45;
    const int W = 80;
    const int C_out = C;
    const int D_out = (D + 2 * PaddingD - (KD - 1) - 1) + 1;
    const int H_out = BLOCK_H;
    const int W_out = BLOCK_W;

    const size_t n_in = static_cast<size_t>(B) * C * D * H * W;
    const size_t n_out = static_cast<size_t>(B) * C_out * D_out * H_out * W_out;
    const size_t n_w = static_cast<size_t>(C_out) * 1 * KD * KH * KW;
    const size_t n_bias = static_cast<size_t>(C_out);

    std::vector<__hip_bfloat16> h_in(n_in), h_w(n_w), h_bias(n_bias), h_out(n_out);
    bf16_fill_pattern(h_in.data(), n_in, 1u);
    bf16_fill_pattern(h_w.data(), n_w, 2u);
    bf16_fill_pattern(h_bias.data(), n_bias, 3u);
    std::memset(h_out.data(), 0, n_out * sizeof(__hip_bfloat16));

    __hip_bfloat16 *d_in = nullptr, *d_out = nullptr, *d_w = nullptr, *d_bias = nullptr;
    HIP_CHECK(hipMalloc(&d_in, n_in * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_out, n_out * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_w, n_w * sizeof(__hip_bfloat16)));
    HIP_CHECK(hipMalloc(&d_bias, n_bias * sizeof(__hip_bfloat16)));

    HIP_CHECK(hipMemcpy(d_in, h_in.data(), n_in * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w, h_w.data(), n_w * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), n_bias * sizeof(__hip_bfloat16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, n_out * sizeof(__hip_bfloat16)));

    dim3 block(256);
    dim3 grid(B, C_out, D_out);

    int dev_id = 0;
    HIP_CHECK(hipGetDevice(&dev_id));
    print_run_header(dev_id);
    print_problem_and_memory(B, C, D, H, W, C_out, D_out, H_out, W_out, n_in, n_out, n_w, n_bias);
    print_kernel_config(grid, block);

    printf("\nRunning warmup (%d iterations)...\n", warmup);
    for (int i = 0; i < warmup; ++i) {
        hipLaunchKernelGGL(conv_depthwise3d_hip, grid, block, 0, 0, d_in, d_out, d_w, d_bias, C, D, H, W,
                           C_out, D_out, H_out, W_out);
    }
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());

    hipEvent_t ev0, ev1;
    HIP_CHECK(hipEventCreate(&ev0));
    HIP_CHECK(hipEventCreate(&ev1));
    printf("Running benchmark (niter=%d)...\n", niter);
    HIP_CHECK(hipEventRecord(ev0));
    for (int i = 0; i < niter; ++i) {
        hipLaunchKernelGGL(conv_depthwise3d_hip, grid, block, 0, 0, d_in, d_out, d_w, d_bias, C, D, H, W,
                           C_out, D_out, H_out, W_out);
    }
    HIP_CHECK(hipEventRecord(ev1));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));

    HIP_CHECK(hipMemcpy(h_out.data(), d_out, n_out * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));

    const float avg_ms = ms / static_cast<float>(niter);
    print_performance(avg_ms, niter, B, C_out, D_out, H_out, W_out, n_in, n_out, n_w, n_bias);

    uint16_t u0;
    std::memcpy(&u0, &h_out[0], sizeof(u0));

    if (do_check) {
        std::vector<float> ref_f32;
        cpu_depthwise_conv3d_ref(h_in, h_w, h_bias, ref_f32, B, C, D, H, W, C_out, D_out, H_out, W_out);
        const float atol = 0.02f;
        const float rtol = 0.02f;
        if (!check_output_bf16_vs_ref(h_out, ref_f32, atol, rtol, B, C_out, D_out, H_out, W_out)) {
            HIP_CHECK(hipEventDestroy(ev0));
            HIP_CHECK(hipEventDestroy(ev1));
            HIP_CHECK(hipFree(d_in));
            HIP_CHECK(hipFree(d_out));
            HIP_CHECK(hipFree(d_w));
            HIP_CHECK(hipFree(d_bias));
            return 1;
        }
    } else {
        printf("correctness skipped (--no-check)\n");
    }

    HIP_CHECK(hipEventDestroy(ev0));
    HIP_CHECK(hipEventDestroy(ev1));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_w));
    HIP_CHECK(hipFree(d_bias));
    return 0;
}
