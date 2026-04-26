#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <chrono>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <random>
#include <cstdint>
#include <cmath>
#include <cstdlib>

using namespace std;

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

using float16 = __hip_bfloat16;
using float8  = __hip_fp8_e4m3_fnuz;

// Naive FP8 GEMM kernel - straightforward implementation without optimizations
// No MFMA, no shared memory, no tiling - just simple computation
// Each thread computes one output element: C[i,j] = sum_k(A[i,k] * B[k,j] * scales)
__global__ void naive_fp8_gemm_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    float16* __restrict__ c,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int M, int N, int K
) {
    // Calculate which output element this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (row >= M || col >= N) return;
    
    // Accumulate result in float for precision
    float sum = 0.0f;
    
    // Simple loop over K dimension
    for (int k = 0; k < K; k++) {
        // Load FP8 values and convert to float
        float8 a_fp8 = reinterpret_cast<const float8*>(a)[row * K + k];
        float8 b_fp8 = reinterpret_cast<const float8*>(b)[k * N + col];
        
        float a_val = static_cast<float>(a_fp8);
        float b_val = static_cast<float>(b_fp8);
        
        // Load scaling factors
        // scale_a: indexed by [k/128][row]
        // scale_b: indexed by [k/128][col/128]
        int k_block = k / 128;
        int col_block = col / 128;
        
        float scale_a_val = scale_a[k_block * M + row];
        float scale_b_val = scale_b[k_block * ((N + 127) / 128) + col_block];
        
        // Compute scaled product and accumulate
        sum += a_val * b_val * scale_a_val * scale_b_val;
    }
    
    // Write result as BF16
    c[row * N + col] = static_cast<float16>(sum);
}

// Helper: Initialize FP8 matrix with random values
void initialize_fp8_matrix(float8* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = static_cast<float8>((rand() % 8192)/1024.0f - 4.0f);
    }
}

// Helper: Initialize scaling factors
void initialize_scales(float* scales, int size) {
    for (int i = 0; i < size; i++) {
        scales[i] = (rand() % 8192)/1024.0f - 4.0f;
    }
}

int main(int argc, char** argv) {
    // Default matrix dimensions
    int M = 1024, N = 1536, K = 7168;
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    
    printf("=== Naive FP8 GEMM (Unoptimized) ===\n");
    printf("Problem size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Compute: C[%dx%d] = A[%dx%d] * B[%dx%d] (FP8 with scaling)\n\n", 
           M, N, M, K, K, N);
    
    // Allocate host memory
    size_t size_A = M * K * sizeof(float8);
    size_t size_B = K * N * sizeof(float8);
    size_t size_C = M * N * sizeof(float16);
    size_t size_scale_a = ((K+127)/128) * M * sizeof(float);
    size_t size_scale_b = ((K+127)/128) * ((N+127)/128) * sizeof(float);
    
    float8* h_A = (float8*)malloc(size_A);
    float8* h_B = (float8*)malloc(size_B);
    float16* h_C = (float16*)malloc(size_C);
    float* h_scale_a = (float*)malloc(size_scale_a);
    float* h_scale_b = (float*)malloc(size_scale_b);
    
    // Initialize matrices
    srand(42);
    initialize_fp8_matrix(h_A, M, K);
    initialize_fp8_matrix(h_B, K, N);
    initialize_scales(h_scale_a, ((K+127)/128) * M);
    initialize_scales(h_scale_b, ((K+127)/128) * ((N+127)/128));
    
    // Allocate device memory
    uint8_t *d_A, *d_B;
    float16 *d_C;
    float *d_scale_a, *d_scale_b;
    HIP_API_CALL(hipMalloc(&d_A, size_A));
    HIP_API_CALL(hipMalloc(&d_B, size_B));
    HIP_API_CALL(hipMalloc(&d_C, size_C));
    HIP_API_CALL(hipMalloc(&d_scale_a, size_scale_a));
    HIP_API_CALL(hipMalloc(&d_scale_b, size_scale_b));
    
    // Copy data to device
    HIP_API_CALL(hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice));
    HIP_API_CALL(hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice));
    HIP_API_CALL(hipMemcpy(d_scale_a, h_scale_a, size_scale_a, hipMemcpyHostToDevice));
    HIP_API_CALL(hipMemcpy(d_scale_b, h_scale_b, size_scale_b, hipMemcpyHostToDevice));
    
    // Setup kernel launch configuration - simple 16x16 thread blocks
    dim3 block(16, 16, 1);  // 256 threads per block
    dim3 grid((N + block.x - 1) / block.x, 
              (M + block.y - 1) / block.y);
    
    printf("Kernel configuration:\n");
    printf("  Block size: %dx%d = %d threads\n", block.x, block.y, block.x * block.y);
    printf("  Grid size: %dx%d = %d blocks\n", grid.x, grid.y, grid.x * grid.y);
    printf("  Each thread computes 1 output element\n");
    printf("  No MFMA, no shared memory, no tiling\n\n");
    
    // Warmup runs
    printf("Running warmup...\n");
    for (int i = 0; i < 10; i++) {
        naive_fp8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K);
    }
    HIP_API_CALL(hipDeviceSynchronize());
    
    // Timed runs - directly call kernel
    printf("Running benchmark (100 iterations)...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        naive_fp8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K);
    }
    
    HIP_API_CALL(hipGetLastError());
    HIP_API_CALL(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    float time_ms = std::chrono::duration<float, std::milli>(end - start).count() / 100.0f;
    
    // Copy result back
    HIP_API_CALL(hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost));
    
    // Calculate performance metrics
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (time_ms / 1000.0)) / 1e12;
    double bandwidth_gb = (size_A + size_B + size_C) / (time_ms / 1000.0) / 1e9;
    
    printf("\n=== Performance Results ===\n");
    printf("Average time: %.3f ms\n", time_ms);
    printf("Throughput: %.3f TFLOPS\n", tflops);
    printf("Effective bandwidth: %.1f GB/s\n", bandwidth_gb);
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_scale_a);
    free(h_scale_b);
    HIP_API_CALL(hipFree(d_A));
    HIP_API_CALL(hipFree(d_B));
    HIP_API_CALL(hipFree(d_C));
    HIP_API_CALL(hipFree(d_scale_a));
    HIP_API_CALL(hipFree(d_scale_b));
    
    return 0;
}
