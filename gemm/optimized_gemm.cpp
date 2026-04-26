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

#define SHMBLOCK 64
#define TBLOCK 16

using float16 = __hip_bfloat16;
using float8  = __hip_fp8_e4m3_fnuz;
using Vec4    = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using IVec4   = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using Vec16   = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using Vec8    = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using float8x8 = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;

static_assert(sizeof(float8) == sizeof(uint8_t));

inline __device__ uint64_t castfrom8x8(const float8x8& f)
{
    return *reinterpret_cast<const uint64_t*>(&f);
}

// MFMA wrapper: performs 16x16x32 FP8 matrix multiply
// Returns 4 float results per thread
inline __device__ Vec4 mfma(const float8x8& a, const float8x8& b)
{
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(castfrom8x8(a), castfrom8x8(b), Vec4{}, 0, 0, 0); 
}

// MFMA with accumulation from two consecutive operations
inline __device__ Vec4 mfma(const float8x8& a0, const float8x8& a1, const float8x8& b0, const float8x8& b1)
{
    Vec4 ret = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(castfrom8x8(a0), castfrom8x8(b0), Vec4{}, 0, 0, 0);
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(castfrom8x8(a1), castfrom8x8(b1), ret, 0, 0, 0);
}

// Optimized FP8 GEMM kernel using MFMA instructions
// Fixed configuration: WIDTH=1, HEIGHT=4 (processes 64x64 output tiles)
// Uses scaled FP8 arithmetic with per-128-element scaling factors
#define WIDTH 1
#define HEIGHT 4

__global__ void __launch_bounds__(TBLOCK*TBLOCK*2, 4)
fp8_gemm_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    float16* __restrict__ c,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int MDIM, int NDIM, int KDIM
) {
    const int X = blockIdx.x * SHMBLOCK * WIDTH;
    const int Y = blockIdx.y * TBLOCK * HEIGHT;
    
    const int TX = threadIdx.x%TBLOCK;
    const int TY = threadIdx.y%4;
    const int TZ = (threadIdx.y/4)%4;
    const int TYZ = threadIdx.y%16;
    const int TW = (threadIdx.y/16)%2;

    // Shared memory for data tiles and scaling factors
    __shared__ uint8_t a_shared[WIDTH][4][4][4][TBLOCK][4];
    __shared__ uint8_t b_shared[4][HEIGHT][4][TBLOCK][4];
    __shared__ float   scalar[4][TBLOCK][WIDTH];

    // Wave 1 (TW=1): Load data from global memory to shared memory
    if (TW != 0)
    {
        for (int k1=0; k1 < KDIM; k1 += 2*SHMBLOCK)
        for (int k2=0; k2 < 2*SHMBLOCK; k2 += SHMBLOCK)
        {
            int k0 = k1 + k2;
            IVec4 temp_a[WIDTH], temp_b;

            int tx3 = (TX & 3) << 2;
            int i1 = TX >> 2;

            // Load B tile (4 uint32_t = 16 bytes per thread)
            for (int j1=0; j1<4; j1++)
            if (k0 + j1*TBLOCK < KDIM && Y + 4*TX < NDIM)
                temp_b[j1] = *reinterpret_cast<const uint32_t*>(&b[(k0 + TZ*TBLOCK + TY + 4*j1)*NDIM + Y + 4*TX]);

            // Load A tiles for all WIDTH blocks
            for (int j1=0; j1<4; j1++)
            if (k0 + j1*TBLOCK < KDIM && X < MDIM)
            for (int n=0; n<WIDTH; n++)
                temp_a[n][j1] = (X + n*SHMBLOCK < MDIM) ? *reinterpret_cast<const uint32_t*>(&a[(k0 + j1*TBLOCK + TYZ)*MDIM + X + 4*TX + n*SHMBLOCK]) : 0;

            if (k0 != 0) __syncthreads();

            // Unpack and store B to shared memory
            if (i1 < HEIGHT)
            for (int j1=0; j1<4; j1++)
            for (int m=0; m<4; m++)
                b_shared[TZ][i1][TY][tx3 + m][j1] = (temp_b[j1] >> (8*m)) & 0xFF;

            // Unpack and store A to shared memory
            for (int j1=0; j1<4; j1++)
            for (int m=0; m<4; m++)
            for (int n=0; n<WIDTH; n++)
                a_shared[n][i1][TZ][TY][tx3 + m][j1] = (temp_a[n][j1] >> (8*m)) & 0xFF;

            __syncthreads();
        }
        return;
    }

    // Wave 0 (TW=0): Compute using MFMA
    Vec16 reg_res[WIDTH];
    for (int n=0; n<WIDTH; n++) reg_res[n] = {};

    float tmp_a[WIDTH];
    float tmp_b;
    
    // Load initial scaling factors
    if (TZ == 0)
    {
        for (int n=0; n<WIDTH; n++)
            tmp_a[n] = (X + TY*TBLOCK + n*SHMBLOCK < MDIM) ? scale_a[X + TY*TBLOCK + TX + n*SHMBLOCK] : 0;
        tmp_b = scale_b[Y/128];
    }

    // Main computation loop over K dimension
    for (int k1=0; k1 < KDIM; k1 += 2*SHMBLOCK)
    {
        // Prepare scaling factors for current K block
        if (TZ == 0)
        {
            for (int n=0; n<WIDTH; n++) scalar[TY][TX][n] = tmp_b * tmp_a[n];

            // Prefetch scaling factors for next K block
            if (k1 + 2*SHMBLOCK < KDIM)
            {
                for (int n=0; n<WIDTH; n++)
                    tmp_a[n] = (X + TY*TBLOCK + n*SHMBLOCK < MDIM) ? scale_a[(k1/128 + 1)*MDIM + X + TY*TBLOCK + TX + n*SHMBLOCK] : 0;
                tmp_b = scale_b[(k1/128 + 1)*((NDIM+127)/128) + (Y/128)];
            }
        }
        
        // Process two SHMBLOCK-sized sub-tiles
        for (int k2=0; k2 < 2*SHMBLOCK; k2 += SHMBLOCK)
        {
            __syncthreads();
            
            // Load data from shared memory into registers
            float8x8 a0_load[WIDTH], a1_load[WIDTH];
            float8x8 b0_load[HEIGHT], b1_load[HEIGHT];

            // Load B data for all HEIGHT blocks
            for (int m=0; m<8; m++)
            for (int n=0; n<HEIGHT; n++)
            {
                b0_load[n][m] = b_shared[m/4 + 0][n][TY][TX][m%4];
                b1_load[n][m] = b_shared[m/4 + 2][n][TY][TX][m%4];
            }

            // Load A data for all WIDTH blocks
            for (int m=0; m<8; m++)
            for (int r=0; r<WIDTH; r++)
            {
                a0_load[r][m] = a_shared[r][TZ][m%4][TY][TX][m/4 + 0];
                a1_load[r][m] = a_shared[r][TZ][m%4][TY][TX][m/4 + 2];
            }

            // Load scaling factors for this computation
            Vec4 scal[WIDTH];
            for (int n=0; n<WIDTH; n++)
            for (int m=0; m<4; m++)
                scal[n][m] = scalar[TZ][TY*4 + m][n];

            __syncthreads();

            // Perform MFMA computations for all combinations of WIDTH x HEIGHT tiles
            for (int n=0; n<HEIGHT; n++)
            for (int r1=0; r1<WIDTH; r1++)
            {
                // Each MFMA computes 16x16 output with 32 K-elements accumulated
                Vec4 res = mfma(a0_load[r1], a1_load[r1], b0_load[n], b1_load[n]);

                // Apply scaling and accumulate results
                for (int m=0; m<4; m++)
                    reg_res[r1][m*HEIGHT + n] += scal[r1][m] * res[m];
            }
        }
    }

    // Write results back to global memory (convert to BF16)
    for (int j=0; j<4; j ++) for (int i=0; i<HEIGHT; i ++) for (int r1=0; r1<WIDTH; r1++)
    if (Y + i*TBLOCK < NDIM && 4*TYZ + X + r1*SHMBLOCK < MDIM)
        c[(4*TYZ + j + X + r1*SHMBLOCK)*NDIM + Y + i*TBLOCK + TX] = (float16) reg_res[r1][j*HEIGHT + i];
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
    
    printf("=== Optimized FP8 GEMM with MFMA ===\n");
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
    
    // Setup kernel launch configuration
    dim3 block(TBLOCK, 32, 1);  // 16 x 32 = 512 threads per block
    dim3 grid((M + SHMBLOCK * WIDTH - 1) / (SHMBLOCK * WIDTH), 
              (N + TBLOCK * HEIGHT - 1) / (TBLOCK * HEIGHT));
    
    printf("Kernel configuration:\n");
    printf("  Block size: %dx%d = %d threads\n", block.x, block.y, block.x * block.y);
    printf("  Grid size: %dx%d = %d blocks\n", grid.x, grid.y, grid.x * grid.y);
    printf("  WIDTH=%d, HEIGHT=%d\n", WIDTH, HEIGHT);
    printf("  Output tile size: %dx%d\n\n", SHMBLOCK * WIDTH, TBLOCK * HEIGHT);
    
    // Warmup runs
    printf("Running warmup...\n");
    for (int i = 0; i < 10; i++) {
        fp8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K);
    }
    HIP_API_CALL(hipDeviceSynchronize());
    
    // Timed runs - directly call kernel
    printf("Running benchmark (100 iterations)...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; i++) {
        fp8_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K);
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
