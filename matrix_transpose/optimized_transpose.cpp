#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cassert>

#define HIP_CHECK(cmd) do { \
  hipError_t e = cmd; \
  if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__, hipGetErrorString(e)); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

static inline double bytesGB(size_t bytes) { return static_cast<double>(bytes) / 1e9; }

#ifndef TILE_DIM
#define TILE_DIM 32
#endif

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 8
#endif

// Launch bounds hint: 256 threads/block, at least 7 blocks per CU if possible.
// Adjust minBlocksPerCU to balance occupancy and perf on your workload.
__launch_bounds__(TILE_DIM * BLOCK_ROWS, 8)
__global__ void transposeTiled(const float* __restrict__ A, float* __restrict__ B,
                               int width, int height, int lda, int ldb) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 avoids LDS bank conflicts

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  // Load tile from A (coalesced)
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    int yy = y + j;
    if (x < width && yy < height) {
      tile[threadIdx.y + j][threadIdx.x] = A[yy * lda + x];
    }
  }

  __syncthreads();

  // Transpose coordinates
  int tx = blockIdx.y * TILE_DIM + threadIdx.x; // becomes column block for B
  int ty = blockIdx.x * TILE_DIM + threadIdx.y; // becomes row block for B

  // Store tile to B (coalesced)
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    int yy = ty + j;
    if (tx < height && yy < width) {
      B[yy * ldb + tx] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

static void checkCorrectness(const std::vector<float>& A, const std::vector<float>& B,
                             int width, int height, int lda, int ldb) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float ref = A[y * lda + x];
      float got = B[x * ldb + y];
      if (std::fabs(ref - got) > 1e-4f) {
        fprintf(stderr, "Mismatch at A(%d,%d) -> B(%d,%d): ref=%f got=%f\n",
                y, x, x, y, ref, got);
        std::exit(EXIT_FAILURE);
      }
    }
  }
}

int main(int argc, char** argv) {
  int device = 0;
  HIP_CHECK(hipSetDevice(device));

  int N = 8192;
  if (argc > 1) N = std::atoi(argv[1]);
  int width = N;
  int height = N;
  int lda = width;
  int ldb = height;

  size_t bytesA = static_cast<size_t>(height) * lda * sizeof(float);
  size_t bytesB = static_cast<size_t>(width) * ldb * sizeof(float);

  std::vector<float> hA(height * lda);
  std::vector<float> hB(width * ldb, 0.0f);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // Initialize with some random data
      hA[y * lda + x] = static_cast<float>((y * 131 + x * 7) % 251);
    }
  }

  float *dA = nullptr, *dB = nullptr;
  HIP_CHECK(hipMalloc(&dA, bytesA));
  HIP_CHECK(hipMalloc(&dB, bytesB));
  HIP_CHECK(hipMemcpy(dA, hA.data(), bytesA, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dB, 0, bytesB));

  dim3 block(TILE_DIM, BLOCK_ROWS);       // 32 x 8 = 256 threads
  dim3 grid((width + TILE_DIM - 1) / TILE_DIM,
            (height + TILE_DIM - 1) / TILE_DIM);

  // Query kernel attributes and occupancy
  hipFuncAttributes attr{};
  HIP_CHECK(hipFuncGetAttributes(&attr, (const void*)transposeTiled));
  int maxBlocksPerCU = 0;
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerCU,
                                                         (const void*)transposeTiled,
                                                         block.x * block.y,
                                                         TILE_DIM * (TILE_DIM + 1) * sizeof(float)));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  int mpCount = prop.multiProcessorCount;

  printf("Device: %s (CUs=%d)\n", prop.name, mpCount);
  printf("Kernel: transposeTiled, regsPerThread=%d, localSizeBytes=%zu, sharedSizeBytes=%zu\n",
         attr.numRegs, (size_t)attr.localSizeBytes, (size_t)attr.sharedSizeBytes);
  printf("Block %dx%d (%d threads); grid %dx%d; Max active blocks/CU (runtime est): %d\n",
         block.x, block.y, block.x * block.y, grid.x, grid.y, maxBlocksPerCU);

  // Warm-up
  for (int i = 0; i < 5; ++i) {
    hipLaunchKernelGGL(transposeTiled, grid, block, 0, 0, dA, dB, width, height, lda, ldb);
  }
  HIP_CHECK(hipDeviceSynchronize());

  // Timed runs
  const int ITERS = 50;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  for (int i = 0; i < ITERS; ++i) {
    hipLaunchKernelGGL(transposeTiled, grid, block, 0, 0, dA, dB, width, height, lda, ldb);
  }
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  ms /= ITERS;

  double gb = bytesGB(bytesA + bytesB);
  double gbps = gb / (ms * 1e-3);
  printf("Avg time: %.3f ms; Effective bandwidth: %.2f GB/s\n", ms, gbps);

  // Validate
  HIP_CHECK(hipMemcpy(hB.data(), dB, bytesB, hipMemcpyDeviceToHost));
  checkCorrectness(hA, hB, width, height, lda, ldb);
  printf("Validation: PASS\n");

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  return 0;
}
