/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#if defined(WITH_CUDA) && defined(USE_HIP)

#include <cublas_v2.h>
#include <string>
#include <vector>
#include "oneflow/core/common/util.h"

namespace {

__device__ float ToFloat(__half x) { return __half2float(x); }
__device__ float ToFloat(__hip_bfloat16 x) { return __bfloat162float(x); }
template<typename T>
__device__ T FromFloat(float x);
template<>
__device__ __half FromFloat<__half>(float x) {
  return __float2half(x);
}
template<>
__device__ __hip_bfloat16 FromFloat<__hip_bfloat16>(float x) {
  return __float2bfloat16(x);
}

// d = (the sum of the slices of tile) + beta * d for a rows x cols block of a column-major matrix,
// where tile holds `slices` column-major rows x cols matrices back to back; d is not read when
// beta is 0, as in BLAS.
template<typename T>
__global__ void StoreTile(int64_t rows, int64_t cols, int64_t slices, const float* tile, float beta,
                          T* d, int64_t ldd) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < rows * cols;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    T* p = d + i % rows + i / rows * ldd;
    float v = 0.0f;
    for (int64_t s = 0; s < slices; ++s) { v += tile[i + s * rows * cols]; }
    if (beta != 0.0f) { v += beta * ToFloat(*p); }
    *p = FromFloat<T>(v);
  }
}

// D in tiles of rows x cols through float. With rocBLAS (chunk = k), gemm_ex with a float D
// accumulates its own k-chunks in float. With hipBLASLt, k is cut into chunks of `chunk` (the last
// may be shorter), each chunk's product goes to its own float slice of the tile in one strided
// batched call (plus one call for a shorter last chunk), and StoreTile sums the slices in float.
// The tile is tile_size floats in all.
template<typename T>
hipblasStatus_t FloatOutputGemm(hipblasHandle_t handle, hipblasLtHandle_t lt_handle,
                                hipStream_t stream, void* workspace, size_t workspace_size,
                                float* tile, int64_t tile_size, bool use_hipblaslt, int64_t chunk,
                                hipblasOperation_t trans_a, hipblasOperation_t trans_b, int64_t m,
                                int64_t n, int64_t k, float alpha, const T* a, int64_t lda,
                                int64_t stride_a, const T* b, int64_t ldb, int64_t stride_b,
                                float beta, T* d, hipDataType type, int64_t ldd, int64_t stride_d,
                                int64_t batch_count) {
  const int64_t full_chunks = k / chunk;
  const int64_t last_chunk = k % chunk;
  const int64_t slices = full_chunks + (last_chunk > 0 ? 1 : 0);
  const int64_t tile_m = std::min(m, std::max<int64_t>(1, tile_size / slices));
  const int64_t tile_n = std::min(n, std::max<int64_t>(1, tile_size / slices / tile_m));
  // Element offsets of k index 1 in A and B.
  const int64_t k_step_a = trans_a == HIPBLAS_OP_N ? lda : 1;
  const int64_t k_step_b = trans_b == HIPBLAS_OP_N ? 1 : ldb;
  const float zero = 0.0f;
  bool written = false;
  for (int64_t batch = 0; batch < batch_count; ++batch) {
    for (int64_t j = 0; j < n; j += tile_n) {
      for (int64_t i = 0; i < m; i += tile_m) {
        const int64_t rows = std::min(tile_m, m - i);
        const int64_t cols = std::min(tile_n, n - j);
        const T* tile_a = a + batch * stride_a + (trans_a == HIPBLAS_OP_N ? i : i * lda);
        const T* tile_b = b + batch * stride_b + (trans_b == HIPBLAS_OP_N ? j * ldb : j);
        hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
        if (!use_hipblaslt) {
          status = hipblasGemmEx(handle, trans_a, trans_b, rows, cols, k, &alpha, tile_a, type, lda,
                                 tile_b, type, ldb, &zero, tile, HIP_R_32F, rows,
                                 HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        } else {
          if (full_chunks > 0) {
            status = OfHipblasLtGemm(lt_handle, stream, workspace, workspace_size, trans_a, trans_b,
                                     rows, cols, chunk, alpha, tile_a, lda, chunk * k_step_a,
                                     tile_b, ldb, chunk * k_step_b, 0.0f, tile, type, HIP_R_32F,
                                     rows, rows * cols, full_chunks);
          }
          if (status == HIPBLAS_STATUS_SUCCESS && last_chunk > 0) {
            status = OfHipblasLtGemm(lt_handle, stream, workspace, workspace_size, trans_a, trans_b,
                                     rows, cols, last_chunk, alpha,
                                     tile_a + full_chunks * chunk * k_step_a, lda, 0,
                                     tile_b + full_chunks * chunk * k_step_b, ldb, 0, 0.0f,
                                     tile + full_chunks * rows * cols, type, HIP_R_32F, rows, 0, 1);
          }
        }
        // Once part of D is written, the caller cannot fall back to another library.
        if (status == HIPBLAS_STATUS_NOT_SUPPORTED && written) {
          status = HIPBLAS_STATUS_INTERNAL_ERROR;
        }
        if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
        const int64_t threads = 256;
        const int64_t blocks = std::min<int64_t>((rows * cols + threads - 1) / threads, 4096);
        StoreTile<T><<<blocks, threads, 0, stream>>>(rows, cols, use_hipblaslt ? slices : 1, tile,
                                                     beta, d + batch * stride_d + i + j * ldd, ldd);
        if (hipGetLastError() != hipSuccess) { return HIPBLAS_STATUS_EXECUTION_FAILED; }
        written = true;
      }
    }
  }
  return HIPBLAS_STATUS_SUCCESS;
}

}  // namespace

// The gfx11 and gfx12 matrix instructions (WMMA) do not round the float accumulator to nearest: a
// chain of 2^20 float16 products drifts about 100 (gfx1100) or 50 (gfx1201) times further than a
// chain of fp32 FMAs, while gfx9 (MFMA) keeps fp32 accuracy.
bool OfMatrixCoreAccumulationDrifts() {
  static const std::vector<bool> drifts = [] {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess) { count = 0; }
    std::vector<bool> result(count);
    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t prop;
      if (hipGetDeviceProperties(&prop, i) != hipSuccess) { continue; }
      const std::string arch = prop.gcnArchName;
      result[i] = arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
    }
    return result;
  }();
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess || device >= static_cast<int>(drifts.size())) {
    return false;
  }
  return drifts[device];
}

hipblasStatus_t OfLongKGemm(hipblasHandle_t handle, hipblasLtHandle_t lt_handle, hipStream_t stream,
                            void* workspace, size_t workspace_size, hipblasOperation_t trans_a,
                            hipblasOperation_t trans_b, int64_t m, int64_t n, int64_t k,
                            float alpha, const void* a, int64_t lda, int64_t stride_a,
                            const void* b, int64_t ldb, int64_t stride_b, float beta, void* d,
                            hipDataType type, int64_t ldd, int64_t stride_d, int64_t batch_count) {
  // BLAS quick return: an empty D needs no write.
  if (m == 0 || n == 0 || batch_count == 0) { return HIPBLAS_STATUS_SUCCESS; }
  const bool use_hipblaslt =
      oneflow::ParseBooleanFromEnv("ONEFLOW_EP_CUDA_LONG_K_GEMM_USE_HIPBLASLT", true);
  // Where the accumulator drifts, hipBLASLt gets no GEMM with k above kOfLongKGemmChain. That also
  // keeps clear of kernels that are wrong only at large k: on gfx1201 the heuristic's first choice
  // for k-major A and small outputs is off by tens of float16 ulp for some k above 2^17, and a
  // hipBLASLt split-k does not change that.
  const bool chunked = use_hipblaslt && OfMatrixCoreAccumulationDrifts();
  if (use_hipblaslt && !chunked) {
    const hipblasStatus_t status = OfHipblasLtGemm(
        lt_handle, stream, workspace, workspace_size, trans_a, trans_b, m, n, k, alpha, a, lda,
        stride_a, b, ldb, stride_b, beta, d, type, type, ldd, stride_d, batch_count);
    if (status != HIPBLAS_STATUS_NOT_SUPPORTED) { return status; }
  }
  if (type != HIP_R_16F && type != HIP_R_16BF) { return HIPBLAS_STATUS_NOT_SUPPORTED; }
  // A float tile of at most ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE output elements (4 MiB by
  // default) per k-chunk, in the workspace when that is large enough and a temporary allocation
  // otherwise; never a library call with the 16-bit D. hipBLASLt gets the rest of the workspace.
  const int64_t max_tile_size = std::max<int64_t>(
      1,
      oneflow::ParseIntegerFromEnv("ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE", int64_t{1} << 20));
  const auto Run = [&](bool hipblaslt) {
    // Equal chunks of at most kOfLongKGemmChain, so that one call does nearly all the work.
    const int64_t slices = hipblaslt ? (k + kOfLongKGemmChain - 1) / kOfLongKGemmChain : 1;
    const int64_t chunk = (k + slices - 1) / slices;
    // Up to max_tile_size output elements in each slice.
    const int64_t tile_size = std::min(m * n, max_tile_size) * slices;
    // 256-byte aligned, as hipBLASLt expects of its workspace.
    const size_t tile_bytes = (tile_size * sizeof(float) + 255) / 256 * 256;
    void* tile = workspace;
    void* lt_workspace = workspace;
    size_t lt_workspace_size = workspace_size;
    if (workspace_size >= tile_bytes) {
      lt_workspace = static_cast<char*>(workspace) + tile_bytes;
      lt_workspace_size = workspace_size - tile_bytes;
    } else if (hipMallocAsync(&tile, tile_bytes, stream) != hipSuccess) {
      return HIPBLAS_STATUS_ALLOC_FAILED;
    }
    hipblasStatus_t status;
    if (type == HIP_R_16F) {
      status = FloatOutputGemm(handle, lt_handle, stream, lt_workspace, lt_workspace_size,
                               static_cast<float*>(tile), tile_size, hipblaslt, chunk, trans_a,
                               trans_b, m, n, k, alpha, static_cast<const __half*>(a), lda,
                               stride_a, static_cast<const __half*>(b), ldb, stride_b, beta,
                               static_cast<__half*>(d), type, ldd, stride_d, batch_count);
    } else {
      status = FloatOutputGemm(handle, lt_handle, stream, lt_workspace, lt_workspace_size,
                               static_cast<float*>(tile), tile_size, hipblaslt, chunk, trans_a,
                               trans_b, m, n, k, alpha, static_cast<const __hip_bfloat16*>(a), lda,
                               stride_a, static_cast<const __hip_bfloat16*>(b), ldb, stride_b, beta,
                               static_cast<__hip_bfloat16*>(d), type, ldd, stride_d, batch_count);
    }
    if (tile != workspace) { (void)hipFreeAsync(tile, stream); }
    return status;
  };
  if (chunked) {
    const hipblasStatus_t status = Run(true);
    if (status != HIPBLAS_STATUS_NOT_SUPPORTED) { return status; }
  }
  return Run(false);
}

#endif  // defined(WITH_CUDA) && defined(USE_HIP)
