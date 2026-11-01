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
#include <hipblaslt/hipblaslt-ext.hpp>
#include <limits>
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

// d = tile + beta * d for a rows x cols block of a column-major matrix; d is not read when beta
// is 0, as in BLAS.
template<typename T>
__global__ void StoreTile(int64_t rows, int64_t cols, const float* tile, float beta, T* d,
                          int64_t ldd) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < rows * cols;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    T* p = d + i % rows + i / rows * ldd;
    float v = tile[i];
    if (beta != 0.0f) { v += beta * ToFloat(*p); }
    *p = FromFloat<T>(v);
  }
}

// rocBLAS gemm_ex with a float D accumulates the k-chunks in float. D goes through a float tile of
// tile_size elements in pieces.
template<typename T>
hipblasStatus_t RocblasFloatOutputGemm(hipblasHandle_t handle, hipStream_t stream, float* tile,
                                       int64_t tile_size, hipblasOperation_t trans_a,
                                       hipblasOperation_t trans_b, int64_t m, int64_t n, int64_t k,
                                       float alpha, const T* a, int64_t lda, int64_t stride_a,
                                       const T* b, int64_t ldb, int64_t stride_b, float beta, T* d,
                                       hipDataType type, int64_t ldd, int64_t stride_d,
                                       int64_t batch_count) {
  const int64_t tile_m = std::min(m, tile_size);
  const int64_t tile_n = std::min(n, tile_size / tile_m);
  const float zero = 0.0f;
  for (int64_t batch = 0; batch < batch_count; ++batch) {
    for (int64_t j = 0; j < n; j += tile_n) {
      for (int64_t i = 0; i < m; i += tile_m) {
        const int64_t rows = std::min(tile_m, m - i);
        const int64_t cols = std::min(tile_n, n - j);
        const T* tile_a = a + batch * stride_a + (trans_a == HIPBLAS_OP_N ? i : i * lda);
        const T* tile_b = b + batch * stride_b + (trans_b == HIPBLAS_OP_N ? j * ldb : j);
        hipblasStatus_t status = hipblasGemmEx(
            handle, trans_a, trans_b, rows, cols, k, &alpha, tile_a, type, lda, tile_b, type, ldb,
            &zero, tile, HIP_R_32F, rows, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
        if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
        const int64_t threads = 256;
        const int64_t blocks = std::min<int64_t>((rows * cols + threads - 1) / threads, 4096);
        StoreTile<T><<<blocks, threads, 0, stream>>>(rows, cols, tile, beta,
                                                     d + batch * stride_d + i + j * ldd, ldd);
        if (hipGetLastError() != hipSuccess) { return HIPBLAS_STATUS_EXECUTION_FAILED; }
      }
    }
  }
  return HIPBLAS_STATUS_SUCCESS;
}

}  // namespace

// The gfx11 and gfx12 matrix instructions (WMMA) do not round the float accumulator to nearest: a
// chain of 2^20 products drifts about 100 times further than a chain of fp32 FMAs, while gfx9
// (MFMA) keeps fp32 accuracy.
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

hipblasStatus_t OfHipblasLtSplitKGemm(hipblasLtHandle_t handle, hipStream_t stream, void* workspace,
                                      size_t workspace_size, hipblasLtMatmulDesc_t desc,
                                      hipblasLtMatrixLayout_t layout_a,
                                      hipblasLtMatrixLayout_t layout_b,
                                      hipblasLtMatrixLayout_t layout_d,
                                      hipblasLtMatmulPreference_t pref, float alpha, const void* a,
                                      const void* b, float beta, void* d, int64_t split_k) {
  if (split_k > std::numeric_limits<uint16_t>::max()) { return HIPBLAS_STATUS_NOT_SUPPORTED; }
  constexpr int kRequested = 8;
  hipblasLtMatmulHeuristicResult_t heuristic[kRequested];
  int count = 0;
  if (hipblasLtMatmulAlgoGetHeuristic(handle, desc, layout_a, layout_b, layout_d, layout_d, pref,
                                      kRequested, heuristic, &count)
      != HIPBLAS_STATUS_SUCCESS) {
    return HIPBLAS_STATUS_NOT_SUPPORTED;
  }
  hipblaslt_ext::Gemm gemm(handle, desc, &alpha, a, layout_a, b, layout_b, &beta, d, layout_d, d,
                           layout_d);
  hipblaslt_ext::GemmTuning tuning;
  tuning.setSplitK(static_cast<uint16_t>(split_k));
  // The first algorithm, in heuristic order, that takes this split within the workspace.
  for (int i = 0; i < count; ++i) {
    size_t needed = 0;
    if (gemm.isAlgoSupported(heuristic[i].algo, tuning, needed) != HIPBLAS_STATUS_SUCCESS
        || needed > workspace_size) {
      continue;
    }
    const hipblasStatus_t status =
        gemm.initialize(heuristic[i].algo, tuning, workspace, false, stream);
    if (status != HIPBLAS_STATUS_SUCCESS) { return status; }
    return gemm.run(stream);
  }
  return HIPBLAS_STATUS_NOT_SUPPORTED;
}

hipblasStatus_t OfLongKGemm(hipblasHandle_t handle, hipblasLtHandle_t lt_handle, hipStream_t stream,
                            void* workspace, size_t workspace_size, hipblasOperation_t trans_a,
                            hipblasOperation_t trans_b, int64_t m, int64_t n, int64_t k,
                            float alpha, const void* a, int64_t lda, int64_t stride_a,
                            const void* b, int64_t ldb, int64_t stride_b, float beta, void* d,
                            hipDataType type, int64_t ldd, int64_t stride_d, int64_t batch_count) {
  // BLAS quick return: an empty D needs no write.
  if (m == 0 || n == 0 || batch_count == 0) { return HIPBLAS_STATUS_SUCCESS; }
  if (oneflow::ParseBooleanFromEnv("ONEFLOW_EP_CUDA_LONG_K_GEMM_USE_HIPBLASLT", true)) {
    // Where the accumulator drifts, keep each partial product within kOfLongKGemmChain.
    const int64_t split_k =
        OfMatrixCoreAccumulationDrifts() ? (k + kOfLongKGemmChain - 1) / kOfLongKGemmChain : 1;
    const hipblasStatus_t status = OfHipblasLtGemm(
        lt_handle, stream, workspace, workspace_size, trans_a, trans_b, m, n, k, alpha, a, lda,
        stride_a, b, ldb, stride_b, beta, d, type, ldd, stride_d, batch_count, split_k);
    if (status != HIPBLAS_STATUS_NOT_SUPPORTED) { return status; }
  }
  if (type != HIP_R_16F && type != HIP_R_16BF) { return HIPBLAS_STATUS_NOT_SUPPORTED; }
  // A float tile of at most ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE elements (4 MiB by default).
  // It is the workspace when that is large enough (rocBLAS does not use it) and a temporary
  // allocation otherwise, never a rocBLAS call with the 16-bit D.
  const int64_t max_tile_size = std::max<int64_t>(
      1,
      oneflow::ParseIntegerFromEnv("ONEFLOW_EP_CUDA_LONG_K_GEMM_MAX_TILE_SIZE", int64_t{1} << 20));
  const int64_t tile_size = std::min(m * n, max_tile_size);
  void* tile = workspace;
  if (workspace_size < tile_size * sizeof(float)) {
    tile = nullptr;
    if (hipMallocAsync(&tile, tile_size * sizeof(float), stream) != hipSuccess) {
      return HIPBLAS_STATUS_ALLOC_FAILED;
    }
  }
  hipblasStatus_t status;
  if (type == HIP_R_16F) {
    status = RocblasFloatOutputGemm(handle, stream, static_cast<float*>(tile), tile_size, trans_a,
                                    trans_b, m, n, k, alpha, static_cast<const __half*>(a), lda,
                                    stride_a, static_cast<const __half*>(b), ldb, stride_b, beta,
                                    static_cast<__half*>(d), type, ldd, stride_d, batch_count);
  } else {
    status = RocblasFloatOutputGemm(
        handle, stream, static_cast<float*>(tile), tile_size, trans_a, trans_b, m, n, k, alpha,
        static_cast<const __hip_bfloat16*>(a), lda, stride_a, static_cast<const __hip_bfloat16*>(b),
        ldb, stride_b, beta, static_cast<__hip_bfloat16*>(d), type, ldd, stride_d, batch_count);
  }
  if (tile != workspace) { (void)hipFreeAsync(tile, stream); }
  return status;
}

#endif  // defined(WITH_CUDA) && defined(USE_HIP)
