/*
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License").
*/
#pragma once
#include "oneflow/core/device/cuda_to_hip.h"
#include <hipblas/hipblas.h>

// cuBLAS keeps C++ overloads of cublasGemmEx, cublasGemmBatchedEx and cublasGemmStridedBatchedEx
// that take the compute type as a data type (the signatures from before cublasComputeType_t
// existed). hipBLAS has only the compute-type forms, so map the data type the same way cuBLAS does.
inline bool OfHipblasComputeType(hipDataType data_type, hipblasComputeType_t* compute_type) {
  switch (data_type) {
    case HIP_R_16F: *compute_type = HIPBLAS_COMPUTE_16F; return true;
    case HIP_R_32F: *compute_type = HIPBLAS_COMPUTE_32F; return true;
    case HIP_R_64F: *compute_type = HIPBLAS_COMPUTE_64F; return true;
    case HIP_R_32I: *compute_type = HIPBLAS_COMPUTE_32I; return true;
    default: return false;
  }
}

inline hipblasStatus_t hipblasGemmEx(hipblasHandle_t handle, hipblasOperation_t transA,
                                     hipblasOperation_t transB, int m, int n, int k,
                                     const void* alpha, const void* A, hipDataType aType, int lda,
                                     const void* B, hipDataType bType, int ldb, const void* beta,
                                     void* C, hipDataType cType, int ldc, hipDataType computeType,
                                     hipblasGemmAlgo_t algo) {
  hipblasComputeType_t compute_type;
  if (!OfHipblasComputeType(computeType, &compute_type)) { return HIPBLAS_STATUS_INVALID_ENUM; }
  return hipblasGemmEx(handle, transA, transB, m, n, k, alpha, A, aType, lda, B, bType, ldb, beta,
                       C, cType, ldc, compute_type, algo);
}

// The pointer arrays follow cuBLAS's `const void* const[]`, which also accepts `void**`.
inline hipblasStatus_t hipblasGemmBatchedEx(
    hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n,
    int k, const void* alpha, const void* const A[], hipDataType aType, int lda,
    const void* const B[], hipDataType bType, int ldb, const void* beta, void* const C[],
    hipDataType cType, int ldc, int batchCount, hipDataType computeType, hipblasGemmAlgo_t algo) {
  hipblasComputeType_t compute_type;
  if (!OfHipblasComputeType(computeType, &compute_type)) { return HIPBLAS_STATUS_INVALID_ENUM; }
  return hipblasGemmBatchedEx(handle, transA, transB, m, n, k, alpha, const_cast<const void**>(A),
                              aType, lda, const_cast<const void**>(B), bType, ldb, beta,
                              const_cast<void**>(C), cType, ldc, batchCount, compute_type, algo);
}

inline hipblasStatus_t hipblasGemmStridedBatchedEx(
    hipblasHandle_t handle, hipblasOperation_t transA, hipblasOperation_t transB, int m, int n,
    int k, const void* alpha, const void* A, hipDataType aType, int lda, hipblasStride strideA,
    const void* B, hipDataType bType, int ldb, hipblasStride strideB, const void* beta, void* C,
    hipDataType cType, int ldc, hipblasStride strideC, int batchCount, hipDataType computeType,
    hipblasGemmAlgo_t algo) {
  hipblasComputeType_t compute_type;
  if (!OfHipblasComputeType(computeType, &compute_type)) { return HIPBLAS_STATUS_INVALID_ENUM; }
  return hipblasGemmStridedBatchedEx(handle, transA, transB, m, n, k, alpha, A, aType, lda, strideA,
                                     B, bType, ldb, strideB, beta, C, cType, ldc, strideC,
                                     batchCount, compute_type, algo);
}
