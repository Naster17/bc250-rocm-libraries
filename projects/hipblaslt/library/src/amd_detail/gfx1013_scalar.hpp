// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once
#include <hipblaslt/hipblaslt.h>

namespace gfx1013_scalar
{
    bool                             is_algo(const hipblasLtMatmulAlgo_t&);
    hipblasLtMatmulHeuristicResult_t algorithm();
    hipblasStatus_t                  supported(hipblasLtHandle_t,
                                               hipblasLtMatmulDesc_t,
                                               hipblasLtMatrixLayout_t,
                                               hipblasLtMatrixLayout_t,
                                               hipblasLtMatrixLayout_t,
                                               hipblasLtMatrixLayout_t);
    hipblasStatus_t heuristic(hipblasLtHandle_t,
                              hipblasLtMatmulDesc_t,
                              hipblasLtMatrixLayout_t,
                              hipblasLtMatrixLayout_t,
                              hipblasLtMatrixLayout_t,
                              hipblasLtMatrixLayout_t,
                              hipblasLtMatmulPreference_t,
                              int,
                              hipblasLtMatmulHeuristicResult_t*,
                              int*);
    hipblasStatus_t matmul(hipblasLtHandle_t,
                           hipblasLtMatmulDesc_t,
                           const void*,
                           const void*,
                           hipblasLtMatrixLayout_t,
                           const void*,
                           hipblasLtMatrixLayout_t,
                           const void*,
                           const void*,
                           hipblasLtMatrixLayout_t,
                           void*,
                           hipblasLtMatrixLayout_t,
                           const hipblasLtMatmulAlgo_t*,
                           void*,
                           size_t,
                           hipStream_t);
}
