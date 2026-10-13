// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gfx1013_scalar.hpp"
#include "handle.h"
#include <climits>
#include <cstring>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>

namespace gfx1013_scalar
{
    namespace
    {
        constexpr unsigned tile = 16;
        constexpr uint8_t  algorithm_id[16]
            = {'g', 'f', 'x', '1', '0', '1', '3', 's', 'c', 'a', 'l', 'a', 'r', 1, 0, 0};

        bool uses_bias(rocblaslt_epilogue epilogue)
        {
            return epilogue == ROCBLASLT_EPILOGUE_BIAS || epilogue == ROCBLASLT_EPILOGUE_RELU_BIAS
                   || epilogue == ROCBLASLT_EPILOGUE_GELU_BIAS;
        }

        bool supported_epilogue(rocblaslt_epilogue epilogue)
        {
            return epilogue == ROCBLASLT_EPILOGUE_DEFAULT || uses_bias(epilogue)
                   || epilogue == ROCBLASLT_EPILOGUE_RELU || epilogue == ROCBLASLT_EPILOGUE_GELU;
        }

        struct Layout
        {
            uint64_t            rows, cols, ld, stride, offset;
            bool                row_major;
            __device__ uint64_t index(uint64_t row, uint64_t col, uint64_t batch) const
            {
                return offset + batch * stride + (row_major ? row * ld + col : col * ld + row);
            }
        };

        Layout layout(const _rocblaslt_matrix_layout& x)
        {
            return {x.m,
                    x.n,
                    uint64_t(x.ld),
                    uint64_t(x.batch_stride),
                    uint64_t(x.batch_offset),
                    x.order == HIPBLASLT_ORDER_ROW};
        }

        // Bound the last byte addressed, including batch and per-matrix padding.
        bool valid_span(const _rocblaslt_matrix_layout& x, bool output)
        {
            if(x.m > INT_MAX || x.n > INT_MAX || x.ld < 0 || x.ld > INT_MAX || x.batch_count < 1
               || x.batch_stride < 0 || x.batch_offset < 0)
                return false;
            const bool     row   = x.order == HIPBLASLT_ORDER_ROW;
            const uint64_t inner = row ? x.n : x.m, outer = row ? x.m : x.n;
            if(x.ld < int64_t(inner))
                return false;
            const uint64_t span = outer && inner ? (outer - 1) * uint64_t(x.ld) + inner : 0;
            if(output && x.batch_count > 1 && uint64_t(x.batch_stride) < span)
                return false;
            const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / 4;
            if(uint64_t(x.batch_offset) > limit || span > limit - uint64_t(x.batch_offset))
                return false;
            return !x.batch_stride
                   || uint64_t(x.batch_count - 1)
                          <= (limit - uint64_t(x.batch_offset) - span) / uint64_t(x.batch_stride);
        }

        hipblasStatus_t validate(rocblaslt_handle        h,
                                 rocblaslt_matmul_desc   op,
                                 rocblaslt_matrix_layout a,
                                 rocblaslt_matrix_layout b,
                                 rocblaslt_matrix_layout c,
                                 rocblaslt_matrix_layout d)
        {
            if(!h)
                return HIPBLAS_STATUS_NOT_INITIALIZED;
            if(!op || !a || !b || !c || !d)
                return HIPBLAS_STATUS_INVALID_VALUE;
            int device = -1;
            if(hipGetDevice(&device) != hipSuccess)
                return HIPBLAS_STATUS_EXECUTION_FAILED;
            if(device != h->device || std::strncmp(h->properties.gcnArchName, "gfx1013", 7)
               || (h->properties.gcnArchName[7] && h->properties.gcnArchName[7] != ':'))
                return HIPBLAS_STATUS_ARCH_MISMATCH;
            if(!supported_epilogue(op->epilogue) || (!uses_bias(op->epilogue) && op->bias)
               || op->scaleA || op->scaleB || op->scaleC || op->scaleD || op->scaleE || op->amaxD
               || op->e || op->scaleAType != RocblasltContractionProblem::ScalingFormat::None
               || op->scaleBType != RocblasltContractionProblem::ScalingFormat::None
               || op->compute_input_typeA != HIPBLASLT_DATATYPE_INVALID
               || op->compute_input_typeB != HIPBLASLT_DATATYPE_INVALID
               || op->scale_type != HIP_R_32F || op->sm_count_target || h->sm_count_target
               || op->streamk_tile_scheduling_ext
               || h->check_numerics != hipblaslt_check_numerics_mode_no_check
               || (op->compute_type_original != rocblaslt_compute_f32
                   && op->compute_type_original != rocblaslt_compute_f32_pedantic)
               || (op->pointermode != rocblaslt_pointer_mode_host
                   && op->pointermode != rocblaslt_pointer_mode_device))
                return HIPBLAS_STATUS_NOT_SUPPORTED;
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
            if(op->fused_epilogue)
                return HIPBLAS_STATUS_NOT_SUPPORTED;
#endif
            if((op->op_A != HIPBLAS_OP_N && op->op_A != HIPBLAS_OP_T)
               || (op->op_B != HIPBLAS_OP_N && op->op_B != HIPBLAS_OP_T))
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            if(a->type != b->type || c->type != d->type
               || (a->type != HIP_R_32F && a->type != HIP_R_16F && a->type != HIP_R_16BF)
               || (d->type != a->type && d->type != HIP_R_32F))
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            if(uses_bias(op->epilogue))
            {
                if(!op->bias || op->bias_stride < 0
                   || (op->bias_stride && uint64_t(op->bias_stride) < d->m))
                    return HIPBLAS_STATUS_INVALID_VALUE;
                const auto bias_type
                    = op->bias_type == HIPBLASLT_DATATYPE_INVALID ? d->type : op->bias_type;
                if(bias_type != d->type && bias_type != HIP_R_32F)
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
            }
            else if(op->bias_stride)
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            for(auto x : {a, b, c, d})
            {
                if((x->order != HIPBLASLT_ORDER_COL && x->order != HIPBLASLT_ORDER_ROW)
                   || x->batch_mode != HIPBLASLT_BATCH_MODE_STRIDED)
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                if(!valid_span(*x, x == d))
                    return HIPBLAS_STATUS_INVALID_VALUE;
            }
            const uint64_t m  = op->op_A == HIPBLAS_OP_N ? a->m : a->n;
            const uint64_t k  = op->op_A == HIPBLAS_OP_N ? a->n : a->m;
            const uint64_t bk = op->op_B == HIPBLAS_OP_N ? b->m : b->n;
            const uint64_t n  = op->op_B == HIPBLAS_OP_N ? b->n : b->m;
            if(k != bk || c->m != m || d->m != m || c->n != n || d->n != n || c->order != d->order
               || a->batch_count != d->batch_count || b->batch_count != d->batch_count
               || c->batch_count != d->batch_count)
                return HIPBLAS_STATUS_INVALID_VALUE;
            const uint64_t blocks = ((m + tile - 1) / tile) * ((n + tile - 1) / tile);
            if(blocks > uint64_t(INT_MAX) / uint64_t(d->batch_count))
                return HIPBLAS_STATUS_INVALID_VALUE;
            return HIPBLAS_STATUS_SUCCESS;
        }

        template <typename Input, typename Output>
        __global__ void gemm(Layout             a,
                             Layout             b,
                             Layout             c,
                             Layout             d,
                             const Input*       A,
                             const Input*       B,
                             const Output*      C,
                             Output*            D,
                             uint64_t           k,
                             bool               trans_a,
                             bool               trans_b,
                             float              alpha_host,
                             float              beta_host,
                             const float*       alpha_device,
                             const float*       beta_device,
                             const void*        bias,
                             hipDataType        bias_type,
                             uint64_t           bias_stride,
                             rocblaslt_epilogue epilogue)
        {
            __shared__ float left[tile][tile];
            __shared__ float right[tile][tile];
            const unsigned   x = threadIdx.x, y = threadIdx.y;
            const uint64_t   tiles_m  = (d.rows + tile - 1) / tile;
            const uint64_t   tiles_n  = (d.cols + tile - 1) / tile;
            const uint64_t   batch    = uint64_t(blockIdx.x) / (tiles_m * tiles_n);
            const uint64_t   position = uint64_t(blockIdx.x) % (tiles_m * tiles_n);
            const uint64_t   row      = (position % tiles_m) * tile + y;
            const uint64_t   col      = (position / tiles_m) * tile + x;
            const float      alpha    = alpha_device ? *alpha_device : alpha_host;
            const float      beta     = beta_device ? *beta_device : beta_host;
            float            acc      = 0;
            // Scalar arithmetic only: gfx1013 does not have MFMA or WMMA instructions.
            if(alpha != 0)
                for(uint64_t start = 0; start < k; start += tile)
                {
                    const uint64_t ak = start + x, bk = start + y;
                    left[y][x]
                        = row < d.rows && ak < k
                              ? float(A[a.index(trans_a ? ak : row, trans_a ? row : ak, batch)])
                              : 0;
                    right[y][x]
                        = col < d.cols && bk < k
                              ? float(B[b.index(trans_b ? col : bk, trans_b ? bk : col, batch)])
                              : 0;
                    __syncthreads();
                    for(unsigned i = 0; i < tile && start + i < k; ++i)
                        acc = __builtin_fmaf(left[y][i], right[i][x], acc);
                    __syncthreads();
                }
            if(row < d.rows && col < d.cols)
            {
                // BLAS beta=0 must not read C: uninitialised/NaN C is valid in this case.
                const float value_c = beta == 0 ? 0 : float(C[c.index(row, col, batch)]);
                float       value   = __builtin_fmaf(alpha, acc, beta * value_c);
                if(bias)
                {
                    // Bias is indexed by logical D row, independent of storage order.
                    const uint64_t index = batch * bias_stride + row;
                    if(bias_type == HIP_R_32F)
                        value += static_cast<const float*>(bias)[index];
                    else if(bias_type == HIP_R_16F)
                        value += float(static_cast<const __half*>(bias)[index]);
                    else
                        value += float(static_cast<const hip_bfloat16*>(bias)[index]);
                }
                if(epilogue == ROCBLASLT_EPILOGUE_RELU || epilogue == ROCBLASLT_EPILOGUE_RELU_BIAS)
                    value = fmaxf(value, 0.0f);
                else if(epilogue == ROCBLASLT_EPILOGUE_GELU
                        || epilogue == ROCBLASLT_EPILOGUE_GELU_BIAS)
                {
                    // Match the tanh approximation used by upstream activation kernels.
                    const float inner
                        = 0.7978845608028654f * value * (1.0f + 0.044715f * value * value);
                    value = 0.5f * value * (1.0f + tanhf(inner));
                }
                D[d.index(row, col, batch)] = Output(value);
            }
        }

        template <typename Input, typename Output>
        hipblasStatus_t launch(rocblaslt_matmul_desc   op,
                               rocblaslt_matrix_layout a,
                               rocblaslt_matrix_layout b,
                               rocblaslt_matrix_layout c,
                               rocblaslt_matrix_layout d,
                               const void*             alpha,
                               const void*             A,
                               const void*             B,
                               const void*             beta,
                               const void*             C,
                               void*                   D,
                               hipStream_t             stream)
        {
            const bool device = op->pointermode == rocblaslt_pointer_mode_device;
            float      ah = 0, bh = 0;
            if(!device)
            {
                std::memcpy(&ah, alpha, sizeof(ah));
                std::memcpy(&bh, beta, sizeof(bh));
            }
            const uint64_t k = op->op_A == HIPBLAS_OP_N ? a->n : a->m;
            if(!D || ((device || bh != 0) && !C) || (k && (device || ah != 0) && (!A || !B)))
                return HIPBLAS_STATUS_INVALID_VALUE;
            if(D == A || D == B || (op->bias && D == op->bias))
                return HIPBLAS_STATUS_INVALID_VALUE;
            if(C == D
               && (c->ld != d->ld || c->batch_stride != d->batch_stride
                   || c->batch_offset != d->batch_offset))
                return HIPBLAS_STATUS_INVALID_VALUE;
            const uint64_t blocks = ((d->m + tile - 1) / tile) * ((d->n + tile - 1) / tile)
                                    * uint64_t(d->batch_count);
            if(!blocks)
                return HIPBLAS_STATUS_SUCCESS;
            gemm<Input, Output><<<dim3(unsigned(blocks)), dim3(tile, tile), 0, stream>>>(
                layout(*a),
                layout(*b),
                layout(*c),
                layout(*d),
                static_cast<const Input*>(A),
                static_cast<const Input*>(B),
                static_cast<const Output*>(C),
                static_cast<Output*>(D),
                k,
                op->op_A == HIPBLAS_OP_T,
                op->op_B == HIPBLAS_OP_T,
                ah,
                bh,
                device ? static_cast<const float*>(alpha) : nullptr,
                device ? static_cast<const float*>(beta) : nullptr,
                uses_bias(op->epilogue) ? op->bias : nullptr,
                op->bias_type == HIPBLASLT_DATATYPE_INVALID ? d->type : op->bias_type,
                uint64_t(op->bias_stride),
                op->epilogue);
            return hipGetLastError() == hipSuccess ? HIPBLAS_STATUS_SUCCESS
                                                   : HIPBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    hipblasStatus_t heuristic(hipblasLtHandle_t                 handle,
                              hipblasLtMatmulDesc_t             desc,
                              hipblasLtMatrixLayout_t           A,
                              hipblasLtMatrixLayout_t           B,
                              hipblasLtMatrixLayout_t           C,
                              hipblasLtMatrixLayout_t           D,
                              hipblasLtMatmulPreference_t       preference,
                              int                               requested,
                              hipblasLtMatmulHeuristicResult_t* results,
                              int*                              count)
    {
        if(!count)
            return HIPBLAS_STATUS_INVALID_VALUE;
        *count = 0;
        if(!preference || !results || requested < 1)
            return HIPBLAS_STATUS_INVALID_VALUE;
        const auto pref = reinterpret_cast<rocblaslt_matmul_preference>(preference);
        if(pref->search_mode || pref->sm_count_target || pref->alg_config_id || pref->alg_max_id
           || pref->search_iterations)
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        auto status = validate(reinterpret_cast<rocblaslt_handle>(handle),
                               reinterpret_cast<rocblaslt_matmul_desc>(desc),
                               reinterpret_cast<rocblaslt_matrix_layout>(A),
                               reinterpret_cast<rocblaslt_matrix_layout>(B),
                               reinterpret_cast<rocblaslt_matrix_layout>(C ? C : D),
                               reinterpret_cast<rocblaslt_matrix_layout>(D));
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;
        results[0] = {};
        std::memcpy(results[0].algo.data, algorithm_id, sizeof(algorithm_id));
        results[0].state = HIPBLAS_STATUS_SUCCESS;
        *count           = 1;
        return HIPBLAS_STATUS_SUCCESS;
    }

    hipblasStatus_t matmul(hipblasLtHandle_t            handle,
                           hipblasLtMatmulDesc_t        desc,
                           const void*                  alpha,
                           const void*                  A,
                           hipblasLtMatrixLayout_t      Adesc,
                           const void*                  B,
                           hipblasLtMatrixLayout_t      Bdesc,
                           const void*                  beta,
                           const void*                  C,
                           hipblasLtMatrixLayout_t      Cdesc,
                           void*                        D,
                           hipblasLtMatrixLayout_t      Ddesc,
                           const hipblasLtMatmulAlgo_t* algo,
                           void*                        workspace,
                           size_t                       workspace_bytes,
                           hipStream_t                  stream)
    {
        auto op     = reinterpret_cast<rocblaslt_matmul_desc>(desc);
        auto a      = reinterpret_cast<rocblaslt_matrix_layout>(Adesc);
        auto b      = reinterpret_cast<rocblaslt_matrix_layout>(Bdesc);
        auto c      = reinterpret_cast<rocblaslt_matrix_layout>(Cdesc ? Cdesc : Ddesc);
        auto d      = reinterpret_cast<rocblaslt_matrix_layout>(Ddesc);
        auto status = validate(reinterpret_cast<rocblaslt_handle>(handle), op, a, b, c, d);
        if(status != HIPBLAS_STATUS_SUCCESS)
            return status;
        if(!alpha || !beta || workspace_bytes > UINT32_MAX
           || (workspace_bytes && (!workspace || (reinterpret_cast<uintptr_t>(workspace) & 15)))
           || (algo
               && (std::memcmp(algo->data, algorithm_id, sizeof(algorithm_id))
                   || algo->max_workspace_bytes)))
            return HIPBLAS_STATUS_INVALID_VALUE;
        if(a->type == HIP_R_32F)
            return launch<float, float>(op, a, b, c, d, alpha, A, B, beta, C, D, stream);
        if(a->type == HIP_R_16F)
            return d->type == HIP_R_32F
                       ? launch<__half, float>(op, a, b, c, d, alpha, A, B, beta, C, D, stream)
                       : launch<__half, __half>(op, a, b, c, d, alpha, A, B, beta, C, D, stream);
        return d->type == HIP_R_32F
                   ? launch<hip_bfloat16, float>(op, a, b, c, d, alpha, A, B, beta, C, D, stream)
                   : launch<hip_bfloat16, hip_bfloat16>(
                         op, a, b, c, d, alpha, A, B, beta, C, D, stream);
    }
}
