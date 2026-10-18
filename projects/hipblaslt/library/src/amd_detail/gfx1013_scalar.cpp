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
                   || epilogue == ROCBLASLT_EPILOGUE_GELU_BIAS
                   || epilogue == ROCBLASLT_EPILOGUE_GELU_AUX_BIAS
                   || epilogue == ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT
                   || epilogue == ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT
                   || epilogue == ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT;
        }

        bool uses_aux(rocblaslt_epilogue epilogue)
        {
            return epilogue == ROCBLASLT_EPILOGUE_GELU_AUX
                   || epilogue == ROCBLASLT_EPILOGUE_GELU_AUX_BIAS
                   || epilogue == ROCBLASLT_EPILOGUE_DGELU
                   || epilogue == ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT
                   || epilogue == ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT;
        }

        bool supported_epilogue(rocblaslt_epilogue epilogue)
        {
            return epilogue == ROCBLASLT_EPILOGUE_DEFAULT || uses_bias(epilogue)
                   || epilogue == ROCBLASLT_EPILOGUE_RELU || epilogue == ROCBLASLT_EPILOGUE_GELU
                   || uses_aux(epilogue) || epilogue == ROCBLASLT_EPILOGUE_SIGMOID
                   || epilogue == ROCBLASLT_EPILOGUE_SWISH_EXT
                   || epilogue == ROCBLASLT_EPILOGUE_CLAMP_EXT;
        }

        struct Layout
        {
            uint64_t            rows, cols, ld, stride, offset, batches;
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
                    uint64_t(x.batch_count),
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
            const bool integer = a->type == HIP_R_8I;
            if(!supported_epilogue(op->epilogue) || (!uses_bias(op->epilogue) && op->bias)
               || op->scaleA || op->scaleB || op->scaleC || op->scaleD || op->scaleE || op->amaxD
               || (!uses_aux(op->epilogue) && op->e)
               || op->scaleAType != RocblasltContractionProblem::ScalingFormat::None
               || op->scaleBType != RocblasltContractionProblem::ScalingFormat::None
               || op->compute_input_typeA != HIPBLASLT_DATATYPE_INVALID
               || op->compute_input_typeB != HIPBLASLT_DATATYPE_INVALID
               || op->scale_type != (integer ? HIP_R_32I : HIP_R_32F) || op->sm_count_target
               || h->sm_count_target || op->streamk_tile_scheduling_ext
               || h->check_numerics != hipblaslt_check_numerics_mode_no_check
               || (integer ? (op->compute_type_original != rocblaslt_compute_i32)
                           : (op->compute_type_original != rocblaslt_compute_f32
                              && op->compute_type_original != rocblaslt_compute_f32_pedantic
                              && !(op->compute_type_original == rocblaslt_compute_f32_fast_f16
                                   && a->type == HIP_R_16F)
                              && !(op->compute_type_original == rocblaslt_compute_f32_fast_bf16
                                   && a->type == HIP_R_16BF)))
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
               || (integer
                       ? d->type != HIP_R_32I
                       : ((a->type != HIP_R_32F && a->type != HIP_R_16F && a->type != HIP_R_16BF)
                          || (d->type != a->type && d->type != HIP_R_32F))))
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            if(integer && op->epilogue != ROCBLASLT_EPILOGUE_DEFAULT
               && op->epilogue != ROCBLASLT_EPILOGUE_BIAS && op->epilogue != ROCBLASLT_EPILOGUE_RELU
               && op->epilogue != ROCBLASLT_EPILOGUE_RELU_BIAS)
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            if(uses_bias(op->epilogue))
            {
                if(!op->bias || op->bias_stride < 0
                   || (op->bias_stride && uint64_t(op->bias_stride) < d->m))
                    return HIPBLAS_STATUS_INVALID_VALUE;
                const auto bias_type
                    = op->bias_type == HIPBLASLT_DATATYPE_INVALID ? d->type : op->bias_type;
                if(integer ? bias_type != HIP_R_32I
                           : (bias_type != d->type && bias_type != HIP_R_32F))
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
            }
            if(integer && uses_bias(op->epilogue))
            {
                auto bias         = *d;
                bias.m            = d->m;
                bias.n            = 1;
                bias.ld           = d->m;
                bias.order        = HIPBLASLT_ORDER_COL;
                bias.batch_offset = 0;
                bias.batch_stride = op->bias_stride;
                if(!valid_span(bias, false))
                    return HIPBLAS_STATUS_INVALID_VALUE;
            }
            if(uses_aux(op->epilogue))
            {
                // Upstream auxiliary tensors are column-major, even for row-major D.
                const auto aux_type
                    = op->aux_type == HIPBLASLT_DATATYPE_INVALID ? d->type : op->aux_type;
                if(aux_type != d->type && aux_type != HIP_R_32F)
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                _rocblaslt_matrix_layout aux = *d;
                aux.order                    = HIPBLASLT_ORDER_COL;
                aux.ld                       = op->lde;
                aux.batch_stride             = op->stride_e;
                aux.batch_offset             = 0;
                if(!op->e || !valid_span(aux, op->epilogue != ROCBLASLT_EPILOGUE_DGELU))
                    return HIPBLAS_STATUS_INVALID_VALUE;
            }
            if(!uses_bias(op->epilogue) && op->bias_stride)
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

        __global__ void gemm_int8(Layout         a,
                                  Layout         b,
                                  Layout         c,
                                  Layout         d,
                                  const int8_t*  A,
                                  const int8_t*  B,
                                  const int32_t* C,
                                  int32_t*       D,
                                  uint64_t       k,
                                  bool           trans_a,
                                  bool           trans_b,
                                  int32_t        alpha_host,
                                  int32_t        beta_host,
                                  const int32_t* alpha_device,
                                  const int32_t* beta_device,
                                  const int32_t* bias,
                                  uint64_t       bias_stride,
                                  bool           relu)
        {
            __shared__ int32_t left[tile][tile];
            __shared__ int32_t right[tile][tile];
            const unsigned     x = threadIdx.x, y = threadIdx.y;
            const uint64_t     tiles_m  = (d.rows + tile - 1) / tile;
            const uint64_t     tiles_n  = (d.cols + tile - 1) / tile;
            const uint64_t     batch    = uint64_t(blockIdx.x) / (tiles_m * tiles_n);
            const uint64_t     position = uint64_t(blockIdx.x) % (tiles_m * tiles_n);
            const uint64_t     row      = (position % tiles_m) * tile + y;
            const uint64_t     col      = (position / tiles_m) * tile + x;
            const uint32_t     alpha    = uint32_t(alpha_device ? *alpha_device : alpha_host);
            const uint32_t     beta     = uint32_t(beta_device ? *beta_device : beta_host);
            uint32_t           acc      = 0;
            // Unsigned accumulation defines modulo-2^32 overflow without signed UB.
            if(alpha != 0)
                for(uint64_t base = 0; base < k; base += tile)
                {
                    const uint64_t ka = base + x, kb = base + y;
                    left[y][x]
                        = row < d.rows && ka < k
                              ? int32_t(A[a.index(trans_a ? ka : row, trans_a ? row : ka, batch)])
                              : 0;
                    right[y][x]
                        = col < d.cols && kb < k
                              ? int32_t(B[b.index(trans_b ? col : kb, trans_b ? kb : col, batch)])
                              : 0;
                    __syncthreads();
                    for(unsigned i = 0; i < tile; ++i)
                        acc += uint32_t(left[y][i] * right[i][x]);
                    __syncthreads();
                }
            if(row < d.rows && col < d.cols)
            {
                uint32_t value = alpha * acc;
                if(beta != 0)
                    value += beta * uint32_t(C[c.index(row, col, batch)]);
                if(bias)
                    value += uint32_t(bias[batch * bias_stride + row]);
                int32_t result;
                __builtin_memcpy(&result, &value, sizeof(result));
                D[d.index(row, col, batch)] = relu && result < 0 ? 0 : result;
            }
        }

        uint64_t span_bytes(const Layout& x, unsigned width)
        {
            if(!x.rows || !x.cols)
                return 0;
            const uint64_t inner = x.row_major ? x.cols : x.rows;
            const uint64_t outer = x.row_major ? x.rows : x.cols;
            return (x.offset + (x.batches - 1) * x.stride + (outer - 1) * x.ld + inner) * width;
        }

        hipblasStatus_t launch_int8(rocblaslt_matmul_desc   op,
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
            int32_t    ah = 0, bh = 0;
            if(!device)
            {
                std::memcpy(&ah, alpha, sizeof(ah));
                std::memcpy(&bh, beta, sizeof(bh));
            }
            const uint64_t k = op->op_A == HIPBLAS_OP_N ? a->n : a->m;
            if(!D || (reinterpret_cast<uintptr_t>(D) & 3) || ((device || bh != 0) && !C)
               || (k && (device || ah != 0) && (!A || !B))
               || (C && (reinterpret_cast<uintptr_t>(C) & 3))
               || (op->bias && (reinterpret_cast<uintptr_t>(op->bias) & 3))
               || (device
                   && ((reinterpret_cast<uintptr_t>(alpha) & 3)
                       || (reinterpret_cast<uintptr_t>(beta) & 3))))
                return HIPBLAS_STATUS_INVALID_VALUE;
            if(C == D
               && (c->ld != d->ld || c->batch_stride != d->batch_stride
                   || c->batch_offset != d->batch_offset))
                return HIPBLAS_STATUS_INVALID_VALUE;
            const uint64_t output_bytes = span_bytes(layout(*d), 4);
            const auto     overlap      = [&](const void* input, uint64_t bytes) {
                if(!input || !bytes || !output_bytes)
                    return false;
                const uintptr_t src = reinterpret_cast<uintptr_t>(input);
                const uintptr_t dst = reinterpret_cast<uintptr_t>(D);
                return bytes > UINTPTR_MAX - src || output_bytes > UINTPTR_MAX - dst
                       || (src < dst + output_bytes && dst < src + bytes);
            };
            const uint64_t bias_bytes
                = uses_bias(op->epilogue)
                      ? (uint64_t(d->batch_count - 1) * uint64_t(op->bias_stride) + d->m) * 4
                      : 0;
            if(overlap(A, span_bytes(layout(*a), 1)) || overlap(B, span_bytes(layout(*b), 1))
               || (C != D && overlap(C, span_bytes(layout(*c), 4))) || overlap(op->bias, bias_bytes)
               || (device && (overlap(alpha, 4) || overlap(beta, 4))))
                return HIPBLAS_STATUS_INVALID_VALUE;
            const uint64_t blocks = ((d->m + tile - 1) / tile) * ((d->n + tile - 1) / tile)
                                    * uint64_t(d->batch_count);
            if(!blocks)
                return HIPBLAS_STATUS_SUCCESS;
            gemm_int8<<<dim3(unsigned(blocks)), dim3(tile, tile), 0, stream>>>(
                layout(*a),
                layout(*b),
                layout(*c),
                layout(*d),
                static_cast<const int8_t*>(A),
                static_cast<const int8_t*>(B),
                static_cast<const int32_t*>(C),
                static_cast<int32_t*>(D),
                k,
                op->op_A == HIPBLAS_OP_T,
                op->op_B == HIPBLAS_OP_T,
                ah,
                bh,
                device ? static_cast<const int32_t*>(alpha) : nullptr,
                device ? static_cast<const int32_t*>(beta) : nullptr,
                uses_bias(op->epilogue) ? static_cast<const int32_t*>(op->bias) : nullptr,
                uint64_t(op->bias_stride),
                op->epilogue == ROCBLASLT_EPILOGUE_RELU
                    || op->epilogue == ROCBLASLT_EPILOGUE_RELU_BIAS);
            return hipGetLastError() == hipSuccess ? HIPBLAS_STATUS_SUCCESS
                                                   : HIPBLAS_STATUS_EXECUTION_FAILED;
        }

        template <typename Input, typename Output, unsigned Outputs = 1, bool Cooperative = false>
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
                             rocblaslt_epilogue epilogue,
                             void*              aux,
                             hipDataType        aux_type,
                             uint64_t           aux_ld,
                             uint64_t           aux_stride,
                             float              act0,
                             float              act1)
        {
            // A register tile reuses each LDS load for Outputs rows and columns.
            // Keep the original reduction order and FP32 FMA accumulation.
            constexpr unsigned block_tile = tile * Outputs;
            __shared__ float   left[block_tile][tile];
            __shared__ float   right[tile][block_tile];
            const unsigned     x = Cooperative ? 0 : threadIdx.x, y = Cooperative ? 0 : threadIdx.y;
            const uint64_t     tiles_m = (d.rows + block_tile - 1) / block_tile;
            const uint64_t     tiles_n = (d.cols + block_tile - 1) / block_tile;
            const uint64_t     linear
                = (uint64_t(blockIdx.x) * tile * tile + threadIdx.y * tile + threadIdx.x)
                  / (Cooperative ? 32 : 1);
            const uint64_t batch    = Cooperative ? linear / (d.rows * d.cols)
                                                  : uint64_t(blockIdx.x) / (tiles_m * tiles_n);
            const uint64_t position = uint64_t(blockIdx.x) % (tiles_m * tiles_n);
            const uint64_t output   = linear % (d.rows * d.cols);
            const uint64_t base_row = Cooperative
                                          ? (d.row_major ? output / d.cols : output % d.rows)
                                          : (position % tiles_m) * block_tile;
            const uint64_t base_col = Cooperative
                                          ? (d.row_major ? output % d.cols : output / d.rows)
                                          : (position / tiles_m) * block_tile;
            const float    alpha    = alpha_device ? *alpha_device : alpha_host;
            const float    beta     = beta_device ? *beta_device : beta_host;
            float          accum[Outputs][Outputs] = {};
            // Choose load lanes independently from output lanes so either storage
            // order can read contiguous input values. All threads still reach barriers.
            const bool a_rows = Outputs == 1 || a.row_major != trans_a;
            const bool b_rows = Outputs == 1 || b.row_major != trans_b;
            if constexpr(Cooperative)
            {
                // One 32-lane subgroup computes each short dot product. More
                // independent blocks than a mostly empty 16x16 output tile;
                // no LDS/barriers. Width32 also isolates halves of wave64.
                if(linear >= d.rows * d.cols * d.batches)
                    return;
                const unsigned lane = (threadIdx.y * tile + threadIdx.x) & 31;
                if(alpha != 0)
                    for(uint64_t i = lane; i < k; i += 32)
                        accum[0][0] = __builtin_fmaf(
                            float(
                                A[a.index(trans_a ? i : base_row, trans_a ? base_row : i, batch)]),
                            float(
                                B[b.index(trans_b ? base_col : i, trans_b ? i : base_col, batch)]),
                            accum[0][0]);
                // FP32 tree reduction changes rounding versus ascending-K FMA.
                // It remains ordinary scalar ALU work, without matrix engines.
                for(unsigned distance = 16; distance; distance /= 2)
                    accum[0][0] += __shfl_down(accum[0][0], distance, 32);
                if(lane != 0)
                    return;
            }
            else if(alpha != 0)
                for(uint64_t start = 0; start < k; start += tile)
                {
                    const unsigned ar = a_rows ? y : x, ac = a_rows ? x : y;
                    const unsigned br = b_rows ? y : x, bc = b_rows ? x : y;
#pragma unroll
                    for(unsigned part = 0; part < Outputs; ++part)
                    {
                        const uint64_t row = base_row + part * tile + ar;
                        const uint64_t col = base_col + part * tile + bc;
                        const uint64_t ak = start + ac, bk = start + br;
                        left[part * tile + ar][ac]
                            = row < d.rows && ak < k
                                  ? float(A[a.index(trans_a ? ak : row, trans_a ? row : ak, batch)])
                                  : 0;
                        right[br][part * tile + bc]
                            = col < d.cols && bk < k
                                  ? float(B[b.index(trans_b ? col : bk, trans_b ? bk : col, batch)])
                                  : 0;
                    }
                    __syncthreads();
                    for(unsigned i = 0; i < tile && start + i < k; ++i)
                    {
#pragma unroll
                        for(unsigned r = 0; r < Outputs; ++r)
#pragma unroll
                            for(unsigned c = 0; c < Outputs; ++c)
                                accum[r][c] = __builtin_fmaf(
                                    left[r * tile + y][i], right[i][c * tile + x], accum[r][c]);
                    }
                    __syncthreads();
                }
#pragma unroll
            for(unsigned r = 0; r < Outputs; ++r)
#pragma unroll
                for(unsigned cpart = 0; cpart < Outputs; ++cpart)
                {
                    const uint64_t row = base_row + r * tile + y;
                    const uint64_t col = base_col + cpart * tile + x;
                    const float    acc = accum[r][cpart];
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
                        const uint64_t aux_index = batch * aux_stride + col * aux_ld + row;
                        if(epilogue == ROCBLASLT_EPILOGUE_GELU_AUX
                           || epilogue == ROCBLASLT_EPILOGUE_GELU_AUX_BIAS
                           || epilogue == ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT
                           || epilogue == ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT)
                        {
                            if(aux_type == HIP_R_32F)
                                static_cast<float*>(aux)[aux_index] = value;
                            else if(aux_type == HIP_R_16F)
                                static_cast<__half*>(aux)[aux_index] = __half(value);
                            else
                                static_cast<hip_bfloat16*>(aux)[aux_index] = hip_bfloat16(value);
                        }
                        if(epilogue == ROCBLASLT_EPILOGUE_CLAMP_EXT
                           || epilogue == ROCBLASLT_EPILOGUE_CLAMP_BIAS_EXT
                           || epilogue == ROCBLASLT_EPILOGUE_CLAMP_AUX_EXT
                           || epilogue == ROCBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT)
                            value = fmaxf(act0, fminf(value, act1));
                        else if(epilogue == ROCBLASLT_EPILOGUE_DGELU)
                        {
                            const float x = aux_type == HIP_R_32F
                                                ? static_cast<float*>(aux)[aux_index]
                                            : aux_type == HIP_R_16F
                                                ? float(static_cast<__half*>(aux)[aux_index])
                                                : float(static_cast<hip_bfloat16*>(aux)[aux_index]);
                            const float t
                                = tanhf(0.7978845608028654f * x * (1 + 0.044715f * x * x));
                            value *= 0.5f * (1 + t)
                                     + 0.3989422804014327f * x * (1 + 0.134145f * x * x)
                                           * (1 - t * t);
                        }
                        else if(epilogue == ROCBLASLT_EPILOGUE_SIGMOID
                                || epilogue == ROCBLASLT_EPILOGUE_SWISH_EXT
                                || epilogue == ROCBLASLT_EPILOGUE_SWISH_BIAS_EXT)
                        {
                            // Stable sigmoid avoids infinity/overflow for large negative inputs.
                            // The public SWISH_EXT epilogue is Swish(x, 1). Activation
                            // arguments parameterize CLAMP; they do not change this beta.
                            const float argument  = value;
                            const float exp_value = expf(-fabsf(argument));
                            const float sigmoid
                                = argument >= 0 ? 1 / (1 + exp_value) : exp_value / (1 + exp_value);
                            value = epilogue == ROCBLASLT_EPILOGUE_SIGMOID ? sigmoid
                                                                           : value * sigmoid;
                        }
                        else if(epilogue == ROCBLASLT_EPILOGUE_RELU
                                || epilogue == ROCBLASLT_EPILOGUE_RELU_BIAS)
                            value = fmaxf(value, 0.0f);
                        else if(epilogue == ROCBLASLT_EPILOGUE_GELU
                                || epilogue == ROCBLASLT_EPILOGUE_GELU_BIAS
                                || epilogue == ROCBLASLT_EPILOGUE_GELU_AUX
                                || epilogue == ROCBLASLT_EPILOGUE_GELU_AUX_BIAS)
                        {
                            // Match the tanh approximation used by upstream activation kernels.
                            const float inner
                                = 0.7978845608028654f * value * (1.0f + 0.044715f * value * value);
                            value = 0.5f * value * (1.0f + tanhf(inner));
                        }
                        D[d.index(row, col, batch)] = Output(value);
                    }
                }
        }

        template <typename Input, typename Output, unsigned Outputs = 1, bool Cooperative = false>
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
            if(D == A || D == B || (op->bias && D == op->bias)
               || (op->e
                   && (op->e == D || op->e == A || op->e == B || op->e == C || op->e == op->bias)))
                return HIPBLAS_STATUS_INVALID_VALUE;
            if(C == D
               && (c->ld != d->ld || c->batch_stride != d->batch_stride
                   || c->batch_offset != d->batch_offset))
                return HIPBLAS_STATUS_INVALID_VALUE;
            if constexpr(Outputs == 1 && !Cooperative)
            {
                // The eight-output subgroup grid can be larger than the tiled
                // grid bounded by validate(). Preserve that API bound even for
                // descriptors too large for this board's physical memory.
                const uint64_t subgroup_blocks = (d->m * d->n * uint64_t(d->batch_count) + 7) / 8;
                if(((d->m <= 32 && d->n <= 32 && d->m * d->n <= 512) || d->m <= 4 || d->n <= 4)
                   && k <= 64 && subgroup_blocks <= uint64_t(INT_MAX))
                    return launch<Input, Output, 1, true>(
                        op, a, b, c, d, alpha, A, B, beta, C, D, stream);
                // Small/skinny problems retain the finer grid; larger matrices reuse
                // four outputs per thread without changing the public algorithm ID.
                if(d->m >= 128 && d->n >= 128)
                    return launch<Input, Output, 2>(
                        op, a, b, c, d, alpha, A, B, beta, C, D, stream);
            }
            constexpr unsigned block_tile = tile * Outputs;
            const uint64_t blocks = Cooperative ? (d->m * d->n * uint64_t(d->batch_count) + 7) / 8
                                                : ((d->m + block_tile - 1) / block_tile)
                                                      * ((d->n + block_tile - 1) / block_tile)
                                                      * uint64_t(d->batch_count);
            if(!blocks)
                return HIPBLAS_STATUS_SUCCESS;
            gemm<Input, Output, Outputs, Cooperative>
                <<<dim3(unsigned(blocks)), dim3(tile, tile), 0, stream>>>(
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
                    op->epilogue,
                    op->e,
                    op->aux_type == HIPBLASLT_DATATYPE_INVALID ? d->type : op->aux_type,
                    uint64_t(op->lde),
                    uint64_t(op->stride_e),
                    op->act0,
                    op->act1);
            return hipGetLastError() == hipSuccess ? HIPBLAS_STATUS_SUCCESS
                                                   : HIPBLAS_STATUS_EXECUTION_FAILED;
        }
    }

    bool is_algo(const hipblasLtMatmulAlgo_t& algo)
    {
        return !std::memcmp(algo.data, algorithm_id, sizeof(algorithm_id))
               && !algo.max_workspace_bytes;
    }

    hipblasLtMatmulHeuristicResult_t algorithm()
    {
        hipblasLtMatmulHeuristicResult_t result{};
        std::memcpy(result.algo.data, algorithm_id, sizeof(algorithm_id));
        result.state = HIPBLAS_STATUS_SUCCESS;
        return result;
    }

    hipblasStatus_t supported(hipblasLtHandle_t       handle,
                              hipblasLtMatmulDesc_t   desc,
                              hipblasLtMatrixLayout_t a,
                              hipblasLtMatrixLayout_t b,
                              hipblasLtMatrixLayout_t c,
                              hipblasLtMatrixLayout_t d)
    {
        return validate(reinterpret_cast<rocblaslt_handle>(handle),
                        reinterpret_cast<rocblaslt_matmul_desc>(desc),
                        reinterpret_cast<rocblaslt_matrix_layout>(a),
                        reinterpret_cast<rocblaslt_matrix_layout>(b),
                        reinterpret_cast<rocblaslt_matrix_layout>(c ? c : d),
                        reinterpret_cast<rocblaslt_matrix_layout>(d));
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
        if(a->type == HIP_R_8I)
            return launch_int8(op, a, b, c, d, alpha, A, B, beta, C, D, stream);
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
