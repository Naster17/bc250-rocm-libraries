// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"
#include <cstdint>
#include <string>
#include <type_traits>

namespace ck_tile {

// Explicit scalar-FMA kernel for devices without a supported matrix instruction.
// This is an independent kernel, not an alias for an MFMA/WMMA pipeline. Tensor
// strides describe logical A[M,K], B[K,N], and C[M,N]; A/B may broadcast batches.
template <typename InputType, typename OutputType = InputType>
struct ScalarGemmKernel
{
    static_assert(std::is_same_v<InputType, float> || std::is_same_v<InputType, fp16_t> ||
                  std::is_same_v<InputType, bf16_t>);
    static_assert(std::is_same_v<OutputType, float> || std::is_same_v<OutputType, InputType>);
    static constexpr index_t Tile       = 16;
    static constexpr index_t kBlockSize = Tile * Tile;

    struct Kargs
    {
        const InputType* a;
        const InputType* b;
        OutputType* c;
        index_t m, n, k, batches;
        long_index_t a_row, a_col, b_row, b_col, c_row, c_col;
        long_index_t a_batch, b_batch, c_batch;
        float alpha, beta;
    };
    using Hargs = Kargs;

    CK_TILE_HOST static constexpr bool IsSupportedArgument(const Hargs& a)
    {
        const auto span = [](index_t rows, index_t cols, long_index_t rs, long_index_t cs) {
            return static_cast<long_index_t>(rows - 1) * rs +
                   static_cast<long_index_t>(cols - 1) * cs + 1;
        };
        const auto valid_layout = [](index_t rows, index_t cols, long_index_t rs, long_index_t cs) {
            // Prevent multiplication overflow and exclude overlapping matrices.
            constexpr long_index_t limit = 1LL << 29;
            return rs > 0 && cs > 0 && rs <= limit / rows && cs <= limit / cols &&
                   ((cs == 1 && rs >= cols) || (rs == 1 && cs >= rows));
        };
        if(a.m <= 0 || a.n <= 0 || a.k < 0 || a.batches <= 0 || a.batches > 65535 || !a.c ||
           a.a_batch < 0 || a.b_batch < 0 || a.c_batch < 0 || a.a_batch > (1LL << 40) / a.batches ||
           a.b_batch > (1LL << 40) / a.batches || a.c_batch > (1LL << 40) / a.batches ||
           !valid_layout(a.m, a.n, a.c_row, a.c_col))
            return false;
        if(a.batches > 1 && a.c_batch < span(a.m, a.n, a.c_row, a.c_col))
            return false;
        if(a.alpha == 0.f || a.k == 0)
            return true;
        if(!a.a || !a.b || !valid_layout(a.m, a.k, a.a_row, a.a_col) ||
           !valid_layout(a.k, a.n, a.b_row, a.b_col))
            return false;
        if((a.a_batch != 0 && a.a_batch < span(a.m, a.k, a.a_row, a.a_col)) ||
           (a.b_batch != 0 && a.b_batch < span(a.k, a.n, a.b_row, a.b_col)))
            return false;
        const auto overlaps_c = [&](const InputType* input,
                                    long_index_t batch_stride,
                                    long_index_t input_span) {
            const auto c_begin = reinterpret_cast<std::uintptr_t>(a.c);
            const auto x_begin = reinterpret_cast<std::uintptr_t>(input);
            const auto c_bytes = static_cast<std::uintptr_t>((a.batches - 1) * a.c_batch +
                                                             span(a.m, a.n, a.c_row, a.c_col)) *
                                 sizeof(OutputType);
            const auto x_bytes =
                static_cast<std::uintptr_t>((a.batches - 1) * batch_stride + input_span) *
                sizeof(InputType);
            // Difference comparisons avoid address-addition overflow.
            return c_begin <= x_begin ? x_begin - c_begin < c_bytes : c_begin - x_begin < x_bytes;
        };
        return !overlaps_c(a.a, a.a_batch, span(a.m, a.k, a.a_row, a.a_col)) &&
               !overlaps_c(a.b, a.b_batch, span(a.k, a.n, a.b_row, a.b_col));
    }

    CK_TILE_HOST static constexpr auto GridSize(const Hargs& a)
    { return dim3(integer_divide_ceil(a.n, Tile), integer_divide_ceil(a.m, Tile), a.batches); }
    CK_TILE_HOST_DEVICE static constexpr index_t BlockSize() { return kBlockSize; }
    CK_TILE_HOST static constexpr Kargs MakeKargs(const Hargs& a) { return a; }
    CK_TILE_HOST static std::string GetName() { return "scalar_gemm_16x16_f32acc"; }

    CK_TILE_DEVICE void operator()(Kargs a) const
    {
        const index_t lr         = get_thread_id() / Tile;
        const index_t lc         = get_thread_id() % Tile;
        const index_t row        = blockIdx.y * Tile + lr;
        const index_t col        = blockIdx.x * Tile + lc;
        const long_index_t batch = blockIdx.z;
        __shared__ float left[Tile * Tile];
        __shared__ float right[Tile * Tile];
        float acc = 0.f;
        if(a.alpha != 0.f && a.k != 0)
        {
            // Scalar buffer accesses preserve odd offsets/strides and tail safety.
            auto av = make_naive_tensor_view<address_space_enum::global,
                                             memory_operation_enum::set,
                                             amd_buffer_coherence_enum::coherence_default,
                                             true>(
                a.a + batch * a.a_batch,
                make_tuple(a.m, a.k),
                make_tuple(static_cast<index_t>(a.a_row), static_cast<index_t>(a.a_col)),
                number<1>{},
                number<1>{});
            auto bv = make_naive_tensor_view<address_space_enum::global,
                                             memory_operation_enum::set,
                                             amd_buffer_coherence_enum::coherence_default,
                                             true>(
                a.b + batch * a.b_batch,
                make_tuple(a.k, a.n),
                make_tuple(static_cast<index_t>(a.b_row), static_cast<index_t>(a.b_col)),
                number<1>{},
                number<1>{});
            for(index_t start = 0; start < a.k; start += Tile)
            {
                const auto ac = make_tensor_coordinate<true>(av.get_tensor_descriptor(),
                                                             make_array(row, start + lc));
                const auto bc = make_tensor_coordinate<true>(bv.get_tensor_descriptor(),
                                                             make_array(start + lr, col));
                left[get_thread_id()] =
                    type_convert<float>(av.template get_vectorized_elements<InputType>(
                        ac, long_index_t{0}, row < a.m && start + lc < a.k));
                right[get_thread_id()] =
                    type_convert<float>(bv.template get_vectorized_elements<InputType>(
                        bc, long_index_t{0}, col < a.n && start + lr < a.k));
                block_sync_lds();
                for(index_t kk = 0; kk < Tile; ++kk)
                    acc = __builtin_fmaf(left[lr * Tile + kk], right[kk * Tile + lc], acc);
                block_sync_lds();
            }
        }
        if(row < a.m && col < a.n)
        {
            const long_index_t offset = batch * a.c_batch + row * a.c_row + col * a.c_col;
            float value               = a.alpha * acc;
            if(a.beta != 0.f)
                value = __builtin_fmaf(a.beta, type_convert<float>(a.c[offset]), value);
            a.c[offset] = type_convert<OutputType>(value);
        }
    }
};

} // namespace ck_tile
