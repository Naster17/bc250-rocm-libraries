// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/*
simple 2d topk implementation, along row (dim=1)
requirement:
    1). each row is within a warp
*/
template <typename Problem_, typename Policy_ = void>
struct BlockTopkStream2D
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using DataType  = typename Problem::DataType;
    using IndexType = typename Problem::IndexType;

    // TODO: if DataType is subdword, need pack into single dword to use argmax
    struct ArgmaxPacket
    {
        DataType arg;
        index_t value;
    };

    template <typename DistributedTensor, typename OutWindow, typename IdxWindow, index_t dim = 1>
    CK_TILE_DEVICE void operator()(const DistributedTensor& x,
                                   const OutWindow& out_window,
                                   const IdxWindow& idx_window,
                                   index_t k,
                                   number<dim>        = {},
                                   index_t valid_cols = numeric<index_t>::max())
    {
        OutWindow out_window_tmp = out_window;
        IdxWindow idx_window_tmp = idx_window;
        static_assert(
            std::is_same_v<typename DistributedTensor::DataType, typename OutWindow::DataType> &&
            std::is_same_v<typename DistributedTensor::DataType, DataType>);
        static_assert(std::is_same_v<typename IdxWindow::DataType, IndexType>);

        auto selected           = tile_elementwise_in([](const auto&) { return index_t{0}; }, x);
        constexpr auto dst_dist = typename IdxWindow::TileDstr{};

        // argmax for topk
        const auto f_argmax = [](ArgmaxPacket e0, ArgmaxPacket e1) {
            if(e0.value == numeric<index_t>::max())
                return e1;
            if(e1.value == numeric<index_t>::max())
                return e0;
            const bool nan0 = e0.arg != e0.arg;
            const bool nan1 = e1.arg != e1.arg;
            if(nan0 != nan1)
                return nan0 ? e0 : e1;
            if(nan0 && nan1)
                return e0.value < e1.value ? e0 : e1;
            return e0.arg > e1.arg || (e0.arg == e1.arg && e0.value < e1.value) ? e0 : e1;
        };

        for(index_t i_k = 0; i_k < k; i_k++)
        {
            constexpr auto span_2d = DistributedTensor::get_distributed_spans();
            auto packet            = [&]() {
                auto tmp = make_static_distributed_tensor<ArgmaxPacket>(x.get_tile_distribution());

                sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            tmp.get_tile_distribution(), make_tuple(idx0, idx1));
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);
                        ArgmaxPacket t;
                        t.arg          = x[i_j_idx];
                        const auto col = tile_idx.at(number<1>{});
                        t.value =
                            selected(i_j_idx) || col >= valid_cols ? numeric<index_t>::max() : col;
                        tmp(i_j_idx) = t;
                    });
                });
                return tmp;
            }();

            auto argmax_init =
                ArgmaxPacket{-numeric<DataType>::infinity(), numeric<index_t>::max()};
            auto r = block_tile_reduce<ArgmaxPacket>(packet, sequence<1>{}, f_argmax, argmax_init);
            block_tile_reduce_xor_sync(r, f_argmax);

            auto o = make_static_distributed_tensor<DataType>(dst_dist);
            auto i = make_static_distributed_tensor<IndexType>(dst_dist);
            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ArgmaxPacket tmp       = r(i_j_idx);
                    o(i_j_idx)             = tmp.arg;
                    i(i_j_idx)             = tmp.value;
                });
            });

            // update value
            sweep_tile_span(span_2d[number<0>{}], [&](auto idx0) {
                sweep_tile_span(span_2d[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        x.get_tile_distribution(), make_tuple(idx0, idx1));
                    auto col_id = tile_idx.at(number<1>{});

                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    selected(i_j_idx) |= col_id == r(i_j_idx).value;
                });
            });

            if(threadIdx.x % Problem::ColLanes == 0)
            {
                store_tile(out_window_tmp, o);
                store_tile(idx_window_tmp, i);
            }
            move_tile_window(out_window_tmp, {number<0>{}, number<1>{}});
            move_tile_window(idx_window_tmp, {number<0>{}, number<1>{}});
        }
    }
};

} // namespace ck_tile
