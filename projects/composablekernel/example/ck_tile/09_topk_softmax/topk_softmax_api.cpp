// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "topk_softmax_api.hpp"

// Identity exposes the same streaming top-k primitive without an activation.
template <typename Input, bool Softmax, bool ApplyActivation>
float dispatch_topk(topk_softmax_kargs a, ck_tile::stream_config s)
{
    const auto launch = [&]<ck_tile::index_t Experts>() {
        using Problem = ck_tile::TopkSoftmaxWarpPerRowProblem<Input,
                                                              float,
                                                              ck_tile::index_t,
                                                              Experts,
                                                              Softmax,
                                                              2,
                                                              sizeof(Input),
                                                              0,
                                                              256,
                                                              ApplyActivation>;
        using Kernel  = ck_tile::TopkSoftmaxKernel<ck_tile::TopkSoftmaxWarpPerRowPipeline<Problem>>;
        return ck_tile::launch_kernel(
            s,
            ck_tile::make_kernel<1>(
                Kernel{}, Kernel::GridSize(a), Kernel::BlockSize(), 0, Kernel::MakeKargs(a)));
    };
    if(a.num_experts <= 8)
        return launch.template operator()<8>();
    if(a.num_experts <= 16)
        return launch.template operator()<16>();
    if(a.num_experts <= 32)
        return launch.template operator()<32>();
    if(a.num_experts <= 64)
        return launch.template operator()<64>();
    if(a.num_experts <= 128)
        return launch.template operator()<128>();
    if(a.num_experts <= 192)
        return launch.template operator()<192>();
    return -1;
}

template <typename Input>
float dispatch_activation(const topk_softmax_trait& t,
                          topk_softmax_kargs a,
                          ck_tile::stream_config s)
{
    if(t.activation == "softmax")
        return dispatch_topk<Input, true, true>(a, s);
    if(t.activation == "sigmoid")
        return dispatch_topk<Input, false, true>(a, s);
    if(t.activation == "identity")
        return dispatch_topk<Input, false, false>(a, s);
    return -1;
}

float topk_softmax(topk_softmax_trait t, topk_softmax_kargs a, ck_tile::stream_config s)
{
    if(t.weight_type != "fp32" || t.experts != a.num_experts || a.num_rows <= 0 ||
       a.num_experts <= 0 || a.topk <= 0 || a.topk > a.num_experts ||
       a.stride_input < a.num_experts || a.stride_output < a.topk || !a.p_input || !a.p_output ||
       !a.p_indices)
        return -1;
    if(t.input_type == "fp16")
        return dispatch_activation<ck_tile::fp16_t>(t, a, s);
    if(t.input_type == "bf16")
        return dispatch_activation<ck_tile::bf16_t>(t, a, s);
    if(t.input_type == "fp32")
        return dispatch_activation<float>(t, a, s);
    return -1;
}
