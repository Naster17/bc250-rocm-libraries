// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <hip/hip_runtime_api.h>

namespace miopen_plugin
{

struct Gfx1013PointwiseDescriptor
{
    bool fp16 = false;
    std::size_t rank = 0;
    std::size_t dims[5]{};
    std::size_t strides[5]{};
    std::size_t elements = 1;
    std::size_t biasStride = 0;
    float lower = 0.0f;
    float upper = 0.0f;
};

hipError_t launchGfx1013BiasActivation(hipStream_t stream,
                                       void* output,
                                       const void* bias,
                                       const Gfx1013PointwiseDescriptor& descriptor);

} // namespace miopen_plugin
