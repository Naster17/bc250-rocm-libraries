// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "MiopenGfx1013Pointwise.hpp"

namespace miopen_plugin
{
namespace
{

__device__ float loadBf16(unsigned short value)
{
    return __uint_as_float(static_cast<unsigned int>(value) << 16);
}

__device__ unsigned short roundBf16(float value)
{
    auto bits = __float_as_uint(value);
    if((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
    {
        return static_cast<unsigned short>((bits >> 16) | 0x40u);
    }
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<unsigned short>(bits >> 16);
}

__device__ float loadStorage(unsigned short value, bool fp16)
{
    return fp16 ? __half2float(__ushort_as_half(value)) : loadBf16(value);
}

__device__ unsigned short roundStorage(float value, bool fp16)
{
    return fp16 ? __half_as_ushort(__float2half_rn(value)) : roundBf16(value);
}

__global__ void biasActivation(unsigned short* output,
                               const unsigned short* bias,
                               Gfx1013PointwiseDescriptor descriptor)
{
    const std::size_t step = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for(std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        linear < descriptor.elements;
        linear += step)
    {
        auto remaining = linear;
        std::size_t offset = 0;
        std::size_t channel = 0;
        for(std::size_t dimension = descriptor.rank; dimension-- > 0;)
        {
            const auto coordinate = remaining % descriptor.dims[dimension];
            remaining /= descriptor.dims[dimension];
            offset += coordinate * descriptor.strides[dimension];
            if(dimension == 1)
            {
                channel = coordinate;
            }
        }
        float value = loadStorage(output[offset], descriptor.fp16);
        if(bias)
        {
            // The graph explicitly stores the bias node's output in FP16/BF16 before
            // the activation node. Preserve that rounding boundary in this kernel.
            value = loadStorage(
                roundStorage(
                    value + loadStorage(bias[channel * descriptor.biasStride], descriptor.fp16),
                    descriptor.fp16),
                descriptor.fp16);
        }
        if(!isnan(value))
        {
            value = fmaxf(descriptor.lower, fminf(descriptor.upper, value));
        }
        output[offset] = roundStorage(value, descriptor.fp16);
    }
}

} // namespace

hipError_t launchGfx1013BiasActivation(hipStream_t stream,
                                       void* output,
                                       const void* bias,
                                       const Gfx1013PointwiseDescriptor& descriptor)
{
    if(descriptor.elements == 0)
    {
        return hipSuccess;
    }
    constexpr unsigned int block = 256;
    const auto requiredBlocks = (descriptor.elements - 1) / block + 1;
    const auto grid = static_cast<unsigned int>(requiredBlocks > 4096 ? 4096 : requiredBlocks);
    hipLaunchKernelGGL(biasActivation,
                       dim3(grid),
                       dim3(block),
                       0,
                       stream,
                       static_cast<unsigned short*>(output),
                       static_cast<const unsigned short*>(bias),
                       descriptor);
    return hipGetLastError();
}

} // namespace miopen_plugin
