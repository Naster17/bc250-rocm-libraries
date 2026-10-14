/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/
#include "contraction_scalar.hpp"

#include <algorithm>
#include <cmath>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <map>
#include <set>

namespace hiptensor
{
    namespace
    {
        constexpr int MaxModes = 16;
        struct ScalarProblem
        {
            int                 freeModes          = 0;
            int                 sumModes           = 0;
            uint64_t            outputElements     = 1;
            uint64_t            sumElements        = 1;
            uint64_t            lengths[MaxModes]  = {};
            uint64_t            aStrides[MaxModes] = {};
            uint64_t            bStrides[MaxModes] = {};
            uint64_t            cStrides[MaxModes] = {};
            uint64_t            dStrides[MaxModes] = {};
            hiptensorOperator_t opA, opB, opC;
        };

        bool checkedMultiply(uint64_t& value, uint64_t factor)
        {
            if(factor == 0 || value > std::numeric_limits<uint64_t>::max() / factor)
                return false;
            value *= factor;
            return true;
        }

        bool supportedOp(hiptensorOperator_t op)
        {
            return op == HIPTENSOR_OP_IDENTITY || op == HIPTENSOR_OP_NEG || op == HIPTENSOR_OP_ABS
                   || op == HIPTENSOR_OP_RELU || op == HIPTENSOR_OP_SQRT || op == HIPTENSOR_OP_RCP
                   || op == HIPTENSOR_OP_SIGMOID || op == HIPTENSOR_OP_TANH
                   || op == HIPTENSOR_OP_EXP || op == HIPTENSOR_OP_LOG || op == HIPTENSOR_OP_SIN
                   || op == HIPTENSOR_OP_COS || op == HIPTENSOR_OP_TAN || op == HIPTENSOR_OP_SINH
                   || op == HIPTENSOR_OP_COSH || op == HIPTENSOR_OP_ASIN || op == HIPTENSOR_OP_ACOS
                   || op == HIPTENSOR_OP_ATAN || op == HIPTENSOR_OP_ASINH
                   || op == HIPTENSOR_OP_ACOSH || op == HIPTENSOR_OP_ATANH
                   || op == HIPTENSOR_OP_CEIL || op == HIPTENSOR_OP_FLOOR;
        }

        hiptensorStatus_t makeProblem(hiptensorOperationDescriptor_t desc, ScalarProblem& problem)
        {
            if(!desc || desc->mOperationType != HIPTENSOR_CONTRACTION || !desc->mDescA
               || !desc->mDescB || !desc->mDescD)
                return HIPTENSOR_STATUS_INVALID_VALUE;
            auto inputType    = desc->mDescA->mType;
            auto outputType   = desc->mDescD->mType;
            bool complexInput = inputType == HIPTENSOR_C_32F || inputType == HIPTENSOR_C_64F;
            auto supported    = [complexInput](hiptensorOperator_t op) {
                return complexInput ? (op == HIPTENSOR_OP_IDENTITY || op == HIPTENSOR_OP_NEG
                                       || op == HIPTENSOR_OP_CONJ)
                                    : supportedOp(op);
            };
            if(!supported(desc->mOpA) || !supported(desc->mOpB)
               || (desc->mDescC && !supported(desc->mOpC)))
                return HIPTENSOR_STATUS_NOT_SUPPORTED;
            // Keep the public C/D dtype contract while permitting useful widened
            // outputs. Inputs are converted before multiplication and accumulation.
            bool fp32Input = inputType == HIPTENSOR_R_16F || inputType == HIPTENSOR_R_16BF
                             || inputType == HIPTENSOR_R_32F;
            bool fp32      = fp32Input && (outputType == inputType || outputType == HIPTENSOR_R_32F)
                             && desc->mDescCompute == HIPTENSOR_COMPUTE_DESC_32F;
            bool fp64      = (inputType == HIPTENSOR_R_32F || inputType == HIPTENSOR_R_64F)
                             && outputType == HIPTENSOR_R_64F
                             && desc->mDescCompute == HIPTENSOR_COMPUTE_DESC_64F;
            bool complex   = outputType == inputType
                             && ((inputType == HIPTENSOR_C_32F
                                  && desc->mDescCompute == HIPTENSOR_COMPUTE_DESC_C32F)
                                 || (inputType == HIPTENSOR_C_64F
                                     && desc->mDescCompute == HIPTENSOR_COMPUTE_DESC_C64F));
            if((!fp32 && !fp64 && !complex) || desc->mDescB->mType != inputType
               || (desc->mDescC && desc->mDescC->mType != outputType))
                return HIPTENSOR_STATUS_NOT_SUPPORTED;

            std::map<int32_t, uint64_t> lengths;
            auto                        collect
                = [&](hiptensorTensorDescriptor_t tensor, std::vector<int32_t> const& modes) {
                      if(!tensor)
                          return modes.empty();
                      if(modes.size() != tensor->mLengths.size()
                         || modes.size() != tensor->mStrides.size() || modes.size() > MaxModes)
                          return false;
                      std::set<int32_t> unique;
                      uint64_t          extent = 1;
                      for(size_t i = 0; i < modes.size(); ++i)
                      {
                          uint64_t len = tensor->mLengths[i], stride = tensor->mStrides[i];
                          if(len == 0 || stride == 0 || len > std::numeric_limits<int64_t>::max()
                             || stride > std::numeric_limits<int64_t>::max()
                             || !unique.insert(modes[i]).second)
                              return false;
                          auto inserted = lengths.emplace(modes[i], len);
                          if(!inserted.second && inserted.first->second != len)
                              return false;
                          if(len - 1 > (std::numeric_limits<uint64_t>::max() - extent) / stride)
                              return false;
                          extent += (len - 1) * stride;
                      }
                      return extent <= std::numeric_limits<int64_t>::max()
                                           / hiptensorDataTypeSize(tensor->mType);
                  };
            if(!collect(desc->mDescA, desc->mModeA) || !collect(desc->mDescB, desc->mModeB)
               || !collect(desc->mDescC, desc->mModeC) || !collect(desc->mDescD, desc->mModeD))
                return HIPTENSOR_STATUS_INVALID_VALUE;
            if(lengths.size() > MaxModes)
                return HIPTENSOR_STATUS_NOT_SUPPORTED;

            // Sum every non-output mode in either multiplicand, including unilateral
            // reductions. An absent mode has stride zero in that operand, so both
            // kernel mappings broadcast it without changing unary-op semantics.
            // Output modes may be shared (batch), or free in either multiplicand.
            auto contains = [](std::vector<int32_t> const& modes, int32_t mode) {
                return std::find(modes.begin(), modes.end(), mode) != modes.end();
            };
            std::vector<int32_t> ordered = desc->mModeD;
            for(auto const& item : lengths)
            {
                bool a = contains(desc->mModeA, item.first);
                bool b = contains(desc->mModeB, item.first);
                bool d = contains(desc->mModeD, item.first);
                if(!a && !b)
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                if(!d)
                    ordered.push_back(item.first);
            }
            if(desc->mDescC
               && (desc->mModeC.size() != desc->mModeD.size()
                   || std::set<int32_t>(desc->mModeC.begin(), desc->mModeC.end())
                          != std::set<int32_t>(desc->mModeD.begin(), desc->mModeD.end())))
                return HIPTENSOR_STATUS_NOT_SUPPORTED;

            // Ensure distinct output elements never race. This admits padded and
            // transposed layouts; interleaved exotic strides are conservatively rejected.
            std::vector<std::pair<uint64_t, uint64_t>> outputDimensions;
            for(size_t i = 0; i < desc->mModeD.size(); ++i)
                if(desc->mDescD->mLengths[i] > 1)
                    outputDimensions.emplace_back(desc->mDescD->mStrides[i],
                                                  desc->mDescD->mLengths[i]);
            std::sort(outputDimensions.begin(), outputDimensions.end());
            uint64_t span = 1;
            for(auto dimension : outputDimensions)
            {
                if(dimension.first < span)
                    return HIPTENSOR_STATUS_NOT_SUPPORTED;
                span += (dimension.second - 1) * dimension.first;
            }

            problem.freeModes = desc->mModeD.size();
            problem.sumModes  = ordered.size() - problem.freeModes;
            problem.opA       = desc->mOpA;
            problem.opB       = desc->mOpB;
            problem.opC       = desc->mOpC;
            auto stride       = [](hiptensorTensorDescriptor_t tensor,
                                   std::vector<int32_t> const& modes,
                                   int32_t                     mode) -> uint64_t {
                auto pos = std::find(modes.begin(), modes.end(), mode);
                return !tensor || pos == modes.end() ? 0 : tensor->mStrides[pos - modes.begin()];
            };
            for(size_t i = 0; i < ordered.size(); ++i)
            {
                problem.lengths[i]  = lengths.at(ordered[i]);
                problem.aStrides[i] = stride(desc->mDescA, desc->mModeA, ordered[i]);
                problem.bStrides[i] = stride(desc->mDescB, desc->mModeB, ordered[i]);
                problem.cStrides[i] = stride(desc->mDescC, desc->mModeC, ordered[i]);
                problem.dStrides[i] = stride(desc->mDescD, desc->mModeD, ordered[i]);
                if(!checkedMultiply(i < static_cast<size_t>(problem.freeModes)
                                        ? problem.outputElements
                                        : problem.sumElements,
                                    problem.lengths[i]))
                    return HIPTENSOR_STATUS_INVALID_VALUE;
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }

        template <class Acc>
        __device__ Acc unary(Acc value, hiptensorOperator_t op)
        {
            if(op == HIPTENSOR_OP_NEG)
                return -value;
            if(op == HIPTENSOR_OP_ABS)
                return value == Acc(0) ? Acc(0) : (value < Acc(0) ? -value : value);
            if(op == HIPTENSOR_OP_RELU)
                return value < Acc(0) ? Acc(0) : value;
            if(op == HIPTENSOR_OP_SQRT)
                return sqrt(value);
            if(op == HIPTENSOR_OP_RCP)
                return Acc(1) / value;
            if(op == HIPTENSOR_OP_SIGMOID)
                return value >= Acc(0) ? Acc(1) / (Acc(1) + exp(-value))
                                       : exp(value) / (Acc(1) + exp(value));
            if(op == HIPTENSOR_OP_TANH)
                return tanh(value);
            if(op == HIPTENSOR_OP_EXP)
                return exp(value);
            if(op == HIPTENSOR_OP_LOG)
                return log(value);
            if(op == HIPTENSOR_OP_SIN)
                return sin(value);
            if(op == HIPTENSOR_OP_COS)
                return cos(value);
            if(op == HIPTENSOR_OP_TAN)
                return tan(value);
            if(op == HIPTENSOR_OP_SINH)
                return sinh(value);
            if(op == HIPTENSOR_OP_COSH)
                return cosh(value);
            if(op == HIPTENSOR_OP_ASIN)
                return asin(value);
            if(op == HIPTENSOR_OP_ACOS)
                return acos(value);
            if(op == HIPTENSOR_OP_ATAN)
                return atan(value);
            if(op == HIPTENSOR_OP_ASINH)
                return asinh(value);
            if(op == HIPTENSOR_OP_ACOSH)
                return acosh(value);
            if(op == HIPTENSOR_OP_ATANH)
                return atanh(value);
            if(op == HIPTENSOR_OP_CEIL)
                return ceil(value);
            if(op == HIPTENSOR_OP_FLOOR)
                return floor(value);
            return value;
        }

        template <>
        __device__ hipFloatComplex unary(hipFloatComplex value, hiptensorOperator_t op)
        {
            if(op == HIPTENSOR_OP_NEG)
                return make_hipFloatComplex(-value.x, -value.y);
            return op == HIPTENSOR_OP_CONJ ? hipConjf(value) : value;
        }

        template <>
        __device__ hipDoubleComplex unary(hipDoubleComplex value, hiptensorOperator_t op)
        {
            if(op == HIPTENSOR_OP_NEG)
                return make_hipDoubleComplex(-value.x, -value.y);
            return op == HIPTENSOR_OP_CONJ ? hipConj(value) : value;
        }

        // HIP complex values are vector pairs. Their vector multiplication is
        // componentwise, so complex products must use the public HIP arithmetic.
        template <class Acc>
        __device__ Acc multiply(Acc lhs, Acc rhs)
        {
            return lhs * rhs;
        }
        __device__ hipFloatComplex multiply(hipFloatComplex lhs, hipFloatComplex rhs)
        {
            return hipCmulf(lhs, rhs);
        }
        __device__ hipDoubleComplex multiply(hipDoubleComplex lhs, hipDoubleComplex rhs)
        {
            return hipCmul(lhs, rhs);
        }
        template <class Acc>
        __device__ Acc add(Acc lhs, Acc rhs)
        {
            return lhs + rhs;
        }
        __device__ hipFloatComplex add(hipFloatComplex lhs, hipFloatComplex rhs)
        {
            return hipCaddf(lhs, rhs);
        }
        __device__ hipDoubleComplex add(hipDoubleComplex lhs, hipDoubleComplex rhs)
        {
            return hipCadd(lhs, rhs);
        }
        template <class Acc>
        __host__ __device__ bool nonzero(Acc value)
        {
            return value != Acc(0);
        }
        __host__ __device__ bool nonzero(hipFloatComplex value)
        {
            return value.x != 0.0f || value.y != 0.0f;
        }
        __host__ __device__ bool nonzero(hipDoubleComplex value)
        {
            return value.x != 0.0 || value.y != 0.0;
        }

        template <class Input, class Output, class Acc>
        __global__ void scalarContraction(ScalarProblem p,
                                          Acc           alpha,
                                          const Input*  a,
                                          const Input*  b,
                                          Acc           beta,
                                          const Output* c,
                                          Output*       d)
        {
            for(uint64_t out = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
                out < p.outputElements;
                out += uint64_t(gridDim.x) * blockDim.x)
            {
                uint64_t linear = out, aBase = 0, bBase = 0, cOffset = 0, dOffset = 0;
                for(int i = 0; i < p.freeModes; ++i)
                {
                    uint64_t coordinate = linear % p.lengths[i];
                    linear /= p.lengths[i];
                    aBase += coordinate * p.aStrides[i];
                    bBase += coordinate * p.bStrides[i];
                    cOffset += coordinate * p.cStrides[i];
                    dOffset += coordinate * p.dStrides[i];
                }
                Acc sum = {};
                // Common GEMM/dot contractions need no mixed-radix division in
                // the inner loop. Preserve the generic mapping for multiple sums.
                if(nonzero(alpha) && p.sumModes == 1)
                    for(uint64_t k = 0; k < p.sumElements; ++k)
                        sum = add(
                            sum,
                            multiply(unary(Acc(a[aBase + k * p.aStrides[p.freeModes]]), p.opA),
                                     unary(Acc(b[bBase + k * p.bStrides[p.freeModes]]), p.opB)));
                else if(nonzero(alpha))
                    for(uint64_t k = 0; k < p.sumElements; ++k)
                    {
                        uint64_t contracted = k, aOffset = aBase, bOffset = bBase;
                        for(int i = p.freeModes; i < p.freeModes + p.sumModes; ++i)
                        {
                            uint64_t coordinate = contracted % p.lengths[i];
                            contracted /= p.lengths[i];
                            aOffset += coordinate * p.aStrides[i];
                            bOffset += coordinate * p.bStrides[i];
                        }
                        sum = add(
                            sum,
                            multiply(unary(Acc(a[aOffset]), p.opA), unary(Acc(b[bOffset]), p.opB)));
                    }
                Acc value = multiply(alpha, sum);
                if(nonzero(beta))
                    value = add(value, multiply(beta, unary(Acc(c[cOffset]), p.opC)));
                d[dOffset] = Output(value);
            }
        }

        template <class Input, class Output, class Acc>
        hiptensorStatus_t launch(ScalarProblem const& problem,
                                 const void*          alpha,
                                 const void*          a,
                                 const void*          b,
                                 const void*          beta,
                                 const void*          c,
                                 void*                d,
                                 hipStream_t          stream)
        {
            Acc alphaValue = *static_cast<const Acc*>(alpha);
            Acc betaValue  = beta ? *static_cast<const Acc*>(beta) : Acc{};
            if(nonzero(betaValue) && !c)
                return HIPTENSOR_STATUS_INVALID_VALUE;
            uint32_t blocks = std::min<uint64_t>(65535, 1 + (problem.outputElements - 1) / 256);
            hipLaunchKernelGGL((scalarContraction<Input, Output, Acc>),
                               dim3(blocks),
                               dim3(256),
                               0,
                               stream,
                               problem,
                               alphaValue,
                               static_cast<const Input*>(a),
                               static_cast<const Input*>(b),
                               betaValue,
                               static_cast<const Output*>(c),
                               static_cast<Output*>(d));
            return hipGetLastError() == hipSuccess ? HIPTENSOR_STATUS_SUCCESS
                                                   : HIPTENSOR_STATUS_EXECUTION_FAILED;
        }
    }

    hiptensorStatus_t validateScalarContraction(hiptensorOperationDescriptor_t desc)
    {
        ScalarProblem problem;
        return makeProblem(desc, problem);
    }

    hiptensorStatus_t executeScalarContraction(hiptensorOperationDescriptor_t desc,
                                               const void*                    alpha,
                                               const void*                    a,
                                               const void*                    b,
                                               const void*                    beta,
                                               const void*                    c,
                                               void*                          d,
                                               hipStream_t                    stream)
    {
        ScalarProblem problem;
        auto          status = makeProblem(desc, problem);
        if(status != HIPTENSOR_STATUS_SUCCESS)
            return status;
        if(!alpha || !a || !b || !d || (desc->mDescC && !beta) || (!desc->mDescC && c))
            return HIPTENSOR_STATUS_INVALID_VALUE;
        auto extent = [](hiptensorTensorDescriptor_t tensor) {
            uint64_t span = 1;
            for(size_t i = 0; i < tensor->mLengths.size(); ++i)
                span += (tensor->mLengths[i] - 1) * tensor->mStrides[i];
            return span * hiptensorDataTypeSize(tensor->mType);
        };
        auto overlaps = [](const void* lhs, uint64_t lhsBytes, const void* rhs, uint64_t rhsBytes) {
            auto x = reinterpret_cast<uintptr_t>(lhs), y = reinterpret_cast<uintptr_t>(rhs);
            return x < y ? y - x < lhsBytes : x - y < rhsBytes;
        };
        auto outputBytes = extent(desc->mDescD);
        if(overlaps(d, outputBytes, a, extent(desc->mDescA))
           || overlaps(d, outputBytes, b, extent(desc->mDescB)))
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        if(c && overlaps(d, outputBytes, c, extent(desc->mDescC))
           && (d != c || desc->mModeC != desc->mModeD
               || desc->mDescC->mStrides != desc->mDescD->mStrides))
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        switch(desc->mDescA->mType)
        {
        case HIPTENSOR_R_16F:
            if(desc->mDescD->mType == HIPTENSOR_R_32F)
                return launch<__half, float, float>(problem, alpha, a, b, beta, c, d, stream);
            return launch<__half, __half, float>(problem, alpha, a, b, beta, c, d, stream);
        case HIPTENSOR_R_16BF:
            if(desc->mDescD->mType == HIPTENSOR_R_32F)
                return launch<hip_bfloat16, float, float>(problem, alpha, a, b, beta, c, d, stream);
            return launch<hip_bfloat16, hip_bfloat16, float>(
                problem, alpha, a, b, beta, c, d, stream);
        case HIPTENSOR_R_32F:
            if(desc->mDescD->mType == HIPTENSOR_R_64F)
                return launch<float, double, double>(problem, alpha, a, b, beta, c, d, stream);
            return launch<float, float, float>(problem, alpha, a, b, beta, c, d, stream);
        case HIPTENSOR_R_64F:
            return launch<double, double, double>(problem, alpha, a, b, beta, c, d, stream);
        case HIPTENSOR_C_32F:
            return launch<hipFloatComplex, hipFloatComplex, hipFloatComplex>(
                problem, alpha, a, b, beta, c, d, stream);
        case HIPTENSOR_C_64F:
            return launch<hipDoubleComplex, hipDoubleComplex, hipDoubleComplex>(
                problem, alpha, a, b, beta, c, d, stream);
        default:
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        }
    }
}
