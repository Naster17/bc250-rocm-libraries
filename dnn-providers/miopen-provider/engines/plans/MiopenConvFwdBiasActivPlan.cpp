// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/DeviceQuery.hpp>
#include <limits>

#include "MiopenConvFwdBiasActivPlan.hpp"

namespace miopen_plugin
{

ConvFwdBiasActivParams::ConvFwdBiasActivParams(
    const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes& convAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* biasAttr,
    const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes& activAttr,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap,
    bool deterministicEnabled)
    : _spatialDimCount(miopen_utils::getSpatialDimCount(
          miopen_utils::findTensorAttributes(tensorMap, convAttr.x_tensor_uid())))
    , _x(miopen_utils::createPaddedTensor(tensorMap, convAttr.x_tensor_uid()))
    , _w(miopen_utils::createPaddedTensor(tensorMap, convAttr.w_tensor_uid()))
    , _y(miopen_utils::createPaddedTensor(tensorMap, activAttr.out_0_tensor_uid()))
{
    using namespace miopen_utils;

    const auto& attrX = findTensorAttributes(tensorMap, _x.uid());
    const auto& attrW = findTensorAttributes(tensorMap, _w.uid());

    const auto xDims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrX.dims());
    const auto wDims
        = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrW.dims());
    const auto groupCount = hipdnn_data_sdk::utilities::calculateGroupCount(xDims, wDims);

    _conv = MiopenConvDescriptor(
        _spatialDimCount, convAttr, static_cast<int>(groupCount), deterministicEnabled);

    if(biasAttr != nullptr)
    {
        if(!biasAttr->in_1_tensor_uid().has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "ConvFwdBiasActivParams: biasAttr missing in_1_tensor_uid");
        }

        if(biasAttr->in_0_tensor_uid() == convAttr.y_tensor_uid())
        {
            _bias = createPaddedTensor(tensorMap, biasAttr->in_1_tensor_uid().value());
        }
        else if(biasAttr->in_1_tensor_uid().value() == convAttr.y_tensor_uid())
        {
            _bias = createPaddedTensor(tensorMap, biasAttr->in_0_tensor_uid());
        }
        else
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "ConvFwdBiasActivParams: biasAttr tensor UIDs do not match convAttr y_tensor_uid");
        }
    }

    if(activAttr.elu_alpha().has_value() || activAttr.softplus_beta().has_value()
       || activAttr.swish_beta().has_value() || activAttr.relu_lower_clip_slope().has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "ConvFwdBiasActivParams: Fusion supports only relu, clipped relu "
            "(relu_upper_clip set), "
            "or CLAMP "
            "(relu_upper_clip and relu_lower_clip set)");
    }

    HIPDNN_PREPEND_MESSAGE_ON_THROW(_activParams = mapPointwiseModeToMiopenActivation(activAttr),
                                    "ConvFwdBiasActivParams: ");

    // Reusing the final output as an intermediate is exact only when each graph
    // edge has the same storage type and shape. Do not silently introduce early
    // rounding into graphs that request FP32 virtual intermediates.
    const auto& attrY = findTensorAttributes(tensorMap, _y.uid());
    const auto sameOutput = [&](int64_t uid) {
        const auto& attr = findTensorAttributes(tensorMap, uid);
        return attr.data_type() == attrY.data_type()
               && hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attr.dims())
                      == hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                          attrY.dims());
    };
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
    const auto type = attrY.data_type();
    bool validBias = !biasAttr;
    if(biasAttr)
    {
        const auto& attrBias = findTensorAttributes(tensorMap, _bias->uid());
        const auto biasDims = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
            attrBias.dims());
        const auto outputDims
            = hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(attrY.dims());
        validBias = attrBias.data_type() == type && biasDims.size() == outputDims.size();
        for(size_t dimension = 0; validBias && dimension < biasDims.size(); ++dimension)
        {
            validBias = biasDims[dimension] == (dimension == 1 ? outputDims[dimension] : 1);
        }
    }
    if((type == DataType::FLOAT || type == DataType::HALF || type == DataType::BFLOAT16)
       && attrX.data_type() == type && attrW.data_type() == type
       && sameOutput(convAttr.y_tensor_uid()) && sameOutput(activAttr.in_0_tensor_uid())
       && validBias)
    {
        _sequentialConvolutionParams
            = std::make_unique<ConvFwdParams>(convAttr, tensorMap, deterministicEnabled, _y.uid());
        if((type == DataType::HALF || type == DataType::BFLOAT16) && attrY.dims()->size() > 5)
        {
            _sequentialConvolutionParams.reset();
        }
        else if(type == DataType::HALF || type == DataType::BFLOAT16)
        {
            Gfx1013PointwiseDescriptor descriptor;
            descriptor.fp16 = type == DataType::HALF;
            descriptor.rank = attrY.dims()->size();
            for(std::size_t dimension = 0; dimension < descriptor.rank; ++dimension)
            {
                const auto length
                    = attrY.dims()->Get(static_cast<flatbuffers::uoffset_t>(dimension));
                const auto stride
                    = attrY.strides()->Get(static_cast<flatbuffers::uoffset_t>(dimension));
                if(length <= 0 || stride <= 0
                   || static_cast<std::size_t>(length)
                          > std::numeric_limits<std::size_t>::max() / descriptor.elements)
                {
                    throw hipdnn_plugin_sdk::HipdnnPluginException(
                        HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                        "Invalid gfx1013 FP16/BF16 pointwise dimensions");
                }
                descriptor.dims[dimension] = static_cast<std::size_t>(length);
                descriptor.strides[dimension] = static_cast<std::size_t>(stride);
                descriptor.elements *= static_cast<std::size_t>(length);
            }
            if(_bias)
            {
                descriptor.biasStride = static_cast<std::size_t>(
                    findTensorAttributes(tensorMap, _bias->uid()).strides()->Get(1));
            }
            switch(_activParams.mode)
            {
            case miopenActivationRELU:
                descriptor.upper = std::numeric_limits<float>::infinity();
                break;
            case miopenActivationCLIPPEDRELU:
                descriptor.upper = static_cast<float>(_activParams.alpha);
                break;
            case miopenActivationCLAMP:
                descriptor.lower = static_cast<float>(_activParams.alpha);
                descriptor.upper = static_cast<float>(_activParams.beta);
                break;
            default:
                // Decline only the native sequential path. Other architectures
                // retain the original fusion implementation and its mode support.
                _sequentialConvolutionParams.reset();
                return;
            }
            _pointwiseDescriptor = descriptor;
        }
    }
}

const MiopenTensor& ConvFwdBiasActivParams::x() const
{
    return _x;
}

const MiopenTensor& ConvFwdBiasActivParams::w() const
{
    return _w;
}

const MiopenConvDescriptor& ConvFwdBiasActivParams::conv() const
{
    return _conv;
}

const std::optional<MiopenTensor>& ConvFwdBiasActivParams::bias() const
{
    return _bias;
}

const miopen_utils::ActivationParams& ConvFwdBiasActivParams::activParams() const
{
    return _activParams;
}

const MiopenTensor& ConvFwdBiasActivParams::y() const
{
    return _y;
}

std::unique_ptr<ConvFwdParams> ConvFwdBiasActivParams::takeSequentialConvolutionParams()
{
    return std::move(_sequentialConvolutionParams);
}

const std::optional<Gfx1013PointwiseDescriptor>& ConvFwdBiasActivParams::pointwiseDescriptor() const
{
    return _pointwiseDescriptor;
}

ConvFwdBiasActivPlan::ConvFwdBiasActivPlan(const HipdnnMiopenHandle& handle,
                                           ConvFwdBiasActivParams&& params,
                                           const HipdnnMiopenSettings& executionSettings,
                                           bool compile,
                                           bool getWsSize)
    : _params(std::move(params))
    , _executionSettings(executionSettings)
{
    const auto arch = hipdnn_plugin_sdk::getDeviceArch(handle.getStream());
    if(arch.substr(0, arch.find(':')) == "gfx1013")
    {
#ifndef MIOPENPROVIDER_GFX1013_SCALAR
        if(_params.pointwiseDescriptor())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx1013 FP16/BF16 sequential fusion requires MIOPENPROVIDER_GFX1013_SCALAR");
        }
#endif
        auto convolutionParams = _params.takeSequentialConvolutionParams();
        if(!convolutionParams)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx1013 sequential convolution fusion requires homogeneous FP32/FP16/BF16 "
                "tensors and intermediate storage types");
        }
        _sequentialConvolution = std::make_unique<ConvFwdPlan>(
            handle, std::move(*convolutionParams), executionSettings);
        _workspaceSize = _sequentialConvolution->getWorkspaceSize(handle);
        miopenActivationDescriptor_t activation = nullptr;
        THROW_ON_MIOPEN_FAILURE(miopenCreateActivationDescriptor(&activation));
        _activationDesc = hipdnn_data_sdk::utilities::ScopedResource<miopenActivationDescriptor_t>(
            activation, [](miopenActivationDescriptor_t desc) {
                LOG_ON_MIOPEN_FAILURE(miopenDestroyActivationDescriptor(desc));
            });
        const auto& activationParams = _params.activParams();
        THROW_ON_MIOPEN_FAILURE(miopenSetActivationDescriptor(activation,
                                                              activationParams.mode,
                                                              activationParams.alpha,
                                                              activationParams.beta,
                                                              activationParams.gamma));
        return;
    }

    // Set tuning policy based on benchmarking flag - RAII ensures restoration
    const ScopedTuningPolicy tuningGuard(handle.miopenHandle,
                                         _executionSettings.benchmarkingEnabled());

    miopenFusionPlanDescriptor_t fusePlanDesc;
    THROW_ON_MIOPEN_FAILURE(miopenCreateFusionPlan(
        &fusePlanDesc, miopenVerticalFusion, _params.x().tensorDescriptor()));
    _fusePlanDesc = hipdnn_data_sdk::utilities::ScopedResource<miopenFusionPlanDescriptor_t>(
        fusePlanDesc, [](miopenFusionPlanDescriptor_t desc) {
            auto status = miopenDestroyFusionPlan(desc);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_PLUGIN_LOG_ERROR(
                    "miopenDestroyFusionPlan failed in ConvFwdBiasActivPlan destructor");
            }
        });

    miopenFusionOpDescriptor_t convOp;
    THROW_ON_MIOPEN_FAILURE(miopenCreateOpConvForward(
        fusePlanDesc, &convOp, _params.conv().convDescriptor(), _params.w().tensorDescriptor()));

    if(_params.bias().has_value())
    {
        miopenFusionOpDescriptor_t biasOp;
        THROW_ON_MIOPEN_FAILURE(miopenCreateOpBiasForward(
            fusePlanDesc, &biasOp, _params.bias().value().tensorDescriptor()));
    }

    miopenFusionOpDescriptor_t activOp;
    THROW_ON_MIOPEN_FAILURE(
        miopenCreateOpActivationForward(fusePlanDesc, &activOp, _params.activParams().mode));

    // Usage scenarios:
    // 1. Applicability check: requires compilation only
    // 2. Workspace size: requires workspace size retrieval only
    // 3. Execution plan: requires both compilation and workspace size retrieval

    if(compile)
    {
        THROW_ON_MIOPEN_FAILURE(miopenCompileFusionPlan(handle.miopenHandle, fusePlanDesc));
    }

    if(!getWsSize)
    {
        return;
    }

    THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetWorkSpaceSize(
        handle.miopenHandle,
        fusePlanDesc,
        &_workspaceSize,
        static_cast<miopenConvFwdAlgorithm_t>(-1))); // Algo is not used in MIOpen
}

size_t
    ConvFwdBiasActivPlan::getWorkspaceSize([[maybe_unused]] const HipdnnMiopenHandle& handle) const
{
    return _workspaceSize;
}

void ConvFwdBiasActivPlan::execute(const HipdnnMiopenHandle& handle,
                                   const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                   uint32_t numDeviceBuffers,
                                   void* workspace) const
{
    if(_sequentialConvolution)
    {
        _sequentialConvolution->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
        const auto output = hipdnn_plugin_sdk::findDeviceBuffer(
            _params.y().uid(), deviceBuffers, numDeviceBuffers);
#ifdef MIOPENPROVIDER_GFX1013_SCALAR
        if(_params.pointwiseDescriptor())
        {
            const void* bias = nullptr;
            if(_params.bias())
            {
                bias = hipdnn_plugin_sdk::findDeviceBuffer(
                           _params.bias()->uid(), deviceBuffers, numDeviceBuffers)
                           .ptr;
            }
            const auto status = launchGfx1013BiasActivation(
                handle.getStream(), output.ptr, bias, *_params.pointwiseDescriptor());
            if(status != hipSuccess)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                    std::string("gfx1013 FP16/BF16 pointwise launch failed: ")
                        + hipGetErrorString(status));
            }
            return;
        }
#endif
        const float alpha = 1.0f;
        // ForwardBias computes alpha*y + alpha*bias + beta*y: beta must
        // be zero because the convolution value is already its first input.
        const float beta = 0.0f;
        if(_params.bias())
        {
            const auto bias = hipdnn_plugin_sdk::findDeviceBuffer(
                _params.bias()->uid(), deviceBuffers, numDeviceBuffers);
            THROW_ON_MIOPEN_FAILURE(miopenConvolutionForwardBias(handle.miopenHandle,
                                                                 &alpha,
                                                                 _params.bias()->tensorDescriptor(),
                                                                 bias.ptr,
                                                                 &beta,
                                                                 _params.y().tensorDescriptor(),
                                                                 output.ptr));
        }
        const float overwrite = 0.0f;
        THROW_ON_MIOPEN_FAILURE(miopenActivationForward(handle.miopenHandle,
                                                        _activationDesc.get(),
                                                        &alpha,
                                                        _params.y().tensorDescriptor(),
                                                        output.ptr,
                                                        &overwrite,
                                                        _params.y().tensorDescriptor(),
                                                        output.ptr));
        return;
    }

    miopenOperatorArgs_t fusionArgs;
    THROW_ON_MIOPEN_FAILURE(miopenCreateOperatorArgs(&fusionArgs));
    auto fusionArgsRes = hipdnn_data_sdk::utilities::ScopedResource<miopenOperatorArgs_t>(
        fusionArgs, [](miopenOperatorArgs_t args) {
            auto status = miopenDestroyOperatorArgs(args);
            if(status != miopenStatusSuccess)
            {
                HIPDNN_PLUGIN_LOG_ERROR(
                    "miopenDestroyOperatorArgs failed in ConvFwdBiasActivPlan destructor");
            }
        });

    auto wBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.w().uid(), deviceBuffers, numDeviceBuffers);

    int opIdx = 0;
    miopenFusionOpDescriptor_t convoOp;
    THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx++, &convoOp));
    THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsConvForward(fusionArgs,
                                                       convoOp,
                                                       nullptr, // Default value for alpha is 1.0f
                                                       nullptr, // Default value for beta is 0.0f
                                                       wBuffer.ptr));

    if(_params.bias().has_value())
    {
        auto biasBuffer = hipdnn_plugin_sdk::findDeviceBuffer(
            _params.bias().value().uid(), deviceBuffers, numDeviceBuffers);

        miopenFusionOpDescriptor_t biasOp;
        THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx++, &biasOp));
        THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsBiasForward(fusionArgs,
                                                           biasOp,
                                                           nullptr, // alpha is not used in MIOpen
                                                           nullptr, // beta is not used in MIOpen
                                                           biasBuffer.ptr));
    }

    miopenFusionOpDescriptor_t activOp;
    THROW_ON_MIOPEN_FAILURE(miopenFusionPlanGetOp(_fusePlanDesc.get(), opIdx, &activOp));
    THROW_ON_MIOPEN_FAILURE(miopenSetOpArgsActivForward(fusionArgs,
                                                        activOp,
                                                        nullptr, // alpha is not used in MIOpen
                                                        nullptr, // beta is not used in MIOpen
                                                        _params.activParams().alpha,
                                                        _params.activParams().beta,
                                                        _params.activParams().gamma));

    size_t workspaceSize = 0;
    if(workspace != nullptr)
    {
        // Assume the provided workspace is large enough
        workspaceSize = _workspaceSize;
    }

    auto xBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.x().uid(), deviceBuffers, numDeviceBuffers);
    auto yBuffer
        = hipdnn_plugin_sdk::findDeviceBuffer(_params.y().uid(), deviceBuffers, numDeviceBuffers);

    THROW_ON_MIOPEN_FAILURE(miopenExecuteFusionPlan_v2(handle.miopenHandle,
                                                       _fusePlanDesc.get(),
                                                       _params.x().tensorDescriptor(),
                                                       xBuffer.ptr,
                                                       _params.y().tensorDescriptor(),
                                                       yBuffer.ptr,
                                                       fusionArgs,
                                                       workspace,
                                                       workspaceSize));
}

} // namespace miopen_plugin
