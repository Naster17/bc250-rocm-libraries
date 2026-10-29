// SPDX-License-Identifier: MIT
// Private API5 prototype. Not part of production plugin admission or installation.
#include <miopen/solver/ck_impl_interface.hpp>
#include <climits>
#include <cstdint>
#include <limits>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/conv/problem_description.hpp>

// Reuse the matching upstream implementation behind a restricted public facade.
#define ck_impl_fwd_fill_valid_kernels bc250_upstream_fwd_fill_valid_kernels
#define ck_impl_fwd_is_applicable bc250_upstream_fwd_is_applicable
#define ck_impl_fwd_is_args_supported bc250_upstream_fwd_is_args_supported
#define ck_impl_fwd_get_workspace_size bc250_upstream_fwd_get_workspace_size
#define ck_impl_fwd_get_solution bc250_upstream_fwd_get_solution
#ifdef MIOPEN_BC250_DL_FP16_PRIVATE
#define ck_impl_fwd_get_all_kernel_type_strings bc250_upstream_fwd_get_all_kernel_type_strings
#endif
#include "ck_grouped_conv_fwd_impl.cpp"
#undef ck_impl_fwd_fill_valid_kernels
#undef ck_impl_fwd_is_applicable
#undef ck_impl_fwd_is_args_supported
#undef ck_impl_fwd_get_workspace_size
#undef ck_impl_fwd_get_solution
#ifdef MIOPEN_BC250_DL_FP16_PRIVATE
#undef ck_impl_fwd_get_all_kernel_type_strings
#endif
#include "ck_grouped_conv_common.cpp"

// The host hides this accessor; retain the exact matching src/tensor.cpp
// implementation locally, without interposing a host symbol.
std::optional<miopenDataType_t> miopen::TensorDescriptor::GetCastType() const
{
    return this->cast_type;
}

namespace {
bool admitted(const miopen::conv::ProblemDescription* p, miopenDataType_t type, bool tf32)
{
    bool supported_type = type == miopenFloat;
#ifdef MIOPEN_BC250_DL_FP16_PRIVATE
    supported_type = supported_type || type == miopenHalf;
#endif
    if(p == nullptr || !supported_type || tf32 || !p->Is2d() || !p->IsDirectionForward() ||
       p->GetInDataType() != type || p->GetWeightsDataType() != type ||
       p->GetOutDataType() != type || p->HasMixedDataTypes() || p->IsTensorsCasted() ||
       p->HasNonPackedTensors() || !(p->IsLayoutDefault() || p->IsLayoutNHWC()) ||
       p->GetConv().attribute.deterministic || p->GetConv().mode != miopenConvolution ||
       p->GetBias() != 0 || p->GetAlpha().GetAsDouble() != 1.0 || p->GetBeta().GetAsDouble() != 0.0)
        return false;
    // This private DL profile binds the int32 CK argument path only.
    for(const auto* descriptor : {&p->GetIn(), &p->GetWeights(), &p->GetOut()})
        if(descriptor->GetNumDims() != 4 || descriptor->GetElementSpace() > INT_MAX ||
           std::any_of(descriptor->GetLengths().begin(),
                       descriptor->GetLengths().end(),
                       [](auto n) { return n == 0 || n > INT_MAX; }))
            return false;
    const auto& x     = p->GetIn().GetLengths();
    const auto& w     = p->GetWeights().GetLengths();
    const auto& y     = p->GetOut().GetLengths();
    const auto groups = p->GetGroupCount();
    if(groups <= 0 || x[0] != y[0] || w[0] != y[1] || x[1] % groups || y[1] % groups ||
       w[1] != x[1] / groups)
        return false;
    for(int axis = 0; axis < 2; ++axis)
    {
        const int stride   = axis ? p->GetKernelStrideW() : p->GetKernelStrideH();
        const int dilation = axis ? p->GetDilationW() : p->GetDilationH();
        const int pad      = axis ? p->GetPadW() : p->GetPadH();
        if(stride <= 0 || dilation <= 0 || pad < 0)
            return false;
        const int64_t extent = static_cast<int64_t>(x[axis + 2]) + 2 * int64_t{pad} -
                               (static_cast<int64_t>(w[axis + 2]) - 1) * dilation - 1;
        if(extent < 0 || extent / stride + 1 != static_cast<int64_t>(y[axis + 2]))
            return false;
    }
    return true;
}
ck_impl_status_t empty_list(CKKernelListHandle** out)
{
    return ck_impl_try_catch([&] {
        CK_IMPL_THROW_IF_NULL(out, CK_IMPL_STATUS_BAD_PARAM, "Null list output");
        *out = new CKKernelListHandle;
    });
}
ck_impl_status_t unsupported()
{
    return ck_impl_try_catch([] {
        throw CkImplException(
            CK_IMPL_STATUS_INVALID_VALUE,
            "Private gfx1013 DL forward request is outside the configured dtype/argument contract");
    });
}
} // namespace
extern "C" ck_impl_status_t
ck_impl_fwd_fill_valid_kernels(const miopen::conv::ProblemDescription* p,
                               miopenDataType_t type,
                               bool tf32,
                               CKKernelListHandle** out)
{
    if(p == nullptr || out == nullptr)
        return CK_IMPL_STATUS_BAD_PARAM;
    return admitted(p, type, tf32) ? bc250_upstream_fwd_fill_valid_kernels(p, type, false, out)
                                   : empty_list(out);
}
extern "C" ck_impl_status_t ck_impl_fwd_is_applicable(const miopen::conv::ProblemDescription* p,
                                                      miopenDataType_t type,
                                                      bool tf32,
                                                      bool* out)
{
    if(p == nullptr || out == nullptr)
        return CK_IMPL_STATUS_BAD_PARAM;
    if(!admitted(p, type, tf32))
    {
        *out = false;
        return CK_IMPL_STATUS_SUCCESS;
    }
    return bc250_upstream_fwd_is_applicable(p, type, false, out);
}
extern "C" ck_impl_status_t ck_impl_fwd_is_args_supported(const miopen::conv::ProblemDescription* p,
                                                          const char* id,
                                                          miopenDataType_t type,
                                                          bool tf32,
                                                          bool* out)
{
    if(p == nullptr || out == nullptr || id == nullptr)
        return CK_IMPL_STATUS_BAD_PARAM;
    if(!admitted(p, type, tf32))
    {
        *out = false;
        return CK_IMPL_STATUS_SUCCESS;
    }
    return bc250_upstream_fwd_is_args_supported(p, id, type, false, out);
}
extern "C" ck_impl_status_t ck_impl_fwd_get_workspace_size(
    const miopen::conv::ProblemDescription* p, miopenDataType_t type, bool tf32, size_t* out)
{
    if(p == nullptr || out == nullptr)
        return CK_IMPL_STATUS_BAD_PARAM;
    return admitted(p, type, tf32) ? bc250_upstream_fwd_get_workspace_size(p, type, false, out)
                                   : unsupported();
}
extern "C" ck_impl_status_t ck_impl_fwd_get_solution(const miopen::ExecutionContext* ctx,
                                                     const miopen::conv::ProblemDescription* p,
                                                     const char* id,
                                                     bool tf32,
                                                     miopen::solver::ConvSolution** out)
{
    if(ctx == nullptr || p == nullptr || id == nullptr || out == nullptr)
        return CK_IMPL_STATUS_BAD_PARAM;
    if(!admitted(p, p->GetInDataType(), tf32))
        return unsupported();
    bool valid  = false;
    auto status = bc250_upstream_fwd_is_args_supported(p, id, p->GetInDataType(), false, &valid);
    if(status != CK_IMPL_STATUS_SUCCESS)
        return status;
    if(!valid)
        return unsupported();
    return ck_impl_try_catch([&] {
        miopen::solver::ConvSolution* raw = nullptr;
        auto created = bc250_upstream_fwd_get_solution(ctx, p, id, false, &raw);
        if(created != CK_IMPL_STATUS_SUCCESS)
            throw CkImplException(created, "Private DL upstream solution creation failed");
        std::unique_ptr<miopen::solver::ConvSolution> solution(raw);
        CK_IMPL_THROW_IF_FALSE(solution && solution->invoker_factory,
                               CK_IMPL_STATUS_INTERNAL_ERROR,
                               "Missing DL invoker");
        size_t alignment = alignof(float);
#ifdef MIOPEN_BC250_DL_FP16_PRIVATE
        // Conservatively cover the qualified half vector loads/stores. Native
        // Torch and owned probe allocations are at least 256-byte aligned.
        if(p->GetInDataType() == miopenHalf)
            alignment = 16;
#endif
        auto factory              = std::move(*solution->invoker_factory);
        solution->invoker_factory = [factory   = std::move(factory),
                                     x         = p->GetIn(),
                                     w         = p->GetWeights(),
                                     y         = p->GetOut(),
                                     workspace = solution->workspace_sz,
                                     alignment](const std::vector<miopen::Kernel>& kernels) {
            auto invoke = factory(kernels);
            return [invoke = std::move(invoke), x, w, y, workspace, alignment](
                       const miopen::Handle& handle, const miopen::AnyInvokeParams& params) {
                const auto& data = params.CastTo<miopen::conv::DataInvokeParams>();
                const auto& t    = data.tensors;
                if(data.alpha.GetAsDouble() != 1.0 || data.beta.GetAsDouble() != 0.0 ||
                   !(t.inDesc == x) || !(t.wDesc == w) || !(t.outDesc == y) ||
                   t.inDesc.GetCastType() || t.wDesc.GetCastType() || t.outDesc.GetCastType() ||
                   !t.in || !t.w || !t.out || data.workSpaceSize < workspace ||
                   (workspace && !data.workSpace))
                    MIOPEN_THROW(miopenStatusBadParm, "Private DL invocation contract mismatch");
                auto range = [alignment](const void* pointer, size_t bytes) {
                    const auto begin = reinterpret_cast<uintptr_t>(pointer);
                    if(begin % alignment || bytes > std::numeric_limits<uintptr_t>::max() - begin)
                        MIOPEN_THROW(miopenStatusBadParm, "Private DL pointer alignment/range");
                    return std::make_pair(begin, begin + bytes);
                };
                const auto in      = range(t.in, x.GetNumBytes());
                const auto weights = range(t.w, w.GetNumBytes());
                const auto out     = range(t.out, y.GetNumBytes());
                auto overlaps      = [](auto a, auto b) {
                    return a.first < b.second && b.first < a.second;
                };
                if(overlaps(out, in) || overlaps(out, weights))
                    MIOPEN_THROW(miopenStatusBadParm, "Private DL output overlaps an input");
                if(workspace)
                {
                    const auto scratch = range(data.workSpace, workspace);
                    if(overlaps(scratch, in) || overlaps(scratch, weights) ||
                       overlaps(scratch, out))
                        MIOPEN_THROW(miopenStatusBadParm, "Private DL workspace overlaps tensors");
                }
                invoke(handle, params);
            };
        };
        *out = solution.release();
    });
}
#ifdef MIOPEN_BC250_DL_FP16_PRIVATE
// Type strings omit the dtype. Publish their union, while all per-problem queries
// retain exact descriptor/type validation and the corresponding typed factory.
extern "C" ck_impl_status_t ck_impl_fwd_get_all_kernel_type_strings(CKKernelListHandle** out)
{
    return ck_impl_try_catch([&] {
        CK_IMPL_THROW_IF_NULL(out, CK_IMPL_STATUS_BAD_PARAM, "Null list output");
        auto result = std::make_unique<CKKernelListHandle>();
        auto add    = [&](auto value) {
            for(const auto& op : DeviceOpGFwdPtrs<decltype(value)>::GetInstances())
            {
                const auto id = op->GetTypeString();
                if(std::find(result->kernels.begin(), result->kernels.end(), id) ==
                   result->kernels.end())
                    result->kernels.push_back(id);
            }
        };
        add(float{});
        add(ck::half_t{});
        *out = result.release();
    });
}
#endif

// The loader requires every solver group even for a forward-only private plugin.
#define BC250_UNSUPPORTED_GROUP(prefix)                                                 \
    extern "C" ck_impl_status_t ck_impl_##prefix##_fill_valid_kernels(                  \
        const miopen::conv::ProblemDescription* p,                                      \
        miopenDataType_t,                                                               \
        bool,                                                                           \
        CKKernelListHandle** out)                                                       \
    {                                                                                   \
        if(p == nullptr || out == nullptr)                                              \
            return CK_IMPL_STATUS_BAD_PARAM;                                            \
        return empty_list(out);                                                         \
    }                                                                                   \
    extern "C" ck_impl_status_t ck_impl_##prefix##_is_applicable(                       \
        const miopen::conv::ProblemDescription* p, miopenDataType_t, bool, bool* out)   \
    {                                                                                   \
        if(p == nullptr || out == nullptr)                                              \
            return CK_IMPL_STATUS_BAD_PARAM;                                            \
        *out = false;                                                                   \
        return CK_IMPL_STATUS_SUCCESS;                                                  \
    }                                                                                   \
    extern "C" ck_impl_status_t ck_impl_##prefix##_is_args_supported(                   \
        const miopen::conv::ProblemDescription* p,                                      \
        const char* id,                                                                 \
        miopenDataType_t,                                                               \
        bool,                                                                           \
        bool* out)                                                                      \
    {                                                                                   \
        if(p == nullptr || id == nullptr || out == nullptr)                             \
            return CK_IMPL_STATUS_BAD_PARAM;                                            \
        *out = false;                                                                   \
        return CK_IMPL_STATUS_SUCCESS;                                                  \
    }                                                                                   \
    extern "C" ck_impl_status_t ck_impl_##prefix##_get_workspace_size(                  \
        const miopen::conv::ProblemDescription* p, miopenDataType_t, bool, size_t* out) \
    {                                                                                   \
        if(p == nullptr || out == nullptr)                                              \
            return CK_IMPL_STATUS_BAD_PARAM;                                            \
        return unsupported();                                                           \
    }                                                                                   \
    extern "C" ck_impl_status_t ck_impl_##prefix##_get_solution(                        \
        const miopen::ExecutionContext* ctx,                                            \
        const miopen::conv::ProblemDescription* p,                                      \
        const char* id,                                                                 \
        bool,                                                                           \
        miopen::solver::ConvSolution** out)                                             \
    {                                                                                   \
        if(ctx == nullptr || p == nullptr || id == nullptr || out == nullptr)           \
            return CK_IMPL_STATUS_BAD_PARAM;                                            \
        return unsupported();                                                           \
    }
BC250_UNSUPPORTED_GROUP(bwd)
BC250_UNSUPPORTED_GROUP(wrw)
BC250_UNSUPPORTED_GROUP(3d_fwd)
BC250_UNSUPPORTED_GROUP(3d_bwd)
BC250_UNSUPPORTED_GROUP(3d_wrw)
BC250_UNSUPPORTED_GROUP(fused_bias_activ)
BC250_UNSUPPORTED_GROUP(fused_bias_res_add_activ)
BC250_UNSUPPORTED_GROUP(fused_grp_activ)
BC250_UNSUPPORTED_GROUP(fused_grp_bias_activ)
BC250_UNSUPPORTED_GROUP(depthwise_fwd)
BC250_UNSUPPORTED_GROUP(depthwise_bwd_data)
#undef BC250_UNSUPPORTED_GROUP
