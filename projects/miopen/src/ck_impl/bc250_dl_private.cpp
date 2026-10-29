// SPDX-License-Identifier: MIT
// Private API5 prototype. Not part of production plugin admission or installation.
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/conv/problem_description.hpp>

// Reuse the matching upstream implementation behind a restricted public facade.
#define ck_impl_fwd_fill_valid_kernels bc250_upstream_fwd_fill_valid_kernels
#define ck_impl_fwd_is_applicable bc250_upstream_fwd_is_applicable
#define ck_impl_fwd_is_args_supported bc250_upstream_fwd_is_args_supported
#define ck_impl_fwd_get_workspace_size bc250_upstream_fwd_get_workspace_size
#define ck_impl_fwd_get_solution bc250_upstream_fwd_get_solution
#include "ck_grouped_conv_fwd_impl.cpp"
#undef ck_impl_fwd_fill_valid_kernels
#undef ck_impl_fwd_is_applicable
#undef ck_impl_fwd_is_args_supported
#undef ck_impl_fwd_get_workspace_size
#undef ck_impl_fwd_get_solution
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
    return p != nullptr && type == miopenFloat && !tf32 && p->Is2d() && p->IsDirectionForward() &&
           p->GetInDataType() == miopenFloat && p->GetWeightsDataType() == miopenFloat &&
           p->GetOutDataType() == miopenFloat && !p->HasMixedDataTypes() && !p->IsTensorsCasted() &&
           !p->HasNonPackedTensors() && (p->IsLayoutDefault() || p->IsLayoutNHWC()) &&
           !p->GetConv().attribute.deterministic;
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
        throw CkImplException(CK_IMPL_STATUS_INVALID_VALUE,
                              "Private gfx1013 DL plugin supports packed 2D FP32 forward only");
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
    return admitted(p, p->GetInDataType(), tf32)
               ? bc250_upstream_fwd_get_solution(ctx, p, id, false, out)
               : unsupported();
}
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
