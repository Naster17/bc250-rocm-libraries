// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once
#include "data_types.hpp"

namespace hiptensor
{
    inline hiptensorStatus_t validatePlanPreference(hiptensorPlanPreference_t pref, bool scalar)
    {
        if(!pref)
            return HIPTENSOR_STATUS_NOT_INITIALIZED;
        if((pref->mSelectionAlgorithm != HIPTENSOR_ALGO_DEFAULT
            && pref->mSelectionAlgorithm != HIPTENSOR_ALGO_DEFAULT_PATIENT
            && pref->mSelectionAlgorithm != HIPTENSOR_ALGO_ACTOR_CRITIC)
           || (pref->mJit != HIPTENSOR_JIT_MODE_NONE && pref->mJit != HIPTENSOR_JIT_MODE_DEFAULT)
           || (pref->mCacheMode != HIPTENSOR_CACHE_MODE_NONE
               && pref->mCacheMode != HIPTENSOR_CACHE_MODE_PEDANTIC)
           || (pref->mAutotuneMode != HIPTENSOR_AUTOTUNE_MODE_NONE
               && pref->mAutotuneMode != HIPTENSOR_AUTOTUNE_MODE_INCREMENTAL)
           || pref->mIncrementalCount < 0 || pref->mKernelRank < 0)
            return HIPTENSOR_STATUS_INVALID_VALUE;
        // A single precompiled scalar implementation has no CK solution UID,
        // alternate candidates, ranked kernels, autotuning or JIT compilation.
        if(scalar
           && (pref->mJit != HIPTENSOR_JIT_MODE_NONE
               || pref->mCacheMode != HIPTENSOR_CACHE_MODE_NONE
               || pref->mAutotuneMode != HIPTENSOR_AUTOTUNE_MODE_NONE
               || pref->mIncrementalCount != 0 || pref->mKernelRank != 0))
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        return HIPTENSOR_STATUS_SUCCESS;
    }
}
