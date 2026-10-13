# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

if(HIPSPARSELT_GFX1013_SCALAR_ONLY)
  target_sources(hipsparselt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gfx1013_scalar.hip")
else()
  target_sources(hipsparselt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt.cpp")
  add_subdirectory(rocsparselt)
endif()
