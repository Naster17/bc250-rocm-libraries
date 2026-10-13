# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

add_subdirectory(rocblaslt)

if(HIPBLASLT_ENABLE_GFX1013_SCALAR)
    target_sources(hipblaslt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/gfx1013_scalar.cpp")
endif()

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_ostream.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt-ext-op.cpp"
)
