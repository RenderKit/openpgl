// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef float (*PGLSamplerNext1DFunction)(void* samplerPtr);

typedef pgl_point2f (*PGLSamplerNext2DFunction)(void* samplerPtr);

struct PGLSampler
{
    void* sampler;
    PGLSamplerNext1DFunction next1D;
    PGLSamplerNext2DFunction next2D;
};

#ifdef __cplusplus
}  // extern "C"
#endif