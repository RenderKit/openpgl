// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "distribution.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
struct Region
{
};
#else
typedef ManagedObject Region;
#endif

typedef Region *PGLRegion;

//bool pglRegionGetValid(PGLRegion region);

//PGLDistribution pglRegionGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool useParallaxComp);


#ifdef __cplusplus
}  // extern "C"
#endif