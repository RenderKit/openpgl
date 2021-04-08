// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
struct Distribution
{
};
#else
typedef ManagedObject Region;
#endif

typedef Distribution *PGLDistribution;


bool pglDistributionIsValid(PGLDistribution distribution);

//void pglGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool &useParallaxComp);


#ifdef __cplusplus
}  // extern "C"
#endif