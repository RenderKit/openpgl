// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
struct Region;
#else
typedef ManagedObject Region;
#endif

#ifdef __cplusplus
struct Distribution;
#else
typedef ManagedObject Distribution;
#endif



typedef Distribution *PGLDistribution;
typedef Region *PGLRegion;

//bool pglDistributionIsValid(PGLDistribution distribution);

//void pglGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool &useParallaxComp);



#ifdef __cplusplus
}  // extern "C"
#endif