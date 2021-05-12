// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "common.h"
#include "region.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
struct SurfaceSamplingDistribution;
#else
typedef ManagedObject SurfaceSamplingDistribution;
#endif

typedef SurfaceSamplingDistribution *PGLSurfaceSamplingDistribution;

//PGLSurfaceSamplingDistribution pglNewSurfaceSamplingDistribution();

void pglReleaseSurfaceSamplingDistribution(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

//void pglSurfaceSamplingDistributionInit(PGLSurfaceSamplingDistribution SurfaceSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp = true);

void pglSurfaceSamplingDistributionApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal);

pgl_vec3f pglSurfaceSamplingDistributionSample(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample);

float pglSurfaceSamplingDistributionPDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction);

bool pglSurfaceSamplingDistributionIsValid(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

void pglSurfaceSamplingDistributionClear(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

PGLRegion pglSurfaceSamplingGetRegion(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

#ifdef __cplusplus
}  // extern "C"
#endif