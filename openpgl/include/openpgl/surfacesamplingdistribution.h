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

//OPENPGL_CORE_INTERFACE PGLSurfaceSamplingDistribution pglNewSurfaceSamplingDistribution();

OPENPGL_CORE_INTERFACE void pglReleaseSurfaceSamplingDistribution(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

//OPENPGL_CORE_INTERFACE void pglSurfaceSamplingDistributionInit(PGLSurfaceSamplingDistribution SurfaceSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp = true);

OPENPGL_CORE_INTERFACE void pglSurfaceSamplingDistributionApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal);

OPENPGL_CORE_INTERFACE bool pglSurfaceSamplingDistributionSupportsApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

OPENPGL_CORE_INTERFACE pgl_vec3f pglSurfaceSamplingDistributionSample(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample);

OPENPGL_CORE_INTERFACE float pglSurfaceSamplingDistributionPDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction);

OPENPGL_CORE_INTERFACE float pglSurfaceSamplingDistributionSamplePDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample, pgl_vec3f &direction);

OPENPGL_CORE_INTERFACE bool pglSurfaceSamplingDistributionValidate(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

OPENPGL_CORE_INTERFACE void pglSurfaceSamplingDistributionClear(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

OPENPGL_CORE_INTERFACE PGLRegion pglSurfaceSamplingGetRegion(PGLSurfaceSamplingDistribution surfaceSamplingDistribution);

#ifdef __cplusplus
}  // extern "C"
#endif