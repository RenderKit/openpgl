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

#ifdef __cplusplus
struct SurfaceSamplingDistribution;
#else
typedef ManagedObject SurfaceSamplingDistribution;
#endif

#ifdef __cplusplus
struct VolumeSamplingDistribution;
#else
typedef ManagedObject VolumeSamplingDistribution;
#endif

typedef SurfaceSamplingDistribution *PGLSurfaceSamplingDistribution;
typedef VolumeSamplingDistribution *PGLVolumeSamplingDistribution;
typedef Distribution *PGLDistribution;
typedef Region *PGLRegion;

bool pglDistributionIsValid(PGLDistribution distribution);

//void pglGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool &useParallaxComp);

PGLSurfaceSamplingDistribution pglNewSurfaceSamplingDistribution();

void pglReleaseSurfaceSamplingDistribution(PGLSurfaceSamplingDistribution SurfaceSamplingDistribution);

void pglSurfaceSamplingDistributionInit(PGLSurfaceSamplingDistribution SurfaceSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, pgl_vec3f normal, bool useParallaxComp = true, bool useCosine = true);

pgl_vec3f pglSurfaceSamplingDistributionSample(PGLSurfaceSamplingDistribution SurfaceSamplingDistribution, pgl_point2f sample);

float pglSurfaceSamplingDistributionPDF(PGLSurfaceSamplingDistribution SurfaceSamplingDistribution, pgl_vec3f direction);

bool pglSurfaceSamplingDistributionIsValid(PGLSurfaceSamplingDistribution SurfaceSamplingDistribution);

void pglSurfaceSamplingDistributionClear(PGLSurfaceSamplingDistribution SurfaceSamplingDistribution);

PGLVolumeSamplingDistribution pglNewVolumeSamplingDistribution();

void pglReleaseVolumeSamplingDistribution(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

void pglVolumeSamplingDistributionInit(PGLVolumeSamplingDistribution VolumeSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp = true);

pgl_vec3f pglVolumeSamplingDistributionSample(PGLVolumeSamplingDistribution VolumeSamplingDistribution, pgl_point2f sample);

float pglVolumeSamplingDistributionPDF(PGLVolumeSamplingDistribution VolumeSamplingDistribution, pgl_vec3f direction);

bool pglVolumeSamplingDistributionIsValid(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

void pglVolumeSamplingDistributionClear(PGLVolumeSamplingDistribution VolumeSamplingDistribution);


#ifdef __cplusplus
}  // extern "C"
#endif