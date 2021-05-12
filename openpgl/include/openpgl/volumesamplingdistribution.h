// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "common.h"
#include "region.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
struct VolumeSamplingDistribution;
#else
typedef ManagedObject VolumeSamplingDistribution;
#endif

typedef VolumeSamplingDistribution *PGLVolumeSamplingDistribution;

//PGLVolumeSamplingDistribution pglNewVolumeSamplingDistribution();

void pglReleaseVolumeSamplingDistribution(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

//void pglVolumeSamplingDistributionInit(PGLVolumeSamplingDistribution VolumeSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp = true);

pgl_vec3f pglVolumeSamplingDistributionSample(PGLVolumeSamplingDistribution VolumeSamplingDistribution, pgl_point2f sample);

float pglVolumeSamplingDistributionPDF(PGLVolumeSamplingDistribution VolumeSamplingDistribution, pgl_vec3f direction);

bool pglVolumeSamplingDistributionIsValid(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

void pglVolumeSamplingDistributionClear(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

PGLRegion pglVolumeSamplingGetRegion(PGLVolumeSamplingDistribution VolumeSamplingDistribution);

#ifdef __cplusplus
}  // extern "C"
#endif