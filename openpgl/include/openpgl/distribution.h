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
struct BSDFSamplingDistribution;
#else
typedef ManagedObject BSDFSamplingDistribution;
#endif

#ifdef __cplusplus
struct PhaseFunctionSamplingDistribution;
#else
typedef ManagedObject PhaseFunctionSamplingDistribution;
#endif

typedef BSDFSamplingDistribution *PGLBSDFSamplingDistribution;
typedef PhaseFunctionSamplingDistribution *PGLPhaseFunctionSamplingDistribution;
typedef Distribution *PGLDistribution;
typedef Region *PGLRegion;

bool pglDistributionIsValid(PGLDistribution distribution);

//void pglGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool &useParallaxComp);

PGLBSDFSamplingDistribution pglNewBSDFSamplingDistribution();

void pglReleaseBSDFSamplingDistribution(PGLBSDFSamplingDistribution bsdfSamplingDistribution);

void pglBSDFSamplingDistributionInit(PGLBSDFSamplingDistribution bsdfSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, pgl_vec3f normal, bool useParallaxComp = true, bool useCosine = true);

pgl_vec3f pglBSDFSamplingDistributionSample(PGLBSDFSamplingDistribution bsdfSamplingDistribution, pgl_point2f sample);

float pglBSDFSamplingDistributionPDF(PGLBSDFSamplingDistribution bsdfSamplingDistribution, pgl_vec3f direction);

bool pglBSDFSamplingDistributionIsValid(PGLBSDFSamplingDistribution bsdfSamplingDistribution);

void pglBSDFSamplingDistributionClear(PGLBSDFSamplingDistribution bsdfSamplingDistribution);


PGLPhaseFunctionSamplingDistribution pglNewPhaseFunctionSamplingDistribution();

void pglReleasePhaseFunctionSamplingDistribution(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution);

void pglPhaseFunctionSamplingDistributionInit(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp = true);

pgl_vec3f pglPhaseFunctionSamplingDistributionSample(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution, pgl_point2f sample);

float pglPhaseFunctionSamplingDistributionPDF(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution, pgl_vec3f direction);

bool pglPhaseFunctionSamplingDistributionIsValid(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution);

void pglPhaseFunctionSamplingDistributionClear(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution);


#ifdef __cplusplus
}  // extern "C"
#endif