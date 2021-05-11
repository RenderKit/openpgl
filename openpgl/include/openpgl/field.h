// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdlib.h>
#endif

#include "common.h"
#include "config.h"
#include "sampler.h"
#include "samplestorage.h"
#include "region.h"
#include "distribution.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
struct Field;
#else
typedef ManagedObject Field;
#endif


typedef Field *PGLField;

PGLField pglNewField(PGLFieldArguments args);

void pglReleaseField(PGLField field);

size_t pglFieldGetIteration(PGLField field);

void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds);

//pgl_box3f pglFieldGetSceneBounds(PGLField field);

void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage, size_t numPerPixelSamples);

//size_t pglGetTrainingIteration(PGLField field);

size_t pglFieldGetTotalSPP(PGLField field);

PGLRegion pglFieldGetSurfaceRegion(PGLField field, pgl_point3f position, PGLSampler* sampler);

PGLRegion pglFieldGetVolumeRegion(PGLField field, pgl_point3f position, PGLSampler* sampler);

PGLSurfaceSamplingDistribution pglFieldNewSurfaceSamplingDistribution(PGLField field);

bool pglFieldInitSurfaceSamplingDistriubtion(PGLField field, PGLSurfaceSamplingDistribution surfaceSamplingDistriubtion, pgl_point3f position, const float sample1D, const bool useParallaxComp);

PGLVolumeSamplingDistribution pglFieldNewVolumeSamplingDistribution(PGLField field);

bool pglFieldInitVolumeSamplingDistriubtion(PGLField field, PGLVolumeSamplingDistribution volumeSamplingDistriubtion, pgl_point3f position, const float sample1D, const bool useParallaxComp);


#ifdef __cplusplus
}  // extern "C"
#endif
