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

uint32_t pglFieldGetIteration(PGLField field);

//void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds);

//pgl_box3f pglFieldGetSceneBounds(PGLField field);

void pglFieldUpdate(PGLField field, pgl_box3f bounds, PGLSampleStorage sampleStorage, uint32_t numPerPixelSamples);

//uint32_t pglGetTrainingIteration(PGLField field);

uint32_t pglFieldGetTotalSPP(PGLField field);

PGLRegion pglFieldGetSurfaceRegion(PGLField field, pgl_point3f position, PGLSampler* sampler);

PGLRegion pglFieldGetVolumeRegion(PGLField field, pgl_point3f position, PGLSampler* sampler);

#ifdef __cplusplus
}  // extern "C"
#endif
