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
#include "surfacesamplingdistribution.h"
#include "volumesamplingdistribution.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
struct Field;
#else
typedef ManagedObject Field;
#endif


typedef Field *PGLField;

OPENPGL_CORE_INTERFACE void pglReleaseField(PGLField field);

OPENPGL_CORE_INTERFACE bool pglFieldStoreToFile(PGLField field, const char* fieldFileName);

OPENPGL_CORE_INTERFACE size_t pglFieldGetIteration(PGLField field);

OPENPGL_CORE_INTERFACE void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds);

OPENPGL_CORE_INTERFACE pgl_box3f pglFieldGetSceneBounds(PGLField field);

//OPENPGL_CORE_INTERFACE pgl_box3f pglFieldGetSceneBounds(PGLField field);

OPENPGL_CORE_INTERFACE void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage);

OPENPGL_CORE_INTERFACE void pglFieldReset(PGLField field);

//OPENPGL_CORE_INTERFACE PGLRegion pglFieldGetSurfaceRegion(PGLField field, pgl_point3f position, PGLSampler* sampler);

//OPENPGL_CORE_INTERFACE PGLRegion pglFieldGetVolumeRegion(PGLField field, pgl_point3f position, PGLSampler* sampler);

OPENPGL_CORE_INTERFACE PGLSurfaceSamplingDistribution pglFieldNewSurfaceSamplingDistribution(PGLField field);

OPENPGL_CORE_INTERFACE bool pglFieldInitSurfaceSamplingDistribution(PGLField field, PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point3f position, float* sample1D, const bool useParallaxComp);

OPENPGL_CORE_INTERFACE PGLVolumeSamplingDistribution pglFieldNewVolumeSamplingDistribution(PGLField field);

OPENPGL_CORE_INTERFACE bool pglFieldInitVolumeSamplingDistribution(PGLField field, PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point3f position, float* sample1D, const bool useParallaxComp);

OPENPGL_CORE_INTERFACE bool pglFieldValidate(PGLField field);

#ifdef __cplusplus
}  // extern "C"
#endif
