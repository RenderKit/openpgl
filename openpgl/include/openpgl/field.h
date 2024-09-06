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
#include "fieldstatistics.h"
#include "region.h"
#include "samplestorage.h"
#include "surfacesamplingdistribution.h"
#include "volumesamplingdistribution.h"

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
    struct Field;
#else
typedef ManagedObject Field;
#endif

    typedef Field *PGLField;

    OPENPGL_CORE_INTERFACE void pglReleaseField(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldStoreToFile(PGLField field, const char *fieldFileName);

    OPENPGL_CORE_INTERFACE size_t pglFieldGetIteration(PGLField field);

    OPENPGL_CORE_INTERFACE void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds);

    OPENPGL_CORE_INTERFACE pgl_box3f pglFieldGetSceneBounds(PGLField field);

    OPENPGL_CORE_INTERFACE void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglFieldUpdateSurface(PGLField field, PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglFieldUpdateVolume(PGLField field, PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglFieldReset(PGLField field);

    OPENPGL_CORE_INTERFACE PGLSurfaceSamplingDistribution pglFieldNewSurfaceSamplingDistribution(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldInitSurfaceSamplingDistribution(PGLField field, PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point3f position,
                                                                        float *sample1D);

    OPENPGL_CORE_INTERFACE PGLVolumeSamplingDistribution pglFieldNewVolumeSamplingDistribution(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldInitVolumeSamplingDistribution(PGLField field, PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point3f position,
                                                                       float *sample1D);

    OPENPGL_CORE_INTERFACE bool pglFieldValidate(PGLField field);

    OPENPGL_CORE_INTERFACE bool pglFieldCompare(PGLField fieldA, PGLField fieldB);

    OPENPGL_CORE_INTERFACE PGLFieldStatistics pglFieldGetSurfaceStatistics(PGLField field);

    OPENPGL_CORE_INTERFACE PGLFieldStatistics pglFieldGetVolumeStatistics(PGLField field);

    OPENPGL_CORE_INTERFACE size_t pglFieldGetNumSurfaceCaches(PGLField field);

    OPENPGL_CORE_INTERFACE size_t pglFieldGetNumVolumeCaches(PGLField field);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglFieldGetSurfaceCachePosition(PGLField field, size_t idx);

    OPENPGL_CORE_INTERFACE pgl_vec3f pglFieldGetVolumeCachePosition(PGLField field, size_t idx);

    OPENPGL_CORE_INTERFACE pgl_box3f pglFieldGetSurfaceCacheSampleBounds(PGLField field, size_t idx);

    OPENPGL_CORE_INTERFACE pgl_box3f pglFieldGetVolumeCacheSampleBounds(PGLField field, size_t idx);

    OPENPGL_CORE_INTERFACE pgl_cacheInfo pglFieldGetSurfaceCacheInfo(PGLField field, size_t idx);

    OPENPGL_CORE_INTERFACE pgl_cacheInfo pglFieldGetVolumeCacheInfo(PGLField field, size_t idx);

#ifdef __cplusplus
}  // extern "C"
#endif
