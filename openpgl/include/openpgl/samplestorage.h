// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdlib.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __cplusplus
    struct SampleStorage;

    struct PGLSampleData;

#else
typedef ManagedObject Region;
#endif

    typedef SampleStorage *PGLSampleStorage;

    typedef PGLSampleData PGLSampleData;

    OPENPGL_CORE_INTERFACE PGLSampleStorage pglNewSampleStorage();

    OPENPGL_CORE_INTERFACE PGLSampleStorage pglNewSampleStorageFromFile(const char *sampleStorageFileName);

    OPENPGL_CORE_INTERFACE void pglReleaseSampleStorage(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE bool pglSampleStorageStoreToFile(PGLSampleStorage sampleStorage, const char *sampleStorageFileName);

    OPENPGL_CORE_INTERFACE void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, const PGLSampleData &sample);

    OPENPGL_CORE_INTERFACE void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData *samples, size_t numSamples);

    OPENPGL_CORE_INTERFACE void pglSampleStorageAddZeroValueSample(PGLSampleStorage sampleStorage, const PGLZeroValueSampleData &sample);

    OPENPGL_CORE_INTERFACE void pglSampleStorageAddZeroValueSamples(PGLSampleStorage sampleStorage, const PGLZeroValueSampleData *samples, size_t numSamples);

    OPENPGL_CORE_INTERFACE void pglSampleStorageReserve(PGLSampleStorage sampleStorage, const size_t sizeSurface, const size_t sizeVolume);

    OPENPGL_CORE_INTERFACE void pglSampleStorageClear(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglSampleStorageClearSurface(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglSampleStorageClearVolume(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE size_t pglSampleStorageGetSizeSurface(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE size_t pglSampleStorageGetSizeVolume(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE size_t pglSampleStorageGetSizeZeroValueSurface(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE size_t pglSampleStorageGetSizeZeroValueVolume(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE PGLSampleData pglSampleStorageGetSampleSurface(PGLSampleStorage sampleStorage, const int idx);

    OPENPGL_CORE_INTERFACE PGLSampleData pglSampleStorageGetSampleVolume(PGLSampleStorage sampleStorage, const int idx);

    OPENPGL_CORE_INTERFACE PGLZeroValueSampleData pglSampleStorageGetZeroValueSampleSurface(PGLSampleStorage sampleStorage, const int idx);

    OPENPGL_CORE_INTERFACE PGLZeroValueSampleData pglSampleStorageGetZeroValueSampleVolume(PGLSampleStorage sampleStorage, const int idx);

    OPENPGL_CORE_INTERFACE bool pglSampleStorageValidate(PGLSampleStorage sampleStorage);

    OPENPGL_CORE_INTERFACE void pglSampleStorageMerge(PGLSampleStorage sampleStorageA, PGLSampleStorage sampleStorageB);

    OPENPGL_CORE_INTERFACE bool pglSampleStorageCompare(PGLSampleStorage sampleStorageA, PGLSampleStorage sampleStorageB);

#ifdef __cplusplus
}  // extern "C"
#endif