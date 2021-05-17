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
extern "C" {
#endif

#ifdef __cplusplus
struct SampleStorage
{
};

struct PGLSampleData;

#else
typedef ManagedObject Region;
#endif

typedef SampleStorage *PGLSampleStorage;

typedef PGLSampleData PGLSampleData;

OPENPGL_CORE_INTERFACE PGLSampleStorage pglNewSampleStorage();

OPENPGL_CORE_INTERFACE void pglReleaseSampleStorage(PGLSampleStorage sampleStorage);

OPENPGL_CORE_INTERFACE void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, PGLSampleData& sample);

OPENPGL_CORE_INTERFACE void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData* samples, size_t numSamples);

OPENPGL_CORE_INTERFACE void pglSampleStorageReserve(PGLSampleStorage sampleStorage, const size_t sizeSurface, const size_t sizeVolume);

OPENPGL_CORE_INTERFACE void pglSampleStorageClear(PGLSampleStorage sampleStorage);

OPENPGL_CORE_INTERFACE size_t pglSampleStorageGetSizeSurface(PGLSampleStorage sampleStorage);

OPENPGL_CORE_INTERFACE size_t pglSampleStorageGetSizeVolume(PGLSampleStorage sampleStorage);

#ifdef __cplusplus
}  // extern "C"
#endif