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

struct PGLDirectionalSampleData;

#else
typedef ManagedObject Region;
#endif

typedef SampleStorage *PGLSampleStorage;

typedef PGLDirectionalSampleData PGLSampleData;

PGLSampleStorage pglNewSampleStorage();

void pglReleaseSampleStorage(PGLSampleStorage sampleStorage);

void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, PGLSampleData& sample);

void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData* samples, uint32_t numSamples);

void pglSampleStorageReserve(PGLSampleStorage sampleStorage, const size_t sizeSurface, const size_t sizeVolume);

void pglSampleStorageClear(PGLSampleStorage sampleStorage);

uint32_t pglSampleStorageGetSizeSurface(PGLSampleStorage sampleStorage);

uint32_t pglSampleStorageGetSizeVolume(PGLSampleStorage sampleStorage);

#ifdef __cplusplus
}  // extern "C"
#endif