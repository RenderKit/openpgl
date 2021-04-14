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

void pglSampleStorageSetSceneBounds(PGLSampleStorage sampleStorage, pgl_box3f bounds);

void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, PGLSampleData& sample);

void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData* samples, uint32_t numSamples);

void pglSampleStorageReserve(PGLSampleStorage sampleStorage, const size_t sizeSurface, const size_t sizeVolume);

void pglSampleStorageClear(PGLSampleStorage sampleStorage);

//void pglReserveVolume(PGLSampleStorage sampleStorage, const size_t size);

//void pglClearSurface(PGLSampleStorage sampleStorage);

//void pglClearVolume(PGLSampleStorage sampleStorage);

uint32_t pglSampleStorageGetSizeSurface(PGLSampleStorage sampleStorage);

uint32_t pglSampleStorageGetSizeVolume(PGLSampleStorage sampleStorage);

/*
void pglSampleDataSetPosition(PGLSampleData& sampleData, const pgl_point3f pos);

void pglSampleDataSetDirection(PGLSampleData& sampleData, const pgl_vec3f direction);

void pglSampleDataSetDistance(PGLSampleData& sampleData, const float distance);

void pglSampleDataSetPDF(PGLSampleData& sampleData, const float pdf);

void pglSampleDataSetWeight(PGLSampleData& sampleData, const float weight);

void pglSampleDataSetFlags(PGLSampleData& sampleData, const int flags);
*/

#ifdef __cplusplus
}  // extern "C"
#endif