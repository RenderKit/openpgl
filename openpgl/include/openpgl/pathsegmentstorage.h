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
#include "region.h"
#include "sampler.h"
#include "samplestorage.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
struct PathSegmentStorage
{
};

struct PGLPathSegmentData;
#else
typedef ManagedObject Region;
#endif

typedef PathSegmentStorage *PGLPathSegmentStorage;

typedef PGLPathSegmentData *PGLPathSegment;

OPENPGL_CORE_INTERFACE PGLPathSegmentStorage pglNewPathSegmentStorage();

OPENPGL_CORE_INTERFACE void pglReleasePathSegmentStorage(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE void pglPathSegmentStorageReserve(PGLPathSegmentStorage pathSegmentStorage, size_t size);

OPENPGL_CORE_INTERFACE void pglPathSegmentStorageClear(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage,const bool &spaltSamples, PGLSampler* sampler, const bool useNEEMiWeights = false, const bool guideDirectLight = false);

OPENPGL_CORE_INTERFACE const PGLSampleData* pglPathSegmentStorageGetSamples(PGLPathSegmentStorage pathSegmentStorage, size_t &nSamples);

//OPENPGL_CORE_INTERFACE void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment sample);

OPENPGL_CORE_INTERFACE void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample);

OPENPGL_CORE_INTERFACE PGLPathSegment pglPathSegmentNextSegment(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE bool pglPathSegmentStorageSamplesValid(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE bool  pglPathSegmentStorageSegmentsValid(PGLPathSegmentStorage pathSegmentStorage);

//OPENPGL_CORE_INTERFACE void pglAddSamples(PGLSampleStorage sampleStorage, PGLSampleData* samples, size_t numSamples);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetPosition(PGLPathSegment pathSegmentStorage, pgl_point3f position);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetNormal(PGLPathSegment pathSegmentStorage, pgl_vec3f normal);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetDirectionIn(PGLPathSegment pathSegmentStorage, pgl_vec3f directionIn);

OPENPGL_CORE_INTERFACE pgl_vec3f pglPathSegmentGetDirectionIn(PGLPathSegment pathSegmentStorage);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetPDFDirectionIn(PGLPathSegment pathSegmentStorage, float pdfDirectionIn);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetDirectionOut(PGLPathSegment pathSegmentStorage, pgl_vec3f directionOut);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetVolumeScatter(PGLPathSegment pathSegmentStorage, bool volumeScatter);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetScatteringWeight(PGLPathSegment pathSegmentStorage, pgl_vec3f scatteringWeight);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetDirectContribution(PGLPathSegment pathSegmentStorage, pgl_vec3f directScatter);

OPENPGL_CORE_INTERFACE void pglPathSegmentAddDirectContribution(PGLPathSegment pathSegmentStorage, pgl_vec3f directScatter);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetScatteredContribution(PGLPathSegment pathSegmentStorage, pgl_vec3f directScatter);

OPENPGL_CORE_INTERFACE void pglPathSegmentAddScatteredContribution(PGLPathSegment pathSegmentStorage, pgl_vec3f directScatter);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetMiWeight(PGLPathSegment pathSegmentStorage, float miWeight);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetRussianRouletteProbability(PGLPathSegment pathSegmentStorage, float russianRouletteProbability);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetEta(PGLPathSegment pathSegmentStorage, float eta);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetIsDelta(PGLPathSegment pathSegmentStorage, bool isDelta);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetRoughness(PGLPathSegment pathSegmentStorage, float roughness);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetRegion(PGLPathSegment pathSegmentStorage, const PGLRegion region);

OPENPGL_CORE_INTERFACE void pglPathSegmentSetTransmittanceWeight(PGLPathSegment pathSegmentStorage, pgl_vec3f transmittanceWeight);


#ifdef __cplusplus
}  // extern "C"
#endif