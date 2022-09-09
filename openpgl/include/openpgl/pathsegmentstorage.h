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

OPENPGL_CORE_INTERFACE void pglPathSegmentSetMaxDistance(PGLPathSegmentStorage pathSegmentStorage, float maxDistance);

OPENPGL_CORE_INTERFACE float pglPathSegmentGetMaxDistance(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE int pglPathSegmentGetNumSegments(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE int pglPathSegmentGetNumSamples(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage, const bool spaltSamples, PGLSampler* sampler, const bool useNEEMiWeights = false, const bool guideDirectLight = false, const bool rrAffectsDirectContribution = true);

OPENPGL_CORE_INTERFACE pgl_vec3f pglPathSegmentStorageCalculatePixelEstimate(PGLPathSegmentStorage pathSegmentStorage, const bool rrAffectsDirectContribution = true);


OPENPGL_CORE_INTERFACE const PGLSampleData* pglPathSegmentStorageGetSamples(PGLPathSegmentStorage pathSegmentStorage, size_t &nSamples);

//OPENPGL_CORE_INTERFACE void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment sample);

OPENPGL_CORE_INTERFACE void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample);

OPENPGL_CORE_INTERFACE PGLPathSegmentData* pglPathSegmentStorageNextSegment(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegmentData segment);

OPENPGL_CORE_INTERFACE bool pglPathSegmentStorageValidateSamples(PGLPathSegmentStorage pathSegmentStorage);

OPENPGL_CORE_INTERFACE bool  pglPathSegmentStorageValidateSegments(PGLPathSegmentStorage pathSegmentStorage);

#ifdef __cplusplus
}  // extern "C"
#endif