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
#include "samplestorage.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
struct PathSegmentStorage
{
};

struct PathSegment
{
};
#else
typedef ManagedObject Region;
#endif

typedef PathSegmentStorage *PGLPathSegmentStorage;

typedef PathSegment *PGLPathSegment;

PGLPathSegmentStorage pglNewPathSegmentStorage();

void pglPathSegmentStorageReserve(PGLPathSegmentStorage pathSegmentStorage, size_t size);

void pglPathSegmentStorageClear(PGLPathSegmentStorage pathSegmentStorage);

size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage, const bool useNEEMiWeights = false, const bool guideDirectLight = false);

void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment sample);

void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample);

//void pglAddSamples(PGLSampleStorage sampleStorage, PGLSampleData* samples, uint32_t numSamples);


#ifdef __cplusplus
}  // extern "C"
#endif