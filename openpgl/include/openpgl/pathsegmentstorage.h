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

struct PGLPathSegmentData;
#else
typedef ManagedObject Region;
#endif

typedef PathSegmentStorage *PGLPathSegmentStorage;

typedef PGLPathSegmentData *PGLPathSegment;

PGLPathSegmentStorage pglNewPathSegmentStorage();

void pglPathSegmentStorageReserve(PGLPathSegmentStorage pathSegmentStorage, size_t size);

void pglPathSegmentStorageClear(PGLPathSegmentStorage pathSegmentStorage);

size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage, const bool useNEEMiWeights = false, const bool guideDirectLight = false);

void pglPathSegmentStorageGetSamples(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData &samples, uint32_t &nSamples);

void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment sample);

void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample);

PGLPathSegment pglPathSegmentNextSegment(PGLPathSegmentStorage pathSegmentStorage);

//void pglAddSamples(PGLSampleStorage sampleStorage, PGLSampleData* samples, uint32_t numSamples);


void pglPathSegmentSetPosition(PGLPathSegment pathSegmentStorage, pgl_point3f position);

void pglPathSegmentSetNormal(PGLPathSegment pathSegmentStorage, pgl_vec3f normal);

void pglPathSegmentSetDirectionIn(PGLPathSegment pathSegmentStorage, pgl_vec3f directionIn);

pgl_vec3f pglPathSegmentGetDirectionIn(PGLPathSegment pathSegmentStorage);

void pglPathSegmentSetPDFDirectionIn(PGLPathSegment pathSegmentStorage, float pdfDirectionIn);

void pglPathSegmentSetDirectionOut(PGLPathSegment pathSegmentStorage, pgl_vec3f directionOut);

void pglPathSegmentSetVolumeScatter(PGLPathSegment pathSegmentStorage, bool volumeScatter);

void pglPathSegmentSetScatteringWeight(PGLPathSegment pathSegmentStorage, pgl_vec3f scatteringWeight);

void pglPathSegmentSetDirectContribution(PGLPathSegment pathSegmentStorage, pgl_vec3f directScatter);

void pglPathSegmentAddDirectContribution(PGLPathSegment pathSegmentStorage, pgl_vec3f directScatter);

void pglPathSegmentSetScatteredContribution(PGLPathSegment pathSegmentStorage, pgl_vec3f directScatter);

void pglPathSegmentAddScatteredContribution(PGLPathSegment pathSegmentStorage, pgl_vec3f directScatter);

void pglPathSegmentSetMiWeight(PGLPathSegment pathSegmentStorage, float miWeight);

void pglPathSegmentSetRussianRouletteProbability(PGLPathSegment pathSegmentStorage, float russianRouletteProbability);

void pglPathSegmentSetEta(PGLPathSegment pathSegmentStorage, float eta);

void pglPathSegmentSetIsDelta(PGLPathSegment pathSegmentStorage, float isDelta);

void pglPathSegmentSetRoughness(PGLPathSegment pathSegmentStorage, float roughness);

void pglPathSegmentSetRegion(PGLPathSegment pathSegmentStorage, const PGLRegion region);

void pglPathSegmentSetTransmittanceWeight(PGLPathSegment pathSegmentStorage, pgl_vec3f transmittanceWeight);


#ifdef __cplusplus
}  // extern "C"
#endif