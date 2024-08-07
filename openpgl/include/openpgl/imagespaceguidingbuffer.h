// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "common.h"
#include "region.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
struct ImageSpaceGuidingBuffer;
#else
typedef ManagedObject ImageSpaceGuidingBuffer;
#endif

typedef ImageSpaceGuidingBuffer *PGLImageSpaceGuidingBuffer;

OPENPGL_CORE_INTERFACE PGLImageSpaceGuidingBuffer pglFieldNewImageSpaceGuidingBuffer(const pgl_point2i resolution);

OPENPGL_CORE_INTERFACE PGLImageSpaceGuidingBuffer pglFieldNewImageSpaceGuidingBufferFromFile(const char* fileName);

OPENPGL_CORE_INTERFACE void pglReleaseImageSpaceGuidingBuffer(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer);

OPENPGL_CORE_INTERFACE void pglImageSpaceGuidingBufferUpdate(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer);

OPENPGL_CORE_INTERFACE void pglImageSpaceGuidingBufferAddSample(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer, const pgl_point2i pixel, const PGLImageSpaceSample sample);

OPENPGL_CORE_INTERFACE void pglImageSpaceGuidingBufferStore(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer, const char* fileName);

OPENPGL_CORE_INTERFACE pgl_vec3f pglImageSpaceGuidingBufferGetPixelContributionEstimate(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer, const pgl_point2i pixel);

OPENPGL_CORE_INTERFACE bool pglImageSpaceGuidingBufferIsReady(PGLImageSpaceGuidingBuffer imageSpaceGuidingBuffer);

#ifdef __cplusplus
}  // extern "C"
#endif