// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
struct FieldStatistics;
#else
typedef ManagedObject FieldStatistics;
#endif

typedef FieldStatistics *PGLFieldStatistics;

OPENPGL_CORE_INTERFACE void pglReleaseFieldStatistics(PGLFieldStatistics fieldStatistics);

OPENPGL_CORE_INTERFACE PGLString pglFieldStatisticsToString(PGLFieldStatistics fieldStatistics);

OPENPGL_CORE_INTERFACE PGLString pglFieldStatisticsHeaderCSVString(PGLFieldStatistics fieldStatistics);

OPENPGL_CORE_INTERFACE PGLString pglFieldStatisticsToCSVString(PGLFieldStatistics fieldStatistics);

#ifdef __cplusplus
}  // extern "C"
#endif
