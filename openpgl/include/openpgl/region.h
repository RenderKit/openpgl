// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "distribution.h"

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __cplusplus
struct Region;
#else
typedef ManagedObject Region;
#endif

typedef Region *PGLRegion;

#ifdef __cplusplus
}  // extern "C"
#endif