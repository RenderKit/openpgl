// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "config.h"
#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
struct Device;
#else
typedef ManagedObject Device;
#endif

typedef Device *PGLDevice;

OPENPGL_CORE_INTERFACE PGLDevice pglNewDevice(PGL_DEVICE_TYPE deviceType);

OPENPGL_CORE_INTERFACE PGLField pglDeviceNewField(PGLDevice device, PGLFieldArguments args);

OPENPGL_CORE_INTERFACE PGLField pglDeviceNewFieldFromFile(PGLDevice device, const char* fieldFileName);

OPENPGL_CORE_INTERFACE void pglReleaseDevice(PGLDevice device);

#ifdef __cplusplus
}  // extern "C"
#endif
