// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define OPENPGL_INLINE inline

//#define OPENPGL_DISABLE_ASSERTS

#ifndef OPENPGL_DISABLE_ASSERTS
#include <assert.h>
#define OPENPGL_ASSERT(cond) assert(cond);
//#define OPENPGL_ASSERT_MSG(cond, msg) SAssertEx(cond, msg);
#else
#define OPENPGL_ASSERT(cond)
//#define OPENPGL_ASSERT_MSG(cond, msg)
#endif

#include "defines.h"
#include "common.h"
#if defined(OPENPGL_DIRECTION_COMPRESSION) || defined(OPENPGL_RADIANCE_COMPRESSION)
#include "compression.h"
#endif
#include "data.h"
#include "region.h"
#include "samplestorage.h"
#include "pathsegmentstorage.h"
#include "device.h"
#include "field.h"
#include "fieldstatistics.h"
#include "surfacesamplingdistribution.h"
#include "volumesamplingdistribution.h"
