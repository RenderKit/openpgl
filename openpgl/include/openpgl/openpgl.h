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

#include "common.h"
#include "data.h"
#include "region.h"
#include "samplestorage.h"
#include "pathsegmentstorage.h"
#include "device.h"
#include "field.h"
#include "surfacesamplingdistribution.h"
#include "volumesamplingdistribution.h"
