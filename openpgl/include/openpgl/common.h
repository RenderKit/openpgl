// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once


#define PGL_VECTOR_SIZE 4
#define PGL_VMM_MAX_COMPONENTS 32
#define PGL_VMM_MAX_KAPPA 320000.f

typedef struct
{
  float x, y, z;
} pgl_vec3f;

typedef pgl_vec3f pgl_point3f;

typedef struct
{
  pgl_vec3f lower, upper;
} pgl_box3f;