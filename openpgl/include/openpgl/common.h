// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef BUILD_SHARED
  #ifdef _WIN32
  #  ifdef openpgl_EXPORTS
  #    define OPENPGL_INTERFACE __declspec(dllexport)
  #  else
  #    define OPENPGL_INTERFACE __declspec(dllimport)
  #  endif
  #  define OPENPGL_DLLEXPORT __declspec(dllexport)
  #else
  #  define OPENPGL_INTERFACE
  #  define OPENPGL_DLLEXPORT __attribute__ ((visibility ("default")))
  #endif
#else
  #define OPENPGL_INTERFACE
  #define OPENPGL_DLLEXPORT
#endif

#define OPENPGL_CORE_INTERFACE OPENPGL_INTERFACE

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
  float x, y;
} pgl_vec2f;

typedef pgl_vec2f pgl_point2f;

typedef struct
{
  pgl_vec3f lower, upper;
} pgl_box3f;

inline void pglVec3f(pgl_vec3f &vec, const float x, const float y, const float z)
{
  vec.x = x;
  vec.y = y;
  vec.z = z;
}

inline void pglVec3fAdd(pgl_vec3f &veca, const pgl_vec3f &vecb)
{
  veca.x += vecb.x;
  veca.y += vecb.y;
  veca.z += vecb.z;
}

inline void pglVec2f(pgl_vec2f &vec, const float x, const float y)
{
  vec.x = x;
  vec.y = y;
}

inline void pglVec2fAdd(pgl_vec2f &veca, const pgl_vec2f &vecb)
{
  veca.x += vecb.x;
  veca.y += vecb.y;
}

inline void pglPoint3f(pgl_point3f &vec, const float x, const float y, const float z)
{
  pglVec3f(vec, x, y, z);
}

inline void pglPoint2f(pgl_point2f &vec, const float x, const float y)
{
  pglVec2f(vec, x, y);
}

inline void pglBox3f(pgl_box3f& box, const float lx, const float ly, const float lz, const float ux, const float uy, const float uz)
{
  pglVec3f(box.lower, lx, ly, lz);
  pglVec3f(box.upper, ux, uy, uz);
}