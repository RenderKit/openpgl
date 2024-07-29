// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef BUILD_SHARED
#ifdef _WIN32
  #  ifdef openpgl_EXPORTS
  #    define OPENPGL_INTERFACE __declspec(dllexport)
  #  else
  #    define OPENPGL_INTERFACE __declspec(dllimport)
  #  endif
  #  define OPENPGL_DLLEXPORT __declspec(dllexport)
#else
#define OPENPGL_INTERFACE
#define OPENPGL_DLLEXPORT __attribute__((visibility("default")))
#endif
#else
#define OPENPGL_INTERFACE
#define OPENPGL_DLLEXPORT
#endif

#define OPENPGL_CORE_INTERFACE OPENPGL_INTERFACE

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

inline pgl_vec3f normalize(pgl_vec3f n)
{
  const float f = 1.f / std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
  return {n.x * f, n.y * f, n.z * f};
}

#if defined(PGL_USE_DIRECTION_COMPRESSION) || defined(OPENPGL_DIRECTION_COMPRESSION)
////////////////////////////////////////////////////////////////////////
//  Directional 32-Bit compression based on octahdral maps
//  as described by Meyer et al. in "On Floating-Point Normal Vectors"
//  (https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01737.x)
////////////////////////////////////////////////////////////////////////

uint32_t quantize_direction(const pgl_vec3f &n);
pgl_vec3f dequantize_direction(const uint32_t word);

struct pgl_direction
{
  uint32_t compressed_direction;
#ifdef __cplusplus
  pgl_direction() {}
  pgl_direction(const float &x, const float &y, const float &z)
  {
    compressed_direction = quantize_direction({x, y, z});
  }

  pgl_direction(const pgl_vec3f &d)
  {
    compressed_direction = quantize_direction({d.x, d.y, d.z});
  }

  pgl_direction &operator=(const pgl_vec3f &direction)
  {
    compressed_direction = quantize_direction(direction);
    return *this;
  }

  bool operator!=(const pgl_direction &b) const
  {
    return compressed_direction != b.compressed_direction;
  }

  operator pgl_vec3f() const
  {
    return dequantize_direction(compressed_direction);
  }
#endif
};
#else
typedef pgl_vec3f pgl_direction;
#endif

#if defined(PGL_USE_COLOR_COMPRESSION) || defined(OPENPGL_RADIANCE_COMPRESSION)

uint32_t vec3f2rgbe(const pgl_vec3f rgb);
pgl_vec3f rgbe2vec3f(const uint32_t rgbe);

////////////////////////////////////////////////////////////////////////
//  RGBE 32-Bit compression for radiance values (linear RGB) using
//  shared exponent encoding as proposed by Greg Ward
//  (https://www.graphics.cornell.edu/~bjw/rgbe.html)
////////////////////////////////////////////////////////////////////////

struct pgl_spectrum
{
  uint32_t spectrum;
#ifdef __cplusplus
  pgl_spectrum() {}
  pgl_spectrum(const pgl_vec3f rgb)
  {
    spectrum = vec3f2rgbe(rgb);
  }
  operator pgl_vec3f() const
  {
    return rgbe2vec3f(spectrum);
  }
  pgl_spectrum &operator=(const pgl_vec3f &rgb)
  {
    spectrum = vec3f2rgbe(rgb);
    return *this;
  }
#endif
};
#else
typedef pgl_vec3f pgl_spectrum;
#endif