// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include <cstdint>
#include <cmath>
#include <algorithm>


////////////////////////////////////////////////////////////////////////
//  RGBE 32-Bit compression for radiance values (linear RGB) using 
//  shared exponent encoding as proposed by Greg Ward
//  (https://www.graphics.cornell.edu/~bjw/rgbe.html)
////////////////////////////////////////////////////////////////////////

/* standard conversion from float pixels to rgbe pixels */
/* note: you can remove the "inline"s if your compiler complains about it */

inline uint32_t vec3f2rgbe(const pgl_vec3f rgb)
{
  uint32_t rgbe;
  float v;
  int e;

  unsigned char* rgbe_ptr = (unsigned char*) &rgbe;

  v = rgb.x;
  if (rgb.y > v) v = rgb.y;
  if (rgb.z > v) v = rgb.z;
  if (v < 1e-32f) {
    rgbe_ptr[0] = rgbe_ptr[1] = rgbe_ptr[2] = rgbe_ptr[3] = 0;
  }
  else {
    v = std::frexp(v,&e) * 256.0f/v;
    rgbe_ptr[0] = (unsigned char) (rgb.x * v);
    rgbe_ptr[1] = (unsigned char) (rgb.y * v);
    rgbe_ptr[2] = (unsigned char) (rgb.z * v);
    rgbe_ptr[3] = (unsigned char) (e + 128);
  }
  return rgbe;
}

/* standard conversion from rgbe to float pixels */
/* note: Ward uses ldexp(col+0.5,exp-(128+8)).  However we wanted pixels */
/*       in the range [0,1] to map back into the range [0,1].            */

inline pgl_vec3f rgbe2vec3f(const uint32_t rgbe)
{
  pgl_vec3f rgb;
  const unsigned char* rgbe_ptr = (const unsigned char*) &rgbe;

  if (rgbe_ptr[3]) {   //nonzero pixel
    const float f = std::ldexp(1.0f,rgbe_ptr[3]-(int)(128+8));
    rgb.x = rgbe_ptr[0] * f;
    rgb.y = rgbe_ptr[1] * f;
    rgb.z = rgbe_ptr[2] * f;
  }
  else {
    rgb.x = rgb.y = rgb.z = 0.f;
  }
  return rgb;
}

////////////////////////////////////////////////////////////////////////
//  Directional 32-Bit compression based on octahdral maps
//  as described by Meyer et al. in "On Floating-Point Normal Vectors"
//  (https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01737.x)
////////////////////////////////////////////////////////////////////////

// represent 0, -1 and 1 precisely by integers
inline uint32_t quantize_direction(const pgl_vec3f& n) {
  const float nl1 = 1.f / (std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z));
  pgl_vec2f pn = {n.x * nl1, n.y * nl1};
  if (n.z <= 0.0f) {
    pgl_vec2f signNotZero = {pn.x >= 0.0f ? 1.0f : -1.0f, pn.y >= 0.0f ? 1.0f : -1.0f};
    pgl_vec2f b = {std::abs(pn.y), std::abs(pn.x)};
    pn = {(1.f - b.x)*signNotZero.x, (1.f - b.y)*signNotZero.y};
  }
  const float f = float(0x8000u);
  const int32_t i = int(0x8000u);

  pn.x *= f;
  pn.y *= f;
  
  const int32_t imin = -0x7FFF;
  const int32_t imax = 0x7FFF;
  const int32_t ix = std::max( imin , std::min(imax, (int32_t) pn.x));
  const int32_t iy = std::max( imin , std::min(imax, (int32_t) pn.y));
  const uint32_t ux = i + ix;
  const uint32_t uy = i + iy;  
  return ux | (uy << 16);
}

inline pgl_vec3f dequantize_direction(const uint32_t word) {
  int32_t ix = word & 0xFFFF;
  ix -= 0x8000;
  int32_t iy = word >> 16;
  iy -= 0x8000;
  const float f = 1.f / float(uint32_t(0x7FFF));
  pgl_vec2f n = {float(ix) * f, float(iy) * f};
  float nl1 = std::abs(n.x) + std::abs(n.y);
  if (nl1 >= 1.0f) {
    pgl_vec2f signNotZero = {n.x >= 0.0f ? 1.0f : -1.0f, n.y >= 0.0f ? 1.0f : -1.0f};
    pgl_vec2f b = {std::abs(n.y), std::abs(n.x)};
    n = {(1.f - b.x)*signNotZero.x, (1.f - b.y)*signNotZero.y};
  }
  return normalize({n.x, n.y, 1.0f - nl1});

////////////////////////////////////////////////////////////////////////
}