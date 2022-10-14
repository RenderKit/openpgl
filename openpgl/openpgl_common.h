// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ONE_OVER_FOUR_PI 0.07957747154594767

#include <embreeSrc/common/math/constants.h>
#include <embreeSrc/common/math/math.h>
#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>
#include <embreeSrc/common/math/bbox.h>

#include <cmath>
#include <algorithm>

#if defined(__WIN32__) || defined(__MACOSX__)

inline void sincosf(const float theta, float* sin, float* cos)
{
    embree::sincosf(theta, sin, cos);
}

#endif

namespace openpgl
{

  inline void* alignedMalloc(size_t size, size_t align)
  {
    if (size == 0)
      return nullptr;

    assert((align & (align-1)) == 0);
    void* ptr = _mm_malloc(size,align);

    if (size != 0 && ptr == nullptr)
      throw std::bad_alloc();

    return ptr;
  }

  inline void alignedFree(void* ptr)
  {
    if (ptr)
      _mm_free(ptr);
  }
}

#define OPENPGL_ALIGNED_STRUCT_(align)                                           \
  void* operator new(size_t size) { return openpgl::alignedMalloc(size,align); } \
  void operator delete(void* ptr) { openpgl::alignedFree(ptr); }                 \
  void* operator new[](size_t size) { return openpgl::alignedMalloc(size,align); } \
  void operator delete[](void* ptr) { openpgl::alignedFree(ptr); }

namespace openpgl
{
    typedef embree::Vec2<float> Vector2;
    typedef embree::Vec3<float> Vector3;
    typedef embree::Vec2<float> Point2;
    typedef embree::Vec3<float> Point3;

    typedef embree::BBox<Vector3> BBox;

    inline float dot(Vector2 &a, Vector2 &b)
    {
        return embree::dot(a, b);
    }

    inline float dot(Vector3 &a, Vector3 &b)
    {
        return embree::dot(a, b);
    }
}

//#define OPENPGL_DISABLE_ASSERTS

#ifndef OPENPGL_DISABLE_ASSERTS
#include <assert.h>
#define OPENPGL_ASSERT(cond) assert(cond);
//#define OPENPGL_ASSERT_MSG(cond, msg) SAssertEx(cond, msg);
#else
#define OPENPGL_ASSERT(cond)
//#define OPENPGL_ASSERT_MSG(cond, msg)
#endif

namespace openpgl
{
    template <int VecSize>
    inline float sum( const embree::vfloat<VecSize> &v)
    {
        float sum = 0.0f;
        for ( int i = 0; i < VecSize; i++)
        {
            sum += v[i];
        }
        return sum;
    }


    inline Vector2 toSphericalCoordinates(const Vector3 &v) {
        Vector2 result(
            std::acos(v.z),
            std::atan2(v.y, v.x)
        );
        if (result.y < 0)
            result.y += 2.0f*(float)M_PI;
        return result;
    }

    inline Vector3 sphericalDirection(const float &cosTheta, const float &sinTheta, const float &cosPhi, const float &sinPhi)
    {
        return Vector3( sinTheta * cosPhi,
	        		    sinTheta * sinPhi,
					    cosTheta);
    };

    inline Vector3 sphericalDirection(const float &theta, const float &phi)
    {
        const float cosTheta = std::cos(theta);
        const float sinTheta = std::sin(theta);
        const float cosPhi = std::cos(phi);
        const float sinPhi = std::sin(phi);

        return sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
    };

    inline Vector3 squareToUniformSphere(const Vector2 sample){
        float z = 1.0f - 2.0f * sample.y;
        float r = std::sqrt(std::max(0.f,(1.0f - z*z)));
        float sinPhi, cosPhi;
        sincosf(2.0f * M_PI * sample.x, &sinPhi, &cosPhi);
        return Vector3(r * cosPhi, r * sinPhi, z);
    }

}