
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

namespace openpgl
{
namespace cpp
{
    //typedef pgl_vec3f Vector3;

    OPENPGL_INLINE pgl_vec3f Vector3(float x, float y, float z)
    {
       pgl_vec3f vec3{x,y,z};
       return vec3; 
    }

    OPENPGL_INLINE pgl_vec2f Vector2(float x, float y)
    {
       pgl_vec2f vec2{x,y};
       return vec2; 
    }

   OPENPGL_INLINE pgl_point3f Point3(float x, float y, float z)
    {
       pgl_point3f point3{x,y,z};
       return point3; 
    }

    OPENPGL_INLINE pgl_vec2f Point2(float x, float y)
    {
       pgl_point2f point2{x,y};
       return point2; 
    }

    OPENPGL_INLINE pgl_box3f Box3(pgl_point3f lower, pgl_point3f upper)
    {
       pgl_box3f box3{lower, upper};
       return box3;
    }

    OPENPGL_INLINE pgl_box3f Box3(float lower_x, float lower_y, float lower_z, float upper_x, float upper_y, float upper_z)
    {
       pgl_box3f box3{pgl_point3f{lower_x, lower_y, lower_z}, pgl_point3f{upper_x,upper_y,upper_z}};
       return box3;
    }

}
}