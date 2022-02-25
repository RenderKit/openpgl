
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

namespace openpgl
{
namespace cpp
{
   /**
    * @brief Wrapper function to simulate a C++ constructur
    * for a 3d vector of type pgl_vec3f.
    * 
    * @param x 
    * @param y 
    * @param z 
    * @return pgl_vec3f 
    */
   OPENPGL_INLINE pgl_vec3f Vector3(float x, float y, float z)
   {
      pgl_vec3f vec3{x,y,z};
      return vec3; 
   }

   /**
    * @brief Wrapper function to simulate a C++ constructor
    * for a 2d vector of type pgl_vec2f.
    * 
    * @param x 
    * @param y 
    * @return pgl_vec2f 
    */
   OPENPGL_INLINE pgl_vec2f Vector2(float x, float y)
   {
      pgl_vec2f vec2{x,y};
      return vec2; 
   }

   /**
    * @brief Wrapper function to simulate a C++ constructor
    * for a 3d position of type pgl_point3f.
    * 
    * @param x 
    * @param y 
    * @param z 
    * @return pgl_point3f 
    */
   OPENPGL_INLINE pgl_point3f Point3(float x, float y, float z)
   {
      pgl_point3f point3{x,y,z};
      return point3; 
   }

   /**
    * @brief Wrapper function to simulate a C++ constructor
    * for a 2d position of type pgl_point2f.
    * 
    * @param x 
    * @param y 
    * @return pgl_point2f 
    */
   OPENPGL_INLINE pgl_vec2f Point2(float x, float y)
   {
      pgl_point2f point2{x,y};
      return point2; 
   }

   /**
    * @brief Wrapper function to simulate a C++ constructor
    * for a bounding box of type pgl_box3f.
    * 
    * @param lower 
    * @param upper 
    * @return pgl_box3f 
    */
   OPENPGL_INLINE pgl_box3f Box3(pgl_point3f lower, pgl_point3f upper)
   {
      pgl_box3f box3{lower, upper};
      return box3;
   }

   /**
    * @brief Wrapper function to simulate a C++ constructor
    * for a bounding box of type pgl_box3f.
    * 
    * @param lower_x 
    * @param lower_y 
    * @param lower_z 
    * @param upper_x 
    * @param upper_y 
    * @param upper_z 
    * @return pgl_box3f 
    */
   OPENPGL_INLINE pgl_box3f Box3(float lower_x, float lower_y, float lower_z, float upper_x, float upper_y, float upper_z)
   {
      pgl_box3f box3{pgl_point3f{lower_x, lower_y, lower_z}, pgl_point3f{upper_x,upper_y,upper_z}};
      return box3;
   }

}
}