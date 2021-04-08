// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

//#define PGL_VERSION_MAJOR 0
//#define PGL_VERSION_MINOR 1
//#define PGL_VERSION_PATCH 0

#if defined(PGL_API_NAMESPACE)
  #define PGL_API_NAMESPACE_BEGIN namespace  {
  #define PGL_API_NAMESPACE_END }
  #define PGL_API_NAMESPACE_USING using namespace ;
  #define PGL_API_EXTERN_C
  #define PGL_NAMESPACE_BEGIN namespace  {
  #define PGL_NAMESPACE_END }
  #define PGL_NAMESPACE_USING using namespace ;
  #undef PGL_API_NAMESPACE
#else
  #define PGL_API_NAMESPACE_BEGIN
  #define PGL_API_NAMESPACE_END
  #define PGL_API_NAMESPACE_USING
  #if defined(__cplusplus)
    #define PGL_API_EXTERN_C extern "C"
  #else
    #define PGL_API_EXTERN_C
  #endif
  #define PGL_NAMESPACE_BEGIN namespace pgl {
  #define PGL_NAMESPACE_END }
  #define PGL_NAMESPACE_USING using namespace pgl;
#endif

#if defined(PGL_EXPORT_API)
  #define PGL_API PGL_API_EXPORT
#else
  #define PGL_API PGL_API_IMPORT
#endif