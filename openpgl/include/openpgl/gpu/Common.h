#ifndef OPENPGL_BUILD
#include <openpgl/defines.h>
#endif

#if defined(OPENPGL_GPU_SYCL)
#include <sycl/sycl.hpp>
#endif

#define USE_TREELETS
#ifndef ONE_OVER_FOUR_PI
#define ONE_OVER_FOUR_PI 0.07957747154594767F
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif

#if defined(OPENPGL_GPU_CPU)
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif
#include <cstring>
#endif

#if defined(OPENPGL_GPU_SYCL)
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif
#ifdef OPENPGL_GPU_CALLABLE
#undef OPENPGL_GPU_CALLABLE
#endif
#define OPENPGL_GPU_CALLABLE
#elif defined(OPENPGL_GPU_CUDA)// && defined(__CUDACC__)
#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290E-07F
#endif
#ifdef OPENPGL_GPU_CALLABLE
#undef OPENPGL_GPU_CALLABLE
#endif
#define OPENPGL_GPU_CALLABLE __host__ __device__
#else
#ifdef OPENPGL_GPU_CALLABLE
#undef OPENPGL_GPU_CALLABLE
#endif
#define OPENPGL_GPU_CALLABLE
#endif
