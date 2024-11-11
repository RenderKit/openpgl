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

#define OPENPGL_CPU_GPU_LAMBDA(...) [=] OPENPGL_GPU_CALLABLE(__VA_ARGS__) mutable

#if defined(OPENPGL_GPU_CUDA)
template <typename F>
inline int MyGetBlockSize(const char *description, F kernel) {
    // Note: this isn't reentrant, but that's fine for our purposes...
    static std::map<std::type_index, int> kernelBlockSizes;

    std::type_index index = std::type_index(typeid(F));

    auto iter = kernelBlockSizes.find(index);
    if (iter != kernelBlockSizes.end())
        return iter->second;

    int minGridSize, blockSize;
    CUDA_CHECK(
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0));
    kernelBlockSizes[index] = blockSize;
    //LOG_VERBOSE("[%s]: block size %d", description, blockSize);

    return blockSize;
}

// GPU Launch Function Declarations
template <typename F>
void ParallelFor(const char *description, int nItems, F &&func);

#ifdef __NVCC__
template <typename F>
__global__ void Kernel(F func, int nItems) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;
    func(tid);
}

template <typename F>
void ParallelFor(const char *description, int nItems, F &&func) {
    auto kernel = &Kernel<F>;
    int blockSize = MyGetBlockSize(description, kernel);
    int gridSize = (nItems + blockSize - 1) / blockSize;
    //std::cout << "gridSize = " << gridSize << "\t blockSize = " << blockSize << std::endl;
    kernel<<<gridSize, blockSize>>>(func, nItems);
}
#endif
#endif
#if defined(OPENPGL_GPU_CPU)
template <typename F>
void ParallelFor(const char *description, int nItems, F &&func) {
}
#endif