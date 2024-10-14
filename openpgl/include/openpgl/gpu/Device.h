#pragma once

#include <cstring>
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
#include <sycl/sycl.hpp>
#endif

#if defined(OPENPGL_GPU_CUDA_SUPPORT)
#include <cuda_runtime.h>

#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline void checkCudaError() {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << " CUDAError " << std::endl;
    }
}

#endif

namespace openpgl
{
namespace gpu
{
struct Device
{
    enum DeviceTypes
    {
        EDeviceType_CPU = 0,
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
        EDeviceType_SYCL,
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
        EDeviceType_CUDA,
#endif
    };

    DeviceTypes m_deviceType{EDeviceType_CPU};
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
    ::sycl::queue* q;
#endif
    Device(DeviceTypes dt, void* ptr = nullptr) : m_deviceType(dt) {
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
    q = reinterpret_cast<::sycl::queue*>(ptr);
#endif        
    }

    ~Device() {}

    void wait()
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                q->wait();
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaDeviceSynchronize();
                break;
            }
#endif
            default:
            {
                break;
            }
        }
    }

    template <class T>
    T *mallocArray(size_t numElements)
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                return new T[numElements];
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                return ::sycl::malloc_shared<T>(numElements, *q);
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                void *devPtr;
                cudaErrchk(cudaMalloc(&devPtr, numElements * sizeof(T)));
                checkCudaError();
                return (T *)devPtr;
                break;
            }
#endif
            default:
            {
                return new T[numElements];
                break;
            }
        }
    }

    template <class T>
    void freeArray(T *ptr)
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                delete[] ptr;
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                ::sycl::free(ptr, *q);
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaErrchk(cudaFree(ptr));
                checkCudaError();
                break;
            }
#endif
            default:
            {
                delete[] ptr;
                break;
            }
        }
    }

    template <class T>
    void memcpyArrayToGPU(T *devicePtr, const T *hostPtr, size_t numElements)
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                std::memcpy(devicePtr, hostPtr, numElements * sizeof(T));
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                q->memcpy(devicePtr, hostPtr, numElements * sizeof(T));
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaErrchk(cudaMemcpy(devicePtr, hostPtr, numElements * sizeof(T), cudaMemcpyHostToDevice));
                checkCudaError();
                break;
            }
#endif
            default:
            {
                std::memcpy(devicePtr, hostPtr, numElements * sizeof(T));
                break;
            }
        }
    }

    template <class T>
    void memcpyArrayFromGPU(T *devicePtr, T *hostPtr, size_t numElements)
    {
        switch (m_deviceType)
        {
            case EDeviceType_CPU:
            {
                std::memcpy(hostPtr, devicePtr, numElements * sizeof(T));
                break;
            }
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
            case EDeviceType_SYCL:
            {
                q->memcpy(hostPtr, devicePtr, numElements * sizeof(T));
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaErrchk(cudaMemcpy(hostPtr, devicePtr, numElements * sizeof(T), cudaMemcpyDeviceToHost));
                checkCudaError();
                break;
            }
#endif
            default:
            {
                std::memcpy(hostPtr, devicePtr, numElements * sizeof(T));
                break;
            }
        }
    }
};
}  // namespace gpu
}  // namespace openpgl