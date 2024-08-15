#pragma once

#include <cstring>
#if defined(OPENPGL_GPU_SYCL_SUPPORT)
#include <sycl/sycl.hpp>
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
    sycl::queue q;
#endif
    Device(DeviceTypes dt) : m_deviceType(dt) {}

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
                q.wait();
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
                return sycl::malloc_shared<T>(numElements, q);
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                void *devPtr;
                cudaMalloc(&devPtr, numElements * sizeof(T));
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
            sycl:
                free(ptr, q);
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaFree(ptr);
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
                q.memcpy(devicePtr, hostPtr, numElements * sizeof(T));
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaMemcpy(devicePtr, hostPtr, numElements * sizeof(T), cudaMemcpyHostToDevice);
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
                q.memcpy(hostPtr, devicePtr, numElements * sizeof(T));
                break;
            }
#endif
#if defined(OPENPGL_GPU_CUDA_SUPPORT)
            case EDeviceType_CUDA:
            {
                cudaMemcpy(hostPtr, devicePtr, numElements * sizeof(T), cudaMemcpyDeviceToHost);
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