// Copyright 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

#include "Field.h"

#include <string>
#include <stdexcept>

namespace openpgl
{
namespace cpp
{
struct Field;
/**
 * @brief The Device class is a key component of OpenPGL. It is used to set
 * compute architecture and optimizations (e.g., SIMD) used for the implementation of
 * guiding structures (e.g., Field, SurfaceSamplingDistribution, or VolumeSamplingDistribution).
 */
struct Device
{
    /**
     * @brief Creates a new Device object.
     * 
     * Creates a new Device object. The object can be optimized
     * for different compute architechtures. On the CPU the device can be 
     * optimized for different SIMD architechtures (e.g., SSE4, AVX, or AVX-512)
     * 
     * @param deviceType The device optimization type.
     */
    Device(PGL_DEVICE_TYPE deviceType);

    ~Device();

    Device(const Device&) = delete;

    friend struct openpgl::cpp::Field;
    private:
        PGLDevice m_deviceHandle {nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE Device::Device(PGL_DEVICE_TYPE deviceType)
{
    m_deviceHandle = pglNewDevice(deviceType);
}

OPENPGL_INLINE Device::~Device()
{
    OPENPGL_ASSERT(m_deviceHandle);
    pglReleaseDevice(m_deviceHandle);
    m_deviceHandle = nullptr;
}

} // api
} // openpgl