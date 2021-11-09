// Copyright 2021 Intel Corporation
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
 * @brief
 *
 */
struct Device
{
    /**
     * @brief Construct a new Device object
     *
     * @param args
     */
    Device(PGL_DEVICE_TYPE deviceType);

    ~Device();

    Device(const Device&) = delete;

    friend struct openpgl::cpp::Field;
    private:
        PGLDevice m_deviceHandle {nullptr};
};

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