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

/**
 * @brief
 *
 */
using VectorSize = PGLVectorSize;

struct Device
{
    /**
     * @brief Construct a new Device object
     *
     * @param args
     */
    Device(VectorSize vectorSize);

    ~Device();

    Device(const Device&) = delete;

    friend class Field;
    private:
        PGLDevice m_deviceHandle {nullptr};
};

OPENPGL_INLINE Device::Device(VectorSize vectorSize)
{
    m_deviceHandle = pglNewDevice(vectorSize);
}

OPENPGL_INLINE Device::~Device()
{
    OPENPGL_ASSERT(m_deviceHandle);
    pglReleaseDevice(m_deviceHandle);
    m_deviceHandle = nullptr;
}

} // api
} // openpgl