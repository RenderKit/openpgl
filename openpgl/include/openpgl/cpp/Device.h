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

    Field NewField(PGLFieldArguments &args) const;

    Field NewFieldFromFile(const std::string& fieldFileName) const;

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

OPENPGL_INLINE Field Device::NewField(PGLFieldArguments &args) const
{
    OPENPGL_ASSERT(m_deviceHandle);
    auto fieldHandle = pglDeviceNewField(m_deviceHandle, args);
    return {fieldHandle};
}

OPENPGL_INLINE Field Device::NewFieldFromFile(const std::string& fieldFileName) const
{
    OPENPGL_ASSERT(m_deviceHandle);
    auto fieldHandle = pglDeviceNewFieldFromFile(m_deviceHandle, fieldFileName.c_str());
    if (!fieldHandle)
        throw std::runtime_error("could not load field from file!");
    return {fieldHandle};
}

} // api
} // openpgl