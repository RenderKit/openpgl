// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

namespace openpgl
{
namespace cpp
{
    struct Field;
    struct PathSegmentStorage;

/**
 * @brief A wrapper class for the random number sampler of the renderer.
 * 
 * @deprecated This will be removed in the future
 * 
 */
struct Sampler
{
    Sampler(PGLSampler samplerHandle);

    friend struct openpgl::cpp::Field;
    friend struct openpgl::cpp::PathSegmentStorage;
    private:
        PGLSampler m_samplerHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE Sampler::Sampler(PGLSampler samplerHandle)
{
    m_samplerHandle = samplerHandle;
}

}
}