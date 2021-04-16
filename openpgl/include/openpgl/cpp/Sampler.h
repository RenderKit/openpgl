// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

namespace openpgl
{
namespace cpp
{
struct Sampler
{
    Sampler(PGLSampler samplerHandle);

    friend class Field;
    friend class PathSegmentStorage;
    private:
        PGLSampler m_samplerHandle{nullptr};
};

OPENPGL_INLINE Sampler::Sampler(PGLSampler samplerHandle)
{
    m_samplerHandle = samplerHandle;
}

}
}