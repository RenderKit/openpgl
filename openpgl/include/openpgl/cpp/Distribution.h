// Copyright 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

namespace openpgl
{
namespace cpp
{
/**
 * @brief The Distribution class represents the approximation of the directional guiding distribution.
 * E.g., this distribution can represent the incident radiance field, its product with a BSDF or phase function,
 * or another target distribution.   
 * 
 */
struct Distribution
{
    Distribution(PGLDistribution distributionHandle);
    private:
        PGLDistribution m_distributionHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE Distribution::Distribution(PGLDistribution distributionHandle)
{
    m_distributionHandle = distributionHandle;
}
}
}