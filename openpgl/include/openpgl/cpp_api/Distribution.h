// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

namespace openpgl
{
namespace cpp
{
struct Distribution
{
    Distribution(PGLDistribution distributionHandle);
    private:
        PGLDistribution m_distributionHandle{nullptr};
};

OPENPGL_INLINE Distribution::Distribution(PGLDistribution distributionHandle)
{
    m_distributionHandle = distributionHandle;
}
}
}