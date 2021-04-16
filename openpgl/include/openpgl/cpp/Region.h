// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

#include "Distribution.h"

namespace openpgl
{
namespace cpp
{

struct Region
{
    Region(PGLRegion regionHandle);

    bool GetValid() const;

    Distribution GetDistribution(pgl_point3f samplePosition, const bool &useParallaxComp) const;

    friend class PathSegment;
    friend class SurfaceSamplingDistribution;
    friend class VolumeSamplingDistribution;
    //private:
        PGLRegion m_regionHandle{nullptr};
};

OPENPGL_INLINE Region::Region(PGLRegion regionHandle)
{
    m_regionHandle = regionHandle;
}

OPENPGL_INLINE bool Region::GetValid()const
{
    if(m_regionHandle)
        return pglRegionGetValid(m_regionHandle);
    else
        return false;
}

OPENPGL_INLINE Distribution Region::GetDistribution(pgl_point3f samplePosition, const bool &useParallaxComp) const
{
    OPENPGL_ASSERT(m_regionHandle);
    PGLDistribution distributionHandle = pglRegionGetDistribution(m_regionHandle, samplePosition, useParallaxComp);
    return Distribution(distributionHandle);
}

}
}