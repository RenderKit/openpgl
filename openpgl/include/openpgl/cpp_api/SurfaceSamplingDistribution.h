// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "Region.h"

namespace openpgl
{
namespace cpp
{
struct SurfaceSamplingDistribution
{
    SurfaceSamplingDistribution();
    ~SurfaceSamplingDistribution();

    pgl_vec3f Sample(const pgl_point2f& sample2D)const;

    float PDF(const pgl_vec3f& direction) const;

    bool IsValid() const;

    void Clear();

    void Init(const Region& region, const pgl_point3f& pos, const pgl_vec3f& normal, const bool& useParallaxCompensation = true, const bool& useCosineProduct= true);


    private:
        PGLBSDFSamplingDistribution m_bsdfSamplingDistributionHandle{nullptr};
};

OPENPGL_INLINE SurfaceSamplingDistribution::SurfaceSamplingDistribution()
{
    m_bsdfSamplingDistributionHandle = pglNewBSDFSamplingDistribution();
}

OPENPGL_INLINE SurfaceSamplingDistribution::~SurfaceSamplingDistribution()
{
    OPENPGL_ASSERT(m_bsdfSamplingDistributionHandle);
    if(m_bsdfSamplingDistributionHandle)
        pglReleaseBSDFSamplingDistribution(m_bsdfSamplingDistributionHandle);
    m_bsdfSamplingDistributionHandle = nullptr;
}


pgl_vec3f SurfaceSamplingDistribution::Sample(const pgl_point2f& sample2D)const
{
    OPENPGL_ASSERT(m_bsdfSamplingDistributionHandle);
    return pglBSDFSamplingDistributionSample(m_bsdfSamplingDistributionHandle, sample2D);
}

float SurfaceSamplingDistribution::PDF(const pgl_vec3f& direction) const
{
    OPENPGL_ASSERT(m_bsdfSamplingDistributionHandle);
    return pglBSDFSamplingDistributionPDF(m_bsdfSamplingDistributionHandle, direction);
}

bool SurfaceSamplingDistribution::IsValid() const
{
    OPENPGL_ASSERT(m_bsdfSamplingDistributionHandle);
    return pglBSDFSamplingDistributionIsValid(m_bsdfSamplingDistributionHandle);
}

void SurfaceSamplingDistribution::Clear()
{
    OPENPGL_ASSERT(m_bsdfSamplingDistributionHandle);
    return pglBSDFSamplingDistributionClear(m_bsdfSamplingDistributionHandle);
}

void SurfaceSamplingDistribution::Init(const Region& region, const pgl_point3f& pos, const pgl_vec3f& normal, const bool& useParallaxCompensation, const bool& useCosineProduct)
{
    OPENPGL_ASSERT(m_bsdfSamplingDistributionHandle);
    pglBSDFSamplingDistributionInit(m_bsdfSamplingDistributionHandle, region.m_regionHandle, pos, normal, useParallaxCompensation, useCosineProduct);
}

}
}