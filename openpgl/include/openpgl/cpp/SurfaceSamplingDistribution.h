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

    SurfaceSamplingDistribution(const SurfaceSamplingDistribution&) = delete;

    pgl_vec3f Sample(const pgl_point2f& sample2D)const;

    float PDF(const pgl_vec3f& direction) const;

    bool IsValid() const;

    void Clear();

    void Init(const Region& region, const pgl_point3f& pos, const pgl_vec3f& normal, const bool& useParallaxCompensation = true, const bool& useCosineProduct= true);


    private:
        PGLSurfaceSamplingDistribution m_surfaceSamplingDistributionHandle{nullptr};
};

SurfaceSamplingDistribution::SurfaceSamplingDistribution()
{
    m_surfaceSamplingDistributionHandle = pglNewSurfaceSamplingDistribution();
}

SurfaceSamplingDistribution::~SurfaceSamplingDistribution()
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    if(m_surfaceSamplingDistributionHandle)
        pglReleaseSurfaceSamplingDistribution(m_surfaceSamplingDistributionHandle);
    m_surfaceSamplingDistributionHandle = nullptr;
}


pgl_vec3f SurfaceSamplingDistribution::Sample(const pgl_point2f& sample2D)const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionSample(m_surfaceSamplingDistributionHandle, sample2D);
}

float SurfaceSamplingDistribution::PDF(const pgl_vec3f& direction) const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionPDF(m_surfaceSamplingDistributionHandle, direction);
}

bool SurfaceSamplingDistribution::IsValid() const
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionIsValid(m_surfaceSamplingDistributionHandle);
}

void SurfaceSamplingDistribution::Clear()
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    return pglSurfaceSamplingDistributionClear(m_surfaceSamplingDistributionHandle);
}

void SurfaceSamplingDistribution::Init(const Region& region, const pgl_point3f& pos, const pgl_vec3f& normal, const bool& useParallaxCompensation, const bool& useCosineProduct)
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    pglSurfaceSamplingDistributionInit(m_surfaceSamplingDistributionHandle, region.m_regionHandle, pos, normal, useParallaxCompensation, useCosineProduct);
}

}
}