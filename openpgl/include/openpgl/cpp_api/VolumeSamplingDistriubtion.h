// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "Region.h"

namespace openpgl
{
namespace cpp
{
struct VolumeSamplingDistribution
{
    VolumeSamplingDistribution();
    ~VolumeSamplingDistribution();

    pgl_vec3f Sample(const pgl_point2f& sample2D)const;

    float PDF(const pgl_vec3f& direction) const;

    bool IsValid() const;

    void Clear();

    void Init(const Region& region, const pgl_point3f& pos, const bool& useParallaxCompensation = true);

    private:
        PGLPhaseFunctionSamplingDistribution m_phaseFunctionSamplingDistributionHandle{nullptr};
};

OPENPGL_INLINE VolumeSamplingDistribution::VolumeSamplingDistribution()
{
    m_phaseFunctionSamplingDistributionHandle = pglNewPhaseFunctionSamplingDistribution();
}

OPENPGL_INLINE VolumeSamplingDistribution::~VolumeSamplingDistribution()
{
    OPENPGL_ASSERT(m_phaseFunctionSamplingDistributionHandle);
    if(m_phaseFunctionSamplingDistributionHandle)
        pglReleasePhaseFunctionSamplingDistribution(m_phaseFunctionSamplingDistributionHandle);
    m_phaseFunctionSamplingDistributionHandle = nullptr;
}

pgl_vec3f VolumeSamplingDistribution::Sample(const pgl_point2f& sample2D)const
{
    OPENPGL_ASSERT(m_phaseFunctionSamplingDistributionHandle);
    return pglPhaseFunctionSamplingDistributionSample(m_phaseFunctionSamplingDistributionHandle, sample2D);
}

float VolumeSamplingDistribution::PDF(const pgl_vec3f& direction) const
{
    OPENPGL_ASSERT(m_phaseFunctionSamplingDistributionHandle);
    return pglPhaseFunctionSamplingDistributionPDF(m_phaseFunctionSamplingDistributionHandle, direction);
}

bool VolumeSamplingDistribution::IsValid() const
{
    OPENPGL_ASSERT(m_phaseFunctionSamplingDistributionHandle);
    return pglPhaseFunctionSamplingDistributionIsValid(m_phaseFunctionSamplingDistributionHandle);
}

void VolumeSamplingDistribution::Clear()
{
    OPENPGL_ASSERT(m_phaseFunctionSamplingDistributionHandle);
    return pglPhaseFunctionSamplingDistributionClear(m_phaseFunctionSamplingDistributionHandle);
}

void VolumeSamplingDistribution::Init(const Region& region, const pgl_point3f& pos, const bool& useParallaxCompensation)
{
    OPENPGL_ASSERT(m_phaseFunctionSamplingDistributionHandle);
    pglPhaseFunctionSamplingDistributionInit(m_phaseFunctionSamplingDistributionHandle, region.m_regionHandle, pos, useParallaxCompensation);
}

}
}