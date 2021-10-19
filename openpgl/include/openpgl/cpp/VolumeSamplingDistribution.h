// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "Region.h"
#include "Field.h"

namespace openpgl
{
namespace cpp
{

/**
 * @brief Guided sampling distribution to be used inside volumes.
 * 
 * 
 */

struct VolumeSamplingDistribution
{
    //VolumeSamplingDistribution();

    VolumeSamplingDistribution(const Field* field);
    
    ~VolumeSamplingDistribution();

    VolumeSamplingDistribution(const VolumeSamplingDistribution&) = delete;

    /**
     * @brief Importance samples a new direction based on the guiding distriubtion.
     * 
     * @param sample2D a 2D random variable
     * @return pgl_vec3f the sampled direction
     */
    pgl_vec3f Sample(const pgl_point2f& sample2D)const;

    /**
     * @brief Returns the sampling PDF for a given direction when is sampled
     * according to the guiding distribution.
     * 
     * @param direction 
     * @return float the PDF for sampling @ref direction
     */
    float PDF(const pgl_vec3f& direction) const;

    /**
     * @brief Combined importance sampling and PDF calculation.
     * Can be more efficient to use for some distributions (e.g. DirectionQuadtree)
     * 
     * @param sample2D a 2D random variable
     * @param direction importance sampled direction
     * @return float the PDF for sampling @ref direction
     */
    float SamplePDF(const pgl_point2f& sample2D, pgl_vec3f& direction) const;

    /**
     * @brief Checks if the current guiding distribution is valid or not.
     * 
     * @return true 
     * @return false 
     */
    bool IsValid() const;

    /**
     * @brief Clears/resets the internal repesentation of the guiding distribution. 
     * 
     */
    void Clear();

    /**
     * @brief Initializes the guided sampling distribution to the approximation of the local incident radiance distriubtion.
     * 
     * @param region the Region containing the local inciden radiance distribution approximation.
     * @param pos the position inside the Region 
     * @param useParallaxCompensation if the local approximation should be adjusted to @ref pos
     */
    //void Init(const Region& region, const pgl_point3f& pos, const bool useParallaxCompensation = true);

    bool Init(const Field* field, const pgl_point3f& pos, float& sample1D, const bool useParallaxCompensation = true);


    Region GetRegion() const;

    ///////////////////////////////////////
    /// Future plans
    ///////////////////////////////////////

/*
    void ApplySingleLobeHGProduct(const float meanCosine);

    void ApplyDualLobeHGProduct(const float meanCosine0, const float meanCosine1, const float mixWeight);
*/

    friend class Field;
    private:
        PGLVolumeSamplingDistribution m_volumeSamplingDistributionHandle{nullptr};
};

/*
VolumeSamplingDistribution::VolumeSamplingDistribution()
{
    m_volumeSamplingDistributionHandle = pglNewVolumeSamplingDistribution();
}
*/

OPENPGL_INLINE VolumeSamplingDistribution::VolumeSamplingDistribution(const Field* field)
{
    m_volumeSamplingDistributionHandle = pglFieldNewVolumeSamplingDistribution(field->m_fieldHandle);
}

OPENPGL_INLINE VolumeSamplingDistribution::~VolumeSamplingDistribution()
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    if(m_volumeSamplingDistributionHandle)
        pglReleaseVolumeSamplingDistribution(m_volumeSamplingDistributionHandle);
    m_volumeSamplingDistributionHandle = nullptr;
}

OPENPGL_INLINE pgl_vec3f VolumeSamplingDistribution::Sample(const pgl_point2f& sample2D)const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionSample(m_volumeSamplingDistributionHandle, sample2D);
}

OPENPGL_INLINE float VolumeSamplingDistribution::PDF(const pgl_vec3f& direction) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionPDF(m_volumeSamplingDistributionHandle, direction);
}

OPENPGL_INLINE float VolumeSamplingDistribution::SamplePDF(const pgl_point2f& sample2D, pgl_vec3f& direction) const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionSamplePDF(m_volumeSamplingDistributionHandle, sample2D, direction);    
}

OPENPGL_INLINE bool VolumeSamplingDistribution::IsValid() const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionIsValid(m_volumeSamplingDistributionHandle);
}

OPENPGL_INLINE void VolumeSamplingDistribution::Clear()
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    return pglVolumeSamplingDistributionClear(m_volumeSamplingDistributionHandle);
}

/*
void VolumeSamplingDistribution::Init(const Region& region, const pgl_point3f& pos, const bool useParallaxCompensation)
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    pglVolumeSamplingDistributionInit(m_volumeSamplingDistributionHandle, region.m_regionHandle, pos, useParallaxCompensation);
}
*/

OPENPGL_INLINE bool VolumeSamplingDistribution::Init(const Field* field, const pgl_point3f& pos, float& sample1D, const bool useParallaxCompensation)
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    OPENPGL_ASSERT(field->m_fieldHandle);
    return pglFieldInitVolumeSamplingDistriubtion(field->m_fieldHandle, m_volumeSamplingDistributionHandle, pos, &sample1D, useParallaxCompensation);
}

OPENPGL_INLINE Region VolumeSamplingDistribution::GetRegion() const
{
    OPENPGL_ASSERT(m_volumeSamplingDistributionHandle);
    //OPENPGL_ASSERT(sampler);
    //OPENPGL_ASSERT(&sampler->m_samplerHandle);
    PGLRegion regionHandle = pglVolumeSamplingGetRegion(m_volumeSamplingDistributionHandle);
    return Region(regionHandle);
}

}
}