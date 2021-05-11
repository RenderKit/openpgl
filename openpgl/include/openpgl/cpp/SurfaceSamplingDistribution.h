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
 * @brief Guided sampling distriubtion to be used on surfaces.
 * 
 */
struct SurfaceSamplingDistribution
{
    //SurfaceSamplingDistribution();

    SurfaceSamplingDistribution(const Field* field);

    ~SurfaceSamplingDistribution();

    SurfaceSamplingDistribution(const SurfaceSamplingDistribution&) = delete;

    /**
     * @brief 
     * 
     * @param sample2D 
     * @return pgl_vec3f 
     */
    pgl_vec3f Sample(const pgl_point2f& sample2D)const;

    /**
     * @brief 
     * 
     * @param direction 
     * @return float 
     */
    float PDF(const pgl_vec3f& direction) const;

    /**
     * @brief 
     * 
     * @return true 
     * @return false 
     */
    bool IsValid() const;

    /**
     * @brief 
     * 
     */
    void Clear();

    /**
     * @brief 
     * 
     * @param region 
     * @param pos 
     * @param normal 
     * @param useParallaxCompensation 
     * @param useCosineProduct 
     */
    //void Init(const Region& region, const pgl_point3f& pos, const bool useParallaxCompensation = true);


    bool Init(const Field* field, const pgl_point3f& pos, const float sample1D, const bool useParallaxCompensation = true);

    /**
     * @brief Applies the product with the cosine to the sampling distriubtion.
     * 
     *  
     * @param normal 
     */
    void ApplyCosineProduct(const pgl_vec3f& normal);

    ///////////////////////////////////////
    /// Future plans
    ///////////////////////////////////////
    
    /**
     * @brief 
     * 
     * @param normal 
     * @param opaque 
     * @param transmission 
     */
    //void ApplyCosineProduct(const pgl_vec3f& normal, const bool opaque, const pgl_vec3f transmission);
    

    private:
        PGLSurfaceSamplingDistribution m_surfaceSamplingDistributionHandle{nullptr};
};
/*
SurfaceSamplingDistribution::SurfaceSamplingDistribution()
{
    m_surfaceSamplingDistributionHandle = pglNewSurfaceSamplingDistribution();
}
*/
SurfaceSamplingDistribution::SurfaceSamplingDistribution(const Field* field)
{
    m_surfaceSamplingDistributionHandle = pglFieldNewSurfaceSamplingDistribution(field->m_fieldHandle);
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
/*
void SurfaceSamplingDistribution::Init(const Region& region, const pgl_point3f& pos, const bool useParallaxCompensation)
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    pglSurfaceSamplingDistributionInit(m_surfaceSamplingDistributionHandle, region.m_regionHandle, pos, useParallaxCompensation);
}
*/
bool SurfaceSamplingDistribution::Init(const Field* field, const pgl_point3f& pos, const float sample1D, const bool useParallaxCompensation)
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    OPENPGL_ASSERT(field->m_fieldHandle);
    return pglFieldInitSurfaceSamplingDistriubtion(field->m_fieldHandle, m_surfaceSamplingDistributionHandle, pos, sample1D, useParallaxCompensation);
}

void SurfaceSamplingDistribution::ApplyCosineProduct(const pgl_vec3f& normal)
{
    OPENPGL_ASSERT(m_surfaceSamplingDistributionHandle);
    pglSurfaceSamplingDistributionApplyCosineProduct(m_surfaceSamplingDistributionHandle, normal);
}
}
}