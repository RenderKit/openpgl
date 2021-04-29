// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "Region.h"
#include "Sampler.h"

namespace openpgl
{
namespace cpp
{
/**
 * @brief Class to store information of the current path segment.
 * 
 * This class stores all required information for a path segment
 * so that a list of succeeding segments (stored in a PathSegmentStorage) 
 * can be used to generate SampleData for training the guiding Field. 
 * 
 */
struct PathSegment
{
    PathSegment();

    PathSegment(PGLPathSegment pathSegmentHandle);

    /// Sets the position of the current event/segment.
    void SetPosition(const pgl_point3f& position);

    /// Sets the surface normal at the current intersection/event.
    /// The normal is equvalent to the surface normal on surfaces and
    /// can be arbritray at volume events.  
    void SetNormal(const pgl_vec3f& normal);

    /// Sets the incomming direction (direction to the next path segment).
    void SetDirectionIn(const pgl_vec3f& directionIn);

    /// returns the direction to the next path segment.
    pgl_vec3f GetDirectionIn() const;

    /// Sets the PDF used for sampling the direction towards the next segment.
    void SetPDFDirectionIn(const float& pdfDirectionIn);

    /// Sets the direction to the previous path segment (i.e., out-scattered direction).
    void SetDirectionOut(const pgl_vec3f& directionOut);

    /// Marks if the current event is inside a volume or not.
    void SetVolumeScatter(const bool& volumeScatter);

    /// Sets the scattering weight (e.g., BSDF/PDF or phase function/PDF).
    void SetScatteringWeight(const pgl_vec3f& scatteringWeight);

    /// Sets the direct/emissive contribution (e.g., emitted radiance).
    void SetDirectContribution(const pgl_vec3f& directContribution);

    /// Adds additional direct/emissive contribution.
    void AddDirectContribution(const pgl_vec3f& directContribution);

    /// Sets the out-scatter contribution.
    void SetScatteredContribution(const pgl_vec3f& scatteredContribution);

    /// Adds additional out-scattered contribtuion to the stored one.
    void AddScatteredContribution(const pgl_vec3f& scatteredContribution);

    /// Sets the MIS weight which would be multiplied to the directContribution 
    /// to account for NEE sampling from the previous segment.
    void SetMiWeight(const float& miWeight);

    /// Sets the Russian roulette probablity for surviving path termination at the current segment. 
    void SetRussianRouletteProbability(const float& russianRouletteProbability);

    /// Sets the eta the current material (1.0 for indexed matched, opaque and volumes).
    void SetEta(const float& eta);

    /// Sets if the current scattering event is a delta Dirac (e.g., perfect glass/mirror with roughness = 0.0f) 
    void SetIsDelta(const bool& isDelta);

    /// Sets the roughness (mean cosines for volumes) of the current material.
    void SetRoughness(const float& roughness);

    /// Sets the transmittance weights (transmittance/PDF) from the current to the next path segment.
    void SetTransmittanceWeight(const pgl_vec3f& transmittanceWeight);

    /// Sets a refence to the spatial Region connected to the starting position of the segment.
    /// Note: it is planned to get rid of this since it is only needed for sample splatting
    /// which should be replace by KNN lookups
    void SetRegion(const Region& region);

    private:
        PGLPathSegment m_pathSegmentHandle{nullptr};
};

OPENPGL_INLINE PathSegment::PathSegment()
{
    m_pathSegmentHandle = nullptr;
}

OPENPGL_INLINE PathSegment::PathSegment(PGLPathSegment pathSegmentHandle)
{
    OPENPGL_ASSERT(pathSegmentHandle);
    m_pathSegmentHandle = pathSegmentHandle;
}

OPENPGL_INLINE void PathSegment::SetPosition(const pgl_point3f& position)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetPosition(m_pathSegmentHandle, position);
}

OPENPGL_INLINE void PathSegment::SetNormal(const pgl_vec3f& normal)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetNormal(m_pathSegmentHandle, normal);
}

OPENPGL_INLINE void PathSegment::SetDirectionIn(const pgl_vec3f& directionIn)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetDirectionIn(m_pathSegmentHandle, directionIn);
}

OPENPGL_INLINE pgl_vec3f PathSegment::GetDirectionIn() const
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    return pglPathSegmentGetDirectionIn(m_pathSegmentHandle);
}

OPENPGL_INLINE void PathSegment::SetPDFDirectionIn(const float& pdfDirectionIn)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetPDFDirectionIn(m_pathSegmentHandle, pdfDirectionIn);
}

OPENPGL_INLINE void PathSegment::SetDirectionOut(const pgl_vec3f& directionOut)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetDirectionOut(m_pathSegmentHandle, directionOut);
}

OPENPGL_INLINE void PathSegment::SetVolumeScatter(const bool& volumeScatter)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetVolumeScatter(m_pathSegmentHandle, volumeScatter);
}

OPENPGL_INLINE void PathSegment::SetScatteringWeight(const pgl_vec3f& scatteringWeight){
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetScatteringWeight(m_pathSegmentHandle, scatteringWeight);
}

OPENPGL_INLINE void PathSegment::SetDirectContribution(const pgl_vec3f& directContribution)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetDirectContribution(m_pathSegmentHandle, directContribution);
}

OPENPGL_INLINE void PathSegment::AddDirectContribution(const pgl_vec3f& directContribution)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentAddDirectContribution(m_pathSegmentHandle, directContribution);
}

OPENPGL_INLINE void PathSegment::SetScatteredContribution(const pgl_vec3f& directScatter)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetScatteredContribution(m_pathSegmentHandle, directScatter);
}

OPENPGL_INLINE void PathSegment::AddScatteredContribution(const pgl_vec3f& directScatter)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentAddScatteredContribution(m_pathSegmentHandle, directScatter);
}

OPENPGL_INLINE void PathSegment::SetMiWeight(const float& miWeight)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetMiWeight(m_pathSegmentHandle, miWeight);
}

OPENPGL_INLINE void PathSegment::SetRussianRouletteProbability(const float& russianRouletteProbability)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetRussianRouletteProbability(m_pathSegmentHandle, russianRouletteProbability);
}

OPENPGL_INLINE void PathSegment::SetEta(const float& eta)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetEta(m_pathSegmentHandle, eta);
}

OPENPGL_INLINE void PathSegment::SetIsDelta(const bool& isDelta)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetIsDelta(m_pathSegmentHandle, isDelta);
}

OPENPGL_INLINE void PathSegment::SetRoughness(const float& roughness)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetRoughness(m_pathSegmentHandle, roughness);
}

OPENPGL_INLINE void PathSegment::SetTransmittanceWeight(const pgl_vec3f& transmittanceWeight)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    pglPathSegmentSetTransmittanceWeight(m_pathSegmentHandle, transmittanceWeight);
}

OPENPGL_INLINE void PathSegment::SetRegion(const Region& region)
{
    OPENPGL_ASSERT(m_pathSegmentHandle);
    //OPENPGL_ASSERT(region.m_regionHandle);
    pglPathSegmentSetRegion(m_pathSegmentHandle, region.m_regionHandle);
}
}
}