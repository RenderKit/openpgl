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
struct PathSegment
{
    PathSegment();

    PathSegment(PGLPathSegment pathSegmentHandle);

    void SetPosition(const pgl_point3f& position);

    void SetNormal(const pgl_vec3f& normal);

    void SetDirectionIn(const pgl_vec3f& directionIn);

    pgl_vec3f GetDirectionIn() const;

    void SetPDFDirectionIn(const float& pdfDirectionIn);

    void SetDirectionOut(const pgl_vec3f& directionOut);

    void SetVolumeScatter(const bool& volumeScatter);

    void SetScatteringWeight(const pgl_vec3f& scatteringWeight);

    void SetDirectContribution(const pgl_vec3f& directContribution);

    void AddDirectContribution(const pgl_vec3f& directContribution);

    void SetScatteredContribution(const pgl_vec3f& directScatter);

    void AddScatteredContribution(const pgl_vec3f& directScatter);

    void SetMiWeight(const float& miWeight);

    void SetRussianRouletteProbability(const float& russianRouletteProbability);

    void SetEta(const float& eta);

    void SetIsDelta(const bool& isDelta);

    void SetRoughness(const float& roughness);

    void SetTransmittanceWeight(const pgl_vec3f& transmittanceWeight);

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