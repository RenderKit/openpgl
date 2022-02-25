// Copyright 2021-2022 Intel Corporation
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
 * The PathSegment class stores all required information for a path segment
 * so that a list of succeeding segments (stored in a PathSegmentStorage) 
 * can be used to generate SampleData for training the guiding Field. 
 * 
 */

using PathSegment = PGLPathSegmentData;

OPENPGL_INLINE void Reset(PathSegment* pathSegment)
{
    OPENPGL_ASSERT(pathSegment);
    *pathSegment = PathSegment();
}

OPENPGL_INLINE void SetPosition(PathSegment* pathSegment, const pgl_point3f& position)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->position = position;
}

OPENPGL_INLINE void SetNormal(PathSegment* pathSegment, const pgl_vec3f& normal)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->normal = normal;
}

OPENPGL_INLINE void SetDirectionIn(PathSegment* pathSegment, const pgl_vec3f& directionIn)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->directionIn = directionIn;
}

OPENPGL_INLINE pgl_vec3f GetDirectionIn(PathSegment* pathSegment)
{
    OPENPGL_ASSERT(pathSegment);
    return pathSegment->directionIn;
}

OPENPGL_INLINE void SetPDFDirectionIn(PathSegment* pathSegment, const float& pdfDirectionIn)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->pdfDirectionIn = pdfDirectionIn;
}

OPENPGL_INLINE void SetDirectionOut(PathSegment* pathSegment, const pgl_vec3f& directionOut)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->directionOut = directionOut;
    
}

OPENPGL_INLINE void SetVolumeScatter(PathSegment* pathSegment, const bool& volumeScatter)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->volumeScatter = volumeScatter;
}

OPENPGL_INLINE void SetScatteringWeight(PathSegment* pathSegment, const pgl_vec3f& scatteringWeight){
    OPENPGL_ASSERT(pathSegment);
    pathSegment->scatteringWeight = scatteringWeight;
}

OPENPGL_INLINE void SetDirectContribution(PathSegment* pathSegment, const pgl_vec3f& directContribution)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->directContribution = directContribution;
}

OPENPGL_INLINE void AddDirectContribution(PathSegment* pathSegment, const pgl_vec3f& directContribution)
{
    OPENPGL_ASSERT(pathSegment);
    pglVec3fAdd(pathSegment->scatteredContribution, directContribution);
}

OPENPGL_INLINE void SetScatteredContribution(PathSegment* pathSegment, const pgl_vec3f& scatteredContribution)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->scatteredContribution = scatteredContribution;
}

OPENPGL_INLINE void AddScatteredContribution(PathSegment* pathSegment, const pgl_vec3f& scatteredContribution)
{
    OPENPGL_ASSERT(pathSegment);
    pglVec3fAdd(pathSegment->scatteredContribution, scatteredContribution);
}

OPENPGL_INLINE void SetMiWeight(PathSegment* pathSegment, const float& miWeight)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->miWeight = miWeight;
}

OPENPGL_INLINE void SetRussianRouletteProbability(PathSegment* pathSegment, const float& russianRouletteProbability)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->russianRouletteProbability = russianRouletteProbability;
}

OPENPGL_INLINE void SetEta(PathSegment* pathSegment, const float& eta)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->eta = eta;
}

OPENPGL_INLINE void SetIsDelta(PathSegment* pathSegment, const bool& isDelta)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->isDelta = isDelta;
}

OPENPGL_INLINE void SetRoughness(PathSegment* pathSegment, const float& roughness)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->roughness = roughness;
}

OPENPGL_INLINE void SetTransmittanceWeight(PathSegment* pathSegment, const pgl_vec3f& transmittanceWeight)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->transmittanceWeight = transmittanceWeight;
}

// This function is deprecated
OPENPGL_INLINE void SetRegion(PathSegment* pathSegment, const Region &region)
{
    OPENPGL_ASSERT(pathSegment);
    pathSegment->regionPtr = (void*)region.m_regionHandle;
}

}
}