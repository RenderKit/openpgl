// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "PathSegment.h"
#include "Sampler.h"
#include "SampleData.h"

namespace openpgl
{
namespace cpp
{
/**
 * @brief 
 * 
 */

struct PathSegmentStorage
{
    PathSegmentStorage();
    ~PathSegmentStorage();

    PathSegmentStorage(const PathSegmentStorage&) = delete;

    /**
     * @brief Reserves memory for the storage. 
     * 
     * @param size 
     */
    void Reserve(size_t size);

    /// Clears all path segments as well as samples stored inside the storage.
    void Clear();

    /**
     * @brief  Generates and internaly stores -radiance- samples from the the stored path segments.
     * 
     * 
     * 
     * @param splatSamples 
     * @param sampler 
     * @param useNEEMiWeights 
     * @param guideDirectLight 
     * @return size_t 
     */
    size_t PrepareSamples(const bool& splatSamples = false, Sampler* sampler = nullptr, const bool useNEEMiWeights = false, const bool guideDirectLight = false);

    pgl_vec3f CalculatePixelEstimate() const;

    /**
     * @brief Get the Samples object
     * 
     * @param nSamples 
     * @return const SampleData* 
     */
    const SampleData* GetSamples(size_t &nSamples);

    /**
     * @brief 
     * 
     * @param sample 
     */
    void AddSample(SampleData sample);

    PathSegment* NextSegment();

    void AddSegment(const PathSegment& segment);

    bool IsValid() const;

    bool SegmentsValid() const;

    bool SamplesValid() const;

    private:
        PGLPathSegmentStorage m_pathSegmentStorageHandle{nullptr};
};

OPENPGL_INLINE PathSegmentStorage::PathSegmentStorage()
{ 
    m_pathSegmentStorageHandle = pglNewPathSegmentStorage();
}

OPENPGL_INLINE PathSegmentStorage::~PathSegmentStorage()
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    pglReleasePathSegmentStorage(m_pathSegmentStorageHandle);
    m_pathSegmentStorageHandle = nullptr;
}

OPENPGL_INLINE void PathSegmentStorage::Reserve(size_t size)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    pglPathSegmentStorageReserve(m_pathSegmentStorageHandle, size);
}

OPENPGL_INLINE void PathSegmentStorage::Clear()
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    pglPathSegmentStorageClear(m_pathSegmentStorageHandle);
}

OPENPGL_INLINE size_t PathSegmentStorage::PrepareSamples(const bool& splatSamples, Sampler* sampler, const bool useNEEMiWeights, const bool guideDirectLight)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    //OPENPGL_ASSERT(&sampler.m_samplerHandle);
    if(sampler)
        return pglPathSegmentStoragePrepareSamples(m_pathSegmentStorageHandle, splatSamples, &sampler->m_samplerHandle, useNEEMiWeights, guideDirectLight);
    else
        return pglPathSegmentStoragePrepareSamples(m_pathSegmentStorageHandle, splatSamples, nullptr, useNEEMiWeights, guideDirectLight);
}

OPENPGL_INLINE pgl_vec3f PathSegmentStorage::CalculatePixelEstimate() const
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageCalculatePixelEstimate(m_pathSegmentStorageHandle);
}

OPENPGL_INLINE const SampleData* PathSegmentStorage::GetSamples(size_t &nSamples)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageGetSamples(m_pathSegmentStorageHandle, nSamples);
}


OPENPGL_INLINE void PathSegmentStorage::AddSample(SampleData sample)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    pglPathSegmentStorageAddSample(m_pathSegmentStorageHandle, sample);
}

OPENPGL_INLINE PathSegment* PathSegmentStorage::NextSegment()
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageNextSegment(m_pathSegmentStorageHandle);
}

OPENPGL_INLINE void PathSegmentStorage::AddSegment(const PathSegment& segment)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    pglPathSegmentStorageAddSegment(m_pathSegmentStorageHandle, segment);
}

OPENPGL_INLINE bool PathSegmentStorage::IsValid() const
{ 
    return SegmentsValid() && SamplesValid();
}

OPENPGL_INLINE bool PathSegmentStorage::SegmentsValid() const
{ 
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageSegmentsValid(m_pathSegmentStorageHandle);
}

OPENPGL_INLINE bool PathSegmentStorage::SamplesValid() const
{ 
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageSamplesValid(m_pathSegmentStorageHandle);
}

}
}