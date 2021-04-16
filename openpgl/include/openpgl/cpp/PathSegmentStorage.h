// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "PathSegment.h"
#include "Sampler.h"


namespace openpgl
{
namespace cpp
{
struct PathSegmentStorage
{
    PathSegmentStorage();
    ~PathSegmentStorage();

    PathSegmentStorage(const PathSegmentStorage&) = delete;

    void Reserve(size_t size);

    void Clear();

    size_t PrepareSamples(const bool& splatSamples, Sampler& sampler, const bool useNEEMiWeights = false, const bool guideDirectLight = false);

    const PGLSampleData* GetSamples(uint32_t &nSamples);

    void AddSample(PGLSampleData sample);

    PathSegment NextSegment();

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

OPENPGL_INLINE size_t PathSegmentStorage::PrepareSamples(const bool& splatSamples, Sampler& sampler, const bool useNEEMiWeights, const bool guideDirectLight)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    OPENPGL_ASSERT(&sampler.m_samplerHandle);
    return pglPathSegmentStoragePrepareSamples(m_pathSegmentStorageHandle, splatSamples, &sampler.m_samplerHandle, useNEEMiWeights, guideDirectLight);
}

OPENPGL_INLINE const PGLSampleData* PathSegmentStorage::GetSamples(uint32_t &nSamples)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageGetSamples(m_pathSegmentStorageHandle, nSamples);
}


OPENPGL_INLINE void PathSegmentStorage::AddSample(PGLSampleData sample)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    pglPathSegmentStorageAddSample(m_pathSegmentStorageHandle, sample);
}

OPENPGL_INLINE PathSegment PathSegmentStorage::NextSegment()
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    PGLPathSegment pathSegmentHandle = pglPathSegmentNextSegment(m_pathSegmentStorageHandle);
    OPENPGL_ASSERT(pathSegmentHandle);
    return PathSegment(pathSegmentHandle);
}

}
}