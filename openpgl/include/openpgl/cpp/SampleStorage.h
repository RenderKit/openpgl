// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "SampleData.h"

namespace openpgl
{
namespace cpp
{

struct SampleStorage
{
    SampleStorage();
    ~SampleStorage();

    SampleStorage(const SampleStorage&) = delete;
    
    void AddSample(SampleData& sample);

    void AddSamples(const SampleData* samples, uint32_t numSamples);

    void Reserve(const uint32_t& sizeSurface, const uint32_t& sizeVolume);

    void Clear();

    uint32_t GetSizeSurface() const;

    uint32_t GetSizeVolume() const;

    friend class Field;
    private:
        PGLSampleStorage m_sampleStorageHandle{nullptr};
};

OPENPGL_INLINE SampleStorage::SampleStorage()
{
    m_sampleStorageHandle = pglNewSampleStorage();
}

OPENPGL_INLINE SampleStorage::~SampleStorage()
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    pglReleaseSampleStorage(m_sampleStorageHandle);
    m_sampleStorageHandle = nullptr;
}

    
OPENPGL_INLINE void SampleStorage::AddSample(SampleData& sample)
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    pglSampleStorageAddSample(m_sampleStorageHandle, sample);
}

OPENPGL_INLINE void SampleStorage::AddSamples(const SampleData* samples, uint32_t numSamples)
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    pglSampleStorageAddSamples(m_sampleStorageHandle, samples, numSamples);
}

OPENPGL_INLINE void SampleStorage::Reserve(const uint32_t& sizeSurface, const uint32_t& sizeVolume)
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    pglSampleStorageReserve(m_sampleStorageHandle, sizeSurface, sizeVolume);
}

OPENPGL_INLINE void SampleStorage::Clear()
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    pglSampleStorageClear(m_sampleStorageHandle);
}

OPENPGL_INLINE uint32_t SampleStorage::GetSizeSurface() const
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    return pglSampleStorageGetSizeSurface(m_sampleStorageHandle);
}

OPENPGL_INLINE uint32_t SampleStorage::GetSizeVolume() const
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    return pglSampleStorageGetSizeVolume(m_sampleStorageHandle);
}

}
}