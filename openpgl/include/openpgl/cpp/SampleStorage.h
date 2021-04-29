// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "SampleData.h"

namespace openpgl
{
namespace cpp
{
/**
 * @brief The container class holding the collected sample data generated during rendering.
 *  This container class stores the (radiance/photon) samples generated during rendering or
 *  or at a pre-processing pass. The container is thread save and supports concurrent adding of
 *  samples by multiple threads. As a result only one instance of this container is needed per 
 *  rendering. The stored samples are later used by the @ref Field class to train/learn the 
 *  guiding field (i.e., radiance field) for a scene.   
 */
struct SampleStorage
{
    SampleStorage();
    ~SampleStorage();

    SampleStorage(const SampleStorage&) = delete;
    
    /**
     * @brief Adds a single sample to the storage container.
     * 
     * @param sample 
     */
    void AddSample(SampleData& sample);

    /**
     * @brief Adds an array of samples to the storage container.
     * 
     * @param samples 
     * @param numSamples 
     */
    void AddSamples(const SampleData* samples, size_t numSamples);

    /**
     * @brief Reserves initial space/memory for the sample storage container.
     * 
     * @param sizeSurface 
     * @param sizeVolume 
     */
    void Reserve(const size_t& sizeSurface, const size_t& sizeVolume);


    /// Clears all internal list of surface and volume samples.
    void Clear();

    /// Returns the number of surface samples currently stored inside the storage container.
    uint32_t GetSizeSurface() const;

    /// Returns the number of volume samples currently stored inside the storage container.
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

OPENPGL_INLINE void SampleStorage::AddSamples(const SampleData* samples, size_t numSamples)
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    pglSampleStorageAddSamples(m_sampleStorageHandle, samples, numSamples);
}

OPENPGL_INLINE void SampleStorage::Reserve(const size_t& sizeSurface, const size_t& sizeVolume)
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