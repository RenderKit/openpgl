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
 * 
 *  This container class stores the (radiance/photon) samples generated during rendering or
 *  or at a pre-processing pass. The container is thread save and supports concurrent adding of
 *  samples by multiple threads. As a result only one instance of this container is needed per 
 *  rendering. The stored samples are later used by the @ref Field class to train/learn the 
 *  guiding field (i.e., radiance field) for a scene.   
 */
struct SampleStorage
{
    SampleStorage();
    
    /**
     * @brief Construct a new Field object from its serialized representation
     *
     * @param fieldFileName path to serialized representation
     */
	SampleStorage(const std::string& sampleStorageFileName);

    ~SampleStorage();

    SampleStorage(const SampleStorage&) = delete;

    /**
     * @brief Stores the SampleStorage to a file.
     * 
     * @param sampleStorageFileName 
     * @return true If saving the SampleStorage was successfull.
     * @return false Otherwise.
     */
    bool Store(const std::string& sampleStorageFileName) const;

    /**
     * @brief Adds a single sample to the storage container.
     * 
     * @param sample 
     */
    void AddSample(SampleData& sample);

    /**
     * @brief Adds an array of samples to the storage container.
     * 
     * @param samples Pointer to the beginning of the SampleData array.
     * @param numSamples Number of SampleData elements stored in the array.
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
    size_t GetSizeSurface() const;

    /// Returns the number of volume samples currently stored inside the storage container.
    size_t GetSizeVolume() const;

    /**
     * @brief Returns a volume sample from the surface sample storage.
     * 
     * @param idx 
     * @return SampleData 
     */
    SampleData GetSampleSurface(const int idx) const;

    /**
     * @brief Returns a volume sample from the volume sample storage.
     * 
     * @param idx 
     * @return SampleData 
     */
    SampleData GetSampleVolume(const int idx) const;

    friend struct Field;
    private:
        PGLSampleStorage m_sampleStorageHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE SampleStorage::SampleStorage()
{
    m_sampleStorageHandle = pglNewSampleStorage();
}

OPENPGL_INLINE SampleStorage::SampleStorage(const std::string& sampleStorageFileName)
{
    m_sampleStorageHandle = pglNewSampleStorageFromFile(sampleStorageFileName.c_str());
    if (!m_sampleStorageHandle)
        throw std::runtime_error("could not load sample storage from file: " + sampleStorageFileName);
}

OPENPGL_INLINE SampleStorage::~SampleStorage()
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    pglReleaseSampleStorage(m_sampleStorageHandle);
    m_sampleStorageHandle = nullptr;
}

OPENPGL_INLINE bool SampleStorage::Store(const std::string& sampleStorageFileName) const
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    return pglSampleStorageStoreToFile(m_sampleStorageHandle, sampleStorageFileName.c_str());
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

OPENPGL_INLINE size_t SampleStorage::GetSizeSurface() const
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    return pglSampleStorageGetSizeSurface(m_sampleStorageHandle);
}

OPENPGL_INLINE size_t SampleStorage::GetSizeVolume() const
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    return pglSampleStorageGetSizeVolume(m_sampleStorageHandle);
}

OPENPGL_INLINE SampleData SampleStorage::GetSampleSurface(const int idx) const
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    return pglSampleStorageGetSampleSurface(m_sampleStorageHandle, idx);
}

OPENPGL_INLINE SampleData SampleStorage::GetSampleVolume(const int idx) const
{
    OPENPGL_ASSERT(m_sampleStorageHandle);
    return pglSampleStorageGetSampleVolume(m_sampleStorageHandle, idx);
}


}
}