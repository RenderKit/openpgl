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
 * @brief The PathSegmentStorage is a utility class to help generating SampleData during 
 * the path/random walk generation process. For the construction of a path/walk each new PathSegment
 * is stored the PathSegmentStorage. When the walk is finished or terminated the -radiance- SampleData is 
 * generated using a back propagation process. The resulting samples can then be passed to the global SampleDataStorage. 
 * 
 */

struct PathSegmentStorage
{
    PathSegmentStorage();
    ~PathSegmentStorage();

    PathSegmentStorage(const PathSegmentStorage&) = delete;

    /**
     * @brief Reserves memory for a given number PathSegments. 
     * 
     * @param size The maximum number of path segments (i.e., max path length)
     */
    void Reserve(size_t size);

    /// Clears all path segments as well as samples stored inside the storage.
    void Clear();

    /**
     * @brief  Generates and internally stores -radiance- samples from the the collected path segments.
     * 
     * @param splatSamples If the samples generated samples should be spatially jittered (i.e., to share information with neighboring cells). DEPRECATED
     * @param sampler The RNG used during splatting. DEPRECATED
     * @param useNEEMiWeights If the direct illumination should be multiplied with the mis weights for NEE.
     * @param guideDirectLight If the gererated samples should include direct illumination.
     * @param rrAffectsDirectContribution If the Russian roulette probability needs to be integrated into the direct illumination.
     * @return size_t The number of generated samples.
     */
    size_t PrepareSamples(const bool& splatSamples = false, Sampler* sampler = nullptr, const bool useNEEMiWeights = false, const bool guideDirectLight = false, const bool rrAffectsDirectContribution = true);

    /**
     * @brief Calculates the color estimate of the random walk/path from the path segments.
     * 
     * This function is mainly used for debug purposes to validate if the path segments stored in the 
     * PathSegmentStorage cover/represent the behavior of the used renderer (e.g., path tracer).
     * Ideally the output for each radom walk should match the pixel value added to the framebuffer.   
     * 
     * @param rrAffectsDirectContribution If the direct contribution of a segment needs to be weighted with the RR probability.
     * @return pgl_vec3f The RGB pixel value estimate for the random walk. 
     */
    pgl_vec3f CalculatePixelEstimate(const bool rrAffectsDirectContribution) const;

    /**
     * @brief Returns a pointer to the samples generated from the path segments.
     * 
     * @param nSamples The size of the array of the returned pointer.
     * @return const SampleData* The pointer to the sample data array.
     */
    const SampleData* GetSamples(size_t &nSamples);

    /**
     * @brief Adds a new PathSegment to the end of the path segment list and returns a pointer to it.  
     * 
     * If the number of PathSegments exceeds the number of reserved elements a nullptr is returned. 
     * 
     * @return PathSegment* The pointer to the -currently- last segment of the storage.
     */
    PathSegment* NextSegment();

    /**
     * @brief Adds a PathSegment at the end of the storage.
     * 
     * If the storage has already reached its limit the segment is not added to the list. 
     * 
     * @param segment 
     */
    void AddSegment(const PathSegment& segment);

    /**
     * @brief Adds a SampleData to the sample list.
     * 
     * @param sample 
     */
    void AddSample(SampleData sample);

    /**
     * @brief Sets the max. distance for a generated SampleData
     * (i.e., the distance used when hitting an environment map).
     * If not set the default value is 1e6f.
     * @param maxDistance 
     */
    void SetMaxDistance(const float maxDistance);

    /**
     * @brief Returns the max. distance for a generated SampleData
     * (i.e., the distance used when hitting an environment map).
     */
    float GetMaxDistance() const;

    /**
     * @brief Gets the number of stored path segments.
     * 
     * @return int number of stored path segments.
     */
    int GetNumSegments() const;

    /**
     * @brief Gets the number of samples generated from the path 
     * segments and the ones added explicitly by the user.
     * 
     * @return int number of generated or added samples.
     */
    int GetNumSamples() const;


    /**
     * @brief Validates the PathSegments as well as the generated SampleData.
     * The function returns false if either a SampleData or a PathSegment is invalid.
     */
    bool Validate() const;

    /**
     * @brief Validates each PathSegment stored in the PathSegmentStorage.
     * The function returns false if one of the PathSegments is invalid.
     */
    bool ValidateSegments() const;

    /**
     * @brief Validates each SampleData generated from the PathSegments.
     * The function returns false if one of the SampleData is invalid.
     */
    bool ValidateSamples() const;

    private:
        PGLPathSegmentStorage m_pathSegmentStorageHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

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

OPENPGL_INLINE size_t PathSegmentStorage::PrepareSamples(const bool& splatSamples, Sampler* sampler, const bool useNEEMiWeights, const bool guideDirectLight, const bool rrAffectsDirectContribution)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    //OPENPGL_ASSERT(&sampler.m_samplerHandle);
    if(sampler)
        return pglPathSegmentStoragePrepareSamples(m_pathSegmentStorageHandle, splatSamples, &sampler->m_samplerHandle, useNEEMiWeights, guideDirectLight, rrAffectsDirectContribution);
    else
        return pglPathSegmentStoragePrepareSamples(m_pathSegmentStorageHandle, splatSamples, nullptr, useNEEMiWeights, guideDirectLight, rrAffectsDirectContribution);
}

OPENPGL_INLINE pgl_vec3f PathSegmentStorage::CalculatePixelEstimate(const bool rrAffectsDirectContribution) const
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageCalculatePixelEstimate(m_pathSegmentStorageHandle, rrAffectsDirectContribution);
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

OPENPGL_INLINE void PathSegmentStorage::SetMaxDistance(const float maxDistance)
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    pglPathSegmentSetMaxDistance(m_pathSegmentStorageHandle, maxDistance);

}

OPENPGL_INLINE float PathSegmentStorage::GetMaxDistance() const
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentGetMaxDistance(m_pathSegmentStorageHandle);
}

OPENPGL_INLINE int PathSegmentStorage::GetNumSegments() const 
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentGetNumSegments(m_pathSegmentStorageHandle);
}

OPENPGL_INLINE int PathSegmentStorage::GetNumSamples() const
{
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentGetNumSamples(m_pathSegmentStorageHandle);
}

OPENPGL_INLINE bool PathSegmentStorage::Validate() const
{ 
    return ValidateSegments() && ValidateSamples();
}

OPENPGL_INLINE bool PathSegmentStorage::ValidateSegments() const
{ 
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageValidateSegments(m_pathSegmentStorageHandle);
}

OPENPGL_INLINE bool PathSegmentStorage::ValidateSamples() const
{ 
    OPENPGL_ASSERT(m_pathSegmentStorageHandle);
    return pglPathSegmentStorageValidateSamples(m_pathSegmentStorageHandle);
}

}
}