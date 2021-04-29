// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

#include "Region.h"
#include "Sampler.h"
#include "SampleStorage.h"

namespace openpgl
{
namespace cpp
{

/**
 * @brief 
 * 
 */
using FieldArguments = PGLFieldArguments;

/**
 * @brief Key component of the guiding libary which holds the guind information
 * (e.g., approximation of the incidance radiance field) for a scene
 * 
 * This class is responsible for storing, learning and accessing the guiding information for a scene.
 * This information can be the incidence radiance field accros the whole scene learned from several training
 * iterations during rendering or from a preprocessing step. The field usually holds seperate approxiamtions
 * for the surface and volumetric radiance field which can be accessed individually. 
 * Based on the used representation the Field separates the positional and directional componetns of the 5D
 * radiance field using a spatial subdivision structure, where each spatial leaf node (a.k.a. Region) contains a directional representation
 * for the local incident radiance distribtuion.
 */
struct Field
{
    /**
     * @brief Construct a new Field object
     * 
     * @param args 
     */
    Field(PGLFieldArguments args);

    ~Field();

    Field(const Field&) = delete;

    /**
     * @brief Sets the bounding box of the scenes.
     * 
     * Sets the bounding box of the scene. This bounding box is used as
     * bounds for the spatial subdivision structures for the surface and 
     * volume guiding fields. If no scene bounding box is set before 
     * @ref Update is called the first time the scene bounds are estimated
     * using the first sample batch.
     * @param bounds
     */
    void SetSceneBounds(const pgl_box3f& bounds);


    /**
     * @brief Upadates the current approximation of the radiance field.
     * 
     * 
     * 
     * @param sampleStorage 
     * @param numPerPixelSamples the number of sample per pixels used to generate the training data
     */
    void Update(const SampleStorage& sampleStorage, const size_t& numPerPixelSamples);


    /// Returns the number of perforemd training iterations.
    size_t GetIteration() const;

    /// Return the over all number of sample per pixel used across all trainin iterations.
    size_t GetTotalSPP() const;

    /**
     * @brief Returns the spatial surface Region containing the approximation of the local incident radiance distriubtion. 
     * 
     * @param position 
     * @param sampler 
     * @return Region 
     */
    Region GetSurfaceRegion(pgl_point3f position, Sampler* sampler);

    /**
     * @brief Returns the spatial volume Region containing the approximation of the local incident radiance distriubtion. 
     * 
     * @param position 
     * @param sampler 
     * @return Region 
     */
    Region GetVolumeRegion(pgl_point3f position, Sampler* sampler);

    private:
        PGLField m_fieldHandle {nullptr};
};

OPENPGL_INLINE Field::Field(PGLFieldArguments args)
{
    m_fieldHandle = pglNewField(args);
}

OPENPGL_INLINE Field::~Field()
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglReleaseField(m_fieldHandle);
    m_fieldHandle = nullptr;
}

OPENPGL_INLINE size_t Field::GetIteration() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetIteration(m_fieldHandle);
}

OPENPGL_INLINE size_t Field::GetTotalSPP() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetTotalSPP(m_fieldHandle);
}

OPENPGL_INLINE void Field::SetSceneBounds(const pgl_box3f& bounds)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldSetSceneBounds(m_fieldHandle, bounds);
}

OPENPGL_INLINE void Field::Update(const SampleStorage& sampleStorage, const size_t& numPerPixelSamples)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldUpdate(m_fieldHandle, sampleStorage.m_sampleStorageHandle, numPerPixelSamples);
}

OPENPGL_INLINE Region Field::GetSurfaceRegion(pgl_point3f position, Sampler* sampler)
{
    OPENPGL_ASSERT(m_fieldHandle);
    //OPENPGL_ASSERT(sampler);
    //OPENPGL_ASSERT(&sampler->m_samplerHandle);
    PGLRegion regionHandle = pglFieldGetSurfaceRegion(m_fieldHandle, position, &sampler->m_samplerHandle);
    return Region(regionHandle);
}

OPENPGL_INLINE Region Field::GetVolumeRegion(pgl_point3f position, Sampler* sampler)
{
    OPENPGL_ASSERT(m_fieldHandle);
    //OPENPGL_ASSERT(sampler);
    //OPENPGL_ASSERT(&sampler->m_samplerHandle);
    PGLRegion regionHandle = pglFieldGetVolumeRegion(m_fieldHandle, position, &sampler->m_samplerHandle);
    return Region(regionHandle);
}



} // api
} // openpgl