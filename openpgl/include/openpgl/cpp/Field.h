// Copyright 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

#include "Region.h"
#include "Device.h"
#include "Sampler.h"
#include "SampleStorage.h"

#include <string>

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
 * @brief Key component of the guiding libary which holds the spatio-directional guiding information
 * (e.g., approximation of the incoming radiance field) for a scene.
 *
 * This class is responsible for storing, learning and accessing the guiding information for a scene.
 * This information can be the incidence radiance field accross the whole scene learned from several training
 * iterations during rendering or from a preprocessing step. The field usually holds separate approximations
 * for the surface and volumetric radiance field which can be accessed individually.
 * Based on the used representation the Field separates the positional and directional components of the 5D
 * radiance field using a spatial subdivision structure, where each spatial leaf node (a.k.a. Region) contains a directional representation
 * for the local incident radiance distribution.
 */
struct Field
{
    /**
     * @brief Creates a new guiding field.
     * 
     * @param device The Device defining the compute architecture and optimization of the Field implementation.
     * @param args The configuration of the Field (e.g., spatial or directional representation).
     */
    Field(Device *device, PGLFieldArguments args);

    /**
     * @brief Creates/Loads a guiding field from a file. 
     * 
     * @param device The Device defining the compute architecture and optimization of the Field implementation.
     * @param fieldFileName The location of the file the Field is loaded from.
     */
	Field(Device *device, const std::string& fieldFileName);

    ~Field();

    Field(const Field&) = delete;

    /**
     * @brief Stores Field as serialized representation to file on disk
     *
     * @param fieldFileName path where to store serialized representation
     * @return if field could be stored to file
     */
    bool Store(const std::string& fieldFileName) const;

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
     * @brief Get the bounding box of the scene/
     * 
     * The returned bounding box merges the bounds for the surface field and
     * the volume field.
     * 
     * @return pgl_box3f 
     */
    pgl_box3f GetSceneBounds() const;

    /**
     * @brief Updates the current approximation of the radiance field.
     *
     * @param sampleStorage
     */
    void Update(const SampleStorage& sampleStorage);

    void Reset();

    /// Returns the number of performed training iterations.
    size_t GetIteration() const;

    /// Checks if the guiding information of the Field is valid (e.g., contains no invalid directional distributions). 
    bool Validate() const;

    /**
     * @brief Returns the spatial surface Region containing the approximation of the local incident radiance Distribution.
     *
     * @param position
     * @param sampler
     * @return Region
     */
    //Region GetSurfaceRegion(pgl_point3f position, Sampler* sampler);

    /**
     * @brief Returns the spatial volume Region containing the approximation of the local incident radiance Distribution.
     *
     * @param position
     * @param sampler
     * @return Region
     */
    //Region GetVolumeRegion(pgl_point3f position, Sampler* sampler);

    friend struct openpgl::cpp::SurfaceSamplingDistribution;
    friend struct openpgl::cpp::VolumeSamplingDistribution;
    private:
        PGLField m_fieldHandle {nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE Field::Field(Device *device, PGLFieldArguments args)
{
    OPENPGL_ASSERT(device);
    OPENPGL_ASSERT(device->m_deviceHandle);
    m_fieldHandle = pglDeviceNewField(device->m_deviceHandle, args);
}

OPENPGL_INLINE Field::Field(Device *device, const std::string& fieldFileName)
{
    OPENPGL_ASSERT(device);
    OPENPGL_ASSERT(device->m_deviceHandle);
    m_fieldHandle = pglDeviceNewFieldFromFile(device->m_deviceHandle, fieldFileName.c_str());
    if (!m_fieldHandle)
        throw std::runtime_error("could not load field from file!");
}

OPENPGL_INLINE Field::~Field()
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglReleaseField(m_fieldHandle);
    m_fieldHandle = nullptr;
}

OPENPGL_INLINE bool Field::Store(const std::string& fieldFileName) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldStoreToFile(m_fieldHandle, fieldFileName.c_str());
}

OPENPGL_INLINE size_t Field::GetIteration() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetIteration(m_fieldHandle);
}

OPENPGL_INLINE void Field::SetSceneBounds(const pgl_box3f& bounds)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldSetSceneBounds(m_fieldHandle, bounds);
}

OPENPGL_INLINE pgl_box3f Field::GetSceneBounds() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetSceneBounds(m_fieldHandle);
}

OPENPGL_INLINE void Field::Update(const SampleStorage& sampleStorage)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldUpdate(m_fieldHandle, sampleStorage.m_sampleStorageHandle);
}

OPENPGL_INLINE void Field::Reset()
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldReset(m_fieldHandle);
}

OPENPGL_INLINE bool Field::Validate() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldValidate(m_fieldHandle);
}

} // api
} // openpgl