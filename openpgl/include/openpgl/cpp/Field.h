// Copyright 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "../openpgl.h"
#include "Device.h"
#include "FieldConfig.h"
#include "FieldStatistics.h"
#include "Region.h"
#include "SampleStorage.h"

namespace openpgl
{
namespace cpp
{

/**
 * @brief Key component of the guiding libary which holds the spatio-directional guiding information
 * (e.g., approximation of the incoming radiance field) for a scene.
 *
 * This class is responsible for storing, learning and accessing the guiding information for a scene.
 * This information can be the incidence radiance field across the whole scene learned from several training
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
    Field(Device *device, const FieldConfig &args);

    /**
     * @brief Creates/Loads a guiding field from a file.
     *
     * @param device The Device defining the compute architecture and optimization of the Field implementation.
     * @param fieldFileName The location of the file the Field is loaded from.
     */
    Field(Device *device, const std::string &fieldFileName);

    ~Field();

    Field(const Field &) = delete;

    /**
     * @brief Stores Field as serialized representation to file on disk
     *
     * @param fieldFileName path where to store serialized representation
     * @return if field could be stored to file
     */
    bool Store(const std::string &fieldFileName) const;

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
    void SetSceneBounds(const pgl_box3f &bounds);

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
     * @brief Updates the current approximation of the surface and radiance fields.
     *
     * @param sampleStorage
     */
    void Update(const SampleStorage &sampleStorage);

    /**
     * @brief Updates the current approximation of the surface radiance field.
     *
     * @param sampleStorage
     */
    void UpdateSurface(const SampleStorage &sampleStorage);

    /**
     * @brief Updates the current approximation of the volume radiance field.
     *
     * @param sampleStorage
     */
    void UpdateVolume(const SampleStorage &sampleStorage);

    void Reset();

    /// Returns the number of performed training iterations.
    size_t GetIteration() const;

    /// Checks if the guiding information of the Field is valid (e.g., contains no invalid directional distributions).
    bool Validate() const;

    /**
     * @brief Gets the statistics for the surface guiding Field.
     *
     * @return FieldStatistics
     */
    FieldStatistics GetSurfaceStatistics() const;

    /**
     * @brief Gets the statistics for the volume guiding Field.
     *
     * @return FieldStatistics
     */
    FieldStatistics GetVolumeStatistics() const;

    /// Checks if the spatial structure and directional distribution of this Field are similar to the ones stored in another Field.
    bool operator==(const Field &b) const;

    size_t GetNumSurfaceCaches() const;

    size_t GetNumVolumeCaches() const;

    pgl_vec3f GetSurfaceCachePosition(size_t idx) const;

    pgl_vec3f GetVolumeCachePosition(size_t idx) const;

    pgl_box3f GetSurfaceCacheSampleBounds(size_t idx) const;

    pgl_box3f GetVolumeCacheSampleBounds(size_t idx) const;

    pgl_cacheInfo GetSurfaceCacheInfo(size_t idx) const;

    pgl_cacheInfo GetVolumeCacheInfo(size_t idx) const;

    friend struct openpgl::cpp::SurfaceSamplingDistribution;
    friend struct openpgl::cpp::VolumeSamplingDistribution;

   private:
    PGLField m_fieldHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE Field::Field(Device *device, const FieldConfig &cfg)
{
    OPENPGL_ASSERT(device);
    OPENPGL_ASSERT(device->m_deviceHandle);
    m_fieldHandle = pglDeviceNewField(device->m_deviceHandle, cfg.m_args);
}

OPENPGL_INLINE Field::Field(Device *device, const std::string &fieldFileName)
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

OPENPGL_INLINE bool Field::Store(const std::string &fieldFileName) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldStoreToFile(m_fieldHandle, fieldFileName.c_str());
}

OPENPGL_INLINE size_t Field::GetIteration() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetIteration(m_fieldHandle);
}

OPENPGL_INLINE void Field::SetSceneBounds(const pgl_box3f &bounds)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldSetSceneBounds(m_fieldHandle, bounds);
}

OPENPGL_INLINE pgl_box3f Field::GetSceneBounds() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetSceneBounds(m_fieldHandle);
}

OPENPGL_INLINE void Field::Update(const SampleStorage &sampleStorage)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldUpdate(m_fieldHandle, sampleStorage.m_sampleStorageHandle);
}

OPENPGL_INLINE void Field::UpdateSurface(const SampleStorage &sampleStorage)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldUpdateSurface(m_fieldHandle, sampleStorage.m_sampleStorageHandle);
}

OPENPGL_INLINE void Field::UpdateVolume(const SampleStorage &sampleStorage)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldUpdateVolume(m_fieldHandle, sampleStorage.m_sampleStorageHandle);
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

OPENPGL_INLINE bool Field::operator==(const Field &b) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    OPENPGL_ASSERT(b.m_fieldHandle);
    return pglFieldCompare(m_fieldHandle, b.m_fieldHandle);
}

OPENPGL_INLINE FieldStatistics Field::GetSurfaceStatistics() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    PGLFieldStatistics fieldStats = pglFieldGetSurfaceStatistics(m_fieldHandle);
    return FieldStatistics(fieldStats);
}

OPENPGL_INLINE FieldStatistics Field::GetVolumeStatistics() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    PGLFieldStatistics fieldStats = pglFieldGetVolumeStatistics(m_fieldHandle);
    return FieldStatistics(fieldStats);
}

OPENPGL_INLINE size_t Field::GetNumSurfaceCaches() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetNumSurfaceCaches(m_fieldHandle);
}

OPENPGL_INLINE size_t Field::GetNumVolumeCaches() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetNumVolumeCaches(m_fieldHandle);
}

OPENPGL_INLINE pgl_vec3f Field::GetSurfaceCachePosition(size_t idx) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetSurfaceCachePosition(m_fieldHandle, idx);
}

OPENPGL_INLINE pgl_vec3f Field::GetVolumeCachePosition(size_t idx) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetVolumeCachePosition(m_fieldHandle, idx);
}

OPENPGL_INLINE pgl_box3f Field::GetSurfaceCacheSampleBounds(size_t idx) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetSurfaceCacheSampleBounds(m_fieldHandle, idx);
}

OPENPGL_INLINE pgl_box3f Field::GetVolumeCacheSampleBounds(size_t idx) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetVolumeCacheSampleBounds(m_fieldHandle, idx);
}

OPENPGL_INLINE pgl_cacheInfo Field::GetSurfaceCacheInfo(size_t idx) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetSurfaceCacheInfo(m_fieldHandle, idx);
}

OPENPGL_INLINE pgl_cacheInfo Field::GetVolumeCacheInfo(size_t idx) const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetVolumeCacheInfo(m_fieldHandle, idx);
}

}  // namespace cpp
}  // namespace openpgl