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

struct Field
{
    Field(PGLFieldArguments args);

    ~Field();

    Field(const Field&) = delete;

    uint32_t GetIteration() const;

    uint32_t GetTotalSPP() const;

    void Update(pgl_box3f bounds, const SampleStorage& sampleStorage, const uint32_t& numPerPixelSamples);

    Region GetSurfaceRegion(pgl_point3f position, Sampler* sampler);

    Region GetVolumeRegion(pgl_point3f position, Sampler* sampler);

    //private:
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

OPENPGL_INLINE uint32_t Field::GetIteration() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetIteration(m_fieldHandle);
}

OPENPGL_INLINE uint32_t Field::GetTotalSPP() const
{
    OPENPGL_ASSERT(m_fieldHandle);
    return pglFieldGetTotalSPP(m_fieldHandle);
}

OPENPGL_INLINE void Field::Update(pgl_box3f bounds, const SampleStorage& sampleStorage, const uint32_t& numPerPixelSamples)
{
    OPENPGL_ASSERT(m_fieldHandle);
    pglFieldUpdate(m_fieldHandle, bounds, sampleStorage.m_sampleStorageHandle, numPerPixelSamples);
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