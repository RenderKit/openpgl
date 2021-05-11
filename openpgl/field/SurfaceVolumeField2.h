// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Field2.h"

namespace openpgl
{


template<class TDirectionalDistributionFactory, template<typename, typename> class TSpatialStructureBuilder>
struct SurfaceVolumeField2
{

private:

    using FieldType = Field2<TDirectionalDistributionFactory, TSpatialStructureBuilder>;
    using SampleContainer = SampleDataStorage::SampleDataContainer;
public:
    
    using Settings = typename FieldType::Settings;
    using RegionType = typename FieldType::RegionType;
    using DirectionalDistribution = typename FieldType::DirectionalDistribution; 
public:

    SurfaceVolumeField2() = default;

    SurfaceVolumeField2(const Settings &settings):
        m_surfaceField(settings),
        m_volumeField(settings)
    {
    }

    void setSceneBounds(const openpgl::BBox &sceneBounds)
    {
        m_surfaceField.setSceneBounds(sceneBounds);
        m_volumeField.setSceneBounds(sceneBounds);
    }


    const RegionType *getSurfaceGuidingRegion( const openpgl::Point3 &p, openpgl::Sampler *sampler) const
    {
        return m_surfaceField.getGuidingRegion(p, sampler);
    }


    const RegionType *getVolumeGuidingRegion( const openpgl::Point3 &p, openpgl::Sampler *sampler) const
    {
        return m_volumeField.getGuidingRegion(p, sampler);
    }


    void buildField(SampleContainer& samplesSurface, SampleContainer& samplesVolume)
    {
        m_surfaceField.buildField(samplesSurface);
        m_volumeField.buildField(samplesVolume);
    }

    void updateField(SampleContainer& samplesSurface, SampleContainer& samplesVolume)
    {
        m_surfaceField.updateField(samplesSurface);
        m_volumeField.updateField(samplesVolume);
    }


    void addTrainingIteration(size_t spp) {
        m_totalSPP += spp;
        ++m_iteration;
    }

    size_t getTotalSPP() const
    {
        return m_totalSPP;
    }

    size_t getIteration() const
    {
        return m_iteration;
    }


private:
    size_t m_iteration {0};
    size_t m_totalSPP  {0};

    FieldType m_surfaceField;
    FieldType m_volumeField;
};

}