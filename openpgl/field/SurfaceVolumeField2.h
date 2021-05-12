// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ISurfaceVolumeField2.h"
#include "Field2.h"

namespace openpgl
{


template<class TDirectionalDistributionFactory, template<typename, typename> class TSpatialStructureBuilder, typename TSurfaceSamplingDistribution, typename TVolumeSamplingDistribution>
struct SurfaceVolumeField2: public ISurfaceVolumeField2
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

    ~SurfaceVolumeField2() override
    {} 

    ISurfaceSamplingDistribution* newSurfaceSamplingDistribution() const override
    {
        return new TSurfaceSamplingDistribution();
    }

    bool initSurfaceSamplingDistribution(ISurfaceSamplingDistribution* surfaceSamplingDistribution, const Point3& position, const float sample1D, const bool useParrallaxComp) const override
    {
        TSurfaceSamplingDistribution* _surfaceSamplingDistribution = (TSurfaceSamplingDistribution*)surfaceSamplingDistribution;
        const RegionType* region = this->getSurfaceGuidingRegion(position, sample1D);
        if(!region || !region->valid)
        {
            return false;
        }
        DirectionalDistribution distribution = region->getDistribution(position, useParrallaxComp);
        _surfaceSamplingDistribution->init(&distribution);
        _surfaceSamplingDistribution->setRegion(region);
        return true;
    }

    IVolumeSamplingDistribution* newVolumeSamplingDistribution() const override
    {
        return new TVolumeSamplingDistribution();    
    }

    bool initVolumeSamplingDistribution(IVolumeSamplingDistribution* volumeSamplingDistribution, const Point3& position, const float sample1D, const bool useParrallaxComp) const override
    {
        TVolumeSamplingDistribution* _volumeSamplingDistribution = (TVolumeSamplingDistribution*)volumeSamplingDistribution;
        const RegionType* region = this->getVolumeGuidingRegion(position, sample1D);
        if(!region || !region->valid)
        {
            return false;
        }
        DirectionalDistribution distribution = region->getDistribution(position, useParrallaxComp);
        _volumeSamplingDistribution->init(&distribution);
        _volumeSamplingDistribution->setRegion(region);
        return true;
    }

    void setSceneBounds(const openpgl::BBox &sceneBounds) override
    {
        m_surfaceField.setSceneBounds(sceneBounds);
        m_volumeField.setSceneBounds(sceneBounds);
    }


    const RegionType *getSurfaceGuidingRegion( const openpgl::Point3 &p, const float sample1D) const
    {
        return m_surfaceField.getGuidingRegion(p, sample1D);
    }


    const RegionType *getVolumeGuidingRegion( const openpgl::Point3 &p, const float sample1D) const
    {
        return m_volumeField.getGuidingRegion(p, sample1D);
    }


    void buildField(SampleContainer& samplesSurface, SampleContainer& samplesVolume) override
    {
        m_surfaceField.buildField(samplesSurface);
        m_volumeField.buildField(samplesVolume);
    }

    void updateField(SampleContainer& samplesSurface, SampleContainer& samplesVolume) override
    {
        m_surfaceField.updateField(samplesSurface);
        m_volumeField.updateField(samplesVolume);
    }


    void addTrainingIteration(size_t spp) override
    {
        m_totalSPP += spp;
        ++m_iteration;
        m_surfaceField.addTrainingIteration(spp);
        m_volumeField.addTrainingIteration(spp);
    }

    size_t getTotalSPP() const override
    {
        return m_totalSPP;
    }

    size_t getIteration() const override
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