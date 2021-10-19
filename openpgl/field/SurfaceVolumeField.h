// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ISurfaceVolumeField.h"
#include "Field.h"

namespace openpgl
{


template<class TDirectionalDistributionFactory, template<typename, typename> class TSpatialStructureBuilder, typename TSurfaceSamplingDistribution, typename TVolumeSamplingDistribution>
struct SurfaceVolumeField: public ISurfaceVolumeField
{

private:

    using FieldType = Field<TDirectionalDistributionFactory, TSpatialStructureBuilder>;
    using SampleContainer = SampleDataStorage::SampleDataContainer;
public:

    using Settings = typename FieldType::Settings;
    using RegionType = typename FieldType::RegionType;
    using DirectionalDistribution = typename FieldType::DirectionalDistribution;
public:

    SurfaceVolumeField() = default;

    SurfaceVolumeField(const Settings &settings):
        m_surfaceField(settings),
        m_volumeField(settings)
    { }

    ~SurfaceVolumeField() override
    {}

    ISurfaceSamplingDistribution* newSurfaceSamplingDistribution() const override
    {
        return new TSurfaceSamplingDistribution(m_surfaceField.getUseParallaxCompensation());
    }

    bool initSurfaceSamplingDistribution(ISurfaceSamplingDistribution* surfaceSamplingDistribution, const Point3& position, float* sample1D, const bool useParrallaxComp) const override
    {
        TSurfaceSamplingDistribution* _surfaceSamplingDistribution = (TSurfaceSamplingDistribution*)surfaceSamplingDistribution;
        const RegionType* region = m_surfaceField.getRegion(position, sample1D);
        if(!region || !region->valid)
        {
            return false;
        }
        const DirectionalDistribution* distribution = region->getDistribution(position);
        _surfaceSamplingDistribution->init(distribution, position);
        _surfaceSamplingDistribution->setRegion(region);
        return true;
    }

    IVolumeSamplingDistribution* newVolumeSamplingDistribution() const override
    {
        return new TVolumeSamplingDistribution(m_volumeField.getUseParallaxCompensation());
    }

    bool initVolumeSamplingDistribution(IVolumeSamplingDistribution* volumeSamplingDistribution, const Point3& position, float* sample1D, const bool useParrallaxComp) const override
    {
        TVolumeSamplingDistribution* _volumeSamplingDistribution = (TVolumeSamplingDistribution*)volumeSamplingDistribution;
        const RegionType* region = m_volumeField.getRegion(position, sample1D);
        if(!region || !region->valid)
        {
            return false;
        }
        const DirectionalDistribution* distribution = region->getDistribution(position);
        _volumeSamplingDistribution->init(distribution, position);
        _volumeSamplingDistribution->setRegion(region);
        return true;
    }

    void setSceneBounds(const openpgl::BBox &sceneBounds) override
    {
        openpgl::BBox scaledSceneBounds = sceneBounds;
        scaledSceneBounds.enlarge_by(1.01f);
        m_surfaceField.setSceneBounds(scaledSceneBounds);
        m_volumeField.setSceneBounds(scaledSceneBounds);
    }

    void buildField(SampleContainer& samplesSurface, SampleContainer& samplesVolume) override
    {
        if(samplesSurface.size() > 0)
            m_surfaceField.buildField(samplesSurface);
        if(samplesVolume.size() > 0)
            m_volumeField.buildField(samplesVolume);
    }

    void updateField(SampleContainer& samplesSurface, SampleContainer& samplesVolume) override
    {
        if(samplesSurface.size() > 0)
            m_surfaceField.updateField(samplesSurface);
        if(samplesVolume.size() > 0)
            m_volumeField.updateField(samplesVolume);
    }


    void addTrainingIteration(size_t spp) override
    {
        m_totalSPP += spp;
        ++m_iteration;
        m_surfaceField.addTrainingIteration(spp);
        m_volumeField.addTrainingIteration(spp);
    }

    PGL_SPATIAL_STRUCTURE_TYPE getSpatialStructureType() const override {
        return FieldType::SpatialStructureBuilder::SPATIAL_STRUCTURE_TYPE;
    }

    PGL_DIRECTIONAL_DISTRIBUTION_TYPE getDirectionalDistributionType() const override {
        return FieldType::DirectionalDistributionFactory::DIRECTIONAL_DISTRIBUTION_TYPE;
    }

    size_t getTotalSPP() const override
    {
        return m_totalSPP;
    }

    size_t getIteration() const override
    {
        return m_iteration;
    }

    void serialize(std::ostream &os) const override
    {
        os.write(reinterpret_cast<const char*>(&m_iteration), sizeof(m_iteration));
        os.write(reinterpret_cast<const char*>(&m_totalSPP), sizeof(m_totalSPP));
        m_surfaceField.serialize(os);
        m_volumeField.serialize(os);
    }

    void deserialize(std::istream &is) override
    {
        is.read(reinterpret_cast<char*>(&m_iteration), sizeof(m_iteration));
        is.read(reinterpret_cast<char*>(&m_totalSPP), sizeof(m_totalSPP));
        m_surfaceField.deserialize(is);
        m_volumeField.deserialize(is);
    }

    virtual bool isValid(const bool checkSurface, const bool checkVolume) const override
    {
        bool valid = true;
        if(m_surfaceField.isInitialized())
            valid = valid & m_surfaceField.isValid();
        if(m_volumeField.isInitialized())
            valid = valid & m_volumeField.isValid();
        return valid;
    }

private:
    size_t m_iteration {0};
    size_t m_totalSPP  {0};

    FieldType m_surfaceField;
    FieldType m_volumeField;
};

}
