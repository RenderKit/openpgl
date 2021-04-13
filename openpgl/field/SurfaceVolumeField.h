// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define USE_TBB

#include "../openpgl.h"
//#include "../data/SampleStatistics.h"
#include "../data/Range.h"
#include "../kdtree/KDTree.h"
#include "../kdtree/KDTreeBuilder.h"
#include "Region.h"
#include "KNN.h"

namespace openpgl
{

template<class TRegion, typename TSampleContainer>
struct SurfaceVolumeField
{
public:
    //typedef TDistribution DistributionType;
    //typedef typename TDistributionFactory::ASMStatistics   StatisticsType;

    //typedef openpgl::Region<DistributionType, StatisticsType> RegionType;
    typedef TRegion RegionType;
    typedef openpgl::Range<TSampleContainer> RangeType;
protected:

    typedef std::pair<RegionType, RangeType > RegionStorageType;
    typedef tbb::concurrent_vector< RegionStorageType > RegionStorageContainerType;

    typedef openpgl::KDTree SpatialSubdivStructure;
    typedef openpgl::KDTreePartitionBuilder<RegionType, RangeType> SpatialSubdivBuilder;

public:

    struct Settings
    {
        typename SpatialSubdivBuilder::Settings spatialSubdivBuilderSettings;
        bool useStochasticNNLookUp {false};
        bool deterministic {false};
        float decayOnSpatialSplit {0.25f};

        std::string toString() const;
    };

    SurfaceVolumeField() = default;

    SurfaceVolumeField(const Settings &settings):
        m_decayOnSpatialSplit(settings.decayOnSpatialSplit),
        m_deterministic(settings.deterministic),
        m_useStochasticNNLookUp(settings.useStochasticNNLookUp),
        m_spatialSubdivBuilderSettings(settings.spatialSubdivBuilderSettings){
        //std::cout << "SurfaceVolumeField(const Settings &settings): " << settings.deterministic << std::endl;
    }

    const RegionType *getSurfaceGuidingRegion( const openpgl::Point3 &p, openpgl::Sampler *sampler) const
    {
        if (m_iteration >0 && embree::inside(m_spatialSubdivSurface.getBounds(), p))
        {
            if(m_useStochasticNNLookUp)
            {
                uint32_t regionIdx =  getClosestRegionIdx(m_regionKNNSearchTreeSurface, p, sampler->next1D());
                if(regionIdx != -1)
                {
                    return &m_regionStorageContainerSurface[regionIdx].first;
                }
                else
                {
                    return nullptr;
                }
            }
            else
            {
                openpgl::BBox regionBounds;
                uint32_t dataIdx = m_spatialSubdivSurface.getDataIdxAtPos(p, regionBounds);
                OPENPGL_ASSERT(dataIdx >= 0);
                return &m_regionStorageContainerSurface[dataIdx].first;
            }
        }
        else
        {
            return nullptr;
        }
    }


    const RegionType *getVolumeGuidingRegion( const openpgl::Point3 &p, openpgl::Sampler *sampler) const
    {
        if (m_iteration >0 && embree::inside(m_spatialSubdivVolume.getBounds(), p))
        {
            if(m_useStochasticNNLookUp)
            {
                uint32_t regionIdx =  getClosestRegionIdx(m_regionKNNSearchTreeVolume, p, sampler->next1D());
                if(regionIdx != -1)
                {
                    return &m_regionStorageContainerVolume[regionIdx].first;
                }
                else
                {
                    return nullptr;
                }
            }
            else
            {
                openpgl::BBox regionBounds;
                uint32_t dataIdx = m_spatialSubdivVolume.getDataIdxAtPos(p, regionBounds);
                OPENPGL_ASSERT(dataIdx >= 0);
                return &m_regionStorageContainerVolume[dataIdx].first;
            }
        }
        else
        {
            return nullptr;
        }
    }

    void buildField(const BBox &bounds, TSampleContainer& samplesSurface, TSampleContainer& samplesVolume)
    {
        m_iteration = 0;
        m_totalSPP  = 0;
        if (m_deterministic)
        {
            std::cout << "SurfaceVolumeField::buildField(): deterministic = " << m_deterministic<< std::endl;
            std::sort(samplesSurface.begin(), samplesSurface.end(), DirectionalSampleDataLess);
            std::sort(samplesVolume.begin(), samplesVolume.end(), DirectionalSampleDataLess);
        }

        std::cout << "BufferSize: " << sizeof(DirectionalSampleData) * m_spatialSubdivBuilderSettings.maxSamples * 1e-6 <<  " MB" << std::endl;
        std::cout << "buildField: samplesSurface = " << samplesSurface.size() << "\t samplesVolume = " << samplesVolume.size() << std::endl;
        buildSpatialStructureSurface(bounds, samplesSurface);
        buildSpatialStructureVolume(bounds, samplesVolume);
        fitRegions();
    }

    void updateField(TSampleContainer& samplesSurface, TSampleContainer& samplesVolume)
    {
        std::cout << "updateField: samplesSurface = " << samplesSurface.size() << "\t samplesVolume = " << samplesVolume.size() << std::endl;
        if (m_deterministic)
        {
            std::cout << "SurfaceVolumeField::buildField(): deterministic = " << m_deterministic << std::endl;
            std::sort(samplesSurface.begin(), samplesSurface.end(), DirectionalSampleDataLess);
            std::sort(samplesVolume.begin(), samplesVolume.end(), DirectionalSampleDataLess);
        }

        updateSpatialStructureSurface(samplesSurface);
        updateSpatialStructureVolume(samplesVolume);
        updateRegions();
    }


    void addTrainingIteration(uint32_t spp) {
        m_totalSPP += spp;
        ++m_iteration;
    }

    uint32_t getTotalSPP() const
    {
        return m_totalSPP;
    }

    uint32_t getIteration() const
    {
        return m_iteration;
    }

    std::string toString() const;

protected:

    void buildSpatialStructureSurface(const BBox &bounds, TSampleContainer& samples)
    {
        //mitsuba::SLog(mitsuba::EInfo, "Begin Tree update");
//        mitsuba::ref<mitsuba::Timer> treeTimer = new mitsuba::Timer();
        m_spatialSubdivBuilder.build(m_spatialSubdivSurface, bounds, samples, m_regionStorageContainerSurface, m_spatialSubdivBuilderSettings, m_nCores);
        //mitsuba::SLog(mitsuba::EInfo, "Tree building time: %s", timeString(treeTimer->getSeconds(), true).c_str());

        if (m_useStochasticNNLookUp)
        {
//            mitsuba::ref<mitsuba::Timer> embreereeTimer = new mitsuba::Timer();
            m_regionKNNSearchTreeSurface.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainerSurface);
            //mitsuba::SLog(mitsuba::EInfo, "Embree BVH update time: %s", timeString(embreereeTimer->getSeconds(), true).c_str());
        }
    }


    void updateSpatialStructureSurface(TSampleContainer& samples)
    {
        //mitsuba::SLog(mitsuba::EInfo, "Begin Tree update");
//        mitsuba::ref<mitsuba::Timer> treeTimer = new mitsuba::Timer();
        m_spatialSubdivBuilder.updateTree(m_spatialSubdivSurface, samples, m_regionStorageContainerSurface, m_spatialSubdivBuilderSettings, m_nCores);
        //mitsuba::SLog(mitsuba::EInfo, "Tree building time: %s", timeString(treeTimer->getSeconds(), true).c_str());

        if (m_useStochasticNNLookUp)
        {
//            mitsuba::ref<mitsuba::Timer> embreereeTimer = new mitsuba::Timer();
            m_regionKNNSearchTreeSurface.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainerSurface);
            //mitsuba::SLog(mitsuba::EInfo, "Embree BVH update time: %s", timeString(embreereeTimer->getSeconds(), true).c_str());
        }
    }


    void buildSpatialStructureVolume(const BBox &bounds, TSampleContainer& samples)
    {
        //mitsuba::SLog(mitsuba::EInfo, "Begin Tree update");
//        mitsuba::ref<mitsuba::Timer> treeTimer = new mitsuba::Timer();
        m_spatialSubdivBuilder.build(m_spatialSubdivVolume, bounds, samples, m_regionStorageContainerVolume, m_spatialSubdivBuilderSettings, m_nCores);
        //mitsuba::SLog(mitsuba::EInfo, "Tree building time: %s", timeString(treeTimer->getSeconds(), true).c_str());

        if (m_useStochasticNNLookUp)
        {
//            mitsuba::ref<mitsuba::Timer> embreereeTimer = new mitsuba::Timer();
            m_regionKNNSearchTreeVolume.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainerVolume);
            //mitsuba::SLog(mitsuba::EInfo, "Embree BVH update time: %s", timeString(embreereeTimer->getSeconds(), true).c_str());
        }
    }


    void updateSpatialStructureVolume(TSampleContainer& samples)
    {
        //mitsuba::SLog(mitsuba::EInfo, "Begin Tree update");
//        mitsuba::ref<mitsuba::Timer> treeTimer = new mitsuba::Timer();
        m_spatialSubdivBuilder.updateTree(m_spatialSubdivVolume, samples, m_regionStorageContainerVolume, m_spatialSubdivBuilderSettings, m_nCores);
        //mitsuba::SLog(mitsuba::EInfo, "Tree building time: %s", timeString(treeTimer->getSeconds(), true).c_str());

        if (m_useStochasticNNLookUp)
        {
//            mitsuba::ref<mitsuba::Timer> embreereeTimer = new mitsuba::Timer();
            m_regionKNNSearchTreeVolume.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainerVolume);
            //mitsuba::SLog(mitsuba::EInfo, "Embree BVH update time: %s", timeString(embreereeTimer->getSeconds(), true).c_str());
        }
    }


/*
    void updateKNNRegionSearchTree()
    {
        m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
    }
*/

    void updateKNNRegionSearchTreeSurface()
    {
        m_regionKNNSearchTreeSurface.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainerSurface);
    }

    void updateKNNRegionSearchTreeVolume()
    {
        m_regionKNNSearchTreeVolume.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainerVolume);
    }

    uint32_t getClosestRegionIdx(const KNearestRegionsSearchTree &knnTree, const openpgl::Point3 &p, float sample) const
    {
        OPENPGL_ASSERT(knnTree.isBuild());
        //OPENPGL_ASSERT(knnTree.numRegions() == m_regionStorageContainer.size());

        const uint32_t regionIdx = knnTree.sampleClosestRegionIdx(p, sample);
        return regionIdx;
        /*
        if (regionIdx != -1)
        {
            return &m_regionStorageContainer[regionIdx].first;
        }
        else
        {
            return nullptr;
        }
        */
    }


    virtual void fitRegions() = 0;

    virtual void updateRegions() = 0;

    void serialize(std::ostream& stream) const;

    void deserialize(std::istream& stream);

protected:
    uint32_t m_iteration {0};
    uint32_t m_totalSPP  {0};

    uint32_t m_nCores {20};

    RegionStorageContainerType m_regionStorageContainerSurface;
    RegionStorageContainerType m_regionStorageContainerVolume;

    float m_decayOnSpatialSplit {0.25f};
    bool m_deterministic {false};
private:
    //typename SpatialSubdivBuilder::Settings spatialSubdivBuilderSettings;
    bool m_useStochasticNNLookUp {false};
    // spatial structure
    SpatialSubdivBuilder m_spatialSubdivBuilder;
    typename SpatialSubdivBuilder::Settings m_spatialSubdivBuilderSettings;

    SpatialSubdivStructure m_spatialSubdivSurface;
    KNearestRegionsSearchTree m_regionKNNSearchTreeSurface;

    SpatialSubdivStructure m_spatialSubdivVolume;
    KNearestRegionsSearchTree m_regionKNNSearchTreeVolume;
};

template<class TRegion, typename TSampleContainer>
inline std::string SurfaceVolumeField<TRegion, TSampleContainer>::toString() const
{
    std::stringstream ss;
    ss << "SurfaceVolumeField:" << std::endl;
    ss << "  private: " << std::endl;
    ss << "    iteration: " << m_iteration << std::endl;
    ss << "    totalSPP: " << m_totalSPP << std::endl;
    ss << "    nCores: " << m_nCores << std::endl;
    ss << "    decayOnSpatialSplit: " << m_decayOnSpatialSplit << std::endl;
    ss << "    deterministic: " << m_deterministic << std::endl;
    ss << "    regionStorageContainerSurface::size: " << m_regionStorageContainerSurface.size() << std::endl;
    ss << "    regionStorageContainerVolume::size: " << m_regionStorageContainerVolume.size() << std::endl;
    ss << "  public: " << std::endl;
    ss << "    spatialSubdivBuilderSettings: " << m_spatialSubdivBuilderSettings.toString() << std::endl;
    ss << "    useStochasticNNLookUp: " << m_useStochasticNNLookUp << std::endl;
    ss << "    spatialSubdivBuilder: " << m_spatialSubdivBuilder.toString() << std::endl;
    ss << "    spatialSubdivBuilderSettings: " << m_spatialSubdivBuilderSettings.toString() << std::endl;
    ss << "    spatialSubdivSurface: " << m_spatialSubdivSurface.toString() << std::endl;
    ss << "    regionKNNSearchTreeSurface: " << m_regionKNNSearchTreeSurface.toString() << std::endl;
    ss << "    spatialSubdivVolume: " << m_spatialSubdivVolume.toString() << std::endl;
    ss << "    regionKNNSearchTreeVolume: " << m_regionKNNSearchTreeVolume.toString() << std::endl;

    return ss.str();
}

template<class TRegion, typename TSampleContainer>
inline std::string SurfaceVolumeField<TRegion, TSampleContainer>::Settings::toString() const
{
    std::stringstream ss;
    ss << "SurfaceVolumeField::Settings:" << std::endl;
    ss << "  spatialSubdivBuilderSettings: " << spatialSubdivBuilderSettings.toString() << std::endl;
    ss << "  useStochasticNNLookUp: " << useStochasticNNLookUp << std::endl;
    ss << "  deterministic: " << deterministic << std::endl;
    ss << "  decayOnSpatialSplit: " << decayOnSpatialSplit << std::endl;

    return ss.str();
}

template<class TRegion, typename TSampleContainer>
inline void SurfaceVolumeField<TRegion, TSampleContainer>::serialize(std::ostream& stream) const
{
    // protected
    stream.write(reinterpret_cast<const char*>(&m_iteration), sizeof(uint32_t));
    stream.write(reinterpret_cast<const char*>(&m_totalSPP), sizeof(uint32_t));
    stream.write(reinterpret_cast<const char*>(&m_nCores), sizeof(uint32_t));

    size_t num_surface_regions = m_regionStorageContainerSurface.size();
    stream.write(reinterpret_cast<const char*>(&num_surface_regions), sizeof(size_t));
    for(size_t n = 0; n < num_surface_regions; n++)
    {
        RegionStorageType region_storage = m_regionStorageContainerSurface[n];
        region_storage.first.serialize(stream);
        //region_storage.second.serialize(stream);
    }

    size_t num_volume_regions = m_regionStorageContainerVolume.size();
    stream.write(reinterpret_cast<const char*>(&num_volume_regions), sizeof(size_t));
    for(size_t n = 0; n < num_volume_regions; n++)
    {
        RegionStorageType region_storage = m_regionStorageContainerVolume[n];
        region_storage.first.serialize(stream);
        //region_storage.second.serialize(stream);
    }

    stream.write(reinterpret_cast<const char*>(&m_decayOnSpatialSplit), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&m_deterministic), sizeof(bool));

    // private
    stream.write(reinterpret_cast<const char*>(&m_useStochasticNNLookUp), sizeof(bool));

    //m_spatialSubdivBuilder.serialize(stream);
    m_spatialSubdivBuilderSettings.serialize(stream);

    m_spatialSubdivSurface.serialize(stream);
    m_spatialSubdivVolume.serialize(stream);

    m_regionKNNSearchTreeSurface.serialize(stream);
    m_regionKNNSearchTreeVolume.serialize(stream);
}

template<class TRegion, typename TSampleContainer>
inline void SurfaceVolumeField<TRegion, TSampleContainer>::deserialize(std::istream& stream)
{
    // protected
    stream.read(reinterpret_cast<char*>(&m_iteration), sizeof(uint32_t));
    stream.read(reinterpret_cast<char*>(&m_totalSPP), sizeof(uint32_t));
    stream.read(reinterpret_cast<char*>(&m_nCores), sizeof(uint32_t));

    size_t num_surface_regions =0;
    stream.read(reinterpret_cast<char*>(&num_surface_regions), sizeof(size_t));
    m_regionStorageContainerSurface.reserve(num_surface_regions);
    for(size_t n = 0; n < num_surface_regions; n++)
    {
        RegionStorageType region_storage;
        region_storage.first.deserialize(stream);
        //region_storage.second.deserialize(stream);
        m_regionStorageContainerSurface.push_back(region_storage);
    }

    size_t num_volume_regions =0;
    stream.read(reinterpret_cast<char*>(&num_volume_regions), sizeof(size_t));
    m_regionStorageContainerVolume.reserve(num_volume_regions);
    for(size_t n = 0; n < num_volume_regions; n++)
    {
        RegionStorageType region_storage;
        region_storage.first.deserialize(stream);

        m_regionStorageContainerVolume.push_back(region_storage);
    }

    stream.read(reinterpret_cast<char*>(&m_decayOnSpatialSplit), sizeof(float));
    stream.read(reinterpret_cast<char*>(&m_deterministic), sizeof(bool));
    // private
    stream.read(reinterpret_cast<char*>(&m_useStochasticNNLookUp), sizeof(bool));

    //m_spatialSubdivBuilder.deserialize(stream);
    m_spatialSubdivBuilderSettings.deserialize(stream);

    m_spatialSubdivSurface.deserialize(stream);
    m_spatialSubdivVolume.deserialize(stream);

    m_regionKNNSearchTreeSurface.deserialize(stream);
    m_regionKNNSearchTreeVolume.deserialize(stream);
}


}