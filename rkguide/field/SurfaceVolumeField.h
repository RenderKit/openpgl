// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define USE_TBB

#include "../rkguide.h"
//#include "../data/SampleStatistics.h"
#include "../data/Range.h"
#include "../kdtree/KDTree.h"
#include "../kdtree/KDTreeBuilder.h"
#include "Region.h"
#include "KNN.h"

namespace rkguide
{

template<class TRegion, typename TSampleContainer>
struct SurfaceVolumeField
{
public:
    //typedef TDistribution DistributionType;
    //typedef typename TDistributionFactory::ASMStatistics   StatisticsType;

    //typedef rkguide::Region<DistributionType, StatisticsType> RegionType;
    typedef TRegion RegionType;
    typedef rkguide::Range<TSampleContainer> RangeType;
protected:

    typedef std::pair<RegionType, RangeType > RegionStorageType;
    typedef tbb::concurrent_vector< RegionStorageType > RegionStorageContainerType;

    typedef rkguide::KDTree SpatialSubdivStructure;
    typedef rkguide::KDTreePartitionBuilder<RegionType, RangeType> SpatialSubdivBuilder;

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
/*
    const RegionType *getGuidingRegion( const rkguide::Point3 &p, rkguide::Sampler *sampler) const
    {
        if (m_iteration >0 && embree::inside(m_spatialSubdiv.getBounds(), p))
        {
            if(m_useStochasticNNLookUp)
            {
                return getClosestRegion(p, sampler->next1D());
            }
            else
            {
                rkguide::BBox regionBounds;
                uint32_t dataIdx = m_spatialSubdiv.getDataIdxAtPos(p, regionBounds);
                RKGUIDE_ASSERT(dataIdx >= 0);
                return &m_regionStorageContainer[dataIdx].first;
            }
        }
        else
        {
            return nullptr;
        }
    }
*/

    const RegionType *getSurfaceGuidingRegion( const rkguide::Point3 &p, rkguide::Sampler *sampler) const
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
                rkguide::BBox regionBounds;
                uint32_t dataIdx = m_spatialSubdivSurface.getDataIdxAtPos(p, regionBounds);
                RKGUIDE_ASSERT(dataIdx >= 0);
                return &m_regionStorageContainerSurface[dataIdx].first;
            }
        }
        else
        {
            return nullptr;
        }
    }


    const RegionType *getVolumeGuidingRegion( const rkguide::Point3 &p, rkguide::Sampler *sampler) const
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
                rkguide::BBox regionBounds;
                uint32_t dataIdx = m_spatialSubdivVolume.getDataIdxAtPos(p, regionBounds);
                RKGUIDE_ASSERT(dataIdx >= 0);
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
            std::sort(samplesSurface.begin(), samplesSurface.end());
            std::sort(samplesVolume.begin(), samplesVolume.end());
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
            std::sort(samplesSurface.begin(), samplesSurface.end());
            std::sort(samplesVolume.begin(), samplesVolume.end());
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

    std::string toString() const
    {
        return "";
    }

protected:
/*
    void buildSpatialStructure(const BBox &bounds, TSampleContainer& samples)
    {
        //mitsuba::SLog(mitsuba::EInfo, "Begin Tree update");
        mitsuba::ref<mitsuba::Timer> treeTimer = new mitsuba::Timer();
        m_spatialSubdivBuilder.build(m_spatialSubdiv, bounds, samples, m_regionStorageContainer, m_spatialSubdivBuilderSettings, m_nCores);
        //mitsuba::SLog(mitsuba::EInfo, "Tree building time: %s", timeString(treeTimer->getSeconds(), true).c_str());

        if (m_useStochasticNNLookUp)
        {
            mitsuba::ref<mitsuba::Timer> embreereeTimer = new mitsuba::Timer();
            m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
            //mitsuba::SLog(mitsuba::EInfo, "Embree BVH update time: %s", timeString(embreereeTimer->getSeconds(), true).c_str());
        }
    }


    void updateSpatialStructure(TSampleContainer& samples)
    {
        //mitsuba::SLog(mitsuba::EInfo, "Begin Tree update");
        mitsuba::ref<mitsuba::Timer> treeTimer = new mitsuba::Timer();
        m_spatialSubdivBuilder.updateTree(m_spatialSubdiv, samples, m_regionStorageContainer, m_spatialSubdivBuilderSettings, m_nCores);
        //mitsuba::SLog(mitsuba::EInfo, "Tree building time: %s", timeString(treeTimer->getSeconds(), true).c_str());

        if (m_useStochasticNNLookUp)
        {
            mitsuba::ref<mitsuba::Timer> embreereeTimer = new mitsuba::Timer();
            m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
            //mitsuba::SLog(mitsuba::EInfo, "Embree BVH update time: %s", timeString(embreereeTimer->getSeconds(), true).c_str());
        }
    }
*/

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

    uint32_t getClosestRegionIdx(const KNearestRegionsSearchTree &knnTree, const rkguide::Point3 &p, float sample) const
    {
        RKGUIDE_ASSERT(knnTree.isBuild());
        //RKGUIDE_ASSERT(knnTree.numRegions() == m_regionStorageContainer.size());

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

protected:
    uint32_t m_iteration {0};
    uint32_t m_totalSPP  {0};

    uint32_t m_nCores {20};

    RegionStorageContainerType m_regionStorageContainerSurface;
    RegionStorageContainerType m_regionStorageContainerVolume;

    float m_decayOnSpatialSplit {0.25f};
    bool m_deterministic {false};
private:
    typename SpatialSubdivBuilder::Settings spatialSubdivBuilderSettings;
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
inline std::string SurfaceVolumeField<TRegion, TSampleContainer>::Settings::toString() const
{
    std::stringstream ss;
    ss << "SurfaceVolumeField::Settings:" << std::endl;
    ss << "spatialSubdivBuilderSettings: " << spatialSubdivBuilderSettings.toString() << std::endl;
    ss << "useStochasticNNLookUp: " << useStochasticNNLookUp << std::endl;
    ss << "deterministic: " << deterministic << std::endl;
    ss << "decayOnSpatialSplit: " << decayOnSpatialSplit << std::endl;

    return ss.str();
}

}