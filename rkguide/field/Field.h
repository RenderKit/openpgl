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
struct Field
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
    };

    Field() = default;

    Field(const Settings &settings):
        m_decayOnSpatialSplit(settings.decayOnSpatialSplit),
        m_deterministic(settings.deterministic),
        m_useStochasticNNLookUp(settings.useStochasticNNLookUp),
        m_spatialSubdivBuilderSettings(settings.spatialSubdivBuilderSettings){
    }

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

    void buildField(const BBox &bounds, TSampleContainer& samples)
    {
        m_iteration = 0;
        m_totalSPP  = 0;
        if (m_deterministic)
        {
            std::sort(samples.begin(), samples.end());
        }
#if defined(MTS_OPENMP)
//        ref<Scheduler> scheduler = Scheduler::getInstance();
//        size_t nCores = scheduler->getCoreCount();

        //m_nCores = nCores;
//        m_nCores = 20;
        //SLog(EWarn, "nCores: %d", nCores);
        mitsuba::Thread::initializeOpenMP(m_nCores);
#endif
        std::cout << "BufferSize: " << sizeof(DirectionalSampleData) * m_spatialSubdivBuilderSettings.maxSamples * 1e-6 <<  " MB" << std::endl;
        buildSpatialStructure(bounds, samples);
        fitRegions();
    }

    void updateField(TSampleContainer& samples)
    {
        if (m_deterministic)
        {
            std::sort(samples.begin(), samples.end());
        }
#if defined(MTS_OPENMP)
//        ref<Scheduler> scheduler = Scheduler::getInstance();
//        size_t nCores = scheduler->getCoreCount();

        //m_nCores = nCores;
//        m_nCores = 20;
        //SLog(EWarn, "nCores: %d", nCores);
        mitsuba::Thread::initializeOpenMP(m_nCores);
#endif
        updateSpatialStructure(samples);
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

    void updateKNNRegionSearchTree()
    {
        m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
    }

    const RegionType *getClosestRegion(const rkguide::Point3 &p, float sample) const
    {
        RKGUIDE_ASSERT(m_regionKNNSearchTree.isBuild());
        RKGUIDE_ASSERT(m_regionKNNSearchTree.numRegions() == m_regionStorageContainer.size());

        const uint32_t regionIdx = m_regionKNNSearchTree.sampleClosestRegionIdx(p, sample);
        if (regionIdx != -1)
        {
            return &m_regionStorageContainer[regionIdx].first;
        }
        else
        {
            return nullptr;
        }
    }


    virtual void fitRegions() = 0;

    virtual void updateRegions() = 0;

protected:
    uint32_t m_iteration {0};
    uint32_t m_totalSPP  {0};

    uint32_t m_nCores {20};

    RegionStorageContainerType m_regionStorageContainer;

    float m_decayOnSpatialSplit {0.25f};
    bool m_deterministic {false};
private:
    typename SpatialSubdivBuilder::Settings spatialSubdivBuilderSettings;
    bool m_useStochasticNNLookUp {false};
    // spatial structure
    SpatialSubdivBuilder m_spatialSubdivBuilder;
    typename SpatialSubdivBuilder::Settings m_spatialSubdivBuilderSettings;
    SpatialSubdivStructure m_spatialSubdiv;
    KNearestRegionsSearchTree m_regionKNNSearchTree;
};


}