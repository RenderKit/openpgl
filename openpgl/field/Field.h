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
struct Field
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

    Field() = default;

    Field(const Settings &settings):
        m_decayOnSpatialSplit(settings.decayOnSpatialSplit),
        m_deterministic(settings.deterministic),
        m_useStochasticNNLookUp(settings.useStochasticNNLookUp),
        m_spatialSubdivBuilderSettings(settings.spatialSubdivBuilderSettings){
    }

    const RegionType *getGuidingRegion( const openpgl::Point3 &p, openpgl::Sampler *sampler) const
    {
        if (m_iteration >0 && embree::inside(m_spatialSubdiv.getBounds(), p))
        {
            if(m_useStochasticNNLookUp)
            {
                return getClosestRegion(p, sampler->next1D());
            }
            else
            {
                openpgl::BBox regionBounds;
                uint32_t dataIdx = m_spatialSubdiv.getDataIdxAtPos(p, regionBounds);
                OPENPGL_ASSERT(dataIdx >= 0);
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

    std::string toString() const;

protected:

    void buildSpatialStructure(const BBox &bounds, TSampleContainer& samples)
    {
        //mitsuba::SLog(mitsuba::EInfo, "Begin Tree update");
//        mitsuba::ref<mitsuba::Timer> treeTimer = new mitsuba::Timer();
        m_spatialSubdivBuilder.build(m_spatialSubdiv, bounds, samples, m_regionStorageContainer, m_spatialSubdivBuilderSettings, m_nCores);
        //mitsuba::SLog(mitsuba::EInfo, "Tree building time: %s", timeString(treeTimer->getSeconds(), true).c_str());

        if (m_useStochasticNNLookUp)
        {
//            mitsuba::ref<mitsuba::Timer> embreereeTimer = new mitsuba::Timer();
            m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
            //mitsuba::SLog(mitsuba::EInfo, "Embree BVH update time: %s", timeString(embreereeTimer->getSeconds(), true).c_str());
        }
    }


    void updateSpatialStructure(TSampleContainer& samples)
    {
        //mitsuba::SLog(mitsuba::EInfo, "Begin Tree update");
//        mitsuba::ref<mitsuba::Timer> treeTimer = new mitsuba::Timer();
        m_spatialSubdivBuilder.updateTree(m_spatialSubdiv, samples, m_regionStorageContainer, m_spatialSubdivBuilderSettings, m_nCores);
        //mitsuba::SLog(mitsuba::EInfo, "Tree building time: %s", timeString(treeTimer->getSeconds(), true).c_str());

        if (m_useStochasticNNLookUp)
        {
//            mitsuba::ref<mitsuba::Timer> embreereeTimer = new mitsuba::Timer();
            m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
            //mitsuba::SLog(mitsuba::EInfo, "Embree BVH update time: %s", timeString(embreereeTimer->getSeconds(), true).c_str());
        }
    }

    void updateKNNRegionSearchTree()
    {
        m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
    }

    const RegionType *getClosestRegion(const openpgl::Point3 &p, float sample) const
    {
        OPENPGL_ASSERT(m_regionKNNSearchTree.isBuild());
        OPENPGL_ASSERT(m_regionKNNSearchTree.numRegions() == m_regionStorageContainer.size());

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

    void serialize(std::ostream& stream) const;

    void deserialize(std::istream& stream);

protected:
    uint32_t m_iteration {0};
    uint32_t m_totalSPP  {0};

    uint32_t m_nCores {20};

    RegionStorageContainerType m_regionStorageContainer;

    float m_decayOnSpatialSplit {0.25f};
    bool m_deterministic {false};
private:
    bool m_useStochasticNNLookUp {false};
    // spatial structure
    SpatialSubdivBuilder m_spatialSubdivBuilder;
    typename SpatialSubdivBuilder::Settings m_spatialSubdivBuilderSettings;
    SpatialSubdivStructure m_spatialSubdiv;
    KNearestRegionsSearchTree m_regionKNNSearchTree;
};

template<class TRegion, typename TSampleContainer>
inline std::string Field<TRegion, TSampleContainer>::toString() const
{
    std::stringstream ss;
    ss << "Field:" << std::endl;
    ss << "  private: " << std::endl;
    ss << "    iteration: " << m_iteration << std::endl;
    ss << "    totalSPP: " << m_totalSPP << std::endl;
    ss << "    nCores: " << m_nCores << std::endl;
    ss << "    decayOnSpatialSplit: " << m_decayOnSpatialSplit << std::endl;
    ss << "    deterministic: " << m_deterministic << std::endl;
    ss << "    regionStorageContainer::size: " << m_regionStorageContainer.size() << std::endl;
    ss << "  public: " << std::endl;
    ss << "    spatialSubdivBuilderSettings: " << m_spatialSubdivBuilderSettings.toString() << std::endl;
    ss << "    useStochasticNNLookUp: " << m_useStochasticNNLookUp << std::endl;
    ss << "    spatialSubdivBuilder: " << m_spatialSubdivBuilder.toString() << std::endl;
    ss << "    spatialSubdivBuilderSettings: " << m_spatialSubdivBuilderSettings.toString() << std::endl;
    ss << "    spatialSubdiv: " << m_spatialSubdiv.toString() << std::endl;
    ss << "    regionKNNSearchTree: " << m_regionKNNSearchTree.toString() << std::endl;

    return ss.str();
}


template<class TRegion, typename TSampleContainer>
inline std::string Field<TRegion, TSampleContainer>::Settings::toString() const
{
    std::stringstream ss;
    ss << "Field::Settings:" << std::endl;
    ss << "  spatialSubdivBuilderSettings: " << spatialSubdivBuilderSettings.toString() << std::endl;
    ss << "  useStochasticNNLookUp: " << useStochasticNNLookUp << std::endl;
    ss << "  deterministic: " << deterministic << std::endl;
    ss << "  decayOnSpatialSplit: " << decayOnSpatialSplit << std::endl;

    return ss.str();
}

template<class TRegion, typename TSampleContainer>
inline void Field<TRegion, TSampleContainer>::serialize(std::ostream& stream) const
{
    // protected
    stream.write(reinterpret_cast<const char*>(&m_iteration), sizeof(uint32_t));
    stream.write(reinterpret_cast<const char*>(&m_totalSPP), sizeof(uint32_t));
    stream.write(reinterpret_cast<const char*>(&m_nCores), sizeof(uint32_t));

    size_t num_regions = m_regionStorageContainer.size();
    stream.write(reinterpret_cast<const char*>(&num_regions), sizeof(size_t));
    for(size_t n = 0; n < num_regions; n++)
    {
        RegionStorageType region_storage = m_regionStorageContainer[n];
        region_storage.first.serialize(stream);
        //region_storage.second.serialize(stream);
    }
    stream.write(reinterpret_cast<const char*>(&m_decayOnSpatialSplit), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&m_deterministic), sizeof(bool));

    // private
    stream.write(reinterpret_cast<const char*>(&m_useStochasticNNLookUp), sizeof(bool));

    //m_spatialSubdivBuilder.serialize(stream);
    m_spatialSubdivBuilderSettings.serialize(stream);

    m_spatialSubdiv.serialize(stream);
    m_regionKNNSearchTree.serialize(stream);
}

template<class TRegion, typename TSampleContainer>
inline void Field<TRegion, TSampleContainer>::deserialize(std::istream& stream)
{
    // protected
    stream.read(reinterpret_cast<char*>(&m_iteration), sizeof(uint32_t));
    stream.read(reinterpret_cast<char*>(&m_totalSPP), sizeof(uint32_t));
    stream.read(reinterpret_cast<char*>(&m_nCores), sizeof(uint32_t));

    size_t num_regions =0;
    stream.read(reinterpret_cast<char*>(&num_regions), sizeof(size_t));
    m_regionStorageContainer.reserve(num_regions);
    for(size_t n = 0; n < num_regions; n++)
    {
        RegionStorageType region_storage;
        region_storage.first.deserialize(stream);
        //region_storage.second.deserialize(stream);
        m_regionStorageContainer.push_back(region_storage);
    }
    stream.read(reinterpret_cast<char*>(&m_decayOnSpatialSplit), sizeof(float));
    stream.read(reinterpret_cast<char*>(&m_deterministic), sizeof(bool));
    // private
    stream.read(reinterpret_cast<char*>(&m_useStochasticNNLookUp), sizeof(bool));

    //m_spatialSubdivBuilder.deserialize(stream);
    m_spatialSubdivBuilderSettings.deserialize(stream);

    m_spatialSubdiv.deserialize(stream);
    m_regionKNNSearchTree.deserialize(stream);
}

}