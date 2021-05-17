// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/Range.h"
#include "../spatial/Region.h"
#include "../spatial/KNN.h"

#if !defined (OPENPGL_USE_OMP_THREADING)
#include <tbb/parallel_for.h>
#endif

namespace openpgl
{


template<class TDirectionalDistributionFactory, template<typename, typename> class TSpatialStructureBuilder>
struct Field
{

public:
    
    using DirectionalDistributionFactory = TDirectionalDistributionFactory;
    using DirectionalDistributionFactorySettings = typename TDirectionalDistributionFactory::Configuration;
    using DirectionalDistribution = typename TDirectionalDistributionFactory::Distribution;
    
    using SampleContainer = SampleDataStorage::SampleDataContainer;

    typedef Region<DirectionalDistribution, typename TDirectionalDistributionFactory::Statistics> RegionType;
    typedef openpgl::Range<SampleDataStorage::SampleDataContainer> RangeType;
    typedef std::pair<RegionType, RangeType > RegionStorageType;
    typedef tbb::concurrent_vector< RegionStorageType > RegionStorageContainerType;

    using SpatialStructureBuilder = TSpatialStructureBuilder<RegionType,RangeType>;
    using SpatialStructure = typename SpatialStructureBuilder::SpatialStructure;
    using SpatialBuilderSettings = typename SpatialStructureBuilder::Settings;



    struct SpatialSettings
    {
        SpatialBuilderSettings spatialSubdivBuilderSettings;
        bool useStochasticNNLookUp {false};
        bool deterministic {false};
        float decayOnSpatialSplit {0.25f};

        std::string toString() const;
    };

    struct Settings
    {
        SpatialSettings settings;
        DirectionalDistributionFactorySettings distributionFactorySettings;
        bool useParallaxCompensation{true};

        std::string toString()const;
    };

public:

    Field() = default;

    Field(const Settings &settings)
    {
        m_decayOnSpatialSplit = settings.settings.decayOnSpatialSplit;
        m_deterministic = settings.settings.deterministic;
        m_useStochasticNNLookUp = settings.settings.useStochasticNNLookUp;
        m_spatialSubdivBuilderSettings = settings.settings.spatialSubdivBuilderSettings;
        
        m_distributionFactorySettings = settings.distributionFactorySettings;
        m_useParallaxCompensation = settings.useParallaxCompensation;
    }

    void setSceneBounds(const openpgl::BBox &sceneBounds)
    {
        m_sceneBounds = sceneBounds;
        m_isSceneBoundsSet = true;
    }

    inline const RegionType *getRegion(const openpgl::Point3 &p, const float sample1D) const
    {
        if (m_iteration >0 && embree::inside(m_spatialSubdiv.getBounds(), p))
        {
            if(m_useStochasticNNLookUp)
            {
                uint32_t regionIdx =  getClosestRegionIdx(m_regionKNNSearchTree, p, sample1D);
                if(regionIdx != -1)
                {
                    return &m_regionStorageContainer[regionIdx].first;
                }
                else
                {
                    return nullptr;
                }
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

    void buildField(SampleContainer& samples)
    {
        m_iteration = 0;
        m_totalSPP  = 0;
        if (m_deterministic)
        {
            //std::cout << "SurfaceVolumeField::buildField(): deterministic = " << m_deterministic<< std::endl;
            std::sort(samples.begin(), samples.end(), SampleDataLess);
            //std::sort(samplesVolume.begin(), samplesVolume.end(), SampleDataLess);
        }

        //std::cout << "BufferSize: " << sizeof(SampleData) * m_spatialSubdivBuilderSettings.maxSamples * 1e-6 <<  " MB" << std::endl;
        //std::cout << "buildField: samplesSurface = " << samplesSurface.size() << "\t samplesVolume = " << samplesVolume.size() << std::endl;
        if(!m_isSceneBoundsSet)
        {
            estimateSceneBounds(samples);
        }
        
        buildSpatialStructure(m_sceneBounds, samples);
        fitRegions();
    }

    void updateField(SampleContainer& samples)
    {
        //std::cout << "updateField: samplesSurface = " << samplesSurface.size() << "\t samplesVolume = " << samplesVolume.size() << std::endl;
        if (m_deterministic)
        {
            //std::cout << "SurfaceVolumeField::buildField(): deterministic = " << m_deterministic << std::endl;
            std::sort(samples.begin(), samples.end(), SampleDataLess);
            //std::sort(samplesVolume.begin(), samplesVolume.end(), SampleDataLess);
        }

        updateSpatialStructure(samples);
        updateRegions();
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

    //std::string toString() const;

    //void serialize(std::ostream& stream) const;

    //void deserialize(std::istream& stream);

private:

    void estimateSceneBounds(const SampleContainer& samples)
    {
        m_sceneBounds.lower = Vector3(std::numeric_limits<float>::max());
        m_sceneBounds.upper = Vector3(std::numeric_limits<float>::min());
        
        // TODO parallize this part (also use some stats?)
        for (const auto& sample : samples)
        {
            m_sceneBounds.extend(Vector3(sample.position.x, sample.position.y, sample.position.z));
        }

        m_sceneBounds.enlarge_by(3.0f);
        m_isSceneBoundsSet = true;
    }

    inline uint32_t getClosestRegionIdx(const KNearestRegionsSearchTree &knnTree, const openpgl::Point3 &p, float sample) const
    {
        OPENPGL_ASSERT(knnTree.isBuild());
        const uint32_t regionIdx = knnTree.sampleClosestRegionIdx(p, sample);
        return regionIdx;
    }

    inline void buildSpatialStructure(const BBox &bounds, SampleContainer& samples)
    {
        m_spatialSubdivBuilder.build(m_spatialSubdiv, bounds, samples, m_regionStorageContainer, m_spatialSubdivBuilderSettings, m_nCores);
        if (m_useStochasticNNLookUp)
        {
            m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
        }
    }

    inline void updateSpatialStructure(SampleContainer& samples)
    {
        m_spatialSubdivBuilder.updateTree(m_spatialSubdiv, samples, m_regionStorageContainer, m_spatialSubdivBuilderSettings, m_nCores);
        if (m_useStochasticNNLookUp)
        {
            m_regionKNNSearchTree.buildRegionSearchTree<RegionStorageContainerType, RegionType>(m_regionStorageContainer);
        }
    }

    inline void fitRegions()
    {
        size_t nGuidingRegions = m_regionStorageContainer.size();
        std::cout << "fitRegion: "<< (m_isSurface? "surface":"volume") << "\tnGuidingRegions = " << nGuidingRegions << std::endl;
#if defined(OPENPGL_USE_OMP_THREADING)
        #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
        for (size_t n=0; n < nGuidingRegions; n++)
#else
        tbb::parallel_for( tbb::blocked_range<int>(0,nGuidingRegions), [&](tbb::blocked_range<int> r)
        {
        for (int n = r.begin(); n<r.end(); ++n) 
#endif
        {
            RegionStorageType &regionStorage = m_regionStorageContainer[n];
            openpgl::Point3 sampleMean = regionStorage.first.sampleStatistics.mean;
            std::vector<openpgl::SampleData> dataPoints;
            for (auto& sample : regionStorage.second)
            {
                if(m_useParallaxCompensation)
                {
                    reorientSample(sample, sampleMean);
                }
                OPENPGL_ASSERT(isValid(sample));
                dataPoints.push_back(sample);
            }
            if (dataPoints.size() > 0)
            {
                typename DirectionalDistributionFactory::FittingStatistics fittingStats;
                m_distributionFactory.fit(regionStorage.first.distribution, regionStorage.first.trainingStatistics, dataPoints.data(), dataPoints.size(), m_distributionFactorySettings, fittingStats);
                regionStorage.first.distribution._pivotPosition = sampleMean;
                regionStorage.first.valid = regionStorage.first.distribution.isValid();
                if(!regionStorage.first.valid)
                    std::cout << "!!!! " << (m_isSurface? "Surface":"Volume") << " regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
                regionStorage.first.splitFlag = false;
            }
            else
            {
                regionStorage.first.valid = false;
                regionStorage.first.splitFlag = false;
            }
        }
#if !defined (OPENPGL_USE_OMP_THREADING)
        });
#endif
    }

    void updateRegions()
    {
        size_t nGuidingRegions = m_regionStorageContainer.size();
        std::cout << "updateRegion: " << (m_isSurface? "surface":"volume") << "\tnGuidingRegions = " << nGuidingRegions << std::endl;
#if defined(OPENPGL_USE_OMP_THREADING)
        #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
        for (size_t n=0; n < nGuidingRegions; n++)
#else
        tbb::parallel_for( tbb::blocked_range<int>(0,nGuidingRegions), [&](tbb::blocked_range<int> r)
        {
        for (int n = r.begin(); n<r.end(); ++n)
#endif
        {
            RegionStorageType &regionStorage = m_regionStorageContainer[n];
            if (regionStorage.first.splitFlag)
            {
                regionStorage.first.trainingStatistics.decay(this->m_decayOnSpatialSplit);
                regionStorage.first.splitFlag = false;
            }

            openpgl::Point3 sampleMean = regionStorage.first.sampleStatistics.mean;
            std::vector<openpgl::SampleData> dataPoints;
            for (auto& sample : regionStorage.second)
            {
                if(m_useParallaxCompensation)
                {
                    reorientSample(sample, sampleMean);
                }
                OPENPGL_ASSERT(isValid(sample));
                dataPoints.push_back(sample);
            }
            if (dataPoints.size() > 0)
            {
                RegionType oldRegion = regionStorage.first;
                if(m_useParallaxCompensation)
                {
                    regionStorage.first.trainingStatistics.sufficientStatistics.applyParallaxShift(regionStorage.first.distribution, regionStorage.first.distribution._pivotPosition - sampleMean);
                    regionStorage.first.distribution.performRelativeParallaxShift(regionStorage.first.distribution._pivotPosition - sampleMean);
                    OPENPGL_ASSERT(regionStorage.first.distribution.isValid());
                    OPENPGL_ASSERT(regionStorage.first.trainingStatistics.sufficientStatistics.isValid());
                }
                typename DirectionalDistributionFactory::FittingStatistics fittingStats;
                m_distributionFactory.update(regionStorage.first.distribution, regionStorage.first.trainingStatistics, dataPoints.data(), dataPoints.size(), m_distributionFactorySettings, fittingStats);
                //regionStorage.first.valid = regionStorage.first.distribution.isValid();
                regionStorage.first.valid = regionStorage.first.isValid();
                if(!regionStorage.first.valid)
                {
                    std::cout << "!!!! " << (m_isSurface? "Surface":"Volume") << " regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
                    storeInvalidRegionData("regionBeforeUpdate_"+ std::string((m_isSurface? "surf":"vol")) + "_itr" + std::to_string(this->m_iteration) + "_region" + std::to_string(n)+".dump", oldRegion, dataPoints, m_distributionFactorySettings);
                }
            }
            else
            {
                regionStorage.first.valid = false;
                regionStorage.first.splitFlag = false;
            }
        }
#if !defined (OPENPGL_USE_OMP_THREADING)
        });
#endif
    }

    static void storeInvalidRegionData(const std::string &fileName, const RegionType &regionBeforeUpdate, const std::vector<SampleData> &samples, const DirectionalDistributionFactorySettings &factorySettings)
    {
        std::filebuf fbDump;
        fbDump.open (fileName,std::ios::out);
        std::ostream dumpStream(&fbDump);

        factorySettings.serialize(dumpStream);

        size_t numSamples = samples.size();
        dumpStream.write(reinterpret_cast<const char*>(&numSamples), sizeof(size_t));
        for (size_t i = 0; i < numSamples; i++)
        {
            dumpStream.write(reinterpret_cast<const char*>(&samples[i]), sizeof(openpgl::SampleData));
        }

        regionBeforeUpdate.serialize(dumpStream);
        fbDump.close();
    }

    static void loadInvalidRegionData(const std::string &fileName, RegionType &regionBeforeUpdate, std::vector<SampleData> &samples, DirectionalDistributionFactorySettings &factorySettings)
    {
        std::filebuf fbDumpIn;
        fbDumpIn.open (fileName,std::ios::in);
        std::istream dumpIStream(&fbDumpIn);

        factorySettings.deserialize(dumpIStream);
        samples.clear();
        size_t numSamples;
        dumpIStream.read(reinterpret_cast<char*>(&numSamples), sizeof(size_t));
        for (size_t i = 0; i < numSamples; i++)
        {
            SampleData dsd;
            dumpIStream.read(reinterpret_cast<char*>(&dsd), sizeof(openpgl::SampleData));
            samples.push_back(dsd);
        }
        regionBeforeUpdate.deserialize(dumpIStream);
        fbDumpIn.close();
    }

    void reorientSample(openpgl::SampleData &sample, const openpgl::Point3 &pivotPoint) const
    {

        if (std::isinf(sample.distance))
        {
            sample.position.x = pivotPoint[0];
            sample.position.y = pivotPoint[1];
            sample.position.z = pivotPoint[2];
            return;
        }
        else if (!(sample.distance > 0.0f))
        {
            return;
        }

        const openpgl::Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
        const openpgl::Vector3 sampleDirection(sample.direction.x, sample.direction.y, sample.direction.z);
        const openpgl::Point3 originPosition = samplePosition + sampleDirection * sample.distance;
        openpgl::Vector3 newDirection = originPosition - pivotPoint;
        const float newDistance = embree::length(newDirection);
        newDirection = newDirection / newDistance;

        sample.position.x = pivotPoint[0];
        sample.position.y = pivotPoint[1];
        sample.position.z = pivotPoint[2];
        sample.distance = newDistance;
        sample.direction.x = newDirection[0];
        sample.direction.y = newDirection[1];
        sample.direction.z = newDirection[2];
    }

private:

    bool m_isSurface{true};

    float m_decayOnSpatialSplit {0.25f};
    bool m_useParallaxCompensation {true};

    size_t m_iteration {0};
    size_t m_totalSPP  {0};

    size_t m_nCores {20};

    bool m_deterministic {false};

    bool m_isSceneBoundsSet{false};
    BBox m_sceneBounds;


    DirectionalDistributionFactory m_distributionFactory;
    DirectionalDistributionFactorySettings m_distributionFactorySettings;
    /////////////////////////////////////////////////////////
    ///////// Spatial Structure        //////////////////////
    /////////////////////////////////////////////////////////

    SpatialStructureBuilder m_spatialSubdivBuilder;
    SpatialBuilderSettings m_spatialSubdivBuilderSettings;
    
    SpatialStructure m_spatialSubdiv;
    RegionStorageContainerType m_regionStorageContainer;

    bool m_useStochasticNNLookUp {false};
    KNearestRegionsSearchTree m_regionKNNSearchTree;
};

}