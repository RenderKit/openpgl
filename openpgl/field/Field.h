// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../data/Range.h"
#include "../spatial/Region.h"
#include "../spatial/KNN.h"

#define TASKING_TBB
#include <embreeSrc/common/algorithms/parallel_for.h>

#if !defined (OPENPGL_USE_OMP_THREADING)
#include <tbb/parallel_for.h>
#endif
#include <tbb/parallel_sort.h>

#define USE_PRECOMPUTED_NN 1
//#define USE_PRESORT

namespace openpgl
{


template<int Vecsize, class TDirectionalDistributionFactory, template<typename, typename> class TSpatialStructureBuilder>
struct Field
{
public:

    using DirectionalDistributionFactory = TDirectionalDistributionFactory;
    using DirectionalDistributionFactorySettings = typename TDirectionalDistributionFactory::Configuration;
    using DirectionalDistribution = typename TDirectionalDistributionFactory::Distribution;

    using SampleContainer = SampleDataStorage::SampleDataContainer;
    using SampleContainerInternal = std::vector<SampleData>;

    typedef Region<DirectionalDistribution, typename TDirectionalDistributionFactory::Statistics> RegionType;
    typedef openpgl::Range RangeType;
    typedef std::pair<RegionType, RangeType > RegionStorageType;
    typedef tbb::concurrent_vector< RegionStorageType > RegionStorageContainerType;

    using SpatialStructureBuilder = TSpatialStructureBuilder<RegionType, SampleContainerInternal>;
    using SpatialStructure = typename SpatialStructureBuilder::SpatialStructure;
    using SpatialBuilderSettings = typename SpatialStructureBuilder::Settings;

    struct DebugSettings
    {
        bool fitRegions {true};
    };

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
        DebugSettings debugSettings;
        std::string toString()const;
    };

public:

    Field() = default;

    Field(const Settings &settings)
    {
        m_decayOnSpatialSplit = settings.settings.decayOnSpatialSplit;
        m_deterministic = settings.settings.deterministic;
        m_fitRegions = settings.debugSettings.fitRegions;
        m_useStochasticNNLookUp = settings.settings.useStochasticNNLookUp;
        m_spatialSubdivBuilderSettings = settings.settings.spatialSubdivBuilderSettings;

        m_distributionFactorySettings = settings.distributionFactorySettings;
        samples_.reserve(1e6);
    }

    ~Field()
    {
        m_regionKNNSearchTree.reset();
    }

    void setSceneBounds(const openpgl::BBox &sceneBounds)
    {
        m_sceneBounds = sceneBounds;
        m_isSceneBoundsSet = true;
    }

    openpgl::BBox getSceneBounds() const
    {
        return m_sceneBounds;
    }

    void setIsSurface(const bool isSurface)
    {
        m_isSurface = isSurface;
    }

    inline const RegionType *getRegion(const openpgl::Point3 &p, float *sample1D, uint32_t &id ) const
    {
        if (m_iteration >0 && embree::inside(m_spatialSubdiv.getBounds(), p))
        {
            if(m_useStochasticNNLookUp)
            {
                if (USE_PRECOMPUTED_NN)
                {
                    uint32_t regionIdx = getApproximateClosestRegionIdx(m_regionKNNSearchTree, p, sample1D, id);
                    return &m_regionStorageContainer[regionIdx].first;
                }
                else
                {
                    uint32_t regionIdx =  getClosestRegionIdx(m_regionKNNSearchTree, p, sample1D, id);
                    if(regionIdx != -1)
                    {
                        return &m_regionStorageContainer[regionIdx].first;
                    }
                    else
                    {
                        return nullptr;
                    }
                }
            }
            else
            {
                uint32_t dataIdx = m_spatialSubdiv.getDataIdxAtPos(p);
                OPENPGL_ASSERT(dataIdx < m_regionStorageContainer.size());
                id = dataIdx;
                return &m_regionStorageContainer[dataIdx].first;
            }
        }
        else
        {
            return nullptr;
        }
    }

    void buildField(const SampleContainer& samples)
    {
        m_iteration = 0;
        m_totalSPP  = 0;
        if(samples.size() > 0)
        {
            if(samples_.capacity() < samples.size()) {
                samples_.reserve(2 * samples.size());
            }
            samples_.resize(samples.size());

            embree::parallel_for( size_t(0), samples.size(), size_t(4*4096), [&](const embree::range<size_t>& r) {
                for (size_t i=r.begin(); i<r.end(); i++) 
                    samples_[i] = samples[i];
            });
#ifdef USE_PRESORT
            if (m_deterministic)
            {
                tbb::parallel_sort(samples_.begin(), samples_.end(), SampleDataLess);
                //std::sort(samples.begin(), samples.end(), SampleDataLess);
            }
#endif
            //std::cout << "BufferSize: " << sizeof(SampleData) * m_spatialSubdivBuilderSettings.maxSamples * 1e-6 <<  " MB" << std::endl;
            //std::cout << "buildField: samplesSurface = " << samplesSurface.size() << "\t samplesVolume = " << samplesVolume.size() << std::endl;
            if(!m_isSceneBoundsSet)
            {
                estimateSceneBounds(samples_);
            }

            buildSpatialStructure(m_sceneBounds, samples_);
            fitRegions(samples_);
        }
        m_iteration++;
    }

    void updateField(const SampleContainer& samples)
    {
        if(samples.size() > 0)
        {
            if(samples_.capacity() < samples.size()) {
                samples_.reserve(2 * samples.size());
            }
            samples_.resize(samples.size());

            embree::parallel_for( size_t(0), samples.size(), size_t(4*4096), [&](const embree::range<size_t>& r) {
                for (size_t i=r.begin(); i<r.end(); i++) 
                    samples_[i] = samples[i];
            });
            //std::cout << "updateField: samplesSurface = " << samplesSurface.size() << "\t samplesVolume = " << samplesVolume.size() << std::endl;
#ifdef USE_PRESORT
            if (m_deterministic)
            {
                tbb::parallel_sort(samples_.begin(), samples_.end(), SampleDataLess);
                //std::sort(samples.begin(), samples.end(), SampleDataLess);
            }
#endif
            updateSpatialStructure(samples_);
            updateRegions(samples_);
        }
        m_iteration++;
    }

    void resetField()
    {
        m_iteration = 0;
        m_totalSPP = 0;
        
        m_isSceneBoundsSet =false;
        m_initialized = false;
        
        m_spatialSubdiv = SpatialStructure();
        m_regionStorageContainer.clear();
        m_regionKNNSearchTree.reset();
    }

    size_t getIteration() const
    {
        return m_iteration;
    }

    void serialize(std::ostream &os) const {
        os.write(reinterpret_cast<const char*>(&m_isSurface), sizeof(m_isSurface));
        os.write(reinterpret_cast<const char*>(&m_decayOnSpatialSplit), sizeof(m_decayOnSpatialSplit));
        os.write(reinterpret_cast<const char*>(&m_iteration), sizeof(m_iteration));
        os.write(reinterpret_cast<const char*>(&m_totalSPP), sizeof(m_totalSPP));
        os.write(reinterpret_cast<const char*>(&m_nCores), sizeof(m_nCores));
        os.write(reinterpret_cast<const char*>(&m_deterministic), sizeof(m_deterministic));
        os.write(reinterpret_cast<const char*>(&m_fitRegions), sizeof(m_fitRegions));
        os.write(reinterpret_cast<const char*>(&m_isSceneBoundsSet), sizeof(m_isSceneBoundsSet));
        os.write(reinterpret_cast<const char*>(&m_sceneBounds), sizeof(m_sceneBounds));
        os.write(reinterpret_cast<const char*>(&m_initialized), sizeof(m_initialized));

        m_distributionFactorySettings.serialize(os);
        m_spatialSubdivBuilderSettings.serialize(os);
        m_spatialSubdiv.serialize(os);
        size_t size = m_regionStorageContainer.size();
        os.write(reinterpret_cast<const char*>(&size), sizeof(size));
        for (size_t i = 0; i < size; i++) {
            m_regionStorageContainer[i].first.serialize(os);
            m_regionStorageContainer[i].second.serialize(os);
        }
        os.write(reinterpret_cast<const char*>(&m_useStochasticNNLookUp), sizeof(m_useStochasticNNLookUp));
        m_regionKNNSearchTree.serialize(os);
    }

    void deserialize(std::istream& is)
    {
        is.read(reinterpret_cast<char*>(&m_isSurface), sizeof(m_isSurface));
        is.read(reinterpret_cast<char*>(&m_decayOnSpatialSplit), sizeof(m_decayOnSpatialSplit));
        is.read(reinterpret_cast<char*>(&m_iteration), sizeof(m_iteration));
        is.read(reinterpret_cast<char*>(&m_totalSPP), sizeof(m_totalSPP));
        is.read(reinterpret_cast<char*>(&m_nCores), sizeof(m_nCores));
        is.read(reinterpret_cast<char*>(&m_deterministic), sizeof(m_deterministic));
        is.read(reinterpret_cast<char*>(&m_fitRegions), sizeof(m_fitRegions));
        is.read(reinterpret_cast<char*>(&m_isSceneBoundsSet), sizeof(m_isSceneBoundsSet));
        is.read(reinterpret_cast<char*>(&m_sceneBounds), sizeof(m_sceneBounds));
        is.read(reinterpret_cast<char*>(&m_initialized), sizeof(m_initialized));

        m_distributionFactorySettings.deserialize(is);
        m_spatialSubdivBuilderSettings.deserialize(is);
        m_spatialSubdiv.deserialize(is);
        size_t size;
        is.read(reinterpret_cast<char*>(&size), sizeof(size));
        m_regionStorageContainer.clear();
        m_regionStorageContainer.reserve(size);
        for (size_t i = 0; i < size; i++) {
            m_regionStorageContainer.emplace_back();
            m_regionStorageContainer[i].first.deserialize(is);
            m_regionStorageContainer[i].second.deserialize(is);

        }
        is.read(reinterpret_cast<char*>(&m_useStochasticNNLookUp), sizeof(m_useStochasticNNLookUp));
        m_regionKNNSearchTree.deserialize(is);

        if (m_useStochasticNNLookUp && USE_PRECOMPUTED_NN && m_regionKNNSearchTree.isBuild()) {
            m_regionKNNSearchTree.buildRegionNeighbours();
        }
    }

    bool isValid() const
    {
        bool valid = true;
        size_t nGuidingRegions = m_regionStorageContainer.size();
        for (int n = 0; n < nGuidingRegions; n++)
        {
            valid = valid && m_regionStorageContainer[n].first.isValid() && m_regionStorageContainer[n].first.valid;
        }
        return valid;
    }

    bool isInitialized() const
    {
        return m_initialized;
    }

private:

    void estimateSceneBounds(const SampleContainerInternal& samples)
    {
        m_sceneBounds.lower = Vector3(std::numeric_limits<float>::max());
        m_sceneBounds.upper = Vector3(std::numeric_limits<float>::min());
        m_isSceneBoundsSet = false;

        if(samples.size() > 0)
        {
            // TODO parallize this part (also use some stats?)
            for (const auto& sample : samples)
            {
                m_sceneBounds.extend(Vector3(sample.position.x, sample.position.y, sample.position.z));
            }
            Vector3 center = m_sceneBounds.center();
            m_sceneBounds.lower = center + 3.0f * (m_sceneBounds.lower - center); 
            m_sceneBounds.upper = center + 3.0f * (m_sceneBounds.upper - center);
            m_isSceneBoundsSet = true;
        }
    }

    inline uint32_t getClosestRegionIdx(const KNearestRegionsSearchTree<Vecsize> &knnTree, const openpgl::Point3 &p, float *sample, uint32_t &id) const
    {
        OPENPGL_ASSERT(knnTree.isBuild());
        const uint32_t regionIdx = knnTree.sampleClosestRegionIdx(p, sample);
        return regionIdx;
    }

    inline uint32_t getApproximateClosestRegionIdx(const KNearestRegionsSearchTree<Vecsize> &knnTree, const openpgl::Point3 &p, float *sample, uint32_t &id) const
    {
        OPENPGL_ASSERT(knnTree.isBuildNeighbours());
        uint32_t dataIdx = m_spatialSubdiv.getDataIdxAtPos(p);
        OPENPGL_ASSERT(dataIdx < m_regionStorageContainer.size());
        id = dataIdx;
        return knnTree.sampleApproximateClosestRegionIdx(dataIdx, p, sample);
    }

    inline void buildSpatialStructure(const BBox &bounds, SampleContainerInternal& samples)
    {
        m_spatialSubdivBuilder.build(m_spatialSubdiv, bounds, samples, m_regionStorageContainer, m_spatialSubdivBuilderSettings, m_nCores);
        if (m_useStochasticNNLookUp)
        {
            m_regionKNNSearchTree.buildRegionSearchTree(m_regionStorageContainer);
            if (USE_PRECOMPUTED_NN)
            {
                m_regionKNNSearchTree.buildRegionNeighbours();
            }
        }
    }

    inline void updateSpatialStructure(SampleContainerInternal& samples)
    {
        m_spatialSubdivBuilder.updateTree(m_spatialSubdiv, samples, m_regionStorageContainer, m_spatialSubdivBuilderSettings, m_nCores);
        if (m_useStochasticNNLookUp)
        {
            m_regionKNNSearchTree.reset();
            m_regionKNNSearchTree.buildRegionSearchTree(m_regionStorageContainer);
            if (USE_PRECOMPUTED_NN)
            {
                m_regionKNNSearchTree.buildRegionNeighbours();
            }
        }
    }

    inline void fitRegions(SampleContainerInternal& samples)
    {
        size_t nGuidingRegions = m_regionStorageContainer.size();
#if defined( OPENPGL_SHOW_PRINT_OUTS)
        std::cout << "fitRegion: "<< (m_isSurface? "surface":"volume") << "\tnGuidingRegions = " << nGuidingRegions << std::endl;
#endif
#if defined(OPENPGL_USE_OMP_THREADING)
        #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
        for (size_t n=0; n < nGuidingRegions; n++)
#else
#ifndef USE_EMBREE_PARALLEL
        tbb::parallel_for( tbb::blocked_range<int>(0,nGuidingRegions), [&](tbb::blocked_range<int> r)
        {
        for (int n = r.begin(); n<r.end(); ++n)
#else
        embree::parallel_for(0,(int)nGuidingRegions, 1, [&] ( const embree::range<unsigned>& r ) {
        for (size_t n=r.begin(); n<r.end(); n++)
#endif
#endif
        {
            RegionStorageType &regionStorage = m_regionStorageContainer[n];
            openpgl::Point3 sampleMean = regionStorage.first.sampleStatistics.mean;
            if(regionStorage.second.size() > 0)
            {

#ifndef USE_PRESORT
                if (m_deterministic)
                {
                    std::sort(samples.begin() + regionStorage.second.m_begin, samples.begin() + regionStorage.second.m_end, SampleDataLess);
                }
#endif
				if(m_fitRegions) {
                	typename DirectionalDistributionFactory::FittingStatistics fittingStats;
                	m_distributionFactory.prepareSamples(samples.data() + regionStorage.second.m_begin, regionStorage.second.m_end - regionStorage.second.m_begin, regionStorage.first.sampleStatistics, m_distributionFactorySettings);
                	m_distributionFactory.fit(regionStorage.first.distribution, regionStorage.first.trainingStatistics, samples.data() + regionStorage.second.m_begin, regionStorage.second.m_end - regionStorage.second.m_begin, m_distributionFactorySettings, fittingStats);
					// TODO: we should move setting the pivot to the factory
                	regionStorage.first.distribution._pivotPosition = sampleMean;
                	regionStorage.first.valid = regionStorage.first.distribution.isValid();
#ifdef OPENPGL_DEBUG_MODE
                	if(!regionStorage.first.valid)
                    	std::cout << "!!!! " << (m_isSurface? "Surface":"Volume") << " regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
#endif
	                regionStorage.first.splitFlag = false;
				}
			}
            else
            {
                regionStorage.first.valid = false;
                regionStorage.first.splitFlag = false;
            }
			regionStorage.second.reset();
        }
#if !defined (OPENPGL_USE_OMP_THREADING)
        });
#endif
        m_initialized = true;
    }

    void updateRegions(SampleContainerInternal& samples)
    {
        size_t nGuidingRegions = m_regionStorageContainer.size();
#if defined( OPENPGL_SHOW_PRINT_OUTS)
        std::cout << "updateRegion: " << (m_isSurface? "surface":"volume") << "\tnGuidingRegions = " << nGuidingRegions << std::endl;
#endif
#if defined(OPENPGL_USE_OMP_THREADING)
        #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
        for (size_t n=0; n < nGuidingRegions; n++)
#else
#ifndef USE_EMBREE_PARALLEL
        tbb::parallel_for( tbb::blocked_range<int>(0,nGuidingRegions), [&](tbb::blocked_range<int> r)
        {
        for (int n = r.begin(); n<r.end(); ++n)
#else
        embree::parallel_for(0,(int)nGuidingRegions, 1, [&] ( const embree::range<unsigned>& r ) {
        for (size_t n=r.begin(); n<r.end(); n++)
#endif
#endif
        {
            RegionStorageType &regionStorage = m_regionStorageContainer[n];
            if (regionStorage.first.splitFlag)
            {
                regionStorage.first.trainingStatistics.decay(this->m_decayOnSpatialSplit);
                regionStorage.first.splitFlag = false;
            }

            openpgl::Point3 sampleMean = regionStorage.first.sampleStatistics.mean;
            if(regionStorage.second.size() > 0)
            {
#ifdef OPENPGL_DEBUG_MODE
                RegionType oldRegion = regionStorage.first;
#endif
#ifndef USE_PRESORT
				if (m_deterministic)
                {
                    std::sort(samples.begin() + regionStorage.second.m_begin, samples.begin() + regionStorage.second.m_end, SampleDataLess);
                }
#endif             
				if (m_fitRegions) {
	                // TODO: we should move applying the paralax comp to the Distribution to the factory
	                if(DirectionalDistribution::ParallaxCompensation == 1)
	                {
						regionStorage.first.trainingStatistics.sufficientStatistics.applyParallaxShift(regionStorage.first.distribution, regionStorage.first.distribution._pivotPosition - sampleMean);
	                    regionStorage.first.distribution.performRelativeParallaxShift(regionStorage.first.distribution._pivotPosition - sampleMean);
	                    OPENPGL_ASSERT(regionStorage.first.distribution.isValid());
	                    OPENPGL_ASSERT(regionStorage.first.trainingStatistics.sufficientStatistics.isValid());
	                }
	                typename DirectionalDistributionFactory::FittingStatistics fittingStats;
	                m_distributionFactory.prepareSamples(samples.data() + regionStorage.second.m_begin, regionStorage.second.m_end - regionStorage.second.m_begin, regionStorage.first.sampleStatistics, m_distributionFactorySettings);
	                m_distributionFactory.update(regionStorage.first.distribution, regionStorage.first.trainingStatistics, samples.data() + regionStorage.second.m_begin, regionStorage.second.m_end - regionStorage.second.m_begin, m_distributionFactorySettings, fittingStats);
	                //regionStorage.first.valid = regionStorage.first.distribution.isValid();
	                regionStorage.first.valid = regionStorage.first.isValid();
#ifdef OPENPGL_DEBUG_MODE
	                if(!regionStorage.first.valid)
	                {
	                    std::cout << "!!!! " << (m_isSurface? "Surface":"Volume") << " regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
	                    storeInvalidRegionData("regionBeforeUpdate_"+ std::string((m_isSurface? "surf":"vol")) + "_itr" + std::to_string(this->m_iteration) + "_region" + std::to_string(n)+".dump", oldRegion, dataPoints, m_distributionFactorySettings);
	                }
#endif
				}
            }
            else
            {
                RegionStorageType &regionStorage = m_regionStorageContainer[n];
                if (regionStorage.first.splitFlag)
                {
                    regionStorage.first.trainingStatistics.decay(this->m_decayOnSpatialSplit);
                    regionStorage.first.splitFlag = false;
                }
            }
            regionStorage.second.reset();
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
public:
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

private:

    bool m_isSurface{true};

    float m_decayOnSpatialSplit {0.25f};

    size_t m_iteration {0};
    size_t m_totalSPP  {0};

    size_t m_nCores {20};

    bool m_fitRegions {true};
    bool m_deterministic {false};

    bool m_isSceneBoundsSet{false};
    BBox m_sceneBounds;

    bool m_initialized {false};

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
    KNearestRegionsSearchTree<Vecsize> m_regionKNNSearchTree;

    SampleContainerInternal samples_;
};

}
