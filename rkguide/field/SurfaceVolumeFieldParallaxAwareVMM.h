// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "../data/SampleStatistics.h"
#include "../field/SurfaceVolumeField.h"

#include "../vmm/ParallaxAwareVMM.h"
#include "../vmm/AdaptiveSplitandMergeFactory.h"

#if defined (USE_TBB_THREADING)
#include <tbb/parallel_for.h>
#endif

namespace rkguide
{

template<int VecSize, int maxComponents, typename TSampleContainer>
struct SurfaceVolumeFieldParallaxAwareVMM: public SurfaceVolumeField<rkguide::Region<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>, typename AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> >::ASMStatistics >, TSampleContainer >
{

    using ParentField = SurfaceVolumeField<rkguide::Region<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>, typename AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> >::ASMStatistics >, TSampleContainer >;
    using DistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>>;
    using DistributionFactorySettings = typename DistributionFactory::ASMConfiguration;

    typedef ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> DistributionType;

    struct Settings //public ParentField::Settings
    {
        typename ParentField::Settings settings;
        DistributionFactorySettings distributionFactorySettings;
        bool useParallaxCompensation{true};

        std::string toString()const;
    };

    SurfaceVolumeFieldParallaxAwareVMM() = default;

    SurfaceVolumeFieldParallaxAwareVMM(const Settings &settings): ParentField(settings.settings)
    {
        m_distributionFactorySettings = settings.distributionFactorySettings;
        m_useParallaxCompensation = settings.useParallaxCompensation;
    }

    void fitRegions() override
    {
        fitRegions(this->m_regionStorageContainerSurface, true);
        fitRegions(this->m_regionStorageContainerVolume, false);
    }

    void updateRegions() override
    {
        updateRegions(this->m_regionStorageContainerSurface, true);
        updateRegions(this->m_regionStorageContainerVolume, false);
    }

    void storeInvalidRegionData(const std::string &fileName, const typename ParentField::RegionType &regionBeforeUpdate, const std::vector<DirectionalSampleData> &samples, const DistributionFactorySettings &factorySettings)
    {
        std::filebuf fbDump;
        fbDump.open (fileName,std::ios::out);
        std::ostream dumpStream(&fbDump);

        factorySettings.serialize(dumpStream);

        size_t numSamples = samples.size();
        dumpStream.write(reinterpret_cast<const char*>(&numSamples), sizeof(size_t));
        for (size_t i = 0; i < numSamples; i++)
        {
            dumpStream.write(reinterpret_cast<const char*>(&samples[i]), sizeof(rkguide::DirectionalSampleData));
        }

        regionBeforeUpdate.serialize(dumpStream);
        fbDump.close();
    }

    void loadInvalidRegionData(const std::string &fileName, typename ParentField::RegionType &regionBeforeUpdate, std::vector<DirectionalSampleData> &samples, DistributionFactorySettings &factorySettings)
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
            DirectionalSampleData dsd;
            dumpIStream.read(reinterpret_cast<char*>(&dsd), sizeof(rkguide::DirectionalSampleData));
            samples.push_back(dsd);
        }
        regionBeforeUpdate.deserialize(dumpIStream);
        fbDumpIn.close();
    }

    void reorientSample(rkguide::DirectionalSampleData &sample, const rkguide::Point3 &pivotPoint) const
    {

        if (std::isinf(sample.distance))
        {
            //std::cout << "inf sample" << std::endl;
            sample.position = pivotPoint;
            return;
        }
        else if (!(sample.distance > 0.0f))
        {
//            mitsuba::SLog(mitsuba::EWarn, "invalid sample distance %f", sample.distance);
            return;
        }

        const rkguide::Point3 originPosition = sample.position + sample.direction * sample.distance;
        const rkguide::Vector3 newDirection = originPosition - pivotPoint;
        const float newDistance = embree::length(newDirection);

        sample.position = pivotPoint;
        sample.distance = newDistance;
        sample.direction = newDirection / newDistance;
    }

private:

    void fitRegions(typename ParentField::RegionStorageContainerType &regionStorageContainer, const bool &isSurface)
    {
        size_t nGuidingRegions = regionStorageContainer.size();
        std::cout << "fitRegion: "<< (isSurface? "surface":"volume") << "\tnGuidingRegions = " << nGuidingRegions << std::endl;
#if defined(USE_OPENMP)
        #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
#endif

#if defined (USE_TBB_THREADING)
        tbb::parallel_for( tbb::blocked_range<int>(0,nGuidingRegions), [&](tbb::blocked_range<int> r)
        {
        for (int n = r.begin(); n<r.end(); ++n)
#else
        for (size_t n=0; n < nGuidingRegions; n++)
#endif
        {
            typename ParentField::RegionStorageType &regionStorage = regionStorageContainer[n];
            rkguide::Point3 sampleMean = regionStorage.first.sampleStatistics.mean;
            std::vector<rkguide::DirectionalSampleData> dataPoints;
            for (auto& sample : regionStorage.second)
            {
                if(m_useParallaxCompensation)
                {
                    reorientSample(sample, sampleMean);
                }
                dataPoints.push_back(sample);
            }
            if (dataPoints.size() > 0)
            {
                typename DistributionFactory::ASMFittingStatistics fittingStats;
                m_distributionFactory.fit(regionStorage.first.distribution, m_distributionFactorySettings.weightedEMCfg.initK, regionStorage.first.trainingStatistics, dataPoints.data(), dataPoints.size(), m_distributionFactorySettings, fittingStats);
                regionStorage.first.distribution._pivotPosition = sampleMean;
                regionStorage.first.valid = regionStorage.first.distribution.isValid();
                if(!regionStorage.first.valid)
                    std::cout << "!!!! " << (isSurface? "Surface":"Volume") << " regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
                regionStorage.first.splitFlag = false;
            }
            else
            {
                regionStorage.first.valid = false;
                regionStorage.first.splitFlag = false;
            }
        }
#if defined (USE_TBB_THREADING)
        });
#endif
    }

    void updateRegions(typename ParentField::RegionStorageContainerType &regionStorageContainer, const bool &isSurface)
    {
        size_t nGuidingRegions = regionStorageContainer.size();
        std::cout << "updateRegion: " << (isSurface? "surface":"volume") << "\tnGuidingRegions = " << nGuidingRegions << std::endl;
#if defined(USE_OPENMP)
        #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
#endif
#if defined (USE_TBB_THREADING)
        tbb::parallel_for( tbb::blocked_range<int>(0,nGuidingRegions), [&](tbb::blocked_range<int> r)
        {
        for (int n = r.begin(); n<r.end(); ++n)
#else
        for (size_t n=0; n < nGuidingRegions; n++)
#endif
        {
            typename ParentField::RegionStorageType &regionStorage = regionStorageContainer[n];
            if (regionStorage.first.splitFlag)
            {
                regionStorage.first.trainingStatistics.decay(this->m_decayOnSpatialSplit);
                regionStorage.first.splitFlag = false;
            }

            rkguide::Point3 sampleMean = regionStorage.first.sampleStatistics.mean;
            std::vector<rkguide::DirectionalSampleData> dataPoints;
            for (auto& sample : regionStorage.second)
            {
                if(m_useParallaxCompensation)
                {
                    reorientSample(sample, sampleMean);
                }
                dataPoints.push_back(sample);
            }
            if (dataPoints.size() > 0)
            {
                typename ParentField::RegionType oldRegion = regionStorage.first;
                if(m_useParallaxCompensation)
                {
                    regionStorage.first.trainingStatistics.sufficientStatistics.applyParallaxShift(regionStorage.first.distribution, regionStorage.first.distribution._pivotPosition - sampleMean);
                    regionStorage.first.distribution.performRelativeParallaxShift(regionStorage.first.distribution._pivotPosition - sampleMean);
                    RKGUIDE_ASSERT(regionStorage.first.distribution.isValid());
                    RKGUIDE_ASSERT(regionStorage.first.trainingStatistics.sufficientStatistics.isValid());
                }
                typename DistributionFactory::ASMFittingStatistics fittingStats;
                m_distributionFactory.update(regionStorage.first.distribution, regionStorage.first.trainingStatistics, dataPoints.data(), dataPoints.size(), m_distributionFactorySettings, fittingStats);
                //regionStorage.first.valid = regionStorage.first.distribution.isValid();
                regionStorage.first.valid = regionStorage.first.isValid();
                if(!regionStorage.first.valid)
                {
                    std::cout << "!!!! " << (isSurface? "Surface":"Volume") << " regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
                    storeInvalidRegionData("regionBeforeUpdate_"+ std::string((isSurface? "surf":"vol")) + "_itr" + std::to_string(this->m_iteration) + "_region" + std::to_string(n)+".dump", oldRegion, dataPoints, m_distributionFactorySettings);
                }
            }
            else
            {
                regionStorage.first.valid = false;
                regionStorage.first.splitFlag = false;
            }
        }
#if defined (USE_TBB_THREADING)
        });
#endif
    }

public:
    void serialize(std::ostream& stream) const;

    void deserialize(std::istream& stream);

private:
    bool m_useParallaxCompensation {true};

    DistributionFactory m_distributionFactory;
    DistributionFactorySettings m_distributionFactorySettings;

};

template<int VecSize, int maxComponents, typename TSampleContainer>
inline std::string SurfaceVolumeFieldParallaxAwareVMM<VecSize, maxComponents, TSampleContainer>::Settings::toString() const
{
    std::stringstream ss;
    ss << "SurfaceVolumeFieldParallaxAwareVMM::Settings:" << std::endl;
    ss << "  settings: " << settings.toString() << std::endl;
    ss << "  distributionFactorySettings: " << distributionFactorySettings.toString() << std::endl;
    ss << "  useParallaxCompensation: " << useParallaxCompensation << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents, typename TSampleContainer>
inline void SurfaceVolumeFieldParallaxAwareVMM<VecSize, maxComponents, TSampleContainer>::serialize(std::ostream& stream) const
{
    ParentField::serialize(stream);
    stream.write(reinterpret_cast<const char*>(&m_useParallaxCompensation), sizeof(bool));

    //m_distributionFactory.serialize(stream);
    m_distributionFactorySettings.serialize(stream);
}

template<int VecSize, int maxComponents, typename TSampleContainer>
inline void SurfaceVolumeFieldParallaxAwareVMM<VecSize, maxComponents, TSampleContainer>::deserialize(std::istream& stream)
{
    ParentField::deserialize(stream);
    stream.read(reinterpret_cast<char*>(&m_useParallaxCompensation), sizeof(bool));

    //m_distributionFactory.deserialize(stream);
    m_distributionFactorySettings.deserialize(stream);
}

}