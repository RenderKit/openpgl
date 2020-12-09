// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "../data/SampleStatistics.h"
#include "../field/SurfaceVolumeField.h"

#include "../vmm/ParallaxAwareVMM.h"
#include "../vmm/AdaptiveSplitandMergeFactory.h"

namespace rkguide
{

template<int VecSize, int maxComponents, typename TSampleContainer>
struct SurfaceVolumeFieldParallaxAwareVMM: public SurfaceVolumeField<rkguide::Region<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>, typename AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> >::ASMStatistics >, TSampleContainer >
{

    using ParentField = SurfaceVolumeField<rkguide::Region<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>, typename AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> >::ASMStatistics >, TSampleContainer >;
    using DistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>>;
    using DistributionFactorySettings = typename DistributionFactory::ASMConfiguration;

    typedef ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> DistributionType;

    struct Settings: public ParentField::Settings
    {
        DistributionFactorySettings distributionFactorySettings;
        bool useParallaxCompensation{true};
    };

    SurfaceVolumeFieldParallaxAwareVMM() = default;

    SurfaceVolumeFieldParallaxAwareVMM(const Settings &settings): ParentField(settings)
    {
        m_distributionFactorySettings = settings.distributionFactorySettings;
        m_useParallaxCompensation = settings.useParallaxCompensation;
    }

    void fitRegions() override
    {
      //mitsuba::ref<mitsuba::Timer> fittingTimer = new mitsuba::Timer();
      size_t nSurfaceGudiginRegions = this->m_regionStorageContainerSurface.size();
      std::cout << "fitRegion: nSurfaceGudiginRegions = " << nSurfaceGudiginRegions << std::endl;
//      mitsuba::SLog(mitsuba::EInfo, "Begin region fitting: nRegions = %d", nGudiginRegions);
#if defined(MTS_OPENMP)
			//SLog(EInfo, "Fit Mixtures: nCores= %d", m_nCores);
      #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
#endif
      for (size_t n=0; n < nSurfaceGudiginRegions; n++)
      {
        typename ParentField::RegionStorageType &regionStorage = this->m_regionStorageContainerSurface[n];
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
                std::cout << "!!!! Surface regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
            regionStorage.first.splitFlag = false;
        }
        else
        {
            regionStorage.first.valid = false;
            regionStorage.first.splitFlag = false;
        }
      }
//      mitsuba::SLog(mitsuba::EInfo, "Region fitting time: %s", timeString(fittingTimer->getSeconds(), true).c_str());

      size_t nVolumeGudiginRegions = this->m_regionStorageContainerVolume.size();
      std::cout << "fitRegion: nVolumeGudiginRegions = " << nVolumeGudiginRegions << std::endl;
//      mitsuba::SLog(mitsuba::EInfo, "Begin region fitting: nRegions = %d", nGudiginRegions);
#if defined(MTS_OPENMP)
			//SLog(EInfo, "Fit Mixtures: nCores= %d", m_nCores);
      #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
#endif
      for (size_t n=0; n < nVolumeGudiginRegions; n++)
      {
        typename ParentField::RegionStorageType &regionStorage = this->m_regionStorageContainerVolume[n];
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
                std::cout << "!!!! Volume regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
            regionStorage.first.splitFlag = false;
        }
        else
        {
            regionStorage.first.valid = false;
            regionStorage.first.splitFlag = false;
        }
      }

    }

    void updateRegions() override
    {
      //mitsuba::ref<mitsuba::Timer> fittingTimer = new mitsuba::Timer();
      size_t nSurfaceGudiginRegions = this->m_regionStorageContainerSurface.size();
      std::cout << "updateRegion: nSurfaceGudiginRegions = " << nSurfaceGudiginRegions << std::endl;
//      mitsuba::SLog(mitsuba::EInfo, "Begin region fitting: nRegions = %d", nGudiginRegions);
#if defined(MTS_OPENMP)
//			mitsuba::SLog(mitsuba::EInfo, "Fit Mixtures: nCores= %d", this->m_nCores);
      #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
#endif
      for (size_t n=0; n < nSurfaceGudiginRegions; n++)
      {
        typename ParentField::RegionStorageType &regionStorage = this->m_regionStorageContainerSurface[n];
//        mitsuba::SLog(mitsuba::EInfo, "Region[%d] = nSamples %d", n, regionStorage.second.size());
        if (regionStorage.first.splitFlag)
        {
            //m_factory.onSpatialSplit(regionStorage.first.distribution, regionStorage.first.trainingStatistics);
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
            if(m_useParallaxCompensation)
            {
                regionStorage.first.trainingStatistics.sufficientStatistics.applyParallaxShift(regionStorage.first.distribution, regionStorage.first.distribution._pivotPosition - sampleMean);
                regionStorage.first.distribution.performRelativeParallaxShift(regionStorage.first.distribution._pivotPosition - sampleMean);
                RKGUIDE_ASSERT(regionStorage.first.distribution.isValid());
                RKGUIDE_ASSERT(regionStorage.first.trainingStatistics.sufficientStatistics.isValid());
            }
            typename DistributionFactory::ASMFittingStatistics fittingStats;
            m_distributionFactory.update(regionStorage.first.distribution, regionStorage.first.trainingStatistics, dataPoints.data(), dataPoints.size(), m_distributionFactorySettings, fittingStats);
            regionStorage.first.valid = regionStorage.first.distribution.isValid();
            if(!regionStorage.first.valid)
                std::cout << "!!!! Surface regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
            
        }
        else
        {
            regionStorage.first.valid = false;
            regionStorage.first.splitFlag = false;
        }
      }
//      mitsuba::SLog(mitsuba::EInfo, "Region fitting time: %s", timeString(fittingTimer->getSeconds(), true).c_str());
      size_t nVolumeGudiginRegions = this->m_regionStorageContainerVolume.size();
      std::cout << "updateRegion: nVolumeGudiginRegions = " << nVolumeGudiginRegions << std::endl;
//      mitsuba::SLog(mitsuba::EInfo, "Begin region fitting: nRegions = %d", nGudiginRegions);
#if defined(MTS_OPENMP)
//			mitsuba::SLog(mitsuba::EInfo, "Fit Mixtures: nCores= %d", this->m_nCores);
      #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
#endif
      for (size_t n=0; n < nVolumeGudiginRegions; n++)
      {
        typename ParentField::RegionStorageType &regionStorage = this->m_regionStorageContainerVolume[n];
//        mitsuba::SLog(mitsuba::EInfo, "Region[%d] = nSamples %d", n, regionStorage.second.size());
        if (regionStorage.first.splitFlag)
        {
            //m_factory.onSpatialSplit(regionStorage.first.distribution, regionStorage.first.trainingStatistics);
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
            if(m_useParallaxCompensation)
            {
                regionStorage.first.trainingStatistics.sufficientStatistics.applyParallaxShift(regionStorage.first.distribution, regionStorage.first.distribution._pivotPosition - sampleMean);
                regionStorage.first.distribution.performRelativeParallaxShift(regionStorage.first.distribution._pivotPosition - sampleMean);
                RKGUIDE_ASSERT(regionStorage.first.distribution.isValid());
                RKGUIDE_ASSERT(regionStorage.first.trainingStatistics.sufficientStatistics.isValid());
            }
            typename DistributionFactory::ASMFittingStatistics fittingStats;
            m_distributionFactory.update(regionStorage.first.distribution, regionStorage.first.trainingStatistics, dataPoints.data(), dataPoints.size(), m_distributionFactorySettings, fittingStats);
            regionStorage.first.valid = regionStorage.first.distribution.isValid();
            if(!regionStorage.first.valid)
                std::cout << "!!!! Volume regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
            
        }
        else
        {
            regionStorage.first.valid = false;
            regionStorage.first.splitFlag = false;
        }
      }
    }

    //const RegionType *getGuidingRegion( const rkguide::Point3 &p, rkguide::Sampler *sampler) const override;

    //template<typename TSampleContainer>
    //void buildField(const BBox &bounds, TSampleContainer& samples) override;

    //void updateField(TSampleContainer& samples) override;

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
    bool m_useParallaxCompensation {true};

    DistributionFactory m_distributionFactory;
    DistributionFactorySettings m_distributionFactorySettings;

};

}