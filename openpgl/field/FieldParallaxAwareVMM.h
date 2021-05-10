// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "../data/SampleStatistics.h"
#include "../field/Field.h"

#include "../directional/vmm/ParallaxAwareVMM.h"
#include "../directional/vmm/AdaptiveSplitandMergeFactory.h"

#if !defined (OPENPGL_USE_OMP_THREADING)
#include <tbb/parallel_for.h>
#endif

namespace openpgl
{

template<int VecSize, int maxComponents, typename TSampleContainer>
struct FieldParallaxAwareVMM: public Field<openpgl::Region<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>, typename AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> >::ASMStatistics >, TSampleContainer >
{

    using ParentField = Field<openpgl::Region<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>, typename AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> >::ASMStatistics >, TSampleContainer >;
    using DistributionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>>;
    using DistributionFactorySettings = typename DistributionFactory::ASMConfiguration;

    typedef ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents> DistributionType;

    struct Settings
    {
        typename ParentField::Settings settings;
        DistributionFactorySettings distributionFactorySettings;
        bool useParallaxCompensation{true};

        std::string toString() const;
    };

    FieldParallaxAwareVMM() = default;

    FieldParallaxAwareVMM(const Settings &settings): ParentField(settings.settings)
    {
        m_distributionFactorySettings = settings.distributionFactorySettings;
        m_useParallaxCompensation = settings.useParallaxCompensation;
    }

    void fitRegions() override
    {
      size_t nGuidingRegions = this->m_regionStorageContainer.size();
      std::cout << "Begin region fitting: nRegions =  " << nGuidingRegions << std::endl;
#if defined(OPENPGL_USE_OMP_THREADING)
      #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
      for (size_t n=0; n < nGuidingRegions; n++)
#else
      tbb::parallel_for( tbb::blocked_range<int>(0,nGuidingRegions), [&](tbb::blocked_range<int> r)
      {
      for (int n = r.begin(); n<r.end(); ++n)      
#endif
      {
        typename ParentField::RegionStorageType &regionStorage = this->m_regionStorageContainer[n];
        openpgl::Point3 sampleMean = regionStorage.first.sampleStatistics.mean;
        std::vector<openpgl::SampleData> dataPoints;
        for (auto& sample : regionStorage.second)
        {
            if(m_useParallaxCompensation)
            {
                reorientSample(sample, sampleMean);
            }
            dataPoints.push_back(sample);
        }
        typename DistributionFactory::ASMFittingStatistics fittingStats;
        m_distributionFactory.fit(regionStorage.first.distribution, m_distributionFactorySettings.weightedEMCfg.initK, regionStorage.first.trainingStatistics, dataPoints.data(), dataPoints.size(), m_distributionFactorySettings, fittingStats);
        regionStorage.first.distribution._pivotPosition = sampleMean;
        regionStorage.first.valid = regionStorage.first.distribution.isValid();
        if(!regionStorage.first.valid)
            std::cout << "!!!! regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
        regionStorage.first.splitFlag = false;
      }
#if !defined (OPENPGL_USE_OMP_THREADING)
      });
#endif
    }

    void updateRegions() override
    {
      size_t nGuidingRegions = this->m_regionStorageContainer.size();
      std::cout << "Begin region updating: nRegions =  " << nGuidingRegions << std::endl;
#if defined(OPENPGL_USE_OMP_THREADING)
      #pragma omp parallel for num_threads(this->m_nCores) schedule(dynamic)
      for (size_t n=0; n < nGuidingRegions; n++)
#else
      tbb::parallel_for( tbb::blocked_range<int>(0,nGuidingRegions), [&](tbb::blocked_range<int> r)
      {
      for (int n = r.begin(); n<r.end(); ++n) 
#endif
      {
        typename ParentField::RegionStorageType &regionStorage = this->m_regionStorageContainer[n];
        if (regionStorage.first.splitFlag)
        {
            //m_factory.onSpatialSplit(regionStorage.first.distribution, regionStorage.first.trainingStatistics);
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
            dataPoints.push_back(sample);
        }

        if(m_useParallaxCompensation)
        {
            regionStorage.first.trainingStatistics.sufficientStatistics.applyParallaxShift(regionStorage.first.distribution, regionStorage.first.distribution._pivotPosition - sampleMean);
            regionStorage.first.distribution.performRelativeParallaxShift(regionStorage.first.distribution._pivotPosition - sampleMean);
            OPENPGL_ASSERT(regionStorage.first.distribution.isValid());
            OPENPGL_ASSERT(regionStorage.first.trainingStatistics.sufficientStatistics.isValid());
        }
        typename DistributionFactory::ASMFittingStatistics fittingStats;
        m_distributionFactory.update(regionStorage.first.distribution, regionStorage.first.trainingStatistics, dataPoints.data(), dataPoints.size(), m_distributionFactorySettings, fittingStats);
        regionStorage.first.valid = regionStorage.first.distribution.isValid();
        if(!regionStorage.first.valid)
            std::cout << "!!!! regionStorage.first.valid !!! " << regionStorage.first.distribution.toString() << std::endl;
      }
#if !defined (OPENPGL_USE_OMP_THREADING)
      });
#endif
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


    void serialize(std::ostream& stream) const;

    void deserialize(std::istream& stream);

    std::string toString() const;

private:
    bool m_useParallaxCompensation {true};

    DistributionFactory m_distributionFactory;
    DistributionFactorySettings m_distributionFactorySettings;

};

template<int VecSize, int maxComponents, typename TSampleContainer>
inline std::string FieldParallaxAwareVMM<VecSize, maxComponents, TSampleContainer>::toString() const
{
    std::stringstream ss;
    ss << "FieldParallaxAwareVMM:" << std::endl;
    ss << "  useParallaxCompensation: " << m_useParallaxCompensation << std::endl;
    ss << "  distributionFactory: " << m_distributionFactory.toString() << std::endl;
    ss << "  distributionFactorySettings: " << m_distributionFactorySettings.toString() << std::endl;
    ss << ParentField::toString() << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents, typename TSampleContainer>
inline std::string FieldParallaxAwareVMM<VecSize, maxComponents, TSampleContainer>::Settings::toString() const
{
    std::stringstream ss;
    ss << "FieldParallaxAwareVMM::Settings:" << std::endl;
    ss << "  settings: " << settings.toString() << std::endl;
    ss << "  distributionFactorySettings: " << distributionFactorySettings.toString() << std::endl;
    ss << "  useParallaxCompensation: " << useParallaxCompensation << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents, typename TSampleContainer>
inline void FieldParallaxAwareVMM<VecSize, maxComponents, TSampleContainer>::serialize(std::ostream& stream) const
{
    ParentField::serialize(stream);
    stream.write(reinterpret_cast<const char*>(&m_useParallaxCompensation), sizeof(bool));
    m_distributionFactorySettings.serialize(stream);
}

template<int VecSize, int maxComponents, typename TSampleContainer>
inline void FieldParallaxAwareVMM<VecSize, maxComponents, TSampleContainer>::deserialize(std::istream& stream)
{
    ParentField::deserialize(stream);
    stream.read(reinterpret_cast<char*>(&m_useParallaxCompensation), sizeof(bool));
    m_distributionFactorySettings.deserialize(stream);
}


}