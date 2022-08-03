// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "WeightedEMVMMFactory.h"

#define USE_HARMONIC_MEAN

//using namespace embree;

namespace openpgl
{

template<class TVMMDistribution>
struct WeightedEMParallaxAwareVonMisesFisherFactory: public WeightedEMVonMisesFisherFactory< TVMMDistribution>
{

    public:

    using VMM = TVMMDistribution;
    using WEMVMMFactory = WeightedEMVonMisesFisherFactory< TVMMDistribution>;
    using FittingStatistics = typename WEMVMMFactory::FittingStatistics;
    using PartialFittingMask = typename WEMVMMFactory:: PartialFittingMask;

    struct Configuration: public WEMVMMFactory::Configuration
    {
        bool parallaxCompensation {true};

        void init();

        void serialize(std::ostream& stream) const;

        void deserialize(std::istream& stream);

        std::string toString() const;
    };

    struct SufficientStatisitcs//: public WEMVMMFactory::SufficientStatisitcs
    {

        //FittingStatistics
        public:
        typename WEMVMMFactory::SufficientStatisitcs wEMSufficientStatisitcs;

        embree::vfloat<VMM::VectorSize> sumOfDistanceWeightes[VMM::NumVectors];

        SufficientStatisitcs() = default;

        void serialize(std::ostream& stream) const;

        void deserialize(std::istream& stream);

        void clear(size_t _numComponents);

        void clearAll();

        void maskedReplace(const PartialFittingMask &mask, const SufficientStatisitcs &stats);

        void applyParallaxShift(const VMM &vmm, const Vector3 shift);

        inline void setNumComponents( const size_t &numComponents)
        {
            wEMSufficientStatisitcs.setNumComponents(numComponents);
        }

        inline size_t getNumComponents( ) const
        {
            return wEMSufficientStatisitcs.getNumComponents();
        }

        inline float getNumSamples() const
        {
            return wEMSufficientStatisitcs.getNumSamples();
        }

        inline float getSumWeights() const
        {
            return wEMSufficientStatisitcs.getSumWeights();
        }

        //void normalize( const float &_numSamples ) override;

        void mergeComponentStats(const size_t &idx0, const size_t &idx1);

        void splitComponentsStats(const size_t &idx0, const size_t &idx1,
                const Vector3 &meanDirection0, const Vector3 &meanDirection1,
                const float &meanCosine0, const float &meanCosine1);

        //void swapComponentStats(const size_t &idx0, const size_t &idx1) override;

        void decay(const float &alpha);

        std::string toString() const;

        bool isValid() const;

    };


public:
    void prepareSamples(SampleData* samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const;
    
    void fitMixture(VMM &vmm, SufficientStatisitcs &stats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void updateMixture(VMM &vmm, SufficientStatisitcs &previousStats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void partialUpdateMixture(VMM &vmm, PartialFittingMask &mask, SufficientStatisitcs &previousStats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    //VMM VMMfromSufficientStatisitcs(const SufficientStatisitcs &suffStats, const Configuration &cfg) const override;
    std::string toString() const
    {
        return "WeightedEMParallaxAwareVonMisesFisherFactory";
    };

//private:

    void initComponentDistances (VMM &vmm, SufficientStatisitcs &sufficientStats, const SampleData* samples, const size_t numSamples) const;

    void updateComponentDistances (VMM &vmm, SufficientStatisitcs &sufficientStats, const SampleData* samples, const size_t numSamples) const;

private:
    void reprojectSample(openpgl::SampleData &sample, const openpgl::Point3 &pivotPoint, const float minDistance) const;

};

////////////////////////////////////////////////////////////
/////////            SufficientStatisitcs
////////////////////////////////////////////////////////////

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::applyParallaxShift(const VMM &vmm, const Vector3 shift)
{
    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    //const int rem = vmm._numComponents % VMM::VectorSize;

    for(uint32_t k=0;k<cnt;k++)
    {
        embree::Vec3< embree::vfloat<VMM::VectorSize> > suffDirections = wEMSufficientStatisitcs.sumOfWeightedDirections[k];
        embree::vfloat<VMM::VectorSize> suffMeanCosines = embree::length(suffDirections);
        suffDirections /= suffMeanCosines;
        suffDirections *= vmm._distances[k];
        suffDirections += embree::Vec3< embree::vfloat<VMM::VectorSize> >(shift);

        suffDirections /= embree::length(suffDirections);
        suffDirections *= suffMeanCosines;
        suffDirections = select(suffMeanCosines > 0.0f, suffDirections, wEMSufficientStatisitcs.sumOfWeightedDirections[k]);
        wEMSufficientStatisitcs.sumOfWeightedDirections[k] = select(vmm._distances[k] > 0.0f, suffDirections, wEMSufficientStatisitcs.sumOfWeightedDirections[k]);
    }
}

template<class TVMMDistribution>
bool WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::isValid() const
{
    bool valid = true;
    valid = wEMSufficientStatisitcs.isValid();


    for(size_t k = 0; k < wEMSufficientStatisitcs.numComponents; k++)
    {
        const div_t tmpK = div( k, VMM::VectorSize );
        valid = valid && embree::isvalid(sumOfDistanceWeightes[tmpK.quot][tmpK.rem]);
        valid = valid && sumOfDistanceWeightes[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);
    }

    for(size_t k = wEMSufficientStatisitcs.numComponents; k < VMM::MaxComponents; k++)
    {
        const div_t tmpK = div( k, VMM::VectorSize );
        valid = valid && embree::isvalid(sumOfDistanceWeightes[tmpK.quot][tmpK.rem]);
        valid = valid && sumOfDistanceWeightes[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);
    }

   return valid;
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::serialize(std::ostream& stream) const
{
    wEMSufficientStatisitcs.serialize(stream);
    for(uint32_t k=0;k<VMM::NumVectors;k++)
    {
        stream.write(reinterpret_cast<const char*>(&sumOfDistanceWeightes[k]), sizeof(embree::vfloat<VMM::VectorSize>));
    }
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::deserialize(std::istream& stream)
{
    wEMSufficientStatisitcs.deserialize(stream);
    for(uint32_t k=0;k<VMM::NumVectors;k++)
    {
        stream.read(reinterpret_cast<char*>(&sumOfDistanceWeightes[k]), sizeof(embree::vfloat<VMM::VectorSize>));
    }
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::clear(size_t _numComponents)
{
    wEMSufficientStatisitcs.clear(_numComponents);

    const embree::vfloat<VMM::VectorSize> zeros(0.0f);
    const embree::vfloat<VMM::VectorSize> infs(std::numeric_limits<float>::infinity());
    const int cnt = (_numComponents+VMM::VectorSize-1) / VMM::VectorSize;

    for(int k = 0; k < cnt;k++)
    {
        sumOfDistanceWeightes[k] = zeros;
    }
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::clearAll()
{
    clear(VMM::MaxComponents);
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::decay(const float &alpha)
{
    wEMSufficientStatisitcs.decay(alpha);

    for(int k = 0; k < VMM::NumVectors; k++)
    {
        sumOfDistanceWeightes[k] *= alpha;
    }
}

template<class TVMMDistribution>
std::string WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::toString() const
{
    std::stringstream ss;
    ss << "SufficientStatisitcs:" << std::endl;
    ss << "\tsumWeights = " << wEMSufficientStatisitcs.sumWeights << std::endl;
    ss << "\tnumSamples = " << wEMSufficientStatisitcs.numSamples << std::endl;
    ss << "\toverallNumSamples = " << wEMSufficientStatisitcs.overallNumSamples << std::endl;
    ss << "\tnumComponents = " << wEMSufficientStatisitcs.numComponents << std::endl;
    ss << "\tisNormalized = " << wEMSufficientStatisitcs.normalized << std::endl;
    //for (size_t k = 0; k < wEMSufficientStatisitcs.numComponents ; k++)
    for (size_t k = 0; k < VMM::MaxComponents ; k++)
    {
        int i = k / VMM::VectorSize;
        int j = k % VMM::VectorSize;
        ss  << "\tstat["<< k <<"]:" << "\tsumWeightedStats = " << wEMSufficientStatisitcs.sumOfWeightedStats[i][j]
            << "\tsumWeightedDirections = [" << wEMSufficientStatisitcs.sumOfWeightedDirections[i].x[j] << ",\t"
            << wEMSufficientStatisitcs.sumOfWeightedDirections[i].y[j] << ",\t" << wEMSufficientStatisitcs.sumOfWeightedDirections[i].z[j] << "]"
            << "\tsumWeightedDistanceWeights = " << sumOfDistanceWeightes[i][j]
            << std::endl;
    }
    return ss.str();
}


template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::maskedReplace(const PartialFittingMask &mask, const SufficientStatisitcs &stats)
{
    wEMSufficientStatisitcs.maskedReplace(mask, stats.wEMSufficientStatisitcs);

    for (size_t k = 0; k < ( (VMM::MaxComponents + (VMM::VectorSize -1)) / VMM::VectorSize); k++)
    {
        sumOfDistanceWeightes[k] = select(mask.mask[k], stats.sumOfDistanceWeightes[k], sumOfDistanceWeightes[k]);
    }

}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::mergeComponentStats(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div( idx0, VMM::VectorSize);
    const div_t tmpIdx1 = div( idx1, VMM::VectorSize);
    const div_t tmpIdx2 = div( wEMSufficientStatisitcs.numComponents-1, VMM::VectorSize);

    sumOfDistanceWeightes[tmpIdx0.quot][tmpIdx0.rem] += sumOfDistanceWeightes[tmpIdx1.quot][tmpIdx1.rem];
    sumOfDistanceWeightes[tmpIdx1.quot][tmpIdx1.rem] = sumOfDistanceWeightes[tmpIdx2.quot][tmpIdx2.rem];
    sumOfDistanceWeightes[tmpIdx2.quot][tmpIdx2.rem] = 0.0f;

    wEMSufficientStatisitcs.mergeComponentStats(idx0, idx1);
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::splitComponentsStats(const size_t &idx0, const size_t &idx1,
                const Vector3 &meanDirection0, const Vector3 &meanDirection1,
                const float &meanCosine0, const float &meanCosine1)
{
    wEMSufficientStatisitcs.splitComponentsStats(idx0, idx1, meanDirection0, meanDirection1, meanCosine0, meanCosine1);

    const div_t tmpIdx0 = div( idx0, VMM::VectorSize);
    const div_t tmpIdx1 = div( idx1, VMM::VectorSize);
    float tmp = sumOfDistanceWeightes[tmpIdx0.quot][tmpIdx0.rem] * 0.5f;
    sumOfDistanceWeightes[tmpIdx0.quot][tmpIdx0.rem] = tmp;
    sumOfDistanceWeightes[tmpIdx1.quot][tmpIdx1.rem] = tmp;
}

////////////////////////////////////////////////////////////
/////////            WeightedEMParallaxAwareVonMisesFisherFactory
////////////////////////////////////////////////////////////


template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::reprojectSample(openpgl::SampleData &sample, const openpgl::Point3 &pivotPoint, const float minDistance) const
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

    const float distance = fmaxf(minDistance, sample.distance);
    const openpgl::Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
    const openpgl::Vector3 sampleDirection(sample.direction.x, sample.direction.y, sample.direction.z);
    const openpgl::Point3 originPosition = samplePosition + sampleDirection * distance;
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

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::prepareSamples(SampleData* samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const
{
    if(cfg.parallaxCompensation) 
    {
        openpgl::Vector3 sampleVariance = sampleStatistics.getVaraince();
        float minDistance = length(sampleVariance);
        minDistance = 3.f * 3.f * sqrt(minDistance);
        for (size_t n = 0; n < numSamples; n++)
        {
            reprojectSample(samples[n], sampleStatistics.mean, minDistance);
        }
    }
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::fitMixture(VMM &vmm, SufficientStatisitcs &stats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    WeightedEMVonMisesFisherFactory< TVMMDistribution>::fitMixture( vmm, stats.wEMSufficientStatisitcs, samples, numSamples, cfg, fitStats);
    /*
    // TODO: implement an init function
    const vfloat<VMM::VectorSize> zeros(0.0f);
    for (size_t k = 0; k < VMM::NumVectors; k++)
    {
        vmm._distances[k] = zeros;
        stats.sumOfDistanceWeightes[k] = zeros;
    }
    updateComponentDistances(vmm, stats, samples, numSamples);
    */
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::updateMixture(VMM &vmm, SufficientStatisitcs &previousStats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    WeightedEMVonMisesFisherFactory< TVMMDistribution>::updateMixture( vmm, previousStats.wEMSufficientStatisitcs, samples, numSamples, cfg, fitStats);
    //updateComponentDistances(vmm, stats, samples, numSamples);
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::partialUpdateMixture(VMM &vmm, PartialFittingMask &mask, SufficientStatisitcs &previousStats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    WeightedEMVonMisesFisherFactory< TVMMDistribution>::partialUpdateMixture( vmm, mask, previousStats.wEMSufficientStatisitcs, samples, numSamples, cfg, fitStats);
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::initComponentDistances (VMM &vmm, SufficientStatisitcs &sufficientStats, const SampleData* samples, const size_t numSamples) const
{
    OPENPGL_ASSERT(vmm.getNumComponents() == sufficientStats.getNumComponents());

    embree::vfloat<VMM::VectorSize> batchDistances[VMM::NumVectors];
    embree::vfloat<VMM::VectorSize> batchSumWeights[VMM::NumVectors];

    const embree::vfloat<VMM::VectorSize> zeros(0.0f);

    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        batchDistances[k] = zeros;
        batchSumWeights[k] = zeros;
    }

    typename VMM::SoftAssignment softAssign;
    float sampleDistance;
    embree::vfloat<VMM::VectorSize> weights;
    for (size_t n = 0; n < numSamples; n++)
    {
#ifdef USE_HARMONIC_MEAN
        sampleDistance = embree::rcp(samples[n].distance);
#else
        sampleDistance = samples[n].distance;
#endif
        const Vector3 sampleDirection(samples[n].direction.x, samples[n].direction.y, samples[n].direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                weights = samples[n].weight * softAssign.assignments[k] * ( (softAssign.assignments[k] * softAssign.pdf) / vmm._weights[k]);
                batchDistances[k] += weights * sampleDistance;
                batchSumWeights[k] += weights;
            }
        }
    }

    for (size_t k = 0; k < cnt; k++)
    {
#ifdef USE_HARMONIC_MEAN
        sufficientStats.sumOfDistanceWeightes[k] = batchSumWeights[k];
        vmm._distances[k] = sufficientStats.sumOfDistanceWeightes[k] / batchDistances[k];
#else
        sufficientStats.sumOfDistanceWeightes[k] = batchSumWeights[k];
        vmm._distances[k] = batchDistances[k] / sufficientStats.sumOfDistanceWeightes[k];
#endif
    }

    if ( rem > 0 )
    {
        for ( size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._distances[cnt-1][i] = 0.0f;
            sufficientStats.sumOfDistanceWeightes[cnt-1][i] = 0.0f;
        }
    }
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::updateComponentDistances (VMM &vmm, SufficientStatisitcs &sufficientStats, const SampleData* samples, const size_t numSamples) const
{
    OPENPGL_ASSERT(vmm.getNumComponents() == sufficientStats.getNumComponents());

    embree::vfloat<VMM::VectorSize> batchDistances[VMM::NumVectors];
    embree::vfloat<VMM::VectorSize> batchSumWeights[VMM::NumVectors];

    const embree::vfloat<VMM::VectorSize> zeros(0.0f);
    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        batchDistances[k] = zeros;
        batchSumWeights[k] = zeros;
    }

    typename VMM::SoftAssignment softAssign;
    float sampleDistance;
    embree::vfloat<VMM::VectorSize> weights;
    for (size_t n = 0; n < numSamples; n++)
    {
        OPENPGL_ASSERT(samples[n].distance > 0);
        OPENPGL_ASSERT(embree::isvalid(samples[n].distance));
#ifdef USE_HARMONIC_MEAN
        sampleDistance = embree::rcp(samples[n].distance);
#else
        sampleDistance = samples[n].distance;
#endif
        const Vector3 sampleDirection(samples[n].direction.x, samples[n].direction.y, samples[n].direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            for (size_t k = 0; k < cnt; k++)
            {
                weights = samples[n].weight * softAssign.assignments[k] * ( (softAssign.assignments[k] * softAssign.pdf) / vmm._weights[k]);
                batchDistances[k] += weights * sampleDistance;
                batchSumWeights[k] += weights;
            }
        }
    }

    for (size_t k = 0; k < cnt; k++)
    {
#ifdef USE_HARMONIC_MEAN
        //embree::vfloat<VMM::VectorSize> sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] / vmm._distances[k]) + batchDistances[k];
        embree::vfloat<VMM::VectorSize> sumInverseDistances = batchDistances[k];
        sumInverseDistances += select( vmm._distances[k] > 0.0f , (sufficientStats.sumOfDistanceWeightes[k] / vmm._distances[k]) , embree::vfloat<VMM::VectorSize>(0.0f));
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = sufficientStats.sumOfDistanceWeightes[k] / sumInverseDistances;
#else
        const embree::vfloat<VMM::VectorSize> sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] * vmm._distances[k]) + batchDistances[k];
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = sumInverseDistances / sufficientStats.sumOfDistanceWeightes[k];
#endif
    }

    if ( rem > 0 )
    {
        for ( size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._distances[cnt-1][i] = 0.0f;
            sufficientStats.sumOfDistanceWeightes[cnt-1][i] = 0.0f;
        }
    }

}
/*
template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::partialUpdateComponentDistances (VMM &vmm, const PartialFittingMask &mask, SufficientStatisitcs &sufficientStats, const SampleData* samples, const size_t numSamples) const
{
    vfloat<VMM::VectorSize> batchDistances[VMM::NumVectors];
    vfloat<VMM::VectorSize> batchSumWeights[VMM::NumVectors];

    const vfloat<VMM::VectorSize> zeros(0.0f);

    for (size_t k = 0; k < VMM::NumVectors; k++)
    {
        batchDistances[k] = zeros;
        batchSumWeights[k] = zeros;
    }

    typename VMM::SoftAssignment softAssign;
    float sampleDistance;
    vfloat<VMM::VectorSize> weights;
    for (size_t n = 0; n < numSamples; n++)
    {
#ifdef USE_HARMONIC_MEAN
        sampleDistance = rcp(samples[n].distance);
#else
        sampleDistance = samples[n].distance;
#endif
        if (vmm.softAssignment(samples[n].direction, softAssign))
        {
            for (size_t k = 0; k < VMM::NumVectors; k++)
            {
                weights = softAssign.assignments[k] * ( (softAssign.assignments[k] * softAssign.pdf) / vmm._weights[k]);
                batchDistances[k] += weights * sampleDistance;
                batchSumWeights[k] += weights;
            }
        }
    }

    for (size_t k = 0; k < VMM::NumVectors; k++)
    {
#ifdef USE_HARMONIC_MEAN
        const vfloat<VMM::VectorSize> sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] / vmm._distances[k]) + batchDistances[k];
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = sufficientStats.sumOfDistanceWeightes[k] / sumInverseDistances;
#else
        const vfloat<VMM::VectorSize> sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] * vmm._distances[k]) + batchDistances[k];
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = select(mask[k], sumInverseDistances / sufficientStats.sumOfDistanceWeightes[k], vmm._distances[k]);
#endif
    }
}
*/

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::Configuration::init()
{
    WEMVMMFactory::Configuration::init();
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::Configuration::serialize(std::ostream& stream) const
{
    WEMVMMFactory::Configuration::serialize(stream);
    stream.write(reinterpret_cast<const char*>(&parallaxCompensation), sizeof(bool));
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::Configuration::deserialize(std::istream& stream)
{
    WEMVMMFactory::Configuration::deserialize(stream);
    stream.read(reinterpret_cast<char*>(&parallaxCompensation), sizeof(bool));
}

template<class TVMMDistribution>
std::string WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::Configuration::toString() const
{
    std::stringstream ss;
    ss << WEMVMMFactory::Configuration::toString();
    ss << "\tparallaxCompensation = " << parallaxCompensation << std::endl;
    return ss.str();
}

}
