// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "WeightedEMVMMFactory.h"

#define USE_HARMONIC_MEAN

using namespace embree;

namespace rkguide
{

template<class TVMMDistribution>
struct WeightedEMParallaxAwareVonMisesFisherFactory: public WeightedEMVonMisesFisherFactory< TVMMDistribution>
{

    public:

    using VMM = TVMMDistribution;
    using WEMVMMFactory = WeightedEMVonMisesFisherFactory< TVMMDistribution>;
    using Configuration = typename WEMVMMFactory::Configuration;
    using FittingStatistics = typename WEMVMMFactory::FittingStatistics;
    using PartialFittingMask = typename WEMVMMFactory:: PartialFittingMask;

    struct SufficientStatisitcs//: public WEMVMMFactory::SufficientStatisitcs
    {

        //FittingStatistics
        public:
        typename WEMVMMFactory::SufficientStatisitcs wEMSufficientStatisitcs;

        vfloat<VMM::VectorSize> sumOfDistanceWeightes[VMM::NumVectors];

        SufficientStatisitcs() = default;

        SufficientStatisitcs(const SufficientStatisitcs &a);// = delete;
/*
        SufficientStatisitcs& operator+=(const SufficientStatisitcs &stats) override;
*/
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
    void fitMixture(VMM &vmm, size_t numComponents, SufficientStatisitcs &stats, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void updateMixture(VMM &vmm, SufficientStatisitcs &previousStats, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void partialUpdateMixture(VMM &vmm, const PartialFittingMask &mask, SufficientStatisitcs &previousStats, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    //VMM VMMfromSufficientStatisitcs(const SufficientStatisitcs &suffStats, const Configuration &cfg) const override;
    std::string toString() const
    {
        return "WeightedEMParallaxAwareVonMisesFisherFactory";
    };

//private:

    void initComponentDistances (VMM &vmm, SufficientStatisitcs &sufficientStats, const DirectionalSampleData* samples, const size_t numSamples) const;

    void updateComponentDistances (VMM &vmm, SufficientStatisitcs &sufficientStats, const DirectionalSampleData* samples, const size_t numSamples) const;
};

////////////////////////////////////////////////////////////
/////////            SufficientStatisitcs
////////////////////////////////////////////////////////////

template<class TVMMDistribution>
WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::SufficientStatisitcs(const SufficientStatisitcs &a)
{


    //std::cout << "WeightedEMParallaxAwareVonMisesFisherFactory::SufficientStatisitcs(const SufficientStatisitcs &a)" << std::endl;
    wEMSufficientStatisitcs = a.wEMSufficientStatisitcs;
    for(uint32_t k=0;k<VMM::NumVectors;k++)
    {
        sumOfDistanceWeightes[k] = a.sumOfDistanceWeightes[k];
    }

}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::applyParallaxShift(const VMM &vmm, const Vector3 shift)
{
    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;
    
    for(uint32_t k=0;k<cnt;k++)
    {
        embree::Vec3< vfloat<VMM::VectorSize> > suffDirections = wEMSufficientStatisitcs.sumOfWeightedDirections[k];
        vfloat<VMM::VectorSize> suffMeanCosines = embree::length(suffDirections);
        suffDirections /= suffMeanCosines;
        suffDirections *= vmm._distances[k];
        suffDirections += embree::Vec3< vfloat<VMM::VectorSize> >(shift);

        suffDirections /= embree::length(suffDirections);
        suffDirections *= suffMeanCosines;
        wEMSufficientStatisitcs.sumOfWeightedDirections[k] = select(vmm._distances[k] > 0.0f, suffDirections, wEMSufficientStatisitcs.sumOfWeightedDirections[k]);
    }
    /*
    if ( rem > 0 )
    {
        for ( size_t i = rem; i < VMM::VectorSize; i++)
        {
            wEMSufficientStatisitcs.sumOfWeightedDirections[cnt].x[i] = 0.0f;
        }
    }
    */

}

template<class TVMMDistribution>
bool WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::isValid() const
{
    //std::cout << "WeightedEMParallaxAwareVonMisesFisherFactory::SufficientStatisitcs(const SufficientStatisitcs &a)" << std::endl;
    bool valid = true;
    valid = wEMSufficientStatisitcs.isValid();


    for(size_t k = 0; k < wEMSufficientStatisitcs.numComponents; k++)
    {
        const div_t tmpK = div( k, VMM::VectorSize );
        valid &= isvalid(sumOfDistanceWeightes[tmpK.quot][tmpK.rem]);
        valid &= sumOfDistanceWeightes[tmpK.quot][tmpK.rem] >= 0.0f;
        RKGUIDE_ASSERT(valid);
    }

    for(size_t k = wEMSufficientStatisitcs.numComponents; k < VMM::MaxComponents; k++)
    {
        const div_t tmpK = div( k, VMM::VectorSize );
        valid &= isvalid(sumOfDistanceWeightes[tmpK.quot][tmpK.rem]);
        valid &= sumOfDistanceWeightes[tmpK.quot][tmpK.rem] == 0.0f;
        RKGUIDE_ASSERT(valid);
    }

   return valid;
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::serialize(std::ostream& stream) const
{
    wEMSufficientStatisitcs.serialize(stream);
    for(uint32_t k=0;k<VMM::NumVectors;k++)
    {
        //stream.write(reinterpret_cast<const char*>(&sumOfWeightedDistances[k]), sizeof(vfloat<VMM::VectorSize>));
        stream.write(reinterpret_cast<const char*>(&sumOfDistanceWeightes[k]), sizeof(vfloat<VMM::VectorSize>));
    }
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::deserialize(std::istream& stream)
{
    wEMSufficientStatisitcs.deserialize(stream);
    for(uint32_t k=0;k<VMM::NumVectors;k++)
    {
        //stream.read(reinterpret_cast<char*>(&sumOfWeightedDistances[k]), sizeof(vfloat<VMM::VectorSize>));
        stream.read(reinterpret_cast<char*>(&sumOfDistanceWeightes[k]), sizeof(vfloat<VMM::VectorSize>));
    }
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::clear(size_t _numComponents)
{
    wEMSufficientStatisitcs.clear(_numComponents);
    //numComponents = _numComponents;
    const vfloat<VMM::VectorSize> zeros(0.0f);
    const vfloat<VMM::VectorSize> infs(std::numeric_limits<float>::infinity());
    const int cnt = (_numComponents+VMM::VectorSize-1) / VMM::VectorSize;

    for(int k = 0; k < cnt;k++)
    {
        //sumOfWeightedDistances[k] = infs;
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
    ss << "\tsumWeights:" << wEMSufficientStatisitcs.sumWeights << std::endl;
    ss << "\tnumSamples:" << wEMSufficientStatisitcs.numSamples << std::endl;
    ss << "\toverallNumSamples:" << wEMSufficientStatisitcs.overallNumSamples << std::endl;
    ss << "\tnumComponents:" << wEMSufficientStatisitcs.numComponents << std::endl;
    ss << "\tisNormalized:" << wEMSufficientStatisitcs.isNormalized << std::endl;
    for (size_t k = 0; k < wEMSufficientStatisitcs.numComponents ; k++)
    {
        int i = k / VMM::VectorSize;
        int j = k % VMM::VectorSize;
        ss  << "\tstat["<< k <<"]:" << "\tsumWeightedStats: " << wEMSufficientStatisitcs.sumOfWeightedStats[i][j]
            << "\tsumWeightedDirections: [" << wEMSufficientStatisitcs.sumOfWeightedDirections[i].x[j] << ",\t"
            << wEMSufficientStatisitcs.sumOfWeightedDirections[i].y[j] << ",\t" << wEMSufficientStatisitcs.sumOfWeightedDirections[i].z[j] << "]"
            << "\tsumWeightedDistanceWeights: " << sumOfDistanceWeightes[i][j]
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
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::fitMixture(VMM &vmm, size_t numComponents, SufficientStatisitcs &stats, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    WeightedEMVonMisesFisherFactory< TVMMDistribution>::fitMixture( vmm, numComponents, stats.wEMSufficientStatisitcs, samples, numSamples, cfg, fitStats);
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
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::updateMixture(VMM &vmm, SufficientStatisitcs &previousStats, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    WeightedEMVonMisesFisherFactory< TVMMDistribution>::updateMixture( vmm, previousStats.wEMSufficientStatisitcs, samples, numSamples, cfg, fitStats);
    //updateComponentDistances(vmm, stats, samples, numSamples);
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::partialUpdateMixture(VMM &vmm, const PartialFittingMask &mask, SufficientStatisitcs &previousStats, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    WeightedEMVonMisesFisherFactory< TVMMDistribution>::partialUpdateMixture( vmm, mask, previousStats.wEMSufficientStatisitcs, samples, numSamples, cfg, fitStats);
}

template<class TVMMDistribution>
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::initComponentDistances (VMM &vmm, SufficientStatisitcs &sufficientStats, const DirectionalSampleData* samples, const size_t numSamples) const
{
    RKGUIDE_ASSERT(vmm._numComponents == sufficientStats.wEMSufficientStatisitcs.numComponents);
    
    vfloat<VMM::VectorSize> batchDistances[VMM::NumVectors];
    vfloat<VMM::VectorSize> batchSumWeights[VMM::NumVectors];

    const vfloat<VMM::VectorSize> zeros(0.0f);

    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
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
        //const vfloat<VMM::VectorSize> sumInverseDistances = batchDistances[k];
        sufficientStats.sumOfDistanceWeightes[k] = batchSumWeights[k];
        vmm._distances[k] = sufficientStats.sumOfDistanceWeightes[k] / batchDistances[k];
#else
        //const vfloat<VMM::VectorSize> sumInverseDistances = batchDistances[k];
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
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::updateComponentDistances (VMM &vmm, SufficientStatisitcs &sufficientStats, const DirectionalSampleData* samples, const size_t numSamples) const
{
    RKGUIDE_ASSERT(vmm._numComponents == sufficientStats.wEMSufficientStatisitcs.numComponents)

    vfloat<VMM::VectorSize> batchDistances[VMM::NumVectors];
    vfloat<VMM::VectorSize> batchSumWeights[VMM::NumVectors];

    const vfloat<VMM::VectorSize> zeros(0.0f);
    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
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
            for (size_t k = 0; k < cnt; k++)
            {
                weights = softAssign.assignments[k] * ( (softAssign.assignments[k] * softAssign.pdf) / vmm._weights[k]);
                batchDistances[k] += weights * sampleDistance;
                batchSumWeights[k] += weights;
            }
        }
    }

    for (size_t k = 0; k < cnt; k++)
    {
#ifdef USE_HARMONIC_MEAN
        const vfloat<VMM::VectorSize> sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] / vmm._distances[k]) + batchDistances[k];
        sufficientStats.sumOfDistanceWeightes[k] += batchSumWeights[k];
        vmm._distances[k] = sufficientStats.sumOfDistanceWeightes[k] / sumInverseDistances;
#else
        const vfloat<VMM::VectorSize> sumInverseDistances = (sufficientStats.sumOfDistanceWeightes[k] * vmm._distances[k]) + batchDistances[k];
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
void WeightedEMParallaxAwareVonMisesFisherFactory< TVMMDistribution>::partialUpdateComponentDistances (VMM &vmm, const PartialFittingMask &mask, SufficientStatisitcs &sufficientStats, const DirectionalSampleData* samples, const size_t numSamples) const
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
}
