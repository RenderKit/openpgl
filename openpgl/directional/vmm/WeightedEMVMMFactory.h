// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"
#include "VMM.h"
#include "../../data/SampleData.h"

#include "VMMFactory.h"

#include <fstream>
#include <iostream>

//#define OPENPGL_MAX_KAPPA 1000000.0f
#define OPENPGL_MAX_KAPPA 32000.0f

namespace openpgl
{

template<class TVMMDistribution>
struct WeightedEMVonMisesFisherFactory: public VonMisesFisherFactory< TVMMDistribution>
{
    public:
    //typedef std::integral_constant<size_t, (maxComponents + (VMM::VectorSize -1)) / VMM::VectorSize> NumVectors;
    typedef TVMMDistribution Distribution;
    using VMM = TVMMDistribution;

    struct Configuration
    {
        size_t initK {VMM::VectorSize};
        float initKappa {5.0f};

        size_t maxK {VMM::MaxComponents};
        size_t maxEMIterrations {100};

        float maxKappa {OPENPGL_MAX_KAPPA};
        float maxMeanCosine { KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA)};
        float convergenceThreshold {0.0025f};

        // MAP prior parameters
        // weight prior
        float weightPrior{0.1f};

        // concentration/meanCosine prior
        float meanCosinePriorStrength {0.1f};
        float meanCosinePrior {0.0f};

        void init();

        void serialize(std::ostream& stream) const;

        void deserialize(std::istream& stream);

        std::string toString() const;

    };

    struct FittingStatistics
    {
        size_t numSamples {0};
        size_t numIterations {0};
        float summedWeightedLogLikelihood {0.0f};
    };

    struct PartialFittingMask
    {
        embree::vbool<VMM::VectorSize> mask[VMM::NumVectors];

        PartialFittingMask() = default;

        void resetToFalse();
        void resetToTrue(const size_t &numComponents);
        void setToTrue(const size_t &idx);
        void setToFalse(const size_t &idx);
        std::string toString() const;
    };

    struct SufficientStatisitcs
    {
        SufficientStatisitcs() = default;

        embree::Vec3< embree::vfloat<VMM::VectorSize> > sumOfWeightedDirections[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> sumOfWeightedStats[VMM::NumVectors];

        float sumWeights {0.f};
        float numSamples {0.f};
        float overallNumSamples {0.f};
        size_t numComponents {VMM::MaxComponents};
        bool normalized {false};
        //SufficientStatisitcs operator+(const SufficientStatisitcs &stats);
        virtual SufficientStatisitcs& operator+=(const SufficientStatisitcs &stats);

        virtual void serialize(std::ostream& stream) const;

        virtual void deserialize(std::istream& stream);

        virtual void clear(size_t _numComponents);

        virtual void clearAll();

        virtual void normalize( const float &_numSamples );

        inline bool isNormalized() const
        {
            return normalized;
        };

        virtual void mergeComponentStats(const size_t &idx0, const size_t &idx1);

        virtual void splitComponentsStats(const size_t &idx0, const size_t &idx1,
                const Vector3 &meanDirection0, const Vector3 &meanDirection1,
                const float &meanCosine0, const float &meanCosine1);

        virtual void swapComponentStats(const size_t &idx0, const size_t &idx1);

        virtual void maskedReplace(const PartialFittingMask &mask, const SufficientStatisitcs &stats);

        virtual void decay(const float &alpha);

        virtual std::string toString() const;

        inline float getNumSamples() const
        {
            return numSamples;
        }

        inline float getSumWeights() const
        {
            return sumWeights;
        }

        inline void setNumComponents( const size_t &numComponents)
        {
            this->numComponents = numComponents;
        }

        inline size_t getNumComponents( ) const
        {
            return this->numComponents;
        }

        bool isValid() const;

    };

    struct UnassignedSamplesStatistics
    {
        float sumOfUnassignedWeights{0.0f};
        Vector3 sumUnassignedWeightedDirections{0.0f, 0.0f, 0.0f};
        void clear();
        bool isValid() const;
    };

public:

    //typedef TVMMDistribution VMM;

    WeightedEMVonMisesFisherFactory();

    virtual void fitMixture(VMM &vmm, SufficientStatisitcs &stats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    virtual void updateMixture(VMM &vmm, SufficientStatisitcs &previousStats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    virtual void partialUpdateMixture(VMM &vmm, PartialFittingMask &mask, SufficientStatisitcs &previousStats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    virtual VMM VMMfromSufficientStatisitcs(const SufficientStatisitcs &suffStats, const Configuration &cfg) const;

    std::string toString() const
    {
        return "WeightedEMVonMisesFisherFactory";
    };

private:

    float weightedExpectationStep(VMM &vmm, SufficientStatisitcs &stats, UnassignedSamplesStatistics &unassignedStats, const SampleData* samples, const size_t numSamples) const;

    void weightedMaximumAPosteriorStep(VMM &vmm, const SufficientStatisitcs &previousStats,
        const SufficientStatisitcs &currentStats,
        const Configuration &cfg) const;

    void estimateMAPWeights( VMM &vmm, const SufficientStatisitcs &currentStats, const SufficientStatisitcs &previousStats, const float &_weightPrior ) const;

    void estimateMAPMeanDirectionAndConcentration( VMM &vmm, const SufficientStatisitcs &currentStats, const SufficientStatisitcs &previousStats, const Configuration &cfg) const;

    void partialWeightedMaximumAPosteriorStep(VMM &vmm, const PartialFittingMask &mask, SufficientStatisitcs &previousStats,
        SufficientStatisitcs &currentStats,
        const Configuration &cfg) const;

    void estimatePartialMAPWeights( VMM &vmm, const PartialFittingMask &mask, SufficientStatisitcs &currentStats, SufficientStatisitcs &previousStats, const float &_weightPrior ) const;

    void estimatePartialMAPMeanDirectionAndConcentration( VMM &vmm, const PartialFittingMask &mask, SufficientStatisitcs &currentStats, SufficientStatisitcs &previousStats, const Configuration &cfg) const;

    void handleUnassignedSampleStats(UnassignedSamplesStatistics &unassignedStats, VMM &vmm, SufficientStatisitcs &currentStats, SufficientStatisitcs &previousStats) const;

};

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::UnassignedSamplesStatistics::clear()
{
    sumOfUnassignedWeights = 0.0f;
    sumUnassignedWeightedDirections = Vector3(0.0f);
}

template<class TVMMDistribution>
bool WeightedEMVonMisesFisherFactory< TVMMDistribution>::UnassignedSamplesStatistics::isValid() const
{
    bool valid = true;
    valid = valid && embree::isvalid(sumOfUnassignedWeights);
    valid = valid && sumOfUnassignedWeights>=0.f;
    OPENPGL_ASSERT(valid);

    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.x);
    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.y);
    valid = valid && embree::isvalid(sumUnassignedWeightedDirections.z);
    OPENPGL_ASSERT(valid);
    return valid;
}

template<class TVMMDistribution>
WeightedEMVonMisesFisherFactory< TVMMDistribution>::WeightedEMVonMisesFisherFactory()
{
    typename VonMisesFisherFactory<TVMMDistribution>::VonMisesFisherFactory( );
}

template<class TVMMDistribution>
typename WeightedEMVonMisesFisherFactory< TVMMDistribution>::VMM WeightedEMVonMisesFisherFactory< TVMMDistribution>::VMMfromSufficientStatisitcs(const SufficientStatisitcs &suffStats, const Configuration &cfg) const
{

    SufficientStatisitcs previousStats;
    previousStats.clear(suffStats.numComponents);
    VMM vmm;
    vmm._numComponents = suffStats.numComponents;
    weightedMaximumAPosteriorStep(vmm, previousStats, suffStats, cfg);

    return vmm;
}


template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::fitMixture(VMM &vmm, SufficientStatisitcs &stats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    const size_t numComponents = cfg.initK;
    //VonMisesFisherFactory< TVMMDistribution>::InitUniformVMM( vmm, numComponents, 5.0f);
    this->InitUniformVMM( vmm, numComponents, cfg.initKappa);
    //OPENPGL_ASSERT(vmm.isValid());
    //SufficientStatisitcs stats;
    stats.clear(numComponents);
    stats.normalized = true;
    //stats.clearAll();
    updateMixture(vmm, stats, samples, numSamples, cfg, fitStats);

}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::handleUnassignedSampleStats(UnassignedSamplesStatistics &unassignedStats, VMM &vmm, SufficientStatisitcs &currentStats, SufficientStatisitcs &previousStats) const
{
    const div_t tmpK = div(currentStats.numComponents, TVMMDistribution::VectorSize);
    currentStats.numComponents++;
    currentStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] = unassignedStats.sumOfUnassignedWeights;
    currentStats.sumOfWeightedDirections[tmpK.quot].x[tmpK.rem] = unassignedStats.sumUnassignedWeightedDirections.x;
    currentStats.sumOfWeightedDirections[tmpK.quot].y[tmpK.rem] = unassignedStats.sumUnassignedWeightedDirections.y;
    currentStats.sumOfWeightedDirections[tmpK.quot].z[tmpK.rem] = unassignedStats.sumUnassignedWeightedDirections.z;

    previousStats.numComponents++;
    previousStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] = 0.0f;
    previousStats.sumOfWeightedDirections[tmpK.quot].x[tmpK.rem] = 0.0f;
    previousStats.sumOfWeightedDirections[tmpK.quot].y[tmpK.rem] = 0.0f;
    previousStats.sumOfWeightedDirections[tmpK.quot].z[tmpK.rem] = 0.0f;

    vmm._numComponents++;
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::updateMixture(VMM &vmm, SufficientStatisitcs &previousStats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    SufficientStatisitcs currentStats;
    // initially clear all stats
    currentStats.clearAll();

    size_t currentEMIteration = 0;
    bool converged = false;
    float previousLogLikelihood = 0.0f;
    float inv_previousLogLikelihood = 1.0f;
    UnassignedSamplesStatistics unassignedStats;
    while ( !converged  && currentEMIteration < cfg.maxEMIterrations )
    {
        float logLikelihood = weightedExpectationStep( vmm, currentStats, unassignedStats, samples, numSamples);
        if(unassignedStats.sumOfUnassignedWeights > 0.0f && currentStats.numComponents < TVMMDistribution::MaxComponents)
        {
            handleUnassignedSampleStats(unassignedStats, vmm, currentStats, previousStats);
        }

        OPENPGL_ASSERT(!currentStats.isNormalized());
        currentStats.normalize(currentStats.numSamples);
        OPENPGL_ASSERT(currentStats.isValid());
        weightedMaximumAPosteriorStep( vmm, currentStats, previousStats, cfg);
        currentEMIteration++;

        // TODO: Add convergence check
        if (currentEMIteration >1)
        {
            float relLogLikelihoodDifference = std::fabs(logLikelihood - previousLogLikelihood) * inv_previousLogLikelihood;
            if(relLogLikelihoodDifference < cfg.convergenceThreshold)
            {
                converged = true;
            }
            //std::cout << "logLikelihood:" <<  logLikelihood << "\t previousLogLikelihood: "<< previousLogLikelihood  << "\t relLogLikelihoodDifference: " << relLogLikelihoodDifference << std::endl;
            previousLogLikelihood = logLikelihood;
            inv_previousLogLikelihood = 1.0f / std::fabs(logLikelihood);
        }
    }
    previousStats += currentStats;

    fitStats.numSamples = numSamples;
    fitStats.numIterations = currentEMIteration;
    fitStats.summedWeightedLogLikelihood = previousLogLikelihood;
    //td::cout << "converged:" <<  currentEMIteration << std::endl;
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::partialUpdateMixture(VMM &vmm, PartialFittingMask &mask, SufficientStatisitcs &previousStats, const SampleData* samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const
{
    SufficientStatisitcs currentStats;
    // initially clear all stats
    currentStats.clearAll();

    size_t currentEMIteration = 0;
    bool converged = false;
    float previousLogLikelihood = 0.0f;
    float inv_previousLogLikelihood = 1.0f;

    UnassignedSamplesStatistics unassignedStats;
    while ( !converged  && currentEMIteration < cfg.maxEMIterrations )
    {
        float logLikelihood = weightedExpectationStep( vmm, currentStats, unassignedStats, samples, numSamples);
        if(unassignedStats.sumOfUnassignedWeights > 0.0f && currentStats.numComponents < TVMMDistribution::MaxComponents)
        {
            handleUnassignedSampleStats(unassignedStats, vmm, currentStats, previousStats);
            mask.setToTrue(vmm._numComponents-1);
        }

        OPENPGL_ASSERT(!currentStats.isNormalized());
        currentStats.normalize(currentStats.numSamples);
        OPENPGL_ASSERT(currentStats.isValid());

        partialWeightedMaximumAPosteriorStep( vmm, mask, currentStats, previousStats, cfg);
        currentEMIteration++;
        // TODO: Add convergence check
        if (currentEMIteration >1)
        {
            float relLogLikelihoodDifference = std::fabs(logLikelihood - previousLogLikelihood) * inv_previousLogLikelihood;
            if(relLogLikelihoodDifference < cfg.convergenceThreshold)
            {
                converged = true;
            }
            //std::cout << "logLikelihood:" <<  logLikelihood << "\t previousLogLikelihood: "<< previousLogLikelihood  << "\t relLogLikelihoodDifference: " << relLogLikelihoodDifference << std::endl;
            previousLogLikelihood = logLikelihood;
            inv_previousLogLikelihood = 1.0f / std::fabs(logLikelihood);
        }
    }
    previousStats += currentStats;

    fitStats.numSamples = numSamples;
    fitStats.numIterations = currentEMIteration;
    fitStats.summedWeightedLogLikelihood = previousLogLikelihood;
    //std::cout << "converged:" <<  currentEMIteration << std::endl;
}


template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::Configuration::serialize(std::ostream& stream) const
{
    stream.write(reinterpret_cast<const char*>(&initK), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(&initKappa), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&maxK), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(&maxEMIterrations), sizeof(size_t));

    stream.write(reinterpret_cast<const char*>(&maxKappa), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&maxMeanCosine), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&convergenceThreshold), sizeof(float));

    stream.write(reinterpret_cast<const char*>(&weightPrior), sizeof(float));

    stream.write(reinterpret_cast<const char*>(&meanCosinePriorStrength), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&meanCosinePrior), sizeof(float));
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::Configuration::deserialize(std::istream& stream)
{
    stream.read(reinterpret_cast<char*>(&initK), sizeof(size_t));
    stream.read(reinterpret_cast<char*>(&initKappa), sizeof(float));
    stream.read(reinterpret_cast<char*>(&maxK), sizeof(size_t));
    stream.read(reinterpret_cast<char*>(&maxEMIterrations), sizeof(size_t));

    stream.read(reinterpret_cast<char*>(&maxKappa), sizeof(float));
    stream.read(reinterpret_cast<char*>(&maxMeanCosine), sizeof(float));
    stream.read(reinterpret_cast<char*>(&convergenceThreshold), sizeof(float));

    stream.read(reinterpret_cast<char*>(&weightPrior), sizeof(float));

    stream.read(reinterpret_cast<char*>(&meanCosinePriorStrength), sizeof(float));
    stream.read(reinterpret_cast<char*>(&meanCosinePrior), sizeof(float));
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::Configuration::init()
{
    maxMeanCosine  = KappaToMeanCosine<float>(maxKappa);
}

template<class TVMMDistribution>
std::string WeightedEMVonMisesFisherFactory< TVMMDistribution>::Configuration::toString() const
{
    std::stringstream ss;
    ss << "Configuration:" << std::endl;
    ss << "\tinitComponent = " << initK << std::endl;
    ss << "\tmaxComponents = " << maxK << std::endl;
    ss << "\tinitNumComponents = " << initK << std::endl;
    ss << "\tmaxEMIterrations = " << maxEMIterrations << std::endl;
    ss << "\tmaxKappa = " << maxKappa << std::endl;
    ss << "\tmaxMeanCosine = " << maxMeanCosine << std::endl;
    ss << "\tconvergenceThreshold = " << convergenceThreshold << std::endl;
    ss << "\tweightPrior = " << weightPrior << std::endl;
    ss << "\tmeanCosinePriorStrength = " << meanCosinePriorStrength << std::endl;
    ss << "\tmeanCosinePrior = " << meanCosinePrior << std::endl;
    return ss.str();
 }

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::PartialFittingMask::resetToFalse()
{
    const embree::vbool<VMM::VectorSize> vFalse(false);
    for (size_t k = 0; k < ( (VMM::MaxComponents + (VMM::VectorSize -1)) / VMM::VectorSize); k++)
    {
        mask[k] = vFalse;
    }
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::PartialFittingMask::resetToTrue(const size_t &numComponents)
{
    const embree::vbool<VMM::VectorSize> vTrue(true);
    const int cnt = (numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    for (size_t k = 0; k < cnt; k++)
    {
        mask[k] = vTrue;
    }

    const div_t tmp = div( numComponents, VMM::VectorSize);
    for (size_t k = tmp.rem; k < VMM::VectorSize; k++)
    {
        clear(mask[tmp.quot], k);
    }
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::PartialFittingMask::setToTrue(const size_t &idx)
{
    //std::cout << "setToTrue: " << idx << std::endl;
    const div_t tmp = div( idx, VMM::VectorSize);
    set(mask[tmp.quot], tmp.rem);
    //std::cout << mask[tmp.quot]<< "\t"<< tmp.quot << "\t"<< tmp.rem <<  std::endl;
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::PartialFittingMask::setToFalse(const size_t &idx)
{
    //std::cout << "setToFalse: " << idx << std::endl;
    const div_t tmp = div( idx, VMM::VectorSize);
    clear(mask[tmp.quot], tmp.rem);
    //std::cout << mask[tmp.quot]<< "\t"<< tmp.quot << "\t"<< tmp.rem <<  std::endl;
}


template<class TVMMDistribution>
std::string WeightedEMVonMisesFisherFactory< TVMMDistribution>::PartialFittingMask::toString() const
{
    std::stringstream ss;
        ss << "PartialFittingMask:" << std::endl;
        for (size_t k = 0; k < VMM::MaxComponents; k++)
        {
            const div_t tmp = div( k, VMM::VectorSize);
            ss << "mask[" << k << "]: " << mask[tmp.quot][tmp.rem] << std::endl;
        }
    return ss.str();
}


template<class TVMMDistribution>
bool WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::isValid() const
{

    bool valid = true;
    for(size_t k = 0; k < numComponents; k++)
    {
        const div_t tmpK = div( k, VMM::VectorSize );
        valid = valid && embree::isvalid(sumOfWeightedDirections[tmpK.quot].x[tmpK.rem]);
        //valid = valid && sumOfWeightedDirections[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(sumOfWeightedDirections[tmpK.quot].y[tmpK.rem]);
        //valid = valid && sumOfWeightedDirections[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(sumOfWeightedDirections[tmpK.quot].z[tmpK.rem]);
        //valid = valid && sumOfWeightedDirections[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(sumOfWeightedStats[tmpK.quot][tmpK.rem]);
        valid = valid && sumOfWeightedStats[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);
    }

    for(size_t k = numComponents; k < VMM::MaxComponents; k++)
    {
        const div_t tmpK = div( k, VMM::VectorSize );
        valid = valid && sumOfWeightedDirections[tmpK.quot].x[tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && sumOfWeightedDirections[tmpK.quot].y[tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && sumOfWeightedDirections[tmpK.quot].z[tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && sumOfWeightedStats[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);
    }

    valid = valid && embree::isvalid(numSamples);
    valid = valid && numSamples >= 0.0f;
    OPENPGL_ASSERT(valid);

    valid = valid && embree::isvalid(sumWeights);
    valid = valid && sumWeights >= 0.0f;
    OPENPGL_ASSERT(valid);

    return valid;
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::decay(const float &alpha)
{
    for (size_t k = 0; k < ( (VMM::MaxComponents + (VMM::VectorSize -1)) / VMM::VectorSize); k++)
    {
        sumOfWeightedDirections[k].x *= alpha;
        sumOfWeightedDirections[k].y *= alpha;
        sumOfWeightedDirections[k].z *= alpha;
        sumOfWeightedStats[k] *= alpha;
    }

    numSamples *= alpha;
    sumWeights *= alpha;

}
template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::maskedReplace(const PartialFittingMask &mask, const SufficientStatisitcs &stats){

    embree::vfloat<VMM::VectorSize> newSumWeights {0.0f};

    for (size_t k = 0; k < ( (VMM::MaxComponents + (VMM::VectorSize -1)) / VMM::VectorSize); k++)
    {
        sumOfWeightedDirections[k] = select(mask.mask[k], stats.sumOfWeightedDirections[k], sumOfWeightedDirections[k]);
        sumOfWeightedStats[k] =  select(mask.mask[k], stats.sumOfWeightedStats[k], sumOfWeightedStats[k]);
        newSumWeights += sumOfWeightedStats[k];
    }
    if (normalized)
    {
        numSamples = embree::reduce_add(newSumWeights);
    }
    else
    {
        sumWeights = embree::reduce_add(newSumWeights);
    }
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::serialize(std::ostream& stream) const
{
    for(uint32_t k=0;k<VMM::NumVectors;k++){
        stream.write(reinterpret_cast<const char*>(&sumOfWeightedDirections[k]), sizeof(embree::Vec3< embree::vfloat<VMM::VectorSize> >));
        stream.write(reinterpret_cast<const char*>(&sumOfWeightedStats[k]), sizeof(embree::vfloat<VMM::VectorSize>));
    }
    stream.write(reinterpret_cast<const char*>(&sumWeights), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&numSamples), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&overallNumSamples), sizeof(float));
    stream.write(reinterpret_cast<const char*>(&numComponents), sizeof(size_t));
    stream.write(reinterpret_cast<const char*>(&normalized), sizeof(bool));
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::deserialize(std::istream& stream)
{
    for(uint32_t k=0;k<VMM::NumVectors;k++){
        stream.read(reinterpret_cast<char*>(&sumOfWeightedDirections[k]), sizeof(embree::Vec3< embree::vfloat<VMM::VectorSize> >));
        stream.read(reinterpret_cast<char*>(&sumOfWeightedStats[k]), sizeof(embree::vfloat<VMM::VectorSize>));
    }
    stream.read(reinterpret_cast<char*>(&sumWeights), sizeof(float));
    stream.read(reinterpret_cast<char*>(&numSamples), sizeof(float));
    stream.read(reinterpret_cast<char*>(&overallNumSamples), sizeof(float));
    stream.read(reinterpret_cast<char*>(&numComponents), sizeof(size_t));
    stream.read(reinterpret_cast<char*>(&normalized), sizeof(bool));
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::swapComponentStats(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div( idx0, VMM::VectorSize);
    const div_t tmpIdx1 = div( idx1, VMM::VectorSize);

    std::swap(sumOfWeightedDirections[tmpIdx0.quot].x[tmpIdx0.rem], sumOfWeightedDirections[tmpIdx1.quot].x[tmpIdx1.rem]);
    std::swap(sumOfWeightedDirections[tmpIdx0.quot].y[tmpIdx0.rem], sumOfWeightedDirections[tmpIdx1.quot].y[tmpIdx1.rem]);
    std::swap(sumOfWeightedDirections[tmpIdx0.quot].z[tmpIdx0.rem], sumOfWeightedDirections[tmpIdx1.quot].z[tmpIdx1.rem]);
    std::swap(sumOfWeightedStats[tmpIdx0.quot][tmpIdx0.rem], sumOfWeightedStats[tmpIdx1.quot][tmpIdx1.rem]);

}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::mergeComponentStats(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div( idx0, VMM::VectorSize);
    const div_t tmpIdx1 = div( idx1, VMM::VectorSize);
    const div_t tmpIdx2 = div( numComponents-1, VMM::VectorSize);

    // merging the statistics of the component 0 and 1
    sumOfWeightedDirections[tmpIdx0.quot].x[tmpIdx0.rem] += sumOfWeightedDirections[tmpIdx1.quot].x[tmpIdx1.rem];
    sumOfWeightedDirections[tmpIdx0.quot].y[tmpIdx0.rem] += sumOfWeightedDirections[tmpIdx1.quot].y[tmpIdx1.rem];
    sumOfWeightedDirections[tmpIdx0.quot].z[tmpIdx0.rem] += sumOfWeightedDirections[tmpIdx1.quot].z[tmpIdx1.rem];
    sumOfWeightedStats[tmpIdx0.quot][tmpIdx0.rem] += sumOfWeightedStats[tmpIdx1.quot][tmpIdx1.rem];

    // copying the statistics of the last component to the position of component 1
    sumOfWeightedDirections[tmpIdx1.quot].x[tmpIdx1.rem] = sumOfWeightedDirections[tmpIdx2.quot].x[tmpIdx2.rem];
    sumOfWeightedDirections[tmpIdx1.quot].y[tmpIdx1.rem] = sumOfWeightedDirections[tmpIdx2.quot].y[tmpIdx2.rem];
    sumOfWeightedDirections[tmpIdx1.quot].z[tmpIdx1.rem] = sumOfWeightedDirections[tmpIdx2.quot].z[tmpIdx2.rem];
    sumOfWeightedStats[tmpIdx1.quot][tmpIdx1.rem] = sumOfWeightedStats[tmpIdx2.quot][tmpIdx2.rem];

    // reseting the statistics of the last component
    sumOfWeightedDirections[tmpIdx2.quot].x[tmpIdx2.rem] = 0.0f;
    sumOfWeightedDirections[tmpIdx2.quot].y[tmpIdx2.rem] = 0.0f;
    sumOfWeightedDirections[tmpIdx2.quot].z[tmpIdx2.rem] = 0.0f;
    sumOfWeightedStats[tmpIdx2.quot][tmpIdx2.rem] = 0.0f;

    numComponents--;
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::splitComponentsStats(const size_t &idx0, const size_t &idx1,
                const Vector3 &meanDirection0, const Vector3 &meanDirection1,
                const float &meanCosine0, const float &meanCosine1)
{
    //OPENPGL_ASSERT(meanCosine0 > 0.f && meanCosine0 <= 1.0f);
    //OPENPGL_ASSERT(meanCosine1 > 0.f && meanCosine1 <= 1.0f);

    const div_t tmpI = div(idx0, static_cast<int>(VMM::VectorSize));
    const div_t tmpJ = div(idx1, static_cast<int>(VMM::VectorSize));

    float sumStatsWeight = sumOfWeightedStats[tmpI.quot][tmpI.rem];
    sumStatsWeight /= 2.0f;

    OPENPGL_ASSERT(sumStatsWeight > 0.f);

    sumOfWeightedStats[tmpI.quot][tmpI.rem] = sumStatsWeight;
    sumOfWeightedDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x * meanCosine0 * sumStatsWeight;
    sumOfWeightedDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y * meanCosine0 * sumStatsWeight;
    sumOfWeightedDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z * meanCosine0 * sumStatsWeight;

    sumOfWeightedStats[tmpJ.quot][tmpJ.rem] = sumStatsWeight;
    sumOfWeightedDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x * meanCosine1 * sumStatsWeight;
    sumOfWeightedDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y * meanCosine1 * sumStatsWeight;
    sumOfWeightedDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z * meanCosine1 * sumStatsWeight;
    numComponents += 1;

    OPENPGL_ASSERT(!std::isnan(sumOfWeightedDirections[tmpI.quot].x[tmpI.rem]) && std::isfinite(sumOfWeightedDirections[tmpI.quot].x[tmpI.rem]));
    OPENPGL_ASSERT(!std::isnan(sumOfWeightedDirections[tmpI.quot].y[tmpI.rem]) && std::isfinite(sumOfWeightedDirections[tmpI.quot].y[tmpI.rem]));
    OPENPGL_ASSERT(!std::isnan(sumOfWeightedDirections[tmpI.quot].z[tmpI.rem]) && std::isfinite(sumOfWeightedDirections[tmpI.quot].z[tmpI.rem]));

}


template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::clear(size_t _numComponents)
{
    const embree::Vec3< embree::vfloat<VMM::VectorSize> > vecZeros(0.0f);
    const embree::vfloat<VMM::VectorSize> zeros(0.0f);

    numComponents = _numComponents;
    const int cnt = (numComponents+VMM::VectorSize-1) / VMM::VectorSize;

    for(int k = 0; k < cnt;k++)
    {
        sumOfWeightedDirections[k] = vecZeros;
        sumOfWeightedStats[k] = zeros;
    }

    sumWeights = 0.0f;
    numSamples = 0;
    normalized = false;
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::clearAll()
{
     clear(VMM::MaxComponents);
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::normalize( const float &_numSamples )
{
    const int cnt = (numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    numSamples = _numSamples;
    embree::vfloat<VMM::VectorSize> sumWeightedStatsVec(0.0f);

    for(int k = 0; k < cnt;k++)
    {
        sumWeightedStatsVec += sumOfWeightedStats[k];
    }
    sumWeights = reduce_add(sumWeightedStatsVec);
    embree::vfloat<VMM::VectorSize> norm ( _numSamples / sumWeights );

    for(int k = 0; k < cnt;k++)
    {
        sumOfWeightedDirections[k] *= norm;
        sumOfWeightedStats[k] *= norm;
    }
    normalized = true;
}

template<class TVMMDistribution>
typename WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs& WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::operator+=(const WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs &stats)
{

    // TODO: chaeck for normalization

    const int cnt = (numComponents+VMM::VectorSize-1) / VMM::VectorSize;

    this->sumWeights += stats.sumWeights;
    this->numSamples += stats.numSamples;
    this->overallNumSamples += stats.numSamples;
    for(int k = 0; k < cnt;k++)
    {
        this->sumOfWeightedDirections[k] += stats.sumOfWeightedDirections[k];
        this->sumOfWeightedStats[k] += stats.sumOfWeightedStats[k];
    }

    return *this;
}

template<class TVMMDistribution>
std::string WeightedEMVonMisesFisherFactory< TVMMDistribution>::SufficientStatisitcs::toString() const
{
    std::stringstream ss;
    ss << "SufficientStatisitcs:" << std::endl;
    ss << "\tsumWeights:" << sumWeights << std::endl;
    ss << "\tnumSamples:" << numSamples << std::endl;
    ss << "\toverallNumSamples:" << overallNumSamples << std::endl;
    ss << "\tnumComponents:" << numComponents << std::endl;
    ss << "\tnormalized:" << normalized << std::endl;
    for (size_t k = 0; k < numComponents ; k++)
    {
        int i = k / VMM::VectorSize;
        int j = k % VMM::VectorSize;
        ss  << "\tstat["<< k <<"]:" << "\tsumWeightedStats: " << sumOfWeightedStats[i][j]
            << "\tsumWeightedDirections: [" << sumOfWeightedDirections[i].x[j] << ",\t"
            << sumOfWeightedDirections[i].y[j] << ",\t" << sumOfWeightedDirections[i].z[j] << "]"
            << std::endl;
    }
    return ss.str();
}

template<class TVMMDistribution>
float WeightedEMVonMisesFisherFactory< TVMMDistribution>::weightedExpectationStep(VMM &vmm,
        SufficientStatisitcs &stats,
        UnassignedSamplesStatistics &unassignedStats,
        const SampleData* samples,
        const size_t numSamples) const
{
    unassignedStats.clear();
    stats.clear(vmm._numComponents);
    stats.numComponents = vmm._numComponents;
    stats.numSamples = numSamples;

    const int cnt = (stats.numComponents+VMM::VectorSize-1) / VMM::VectorSize;

    float summedWeightedLogLikelihood {0.f};

    typename VMM::SoftAssignment softAssign;

    for (size_t n = 0; n < numSamples; n++ )
    {
        const SampleData sampleData = samples[n];
        const embree::vfloat<VMM::VectorSize> sampleWeight = sampleData.weight;
        const Vector3 sampleDirection(sampleData.direction.x, sampleData.direction.y, sampleData.direction.z);
        const embree::Vec3< embree::vfloat<VMM::VectorSize> > sampleDirectionSIMD( sampleDirection );

        // check if the samples is covered by any of the components
        if ( !vmm.softAssignment( sampleDirection, softAssign) )
        {
            unassignedStats.sumOfUnassignedWeights += sampleData.weight;
            unassignedStats.sumUnassignedWeightedDirections += sampleDirection * sampleData.weight;
#ifdef OPENPGL_DEBUG_MODE
            std::cout << "continue" << std::endl;
#endif
            continue;
        }

        summedWeightedLogLikelihood += sampleData.weight * embree::log( softAssign.pdf );

        for (size_t k =0; k < cnt; k++)
        {
            stats.sumOfWeightedDirections[k] += sampleDirectionSIMD * softAssign.assignments[k] * sampleWeight;
            stats.sumOfWeightedStats[k] += softAssign.assignments[k] * sampleWeight;
        }

    }
    return summedWeightedLogLikelihood;
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::estimateMAPWeights( VMM &vmm,
        const SufficientStatisitcs &currentStats,
        const SufficientStatisitcs &previousStats,
        const float &_weightPrior ) const
{
    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;

    const size_t numComponents = vmm._numComponents;

    const embree::vfloat<VMM::VectorSize> weightPrior(_weightPrior);

    const embree::vfloat<VMM::VectorSize> numSamples = currentStats.numSamples + previousStats.numSamples;
    //const vfloat<VMM::VectorSize> numSamples = currentStats.numSamples + previousStats.overallNumSamples;

    for ( size_t k = 0; k < cnt; k ++ )
    {
        //_sumWeights += currentStats.sumOfWeightedStats[k];
        embree::vfloat<VMM::VectorSize>  weight = ( currentStats.sumOfWeightedStats[k] + previousStats.sumOfWeightedStats[k] ) ;
        weight = ( weightPrior + ( weight ) ) / (( weightPrior * numComponents ) + numSamples );
        //vfloat<VMM::VectorSize>  weight = ( currentStats.sumOfWeightedStats[k]/* + previousStats.sumOfWeightedStats[k]*/ ) / ( sumWeights );
        //weight = ( weightPrior + ( weight * numSamples ) ) / (( weightPrior * numComponents ) + numSamples );
        vmm._weights[k] = weight;
    }

    // TODO: find better more efficient way
    if ( vmm._numComponents % VMM::VectorSize > 0 )
    {
            for (size_t i = vmm._numComponents % VMM::VectorSize; i < VMM::VectorSize; i++ )
            {
                vmm._weights[cnt-1][i] = 0.0f;
            }
    }
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::estimateMAPMeanDirectionAndConcentration( VMM &vmm,
        const SufficientStatisitcs &currentStats,
        const SufficientStatisitcs &previousStats ,
        const Configuration &cfg) const
{
    const embree::vfloat<VMM::VectorSize> currentNumSamples = currentStats.numSamples;
    const embree::vfloat<VMM::VectorSize> previousNumSamples = previousStats.numSamples;
    const embree::vfloat<VMM::VectorSize> numSamples = currentNumSamples + previousNumSamples;
    const embree::vfloat<VMM::VectorSize> overallNumSamples = currentStats.numSamples + previousStats.overallNumSamples;


    const embree::vfloat<VMM::VectorSize> currentEstimationWeight = currentNumSamples / numSamples;
    const embree::vfloat<VMM::VectorSize> previousEstimationWeight = 1.0f - currentEstimationWeight;

    const embree::vfloat<VMM::VectorSize> meanCosinePrior = cfg.meanCosinePrior;
    const embree::vfloat<VMM::VectorSize> meanCosinePriorStrength = cfg.meanCosinePriorStrength;
    const embree::vfloat<VMM::VectorSize> maxMeanCosine = cfg.maxMeanCosine;
    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k ++)
    {
        //const vfloat<VMM::VectorSize> partialNumSamples = vmm._weights[k] * numSamples;
        const embree::vfloat<VMM::VectorSize> partialNumSamples = vmm._weights[k] * overallNumSamples;
        embree::Vec3< embree::vfloat<VMM::VectorSize> > currentMeanDirection;
        currentMeanDirection.x = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].x / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.y = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].y / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.z = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].z / currentStats.sumOfWeightedStats[k], 0.0f);

        // TODO: find a better design to precompute the previousMeanDirection
        embree::Vec3< embree::vfloat<VMM::VectorSize> > previousMeanDirection;
        previousMeanDirection.x = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].x / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.y = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].y / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.z = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].z / previousStats.sumOfWeightedStats[k], 0.0f);

        embree::Vec3< embree::vfloat<VMM::VectorSize> > meanDirection =  currentMeanDirection * currentEstimationWeight
            + previousMeanDirection * previousEstimationWeight;

        embree::vfloat<VMM::VectorSize> meanCosine = length(meanDirection);

        vmm._meanDirections[k].x = select(meanCosine > 0.0f, meanDirection.x / meanCosine, vmm._meanDirections[k].x);
        vmm._meanDirections[k].y = select(meanCosine > 0.0f, meanDirection.y / meanCosine, vmm._meanDirections[k].y);
        vmm._meanDirections[k].z = select(meanCosine > 0.0f, meanDirection.z / meanCosine, vmm._meanDirections[k].z);

        meanCosine = ( meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples ) / ( meanCosinePriorStrength + partialNumSamples );

        meanCosine = embree::min( maxMeanCosine, meanCosine );
        vmm._meanCosines[k] = meanCosine;
        vmm._kappas[k] = MeanCosineToKappa< embree::vfloat<VMM::VectorSize> >( meanCosine );
    }

    // TODO: find better more efficient way
    if ( rem > 0 )
    {
        for ( size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._meanDirections[cnt-1].x[i] = 0.0f;
            vmm._meanDirections[cnt-1].y[i] = 0.0f;
            vmm._meanDirections[cnt-1].z[i] = 1.0f;

            vmm._meanCosines[cnt-1][i] = 0.0f;
            vmm._kappas[cnt-1][i] = 0.0f;

            vmm._normalizations[cnt-1][i] = ONE_OVER_FOUR_PI;
            vmm._eMinus2Kappa[cnt-1][i] = 1.0f;
        }
    }

    vmm._calculateNormalization();
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::weightedMaximumAPosteriorStep(VMM &vmm,
        const SufficientStatisitcs &currentStats,
        const SufficientStatisitcs &previousStats,
        const Configuration &cfg) const
{
    // Estimating components weights
    estimateMAPWeights( vmm, currentStats, previousStats, cfg.weightPrior );

    // Estimating mean and concentration
    estimateMAPMeanDirectionAndConcentration( vmm, currentStats, previousStats, cfg);
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::partialWeightedMaximumAPosteriorStep(VMM &vmm,
        const PartialFittingMask &mask,
        SufficientStatisitcs &currentStats,
        SufficientStatisitcs &previousStats,
        const Configuration &cfg) const
{
    // Estimating components weights
    estimatePartialMAPWeights( vmm, mask, currentStats, previousStats, cfg.weightPrior );

    // Estimating mean and concentration
    estimatePartialMAPMeanDirectionAndConcentration( vmm, mask, currentStats, previousStats, cfg);
}


template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::estimatePartialMAPWeights( VMM &vmm,
        const PartialFittingMask &mask,
        SufficientStatisitcs &currentStats,
        SufficientStatisitcs &previousStats,
        const float &_weightPrior ) const
{
    const embree::vfloat<VMM::VectorSize> zeros(0.0f);
    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;

    const size_t numComponents = vmm._numComponents;

    const embree::vfloat<VMM::VectorSize> weightPrior(_weightPrior);

    const embree::vfloat<VMM::VectorSize> numSamples = currentStats.numSamples + previousStats.numSamples;
    //const vfloat<VMM::VectorSize> numSamples = currentStats.numSamples + previousStats.overallNumSamples;

    embree::vfloat<VMM::VectorSize> sumWeights(0.0f);
    embree::vfloat<VMM::VectorSize> sumPartialWeights(0.0f);

    for ( size_t k = 0; k < cnt; k ++ )
    {
        //_sumWeights += currentStats.sumOfWeightedStats[k];
        embree::vfloat<VMM::VectorSize>  weight = ( currentStats.sumOfWeightedStats[k] + previousStats.sumOfWeightedStats[k] ) ;
        weight = ( weightPrior + ( weight ) ) / (( weightPrior * numComponents ) + numSamples );
        //vfloat<VMM::VectorSize>  weight = ( currentStats.sumOfWeightedStats[k]/* + previousStats.sumOfWeightedStats[k]*/ ) / ( sumWeights );
        //weight = ( weightPrior + ( weight * numSamples ) ) / (( weightPrior * numComponents ) + numSamples );

        sumPartialWeights += select(mask.mask[k], weight, zeros);
        sumWeights += select(mask.mask[k], zeros, vmm._weights[k]);

        vmm._weights[k] =select(mask.mask[k], weight,  vmm._weights[k]);
    }

    embree::vfloat<VMM::VectorSize> inv_sumPartialWeights = 1.0f / reduce_add(sumPartialWeights);
    inv_sumPartialWeights *= 1.0f - reduce_add(sumWeights);
    for ( size_t k = 0; k < cnt; k ++ )
    {
        vmm._weights[k] =select(mask.mask[k], vmm._weights[k] * inv_sumPartialWeights,  vmm._weights[k]);
    }

    // TODO: find better more efficient way
    if ( vmm._numComponents % VMM::VectorSize > 0 )
    {
            for (size_t i = vmm._numComponents % VMM::VectorSize; i < VMM::VectorSize; i++ )
            {
                vmm._weights[cnt-1][i] = 0.0f;
            }
    }
}

template<class TVMMDistribution>
void WeightedEMVonMisesFisherFactory< TVMMDistribution>::estimatePartialMAPMeanDirectionAndConcentration( VMM &vmm,
        const PartialFittingMask &mask,
        SufficientStatisitcs &currentStats,
        SufficientStatisitcs &previousStats ,
        const Configuration &cfg) const
{
    const embree::vfloat<VMM::VectorSize> currentNumSamples = currentStats.numSamples;
    const embree::vfloat<VMM::VectorSize> previousNumSamples = previousStats.numSamples;
    const embree::vfloat<VMM::VectorSize> numSamples = currentNumSamples + previousNumSamples;
    const embree::vfloat<VMM::VectorSize> overallNumSamples = currentStats.numSamples + previousStats.overallNumSamples;


    const embree::vfloat<VMM::VectorSize> currentEstimationWeight = currentNumSamples / numSamples;
    const embree::vfloat<VMM::VectorSize> previousEstimationWeight = 1.0f - currentEstimationWeight;

    const embree::vfloat<VMM::VectorSize> meanCosinePrior = cfg.meanCosinePrior;
    const embree::vfloat<VMM::VectorSize> meanCosinePriorStrength = cfg.meanCosinePriorStrength;
    const embree::vfloat<VMM::VectorSize> maxMeanCosine = cfg.maxMeanCosine;
    const int cnt = (vmm._numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    const int rem = vmm._numComponents % VMM::VectorSize;

    for (size_t k = 0; k < cnt; k ++)
    {
        //const vfloat<VMM::VectorSize> partialNumSamples = vmm._weights[k] * numSamples;
        const embree::vfloat<VMM::VectorSize> partialNumSamples = vmm._weights[k] * overallNumSamples;
        embree::Vec3< embree::vfloat<VMM::VectorSize> > currentMeanDirection;
        currentMeanDirection.x = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].x / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.y = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].y / currentStats.sumOfWeightedStats[k], 0.0f);
        currentMeanDirection.z = select(currentStats.sumOfWeightedStats[k] > 0.0f, currentStats.sumOfWeightedDirections[k].z / currentStats.sumOfWeightedStats[k], 0.0f);

        // TODO: find a better design to precompute the previousMeanDirection
        embree::Vec3< embree::vfloat<VMM::VectorSize> > previousMeanDirection;
        previousMeanDirection.x = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].x / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.y = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].y / previousStats.sumOfWeightedStats[k], 0.0f);
        previousMeanDirection.z = select(previousStats.sumOfWeightedStats[k] > 0.0f, previousStats.sumOfWeightedDirections[k].z / previousStats.sumOfWeightedStats[k], 0.0f);

        embree::Vec3< embree::vfloat<VMM::VectorSize> > meanDirection =  currentMeanDirection * currentEstimationWeight
            + previousMeanDirection * previousEstimationWeight;

        embree::vfloat<VMM::VectorSize> meanCosine = embree::length(meanDirection);

        vmm._meanDirections[k].x = select( mask.mask[k], select(meanCosine > 0.0f, meanDirection.x / meanCosine, vmm._meanDirections[k].x) , vmm._meanDirections[k].x);
        vmm._meanDirections[k].y = select( mask.mask[k], select(meanCosine > 0.0f, meanDirection.y / meanCosine, vmm._meanDirections[k].y), vmm._meanDirections[k].y);
        vmm._meanDirections[k].z = select( mask.mask[k], select(meanCosine > 0.0f, meanDirection.z / meanCosine, vmm._meanDirections[k].z), vmm._meanDirections[k].z);

        meanCosine = ( meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples ) / ( meanCosinePriorStrength + partialNumSamples );

        meanCosine = embree::min( maxMeanCosine, meanCosine );
        vmm._meanCosines[k] = select( mask.mask[k], meanCosine, vmm._meanCosines[k]);
        vmm._kappas[k] = select( mask.mask[k], MeanCosineToKappa< embree::vfloat<VMM::VectorSize> >( meanCosine ), vmm._kappas[k]);
    }

    // TODO: find better more efficient way
    if ( rem > 0 )
    {
        for ( size_t i = rem; i < VMM::VectorSize; i++)
        {
            vmm._meanDirections[cnt-1].x[i] = 0.0f;
            vmm._meanDirections[cnt-1].y[i] = 0.0f;
            vmm._meanDirections[cnt-1].z[i] = 1.0f;

            vmm._meanCosines[cnt-1][i] = 0.0f;
            vmm._kappas[cnt-1][i] = 0.0f;
        }
    }

    vmm._calculateNormalization();
}

}