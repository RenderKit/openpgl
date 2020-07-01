// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "VMM.h"
#include "../data/DirectionalSampleData.h"

#include "VMMFactory.h"

namespace rkguide
{

template<int VecSize, int maxComponents>
struct WeightedEMVonMisesFisherFactory: public VonMisesFisherFactory< VecSize, maxComponents>
{
    public:
    typedef std::integral_constant<size_t, (maxComponents + (VecSize -1)) / VecSize> NumVectors;

    struct Configuration
    {
        size_t maK {maxComponents};
        size_t maxEMIterrations {100};

        float maxKappa {10000.0f};
        float maxMeanCosine { KappaToMeanCosine<float>(10000.0f)};
        float convergenceThreshold {0.1f};

        // MAP prior parameters
        // weight prior
        float weightPrior{0.1f};

        // concentration/meanCosine prior
        float meanCosinePriorStrength {0.1f};
        float meanCosinePrior {0.0f};

        void init();

        std::string toString() const;

    };

    struct SufficientStatisitcs
    {
        embree::Vec3< vfloat<VecSize> > sumOfWeightedDirections[NumVectors::value];
        vfloat<VecSize> sumOfWeightedStats[NumVectors::value];

        float sumWeights {0.f};
        float numSamples {0};
        size_t numComponents {maxComponents};

        void clear(size_t _numComponents);

        void clearAll();

        std::string toString() const;

    };

public:

    typedef VonMisesFisherMixture<VecSize, maxComponents> VMM;

    WeightedEMVonMisesFisherFactory();

    void fitMixture(VMM &vmm, size_t numComponents, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg) const;

    void updateMixture(VMM &vmm, SufficientStatisitcs &previousStats, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg) const;

private:

    float weightedExpectationStep(VMM &vmm, SufficientStatisitcs &stats, const DirectionalSampleData* samples, const size_t numSamples) const;

    void weightedMaximumAPosteriorStep(VMM &vmm, SufficientStatisitcs &previousStats,
        SufficientStatisitcs &currentStats,
        const Configuration &cfg) const;

    void estimateMAPWeights( VMM &vmm, SufficientStatisitcs &currentStats, SufficientStatisitcs &previousStats, const float &_weightPrior ) const;

    void estimateMAPMeanDirectionAndConcentration( VMM &vmm, SufficientStatisitcs &currentStats, SufficientStatisitcs &previousStats, const Configuration &cfg) const;


};


template<int VecSize, int maxComponents>
WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::WeightedEMVonMisesFisherFactory()
{
    typename VonMisesFisherFactory<VecSize, maxComponents>::VonMisesFisherFactory( );
}


template<int VecSize, int maxComponents>
void WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::fitMixture(VMM &vmm, size_t numComponents, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg) const
{
    //VonMisesFisherFactory< VecSize, maxComponents>::InitUniformVMM( vmm, numComponents, 5.0f);
    this->InitUniformVMM( vmm, numComponents, 50.0f);
    SufficientStatisitcs stats;
    stats.clear(numComponents);
    stats.clearAll();
    updateMixture(vmm, stats, samples, numSamples, cfg);

}

template<int VecSize, int maxComponents>
void WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::updateMixture(VMM &vmm, SufficientStatisitcs &previousStats, const DirectionalSampleData* samples, const size_t numSamples, const Configuration &cfg) const
{
    SufficientStatisitcs currentStats;
    //stats.clear();
    size_t currentEMIteration = 0;
    bool converged = false;
    float logLikelihood;
    while ( !converged  && currentEMIteration < cfg.maxEMIterrations )
    {
        logLikelihood = weightedExpectationStep( vmm, currentStats, samples, numSamples);
        weightedMaximumAPosteriorStep( vmm, currentStats, previousStats, cfg);
        currentEMIteration++;
    }



}

template<int VecSize, int maxComponents>
void WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::Configuration::init()
{
    maxMeanCosine  = KappaToMeanCosine<float>(maxKappa);
}

template<int VecSize, int maxComponents>
std::string WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::Configuration::toString() const
{
    std::stringstream ss;
    ss << "Configuration:" << std::endl;
    ss << "\tmaxComponents:" << maxComponents << std::endl;
    ss << "\tmaxEMIterrations:" << maxEMIterrations << std::endl;
    ss << "\tmaxKappa:" << maxKappa << std::endl;
    ss << "\tmaxMeanCosine:" << maxMeanCosine << std::endl;
    ss << "\tconvergenceThreshold:" << convergenceThreshold << std::endl;
    ss << "\tweightPrior:" << weightPrior << std::endl;
    ss << "\tmeanCosinePriorStrength:" << meanCosinePriorStrength << std::endl;
    ss << "\tmeanCosinePrior:" << meanCosinePrior << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents>
void WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::SufficientStatisitcs::clear(size_t _numComponents)
{
    numComponents = _numComponents;
    const int cnt = (numComponents+VecSize-1) / VecSize;

    for(int k = 0; k < cnt;k++)
    {
        sumOfWeightedDirections[k] = embree::Vec3< vfloat<VecSize> >(0.0f);
        sumOfWeightedStats[k] = vfloat<VecSize>(0.0f);
    }

    sumWeights = 0.0f;
    numSamples = 0;
}

template<int VecSize, int maxComponents>
void WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::SufficientStatisitcs::clearAll()
{
     clear(maxComponents);
}

template<int VecSize, int maxComponents>
std::string WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::SufficientStatisitcs::toString() const
{
    std::stringstream ss;
    ss << "SufficientStatisitcs:" << std::endl;
    ss << "\tsumWeights:" << sumWeights << std::endl;
    ss << "\tnumSamples:" << numSamples << std::endl;
    ss << "\tnumComponents:" << numComponents << std::endl;
    for (size_t k = 0; k < numComponents ; k++)
    {
        int i = k / VecSize;
        int j = k % VecSize;
        ss  << "\tstat["<< k <<"]:" << "\tsumWeightedStats: " << sumOfWeightedStats[i][j]
            << "\tsumWeightedDirections: [" << sumOfWeightedDirections[i].x[j] << ",\t"
            << sumOfWeightedDirections[i].y[j] << ",\t" << sumOfWeightedDirections[i].z[j] << "]"
            << std::endl;
    }
    return ss.str();
}

template<int VecSize, int maxComponents>
float WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::weightedExpectationStep(VMM &vmm,
        SufficientStatisitcs &stats,
        const DirectionalSampleData* samples,
        const size_t numSamples) const
{
    stats.clear(vmm._numComponents);
    stats.clearAll();
    stats.numComponents = vmm._numComponents;
    stats.numSamples = numSamples;

    const int cnt = (stats.numComponents+VecSize-1) / VecSize;

    typename VMM::SoftAssignment softAssign;

    for (size_t n = 0; n < numSamples; n++ )
    {
        const DirectionalSampleData sampleData = samples[n];
        const vfloat<VecSize> sampleWeight = sampleData.weight;
        const embree::Vec3< vfloat<VecSize> > sampleDirection( sampleData.direction[0], sampleData.direction[1], sampleData.direction[2] );
        vfloat<VecSize> sumSoft(0.f);

        if ( !vmm.softAssignment( sampleData.direction, softAssign) )
        {
            std::cout << "continue" << std::endl;
            continue;
        }

        for (size_t k =0; k < cnt; k++)
        {
            stats.sumOfWeightedDirections[k] += sampleDirection * softAssign.assignments[k] * sampleWeight;
            stats.sumOfWeightedStats[k] += softAssign.assignments[k] * sampleWeight;
            sumSoft += softAssign.assignments[k];
        }

        //std::cout << "sumSoftAssignment: " << reduce_add(sumSoft)<< std::endl;

    }

    for (size_t k =0; k < cnt; k++)
    {
        stats.sumWeights += reduce_add(stats.sumOfWeightedStats[k]);
    }

    // TODO: calculate summedLogLikelihood
    return 1.0f;
}

template<int VecSize, int maxComponents>
void WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::estimateMAPWeights( VMM &vmm,
        SufficientStatisitcs &currentStats,
        SufficientStatisitcs &previousStats,
        const float &_weightPrior ) const
{
    const int cnt = (vmm._numComponents+VecSize-1) / VecSize;

    const size_t numComponents = vmm._numComponents;

    const vfloat<VecSize> weightPrior(_weightPrior);

    const vfloat<VecSize> currentNumSamples = currentStats.numSamples;
    const vfloat<VecSize> previousNumSamples = previousStats.numSamples;
    const vfloat<VecSize> numSamples = currentNumSamples + previousNumSamples;

    const vfloat<VecSize> currentSumWeights = currentStats.sumWeights;
    const vfloat<VecSize> previousSumWeights = previousStats.sumWeights;
    const vfloat<VecSize> sumWeights = currentSumWeights + previousSumWeights;


    size_t k;

    vfloat<VecSize> _sumWeights(0.0f);
    for (k = 0; k < cnt; k ++)
    {
        _sumWeights += currentStats.sumOfWeightedStats[k];
        vfloat<VecSize>  weight = ( currentStats.sumOfWeightedStats[k]/* + previousStats.sumOfWeightedStats[k]*/ ) / ( sumWeights );
        //weight = ( weightPrior + ( weight * numSamples ) ) / (( weightPrior * numComponents ) + numSamples );
        vmm._weights[k] = weight;
    }

    if ( vmm._numComponents % VecSize > 0 )
    {
            for (size_t i = vmm._numComponents % VecSize; i <= VecSize; i++ )
            {
                vmm._weights[k-1][i-1] = 0.0f;
            }
    }

    std::cout << "_sumWeights: " << reduce_add(_sumWeights) << "\tsumWeights: " << currentStats.sumWeights << std::endl;
    std::cout << "_sumWeights: " << reduce_add(_sumWeights) << "\tsumWeights: " << previousStats.sumWeights << std::endl;


}

template<int VecSize, int maxComponents>
void WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::estimateMAPMeanDirectionAndConcentration( VMM &vmm,
        SufficientStatisitcs &currentStats,
        SufficientStatisitcs &previousStats ,
        const Configuration &cfg) const
{
    const vfloat<VecSize> currentNumSamples = currentStats.numSamples;
    const vfloat<VecSize> previousNumSamples = previousStats.numSamples;
    const vfloat<VecSize> numSamples = currentNumSamples + previousNumSamples;

    const vfloat<VecSize> currentSumWeights = currentStats.sumWeights;
    const vfloat<VecSize> previousSumWeights = previousStats.sumWeights;
    const vfloat<VecSize> sumWeights = currentSumWeights + previousSumWeights;

    const vfloat<VecSize> meanCosinePrior = cfg.meanCosinePrior;
    const vfloat<VecSize> meanCosinePriorStrength = cfg.meanCosinePriorStrength;
    const vfloat<VecSize> maxMeanCosine = cfg.maxMeanCosine;
    const int cnt = (vmm._numComponents+VecSize-1) / VecSize;

    for (size_t k = 0; k < cnt; k ++)
    {
        const vfloat<VecSize> partialNumSamples = vmm._weights[k] * numSamples;
        embree::Vec3< vfloat<VecSize> > meanDirection =  currentStats.sumOfWeightedDirections[k] / currentStats.sumOfWeightedStats[k] ;/*+ previousStats.sumOfWeightedDirections[k] ) / ( sumWeights ); */
        vfloat<VecSize> meanCosine = length(meanDirection);

        vmm._meanDirections[k].x = select(meanCosine > 0.0f, meanDirection.x / meanCosine, vmm._meanDirections[k].x);
        vmm._meanDirections[k].y = select(meanCosine > 0.0f, meanDirection.y / meanCosine, vmm._meanDirections[k].y);
        vmm._meanDirections[k].z = select(meanCosine > 0.0f, meanDirection.z / meanCosine, vmm._meanDirections[k].z);

        meanCosine = ( meanCosinePrior * meanCosinePriorStrength + meanCosine * partialNumSamples ) / ( meanCosinePriorStrength + partialNumSamples );
        meanCosine = embree::min( maxMeanCosine, meanCosine );
        vmm._meanCosines[k] = meanCosine;
        vmm._kappas[k] = MeanCosineToKappa< vfloat<VecSize> >( meanCosine );
    }
    vmm._calculateNormalization();
}

template<int VecSize, int maxComponents>
void WeightedEMVonMisesFisherFactory< VecSize, maxComponents>::weightedMaximumAPosteriorStep(VMM &vmm,
        SufficientStatisitcs &currentStats,
        SufficientStatisitcs &previousStats,
        const Configuration &cfg) const
{
    // Estimating components weights
    estimateMAPWeights( vmm, currentStats, previousStats, cfg.weightPrior );

    // Estimating mean and concentration
    estimateMAPMeanDirectionAndConcentration( vmm, currentStats, previousStats, cfg);
}



}