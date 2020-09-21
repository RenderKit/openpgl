// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "VMM.h"
#include "../data/DirectionalSampleData.h"

#include "WeightedEMVMMFactory.h"
#include "VMMChiSquareComponentSplitter.h"
#include "VMMChiSquareComponentMerger.h"

namespace rkguide
{

template<int VecSize, int maxComponents>
struct AdaptiveSplitAndMergeFactory
{

public:
    typedef VonMisesFisherMixture<VecSize, maxComponents> VMM;
    //typedef std::integral_constant<size_t, (maxComponents + (VecSize -1)) / VecSize> NumVectors;
    //typedef typename WeightedEMVonMisesFisherFactory<VecSize, maxComponents>::SufficientStatisitcs SufficientStatisitcs;
    //typedef typename WeightedEMVonMisesFisherFactory<VecSize, maxComponents>::PartialFittingMask PartialFittingMask;
    typedef WeightedEMVonMisesFisherFactory<VecSize, maxComponents> WeightedEMFactory;
    typedef VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents> Splitter;
    typedef VonMisesFisherChiSquareComponentMerger<VecSize, maxComponents> Merger;


    struct ASMConfiguration
    {
        typename WeightedEMFactory::Configuration weightedEMCfg;

        float splittingThreshold { 0.75 };
        float mergingThreshold { 0.00625 };

        bool partialReFit { false };
        int maxSplitItr { 1 };

        int minSamplesForSplitting { 0 };
        int minSamplesForMerging { 0 };
        //int minSamplesForSplitting { 4096 };
        //int minSamplesForMerging { 4096 * 4 };

        std::string toString() const;
    };


    struct ASMStatistics
    {
        typename WeightedEMFactory::SufficientStatisitcs sufficientStatistics;
        typename Splitter::ComponentSplitStatistics splittingStatistics;

        size_t numSamplesAfterLastSplit {0};
        size_t numSamplesAfterLastMerge {0};
        void clear(const size_t &_numComponents);
        void clearAll();

        void decay(const float &alpha);

        std::string toString() const;

    };


    struct ASMFittingStatistics
    {
        size_t numSamples {0};
        size_t numSplits {0};
        size_t numMerges {0};

        size_t numComponents {0};

        size_t numUpdateWEMIterations {0};
        size_t numPartialUpdateWEMIterations {0};

        std::string toString() const;
    };


    void fit(VMM &vmm, size_t numComponents, ASMStatistics &stats, const DirectionalSampleData* samples, const size_t numSamples, const ASMConfiguration &cfg, ASMFittingStatistics &fitStats) const;

    void update(VMM &vmm, ASMStatistics &stats, const DirectionalSampleData* samples, const size_t numSamples, const ASMConfiguration &cfg, ASMFittingStatistics &fitStats) const;

};

template<int VecSize, int maxComponents>
void AdaptiveSplitAndMergeFactory<VecSize, maxComponents>::ASMStatistics::decay(const float &alpha)
{
    sufficientStatistics.decay(alpha);
    splittingStatistics.decay(alpha);
}

template<int VecSize, int maxComponents>
std::string AdaptiveSplitAndMergeFactory<VecSize, maxComponents>::ASMStatistics::toString() const
{
    return "";
}

template<int VecSize, int maxComponents>
std::string AdaptiveSplitAndMergeFactory<VecSize, maxComponents>::ASMFittingStatistics::toString() const
{
    std::stringstream ss;
    ss << "ASMFittingStatistics:" << std::endl;
    ss << "\tnumSamples:" << numSamples << std::endl;
    ss << "\tnumSplits:" << numSplits << std::endl;
    ss << "\tnumMerges:" << numMerges << std::endl;
    ss << "\tnumComponents:" << numComponents << std::endl;
    ss << "\tnumUpdateWEMIterations:" << numUpdateWEMIterations << std::endl;
    ss << "\tnumPartialUpdateWEMIterations:" << numPartialUpdateWEMIterations << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents>
void AdaptiveSplitAndMergeFactory<VecSize, maxComponents>::ASMStatistics::clear(const size_t &_numComponents)
{
    sufficientStatistics.clear(_numComponents);
    splittingStatistics.clear(_numComponents);

    numSamplesAfterLastSplit = 0;
    numSamplesAfterLastMerge = 0;
}

template<int VecSize, int maxComponents>
void AdaptiveSplitAndMergeFactory<VecSize, maxComponents>::ASMStatistics::clearAll()
{
    clear(maxComponents);
}

template<int VecSize, int maxComponents>
std::string AdaptiveSplitAndMergeFactory<VecSize, maxComponents>::ASMConfiguration::toString() const
{
    return "";
}

template<int VecSize, int maxComponents>
void AdaptiveSplitAndMergeFactory<VecSize, maxComponents>::fit(VMM &vmm, size_t numComponents, ASMStatistics &stats, const DirectionalSampleData* samples, const size_t numSamples, const ASMConfiguration &cfg, ASMFittingStatistics &fitStats) const
{
    // intial fit
    WeightedEMFactory factory = WeightedEMFactory();
    typename WeightedEMFactory::FittingStatistics wemFitStats;
    factory.fitMixture(vmm, numComponents, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);

    // calculate the estimate of the integral of the function (e.g. radiance or importance) fitted by the VMM
    float mcEstimate = stats.sufficientStatistics.sumWeights / stats.sufficientStatistics.numSamples;

    // split the fitted components of the inital fit to match
    // the observed samples

    std::cout << stats.sufficientStatistics.toString() << std::endl;

    Splitter splitter = Splitter();
    splitter.PerformRecursiveSplitting(vmm, stats.sufficientStatistics, cfg.splittingThreshold, mcEstimate, samples, numSamples, cfg.weightedEMCfg);

    splitter.CalculateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);

    //std::cout << stats.sufficientStatistics.toString() << std::endl;

    RKGUIDE_ASSERT(vmm._numComponents == stats.sufficientStatistics.numComponents);
    RKGUIDE_ASSERT(vmm._numComponents == stats.splittingStatistics.numComponents);

    Merger merger = Merger();
    merger.PerformMerging(vmm, cfg.mergingThreshold, stats.sufficientStatistics, stats.splittingStatistics);

    //stats.splittingStatistics.clear(vmm._numComponents);
    stats.numSamplesAfterLastSplit = 0.0f;
    stats.numSamplesAfterLastMerge = 0.0f;
}



template<int VecSize, int maxComponents>
void AdaptiveSplitAndMergeFactory<VecSize, maxComponents>::update(VMM &vmm, ASMStatistics &stats, const DirectionalSampleData* samples, const size_t numSamples, const ASMConfiguration &cfg, ASMFittingStatistics &fitStats) const
{
    
    //std::cout << "Before UPDATE: "<< std::endl;
    //std::cout << vmm.toString()<< std::endl;
    RKGUIDE_ASSERT(vmm.isValid());

    // first update the mixture
    WeightedEMFactory factory = WeightedEMFactory();
    typename WeightedEMFactory::FittingStatistics wemFitStats;
    //stats.sufficientStatistics.clear(vmm._numComponents);
    factory.updateMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);

    float mcEstimate = stats.sufficientStatistics.sumWeights / stats.sufficientStatistics.numSamples;
    //std::cout << "After UPDATE: "<< std::endl;
    //std::cout << vmm.toString()<< std::endl;
    RKGUIDE_ASSERT(vmm.isValid());

    fitStats.numSamples = numSamples;
    fitStats.numUpdateWEMIterations = wemFitStats.numIterations;

    //stats.splittingStatistics.clear(vmm._numComponents);

/* 
    if (stats.numSamplesAfterLastMerge >= cfg.minSamplesForMerging)
    //if(stats.numSamplesAfterLastMerge % 4 == 5)
    {
        Merger merger = Merger();
        size_t numMerges = merger.PerformMerging(vmm, cfg.mergingThreshold, stats.sufficientStatistics, stats.splittingStatistics);
        fitStats.numMerges = numMerges;
        stats.numSamplesAfterLastSplit = 0.0f;
        stats.numSamplesAfterLastMerge = 0.0f;
        stats.splittingStatistics.clear(vmm._numComponents);
    }
*/
//    Merger merger = Merger();
//    merger.PerformMerging(vmm, cfg.mergingThreshold, stats.sufficientStatistics, stats.splittingStatistics);


    Splitter splitter = Splitter();
    //stats.splittingStatistics.clear(vmm._numComponents);
    RKGUIDE_ASSERT(stats.splittingStatistics.isValid());
    splitter.UpdateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);
    RKGUIDE_ASSERT(stats.splittingStatistics.isValid());
    //splitter.CalculateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);

    if (stats.numSamplesAfterLastSplit >= cfg.minSamplesForSplitting)
    {
        typename WeightedEMFactory::PartialFittingMask mask;
        mask.resetToFalse();
        //splitStatistics.clearAll();

        std::vector<typename Splitter::SplitCandidate> splitComps = stats.splittingStatistics.getSplitCandidates();
        int totalSplitCount = 0;
        //const size_t numComp = vmm._numComponents;
        for (size_t k = 0; k < splitComps.size(); k++)
        {
            if (splitComps[k].chiSquareEst > cfg.splittingThreshold && vmm._numComponents  < maxComponents)
            {
                bool splitSucess = splitter.SplitComponent(vmm, stats.splittingStatistics, stats.sufficientStatistics, splitComps[k].componentIndex);
                mask.setToTrue(splitComps[k].componentIndex);
                mask.setToTrue(vmm._numComponents-1);
                std::cout << "split[" << totalSplitCount << "]: " << "\tidx0: " << splitComps[k].componentIndex << "\tidx1: " << vmm._numComponents-1 << std::endl;
                totalSplitCount++;

            }
        }

        RKGUIDE_ASSERT(stats.splittingStatistics.isValid());
        RKGUIDE_ASSERT(vmm._numComponents == stats.sufficientStatistics.numComponents);
        RKGUIDE_ASSERT(vmm._numComponents == stats.splittingStatistics.numComponents);

        if (totalSplitCount > 0)
        {
            //std::cout << "Before Partial FIT" << std::endl;
            //std::cout << vmm.toString() << std::endl;
            RKGUIDE_ASSERT(vmm.isValid());

            typename WeightedEMFactory::SufficientStatisitcs tempSuffStatistics;
            tempSuffStatistics.clear(vmm._numComponents);
            factory.partialUpdateMixture(vmm, mask, tempSuffStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
            stats.sufficientStatistics.numComponents = vmm._numComponents;
            stats.sufficientStatistics.maskedReplace(mask, tempSuffStatistics);

            fitStats.numPartialUpdateWEMIterations = wemFitStats.numIterations;
            //std::cout << "After Partial FIT" << std::endl;
            //std::cout << vmm.toString() << std::endl;
            RKGUIDE_ASSERT(vmm.isValid());

        }

        fitStats.numSplits = totalSplitCount;



        //stats.splittingStatistics.clearAll();
        std::cout << "update: totalSplitCount = " << totalSplitCount << "\t splitThreshold: " << cfg.splittingThreshold<< std::endl;

        stats.numSamplesAfterLastSplit = 0.0f;
        //stats.numSamplesAfterLastMerge = numSamples;
        stats.numSamplesAfterLastMerge++;

//    factory.updateMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg);
        //stats.splittingStatistics.clear(vmm._numComponents);
    }
/* */
    RKGUIDE_ASSERT(stats.splittingStatistics.isValid());
    if (stats.numSamplesAfterLastMerge >= cfg.minSamplesForMerging)
    //if(stats.numSamplesAfterLastMerge % 4 == 0)
    {
        Merger merger = Merger();
        size_t numMerges = merger.PerformMerging(vmm, cfg.mergingThreshold, stats.sufficientStatistics, stats.splittingStatistics);
        fitStats.numMerges = numMerges;
        stats.numSamplesAfterLastSplit = 0.0f;
        stats.numSamplesAfterLastMerge = 0.0f;
        //stats.splittingStatistics.clear(vmm._numComponents);
        RKGUIDE_ASSERT(stats.splittingStatistics.isValid());
    }

    fitStats.numComponents = vmm._numComponents;
/* */
    /*
    stats.numSamplesAfterLastSplit += numSamples;
    stats.numSamplesAfterLastMerge += numSamples;


        Merger merger = Merger();
        merger.PerformMerging(vmm, cfg.mergingThreshold);
        stats.numSamplesAfterLastSplit = numSamples;
        stats.numSamplesAfterLastMerge = 0.0f;
        stats.splittingStatistics.clearAll();
    }

    // calculate the estimate of the integral of the function (e.g. radiance or importance) fitted by the VMM
    float mcEstimate = stats.sufficientStatistics.sumWeights / stats.sufficientStatistics.numSamples;

    Splitter splitter = Splitter();
    splitter->UpdateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);

    if (stats.numSamplesAfterLastSplit >= cfg.minSamplesForSplitting)
    {
        std::vector<SplitCandidate> splitComps = stats.splittingStatistics.getSplitCandidates();

        const size_t numComp = vmm._numComponents;
        for (size_t k = 0; k < numComp; k++)
        {
            if (splitComps[k].chiSquareEst > splitThreshold && vmm._numComponents  < maxComponents)
            {
                bool splitSucess = SplitComponent(vmm, stats.splittingStatistics, stats.sufficientStatistics, splitComps[k].componentIndex);
            }
        }

        stats.splittingStatistics.clearAll();

        stats.numSamplesAfterLastSplit = 0.0f;
        stats.numSamplesAfterLastMerge = 0.0f;
    }
    */
}


}

