// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <iostream>

#include "../../data/SampleData.h"
#include "../../include/openpgl/types.h"
#include "../../openpgl_common.h"
#include "ParallaxAwareVonMisesFisherWeightedEMFactory.h"
#include "VMMChiSquareComponentMerger.h"
#include "VMMChiSquareComponentSplitterV2.h"

namespace openpgl
{

template <class TVMMDistribution>
struct AdaptiveSplitAndMergeFactoryV2
{
   public:
    const static PGL_DIRECTIONAL_DISTRIBUTION_TYPE DIRECTIONAL_DISTRIBUTION_TYPE =
        TVMMDistribution::ParallaxCompensation ? PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM_V2 : PGL_DIRECTIONAL_DISTRIBUTION_VMM_V2;

    typedef TVMMDistribution Distribution;
    typedef TVMMDistribution VMM;

    // typedef WeightedEMVonMisesFisherFactory<VMM> WeightedEMFactory;
    typedef ParallaxAwareVonMisesFisherWeightedEMFactory<VMM> WeightedEMFactory;
    typedef VonMisesFisherChiSquareComponentSplitterV2<WeightedEMFactory> Splitter;
    typedef VonMisesFisherChiSquareComponentMerger<WeightedEMFactory, Splitter> Merger;

    struct Configuration
    {
        typename WeightedEMFactory::Configuration weightedEMCfg;

        float splittingThreshold{0.75f};
        float mergingThreshold{0.00625f};

        bool useSplitAndMerge{true};

        bool partialReFit{false};
        int maxSplitItr{1};

        int minSamplesForSplitting{0};
        int minSamplesForPartialRefitting{0};
        int minSamplesForMerging{0};

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        std::string toString() const;

        bool operator==(const Configuration &b) const
        {
            bool equal = true;
            if (splittingThreshold != b.splittingThreshold || mergingThreshold != b.mergingThreshold || useSplitAndMerge != b.useSplitAndMerge || partialReFit != b.partialReFit ||
                maxSplitItr != b.maxSplitItr || minSamplesForSplitting != b.minSamplesForSplitting || minSamplesForPartialRefitting != b.minSamplesForPartialRefitting ||
                minSamplesForMerging != b.minSamplesForMerging || !weightedEMCfg.operator==(b.weightedEMCfg))
            {
                equal = false;
            }
            return equal;
        }
    };

    struct Statistics
    {
        typename WeightedEMFactory::SufficientStatistics sufficientStatistics;
        typename Splitter::ComponentSplitStatistics splittingStatistics;

        // size_t numComponents {0};

        size_t numSamplesAfterLastSplit{0};
        size_t numSamplesAfterLastMerge{0};

        Statistics() = default;

        void clear(const size_t &_numComponents);
        void clearAll();

        void decay(const float &alpha);

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        bool isValid() const;

        inline size_t getNumComponents() const
        {
            OPENPGL_ASSERT(sufficientStatistics.getNumComponents() == splittingStatistics.getNumComponents());
            return sufficientStatistics.getNumComponents();
        }

        std::string toString() const;

        bool operator==(const Statistics &b) const;
    };

    struct FittingStatistics
    {
        size_t numSamples{0};
        size_t numSplits{0};
        size_t numMerges{0};

        size_t numComponents{0};

        size_t numUpdateWEMIterations{0};
        size_t numPartialUpdateWEMIterations{0};

        std::string toString() const;
    };

    void prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics, const Configuration &cfg) const;

    void fit(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void update(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg, FittingStatistics &fitStats) const;

    void updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples, const SampleStatistics &sampleStatistics) const;

    std::string toString() const
    {
        std::ostringstream oss;
        WeightedEMFactory vmmFactory;
        oss << "AdaptiveSplitAndMergeFactoryV2[\n";
        oss << "  VMMFactory: " << vmmFactory.toString() << '\n';
        oss << ']';

        return oss.str();
    }
};

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::serialize(std::ostream &stream) const
{
    sufficientStatistics.serialize(stream);
    splittingStatistics.serialize(stream);

    // stream.write(reinterpret_cast<const char*>(&numComponents), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&numSamplesAfterLastSplit), sizeof(size_t));
    stream.write(reinterpret_cast<const char *>(&numSamplesAfterLastMerge), sizeof(size_t));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::deserialize(std::istream &stream)
{
    sufficientStatistics.deserialize(stream);
    splittingStatistics.deserialize(stream);

    // stream.read(reinterpret_cast<char*>(&numComponents), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&numSamplesAfterLastSplit), sizeof(size_t));
    stream.read(reinterpret_cast<char *>(&numSamplesAfterLastMerge), sizeof(size_t));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::decay(const float &alpha)
{
    sufficientStatistics.decay(alpha);
    splittingStatistics.decay(alpha);
}

template <class TVMMDistribution>
bool AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::isValid() const
{
    bool valid = true;
    valid = valid && sufficientStatistics.isValid();
    OPENPGL_ASSERT(valid);
    valid = valid && splittingStatistics.isValid();
    OPENPGL_ASSERT(valid);
    valid = valid && embree::isvalid(numSamplesAfterLastSplit);
    OPENPGL_ASSERT(valid);
    valid = valid && embree::isvalid(numSamplesAfterLastMerge);
    OPENPGL_ASSERT(valid);

    return valid;
}

template <class TVMMDistribution>
std::string AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::toString() const
{
    std::stringstream ss;
    ss << "Statistics:" << std::endl;
    ss << "\tsufficientStatistics:" << sufficientStatistics.toString() << std::endl;
    ss << "\tsplittingStatistics:" << splittingStatistics.toString() << std::endl;
    ss << "\tnumSamplesAfterLastSplit = " << numSamplesAfterLastSplit << std::endl;
    ss << "\tnumSamplesAfterLastMerge = " << numSamplesAfterLastMerge << std::endl;
    return ss.str();
}

template <class TVMMDistribution>
std::string AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::FittingStatistics::toString() const
{
    std::stringstream ss;
    ss << "FittingStatistics:" << std::endl;
    ss << "\tnumSamples:" << numSamples << std::endl;
    ss << "\tnumSplits:" << numSplits << std::endl;
    ss << "\tnumMerges:" << numMerges << std::endl;
    ss << "\tnumComponents:" << numComponents << std::endl;
    ss << "\tnumUpdateWEMIterations:" << numUpdateWEMIterations << std::endl;
    ss << "\tnumPartialUpdateWEMIterations:" << numPartialUpdateWEMIterations << std::endl;
    return ss.str();
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::clear(const size_t &_numComponents)
{
    sufficientStatistics.clear(_numComponents);
    splittingStatistics.clear(_numComponents);

    // numComponents = _numComponents;
    numSamplesAfterLastSplit = 0;
    numSamplesAfterLastMerge = 0;
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::clearAll()
{
    clear(VMM::MaxComponents);
}

template <class TVMMDistribution>
bool AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Statistics::operator==(const Statistics &b) const
{
    bool equal = true;
    if (numSamplesAfterLastSplit != b.numSamplesAfterLastSplit || numSamplesAfterLastMerge != b.numSamplesAfterLastMerge)
    {
        equal = false;
    }

    if (!sufficientStatistics.operator==(b.sufficientStatistics) || !splittingStatistics.operator==(b.splittingStatistics))
    {
        equal = false;
    }
    return equal;
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Configuration::serialize(std::ostream &stream) const
{
    weightedEMCfg.serialize(stream);

    stream.write(reinterpret_cast<const char *>(&splittingThreshold), sizeof(float));
    stream.write(reinterpret_cast<const char *>(&mergingThreshold), sizeof(float));

    stream.write(reinterpret_cast<const char *>(&partialReFit), sizeof(bool));
    stream.write(reinterpret_cast<const char *>(&maxSplitItr), sizeof(int));

    stream.write(reinterpret_cast<const char *>(&minSamplesForSplitting), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&minSamplesForMerging), sizeof(int));
    stream.write(reinterpret_cast<const char *>(&minSamplesForPartialRefitting), sizeof(int));
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Configuration::deserialize(std::istream &stream)
{
    weightedEMCfg.deserialize(stream);

    stream.read(reinterpret_cast<char *>(&splittingThreshold), sizeof(float));
    stream.read(reinterpret_cast<char *>(&mergingThreshold), sizeof(float));

    stream.read(reinterpret_cast<char *>(&partialReFit), sizeof(bool));
    stream.read(reinterpret_cast<char *>(&maxSplitItr), sizeof(int));

    stream.read(reinterpret_cast<char *>(&minSamplesForSplitting), sizeof(int));
    stream.read(reinterpret_cast<char *>(&minSamplesForMerging), sizeof(int));
    stream.read(reinterpret_cast<char *>(&minSamplesForPartialRefitting), sizeof(int));
}

template <class TVMMDistribution>
std::string AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::Configuration::toString() const
{
    std::stringstream ss;
    ss << "Configuration:" << std::endl;
    ss << "\tweightedEMCfg = " << weightedEMCfg.toString() << std::endl;
    ss << "\tsplittingThreshold = " << splittingThreshold << std::endl;
    ss << "\tmergingThreshold = " << mergingThreshold << std::endl;
    ss << "\tuseSplitAndMerge = " << useSplitAndMerge << std::endl;
    ss << "\tpartialReFit = " << partialReFit << std::endl;
    ss << "\tmaxSplitItr = " << maxSplitItr << std::endl;
    ss << "\tminSamplesForSplitting = " << minSamplesForSplitting << std::endl;
    ss << "\tminSamplesForPartialRefitting = " << minSamplesForPartialRefitting << std::endl;
    ss << "\tminSamplesForMerging = " << minSamplesForMerging << std::endl;
    return ss.str();
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::prepareSamples(SampleData *samples, const size_t numSamples, const SampleStatistics &sampleStatistics,
                                                                    const Configuration &cfg) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.prepareSamples(samples, numSamples, sampleStatistics, cfg.weightedEMCfg);
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::fit(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                         FittingStatistics &fitStats) const
{
    const size_t numComponents = cfg.weightedEMCfg.initK;
    stats.clear(numComponents);
    // intial fit
    WeightedEMFactory factory = WeightedEMFactory();
    typename WeightedEMFactory::FittingStatistics wemFitStats;
    factory.fitMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
    factory.initComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(vmm.getNumComponents() == stats.sufficientStatistics.getNumComponents());
    OPENPGL_ASSERT(stats.isValid());
    /* */

    if (cfg.useSplitAndMerge)
    {
        // calculate the estimate of the integral of the function (e.g. radiance or importance) fitted by the VMM
        float mcEstimate = stats.sufficientStatistics.getSumWeights() / stats.sufficientStatistics.getNumSamples();

        // split the fitted components of the inital fit to match
        // the observed samples
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << stats.sufficientStatistics.toString() << std::endl;
#endif
        Splitter splitter = Splitter();
        splitter.PerformRecursiveSplitting(vmm, stats.sufficientStatistics, cfg.splittingThreshold, mcEstimate, samples, numSamples, cfg.weightedEMCfg);

        splitter.CalculateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);

        OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
        OPENPGL_ASSERT(vmm.isValid());

        Merger merger = Merger();
        merger.PerformMerging(vmm, cfg.mergingThreshold, cfg.splittingThreshold, false, stats.sufficientStatistics, stats.splittingStatistics);
        OPENPGL_ASSERT(vmm.isValid());
    }

    stats.numSamplesAfterLastSplit = 0.0f;
    stats.numSamplesAfterLastMerge = 0.0f;

    factory.initComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
    OPENPGL_ASSERT(vmm.isValid());
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::update(VMM &vmm, Statistics &stats, const SampleData *samples, const size_t numSamples, const Configuration &cfg,
                                                            FittingStatistics &fitStats) const
{
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
    OPENPGL_ASSERT(stats.isValid());

    // first update the mixture
    WeightedEMFactory factory = WeightedEMFactory();
    typename WeightedEMFactory::FittingStatistics wemFitStats;
    // stats.sufficientStatistics.clear(vmm._numComponents);
    const size_t prevNumberOfComponents = vmm._numComponents;
    factory.updateMixture(vmm, stats.sufficientStatistics, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
    OPENPGL_ASSERT(vmm.isValid());
    // check if the update step added a new component.
    // This happnes if samples are not covered by any existing component
    // and we need to extend the splittingStats.
    if (prevNumberOfComponents < vmm._numComponents)
    {
        stats.splittingStatistics.setNumComponents(vmm._numComponents);
    }
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());

    if (cfg.useSplitAndMerge)
    {
        float mcEstimate = stats.sufficientStatistics.getSumWeights() / stats.sufficientStatistics.getNumSamples();

        fitStats.numSamples = numSamples;
        fitStats.numUpdateWEMIterations = wemFitStats.numIterations;

        stats.numSamplesAfterLastSplit += numSamples;
        stats.numSamplesAfterLastMerge += numSamples;


        Splitter splitter = Splitter();
        // OPENPGL_ASSERT(stats.splittingStatistics.isValid());
        splitter.UpdateSplitStatistics(vmm, stats.splittingStatistics, mcEstimate, samples, numSamples);
        OPENPGL_ASSERT(stats.splittingStatistics.isValid());

        if (stats.numSamplesAfterLastSplit >= cfg.minSamplesForSplitting)
        {
            typename WeightedEMFactory::PartialFittingMask mask;
            mask.resetToFalse();

            std::vector<typename Splitter::SplitCandidate> splitComps = stats.splittingStatistics.getSplitCandidates(cfg.splittingThreshold);
            int totalSplitCount = 0;
            // const size_t numComp = vmm._numComponents;
            for (size_t k = 0; k < splitComps.size(); k++)
            {
                if (splitComps[k].chiSquareEst > cfg.splittingThreshold && vmm._numComponents < VMM::MaxComponents)
                {
                    splitter.SplitComponent(vmm, stats.splittingStatistics, stats.sufficientStatistics, splitComps[k].componentIndex);
                    mask.setToTrue(splitComps[k].componentIndex);
                    mask.setToTrue(vmm._numComponents - 1);
                    // std::cout << "split[" << totalSplitCount << "]: " << "\tidx0: " << splitComps[k].componentIndex << "\tidx1: " << vmm._numComponents-1 << std::endl;
                    totalSplitCount++;
                }
            }

            OPENPGL_ASSERT(vmm.isValid());
            OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
            OPENPGL_ASSERT(stats.isValid());

            if (totalSplitCount > 0 && cfg.partialReFit && numSamples >= cfg.minSamplesForPartialRefitting)
            {
                factory.partialUpdateMixture(vmm, mask, stats.sufficientStatistics, false, samples, numSamples, cfg.weightedEMCfg, wemFitStats);
                // update number of components for the splitStats to
                // account for additionaly added componetes based on not covered samples.
                stats.splittingStatistics.setNumComponents(vmm._numComponents);
                fitStats.numPartialUpdateWEMIterations = wemFitStats.numIterations;
                OPENPGL_ASSERT(vmm.isValid());
                OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
                OPENPGL_ASSERT(stats.isValid());
            }

            fitStats.numSplits = totalSplitCount;
            stats.numSamplesAfterLastSplit = 0.0f;

#ifdef OPENPGL_SHOW_PRINT_OUTS
            std::cout << "update: totalSplitCount = " << totalSplitCount << "\t splitThreshold: " << cfg.splittingThreshold << std::endl;
#endif
        }

        OPENPGL_ASSERT(vmm.isValid());
        OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
        OPENPGL_ASSERT(stats.isValid());

        if (stats.numSamplesAfterLastMerge >= cfg.minSamplesForMerging)
        {
            Merger merger = Merger();
            size_t numMerges = merger.PerformMerging(vmm, cfg.mergingThreshold, cfg.splittingThreshold, false, stats.sufficientStatistics, stats.splittingStatistics);
            fitStats.numMerges = numMerges;
            stats.numSamplesAfterLastMerge = 0.0f;

            OPENPGL_ASSERT(vmm.isValid());
            OPENPGL_ASSERT(vmm.getNumComponents() == stats.getNumComponents());
            OPENPGL_ASSERT(stats.isValid());
        }

        fitStats.numComponents = vmm._numComponents;
    }
    OPENPGL_ASSERT(vmm.isValid());
    factory.updateComponentDistances(vmm, stats.sufficientStatistics, samples, numSamples);

    OPENPGL_ASSERT(vmm.getNumComponents() == stats.sufficientStatistics.getNumComponents());
    OPENPGL_ASSERT(vmm.isValid());
    OPENPGL_ASSERT(stats.sufficientStatistics.isValid());
}

template <class TVMMDistribution>
void AdaptiveSplitAndMergeFactoryV2<TVMMDistribution>::updateFluenceEstimate(VMM &vmm, const SampleData *samples, const size_t numSamples, const size_t numZeroValueSamples,
                                                                           const SampleStatistics &sampleStatistics) const
{
    WeightedEMFactory factory = WeightedEMFactory();
    factory.updateFluenceEstimate(vmm, samples, numSamples, numZeroValueSamples, sampleStatistics);
}

}  // namespace openpgl
