// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#include "../../data/SampleData.h"
#include "../../openpgl_common.h"
#include "ParallaxAwareVonMisesFisherMixture.h"

#define OPENPGL_USE_LOGMAP
#define OPENPGL_ZERO_MEAN
// #define OPENPGL_USE_THREE_SPLIT

#define APPLY_PATCH

namespace openpgl
{

struct ComponentSplitinfoV2
{
    Vector2 mean{0.0f};
    Vector3 covariance{0.0f};

    float eigenValue0{0.0f};
    float eigenValue1{0.0f};

    Vector2 eigenVector0{0.0f};
    Vector2 eigenVector1{0.0f};

    std::string toString() const;
};

template <class TVMMFactory>
struct VonMisesFisherChiSquareComponentSplitterV2
{
   public:
    typedef typename TVMMFactory::Distribution VMM;
    typedef TVMMFactory VMMFactory;

    typedef typename VMMFactory::Configuration Configuration;
    typedef typename VMMFactory::SufficientStatistics SufficientStatistics;
    typedef typename VMMFactory::PartialFittingMask PartialFittingMask;

    enum SplitType
    {
        EFirefly = 0,
        EMultiModal,
        ENone
    };

    struct SplitCandidate
    {
        size_t componentIndex;
        float chiSquareEst;
        float chiSquareVar;

        SplitType splitType;

        bool operator<(const SplitCandidate &sc) const
        {
            return chiSquareEst < sc.chiSquareEst;
        }

        bool operator>(const SplitCandidate &sc) const
        {
            return chiSquareEst > sc.chiSquareEst;
        }
    };

    struct ComponentSplitStatistics
    {
        ComponentSplitStatistics() = default;

        embree::vfloat<VMM::VectorSize> chiSquareMCEstimates[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> chiSquareMCEstimate2ndMoments[VMM::NumVectors];
        embree::Vec2<embree::vfloat<VMM::VectorSize> > splitMeans[VMM::NumVectors];
        embree::Vec3<embree::vfloat<VMM::VectorSize> > splitWeightedSampleCovariances[VMM::NumVectors];

        embree::vfloat<VMM::VectorSize> weightsEstimates[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> weights2ndmomentEstimates[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> weightsVarianceEstimates[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> numWeightsEstimatesSamples[VMM::NumVectors];

        embree::vint<VMM::VectorSize> splitType[VMM::NumVectors];

        embree::vfloat<VMM::VectorSize> numSamples[VMM::NumVectors];
        embree::vfloat<VMM::VectorSize> sumWeights[VMM::NumVectors];

        embree::vfloat<VMM::VectorSize> sumAssignedSamples[VMM::NumVectors];

        size_t numComponents{0};

        // sufficient stats for a single (firefly) component of the last update step
        embree::vfloat<VMM::VectorSize> weights[VMM::NumVectors];
        embree::Vec3<embree::vfloat<VMM::VectorSize> > weightedMeans[VMM::NumVectors];

        void clear(const size_t &_numComponents);
        void clearMasked(const size_t &_numComponents, const PartialFittingMask &mask);
        void clearAll();

        float getChiSquareEst(const size_t &idx) const;
        float getChiSquare2ndMomentEst(const size_t &idx) const;

        float getWeightsEst(const size_t &idx) const;
        float getRelVarianceEst(const size_t &idx) const;
        float getVarianceEst(const size_t &idx) const;
        float getWeights2ndMomentEst(const size_t &idx) const;
        float getWeightsVarianceEst(const size_t &idx) const;

        // float getChiSquareVar(const size_t &idx) const;
        float getSumChiSquareEst() const;
        size_t getHighestChiSquareIdx() const;

        SplitCandidate getHighestValidChiSquareSplitComponent(const VMM &vmm, const VMM &previousVMM, const ComponentSplitStatistics &previousSplitStats,
                                                              const bool *alreadySplitted, const bool useConfidence) const;

        void mergeComponentStats(const size_t &idxI, const size_t &idxJ, const float &weightI, const Vector3 &meanDirectionI, const float &weightJ, const Vector3 &meanDirectionJ,
                                 const float &weightK, const Vector3 &meanDirectionK);

        Vector2 getSplitMean(const size_t &idx) const;

        Vector3 getSplitCovariance(const size_t &idx) const;

        SplitType getSplitType(const size_t &idx) const;

        SplitType getSplitType(const size_t &idx, const VMM &vmm, const VMM &previousVMM, const ComponentSplitStatistics &previousSplitStats) const;

        std::vector<SplitCandidate> getSplitCandidates(const float splitThreshold, const bool useConfidence) const;

        void decay(const float &alpha);

        bool isValid() const;

        void serialize(std::ostream &stream) const;

        void deserialize(std::istream &stream);

        inline size_t getNumComponents() const
        {
            return numComponents;
        }

        void setNumComponents(const size_t &n)
        {
            numComponents = n;
        }

        std::string toString() const;

        bool operator==(const ComponentSplitStatistics &b) const;
    };

    void PerformSplitting(VMM &vmm, const float &splitThreshold, const float &mcEstimate, const SampleData *data, const size_t &numData,
                          const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit, const int &maxSplittingItr = -1) const;

    bool SplitAndRefit(VMM &vmm, const float &mcEstimate, const size_t idx, ComponentSplitStatistics &splitStatistics, SufficientStatistics &suffStatistics, const SampleData *data,
                       const size_t &numData, const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit) const;

    bool SplitAndUpdate(VMM &vmm, const float &mcEstimate, const SplitCandidate &candidate, ComponentSplitStatistics &splitStatistics, SufficientStatistics &suffStatistics,
                        const SampleData *data, const size_t &numData, const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit) const;

    void SplitAndRefitNext(VMM &vmm, const float &mcEstimate, ComponentSplitStatistics &splitStatistics, SufficientStatistics &suffStatistics, const SampleData *data,
                           const size_t &numData, const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit) const;

    void PerformRecursiveSplitting(VMM &vmm, typename VMMFactory::SufficientStatistics &suffStats, const float &splitThreshold, const float &mcEstimate, const SampleData *data,
                                   const size_t &numData, const typename VMMFactory::Configuration factoryCfg) const;

    void PerformSplittingIteration(VMM &vmm, const float &splitThreshold) const;

    void CalucalteWeightsEstimates(const VMM &vmm, ComponentSplitStatistics &splitStats, const SampleData *data, const size_t &numData) const;

    void CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data, const size_t &numData) const;

    void PartialCalculateSplitStatistics(const VMM &vmm, const PartialFittingMask &mask, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data,
                                         const size_t &numData) const;

    void UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data, const size_t &numData,
                               bool updateWeightsEstimates, bool onlyConsiderFireflySamples) const;

    void PartialUpdateSplitStatistics(const VMM &vmm, const PartialFittingMask &mask, ComponentSplitStatistics &splitStats, const float &mcEstimate, const SampleData *data,
                                      const size_t &numData) const;

    bool SplitComponent(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx) const;

    bool SplitComponentFireFly(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx, const Configuration &cfg) const;

    bool SplitComponentIntoThree(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats, const size_t idx) const;

    ComponentSplitinfoV2 GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const SampleData *data, const size_t &numData, Vector3 *local2D) const;
};
/*
#ifndef OPENPGL_USE_LOGMAP

template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec2Type Map3DTo2D(const Vec3Type &vec3D)
{
    return Vec2Type(vec3D.x, vec3D.y);
}

template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec3Type Map2DTo3D(const Vec2Type &vec2D)
{
    Vec3Type vec3D = Vec3Type(0.0f);
    vec3D.x = vec2D.x;
    vec3D.y = vec2D.y;
    vec3D.z = embree::sqrt(1.0f - vec2D.x * vec2D.x - vec2D.y * vec2D.y);
    return vec3D;
}

#else

// logMapping https://ronnybergmann.net/mvirt/manifolds/Sn/log.html
template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec2Type Map3DTo2D(const Vec3Type &vec3D)
{
    Vec2Type vec2D(0.0f);

    // OPENPGL_ASSERT((vec3D.z <= 1.0f &&  vec3D.z >= -1.0f));
    ScalarType alpha = embree::fastapprox::acos(vec3D.z);
    ScalarType inv_sinc = alpha / embree::fastapprox::sin(alpha);
    // TODO: Needs to be implmented

    vec2D.x = embree::select(alpha > 0.0f, vec3D.x * inv_sinc, vec2D.x);
    vec2D.y = embree::select(alpha > 0.0f, vec3D.y * inv_sinc, vec2D.y);
    return vec2D;
}

// expMapping https://ronnybergmann.net/mvirt/manifolds/Sn/exp.html
template <typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec3Type Map2DTo3D(const Vec2Type &vec2D)
{
    Vec3Type vec3D = Vec3Type(0.0f);
    ScalarType length = embree::sqrt(vec2D.x * vec2D.x + vec2D.y * vec2D.y);
    OPENPGL_ASSERT(length < M_PI_F);
    ScalarType sinc = embree::fastapprox::sin(length) / length;

    vec3D.x = embree::select(length > 0.0f, vec2D.x * sinc, vec3D.x);
    vec3D.y = embree::select(length > 0.0f, vec2D.y * sinc, vec3D.y);
    vec3D.z = embree::cos(length);

    return vec3D;
}

#endif
*/
inline std::string ComponentSplitinfoV2::toString() const
{
    std::stringstream ss;
    ss << "ComponentSplitinfoV2:" << std::endl;
    // ss << "mean: " << mean << std::endl;
    // ss << "covariance: " << covariance << std::endl;
    ss << "eigenValue0: " << eigenValue0 << std::endl;
    ss << "eigenValue1: " << eigenValue1 << std::endl;
    // ss << "eigenVector0: " << eigenVector0 << std::endl;
    // ss << "eigenVector1: " << eigenVector1 << std::endl;
    return ss.str();
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate,
                                                                                       const SampleData *data, const size_t &numData) const
{
    splitStats.clear(vmm._numComponents);
    this->CalucalteWeightsEstimates(vmm, splitStats, data, numData);
    OPENPGL_ASSERT(splitStats.isValid());
    this->UpdateSplitStatistics(vmm, splitStats, mcEstimate, data, numData, false, true);
    OPENPGL_ASSERT(splitStats.isValid());
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::PartialCalculateSplitStatistics(const VMM &vmm, const PartialFittingMask &mask, ComponentSplitStatistics &splitStats,
                                                                                              const float &mcEstimate, const SampleData *data, const size_t &numData) const
{
    splitStats.clearMasked(vmm._numComponents, mask);
    this->PartialUpdateSplitStatistics(vmm, mask, splitStats, mcEstimate, data, numData);
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitAndRefit(VMM &vmm, const float &mcEstimate, const size_t idx, ComponentSplitStatistics &splitStatistics,
                                                                            SufficientStatistics &suffStatistics, const SampleData *data, const size_t &numData,
                                                                            const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit) const
{
    // std::cout << "SplitAndRefit: " << idx << std::endl;
    PartialFittingMask mask;
    PartialFittingMask previousAsPriorMask;

    bool stopSplitting = false;

    size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;

    this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);
    bool splitSucess = false;
#ifndef OPENPGL_USE_THREE_SPLIT
    if (vmm._numComponents < VMM::MaxComponents)
#else
    if (vmm._numComponents < VMM::MaxComponents - 1)
#endif
    {
        previousAsPriorMask.resetToFalse();
        mask.resetToFalse();
#ifndef OPENPGL_USE_THREE_SPLIT
        splitSucess = SplitComponent(vmm, splitStatistics, suffStatistics, idx);
        mask.setToTrue(idx);
        mask.setToTrue(vmm._numComponents - 1);
#else
        splitSucess = SplitComponentIntoThree(vmm, splitStatistics, suffStatistics, idx);
        mask.setToTrue(idx);
        mask.setToTrue(vmm._numComponents - 2);
        mask.setToTrue(vmm._numComponents - 1);
#endif
        if (splitSucess)
        {
            if (doPartialRefit)
            {
                //            vmmFactory.partialFitMixture(vmm, mask, suffStatistics, data, numData, factoryCfg, vmmFitStats);
                vmmFactory.partialUpdateMixtureV2(vmm, mask, previousAsPriorMask, suffStatistics, data, numData, factoryCfg, vmmFitStats);
                //            vmmFactory.partialUpdateMixture(vmm, mask, suffStatistics, data, numData, factoryCfg, vmmFitStats);
            }
            splitStatistics.clearAll();
            this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);
        }
    }
    return splitSucess;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitAndRefitNext(VMM &vmm, const float &mcEstimate, ComponentSplitStatistics &splitStatistics,
                                                                                SufficientStatistics &suffStatistics, const SampleData *data, const size_t &numData,
                                                                                const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit) const
{
    // std::cout << "SplitAndRefit: " << idx << std::endl;
    PartialFittingMask mask;

    bool success = false;

    size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;

    this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);

#ifndef OPENPGL_USE_THREE_SPLIT
    if (vmm._numComponents < VMM::MaxComponents)
#else
    if (vmm._numComponents < VMM::MaxComponents - 1)
#endif
    {
        std::vector<SplitCandidate> splitComps = splitStatistics.getSplitCandidates();

        mask.resetToFalse();
#ifndef OPENPGL_USE_THREE_SPLIT
        int k = 0;
        while (!SplitComponent(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex) && k < vmm._numComponents)
        {
            k++;
        }
        // bool splitSucess = SplitComponent(vmm, splitStatistics, suffStatistics, idx);
        if (k < vmm._numComponents)
            success = true;

        if (success)
        {
            mask.setToTrue(splitComps[k].componentIndex);
            mask.setToTrue(vmm._numComponents - 1);
#else
        bool splitSucess = SplitComponentIntoThree(vmm, splitStatistics, suffStatistics, idx);
        mask.setToTrue(idx);
        mask.setToTrue(vmm._numComponents - 2);
        mask.setToTrue(vmm._numComponents - 1);
#endif

            //            std::cout << "SplitComponent: idx = " << splitComps[k].componentIndex << "\t sucess = " << success << std::endl;
            suffStatistics.clear(vmm._numComponents);
            // std::cout << "mask: " << mask.toString() << std::endl;
            // std::cout << "vmmSplit: " << vmm.toString() << std::endl;
            // std::cout << "factoryCfg: " << factoryCfg.toString() << std::endl;
            // std::cout << "suffStatistics: " << suffStatistics.toString() << std::endl;
            if (doPartialRefit)
            {
                vmmFactory.partialUpdateMixture(vmm, mask, suffStatistics, data, numData, factoryCfg, vmmFitStats);
                // std::cout << "vmmpartialUpdate: " << vmm.toString() << std::endl;
                // splitItr++;
            }

            splitStatistics.clearAll();
            this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);
        }
    }
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitAndUpdate(VMM &vmm, const float &mcEstimate, const SplitCandidate &candidate,
                                                                             ComponentSplitStatistics &splitStatistics, SufficientStatistics &suffStatistics,
                                                                             const SampleData *data, const size_t &numData, const typename VMMFactory::Configuration factoryCfg,
                                                                             const bool &doPartialRefit) const
{
    OPENPGL_ASSERT(vmm.isValid());
    // std::cout << "SplitAndRefit: " << idx << std::endl;
    PartialFittingMask mask;
    PartialFittingMask previousAsPriorMask;

    bool stopSplitting = false;

    size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;

    // this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);
    bool splitSucess = false;
#ifndef OPENPGL_USE_THREE_SPLIT
    if (vmm._numComponents < VMM::MaxComponents)
#else
    if (vmm._numComponents < VMM::MaxComponents - 1)
#endif
    {
        // std::vector<SplitCandidate> splitComps = splitStatistics.getSplitCandidates();
        previousAsPriorMask.resetToTrue();
        mask.resetToFalse();
#ifndef OPENPGL_USE_THREE_SPLIT
        size_t idx = candidate.componentIndex;
        if (candidate.splitType == EFirefly)
        {
            splitSucess = SplitComponentFireFly(vmm, splitStatistics, suffStatistics, idx, factoryCfg);
            previousAsPriorMask.setToTrue(idx);
            previousAsPriorMask.setToFalse(vmm._numComponents - 1);
        }
        else
        {
            splitSucess = SplitComponent(vmm, splitStatistics, suffStatistics, idx);
            previousAsPriorMask.setToTrue(idx);
            previousAsPriorMask.setToTrue(vmm._numComponents - 1);
            // previousAsPriorMask.setToFalse(idx);
            // previousAsPriorMask.setToFalse(vmm._numComponents - 1);
        }

        mask.setToTrue(idx);
        mask.setToTrue(vmm._numComponents - 1);
#else
        bool splitSucess = SplitComponentIntoThree(vmm, splitStatistics, suffStatistics, idx);
        mask.setToTrue(idx);
        mask.setToTrue(vmm._numComponents - 2);
        mask.setToTrue(vmm._numComponents - 1);
#endif

        if (true && splitSucess)
        {
            OPENPGL_ASSERT(vmm.isValid());
            vmmFactory.partialUpdateMixtureV2(vmm, mask, previousAsPriorMask, suffStatistics, data, numData, factoryCfg, vmmFitStats);
            OPENPGL_ASSERT(vmm.isValid());
            this->PartialCalculateSplitStatistics(vmm, mask, splitStatistics, mcEstimate, data, numData);
            OPENPGL_ASSERT(vmm.isValid());
        }
    }
    return splitSucess;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::PerformSplitting(VMM &vmm, const float &splitThreshold, const float &mcEstimate, const SampleData *data,
                                                                               const size_t &numData, const typename VMMFactory::Configuration factoryCfg,
                                                                               const bool &doPartialRefit, const int &maxSplittingItr) const
{
    PartialFittingMask mask;
    ComponentSplitStatistics splitStatistics;
    SufficientStatistics suffStatistics;

    bool stopSplitting = false;

    size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;

#ifndef OPENPGL_USE_THREE_SPLIT
    while (vmm._numComponents < VMM::MaxComponents && !stopSplitting)
#else
    while (vmm._numComponents < VMM::MaxComponents - 1 && !stopSplitting)
#endif
    {
        stopSplitting = true;
        splitStatistics.clearAll();
        this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);

        std::vector<SplitCandidate> splitComps = splitStatistics.getSplitCandidates(splitThreshold);

        mask.resetToFalse();
        const size_t numComp = vmm._numComponents;
        for (size_t k = 0; k < numComp; k++)
        {
            if (splitComps[k].chiSquareEst > splitThreshold && vmm._numComponents < VMM::MaxComponents)
            {
                // std::cout << "split[" << k << "]: idx:" << splitComps[k].componentIndex << "\t chi2: " << splitComps[k].chiSquareEst << std::endl;
#ifndef OPENPGL_USE_THREE_SPLIT
                bool splitSucess = SplitComponent(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex);
                mask.setToTrue(splitComps[k].componentIndex);
                mask.setToTrue(vmm._numComponents - 1);
#else
                bool splitSucess = SplitComponentIntoThree(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex);
                mask.setToTrue(splitComps[k].componentIndex);
                mask.setToTrue(vmm._numComponents - 2);
                mask.setToTrue(vmm._numComponents - 1);
#endif
                if (splitSucess)
                {
                    stopSplitting = false;
                }
            }
            else
            {
                continue;
            }
        }
        suffStatistics.clear(vmm._numComponents);
        // std::cout << "mask: " << mask.toString() << std::endl;
        // std::cout << "vmmSplit: " << vmm.toString() << std::endl;
        // std::cout << "factoryCfg: " << factoryCfg.toString() << std::endl;
        // std::cout << "suffStatistics: " << suffStatistics.toString() << std::endl;
        if (doPartialRefit)
        {
            vmmFactory.partialUpdateMixture(vmm, mask, suffStatistics, true, data, numData, factoryCfg, vmmFitStats);
            // std::cout << "vmmpartialUpdate: " << vmm.toString() << std::endl;
            splitItr++;
        }
        else
        {
            stopSplitting = true;
        }

        if (splitItr >= maxSplittingItr)
        {
            stopSplitting = true;
        }
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::PerformRecursiveSplitting(VMM &vmm, typename VMMFactory::SufficientStatistics &suffStatistics,
                                                                                        const float &splitThreshold, const float &mcEstimate, const SampleData *data,
                                                                                        const size_t &numData, const typename VMMFactory::Configuration factoryCfg) const
{
    PartialFittingMask mask;
    ComponentSplitStatistics splitStatistics;

    // bool stopSplitting = false;
    // size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;
    // std::cout << "vmm: " << vmm.toString() << std::endl;
    int numSplits = -1;
#ifndef OPENPGL_USE_THREE_SPLIT
    while (vmm._numComponents < VMM::MaxComponents && numSplits != 0)
#else

#endif
    // for (size_t j =0; j<1; j++)
    {
        numSplits = 0;
        splitStatistics.clearAll();
        this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);

        std::vector<SplitCandidate> splitComps = splitStatistics.getSplitCandidates(splitThreshold, false);
        // SplitCandidate splitComp = splitStatistics.getHighestValidChiSquareSplitComponent(splitThreshold, false);
        // SplitCandidate splitCandidate = stats.splittingStatistics.getHighestValidChiSquareSplitComponent(vmm, vmmOld, splitStatsBefore, alreadySplitted, true);
        mask.resetToFalse();
        // const size_t numComp = vmm._numComponents;
        for (size_t k = 0; k < splitComps.size(); k++)
        // int k = 0;
        // if(splitComps.size() > 0)
        {
            if (splitComps[k].chiSquareEst > splitThreshold && vmm._numComponents < VMM::MaxComponents)
            {
#ifndef OPENPGL_USE_THREE_SPLIT
                const div_t tmpK = div(splitComps[k].componentIndex, static_cast<int>(VMM::VectorSize));
                SplitType splitType = (SplitType)splitStatistics.splitType[tmpK.quot][tmpK.rem];
                //std::cout << "splitIdx = " << splitComps[k].componentIndex << "\t splitType = " << (splitType == EFirefly ? "FireFly" : "MultiModal") << std::endl;
                bool splitSucess = true;
                if (splitType == EFirefly)
                //if (false)
                {
                    splitSucess = SplitComponentFireFly(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex, factoryCfg);
                }
                else
                {
                    splitSucess = SplitComponent(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex);
                }
                mask.setToTrue(splitComps[k].componentIndex);
                if (splitSucess)
                {
                    mask.setToTrue(vmm._numComponents - 1);
                }
                // std::cout << "sucessfull split: " << (splitSucess ? "True" : "False") << std::endl;
#else
                bool splitSucess = SplitComponentIntoThree(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex);
                mask.setToTrue(splitComps[k].componentIndex);
                if (splitSucess)
                {
                    mask.setToTrue(vmm._numComponents - 1);
                    mask.setToTrue(vmm._numComponents - 2);
                }
#endif
                if (splitSucess)
                {
                    numSplits++;
                }
            }
            else
            {
                continue;
            }
        }
        if (numSplits > 0)
        {
            vmmFactory.partialUpdateMixture(vmm, mask, suffStatistics, false, data, numData, factoryCfg, vmmFitStats);
        }
        // std::cout << "vmmpartialUpdate: " << vmm.toString() << std::endl;
        // splitItr++;
    }
}

template <class TVMMFactory>
ComponentSplitinfoV2 VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const SampleData *data,
                                                                                                          const size_t &numData, Vector3 *local2D) const
{
    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    // const int cnt = (vmm._numComponents + VMM::VectorSize-1) / VMM::VectorSize;
    // size_t validDataCount = 0.0f;

    ComponentSplitinfoV2 splitInfo;

    Vector2 mean(0.0f);
    Vector3 covarianceStats(0.0f);
    float sumWeights = 0.0f;

    for (size_t n = 0; n < numData; n++)
    {
        const SampleData sample = data[n];
        pgl_vec3f direction = sample.direction;
        openpgl::Vector3 sampleDirection(direction.x, direction.y, direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));

            const embree::vfloat<VMM::VectorSize> weight = sample.weight;
            // const embree::vfloat<VMM::VectorSize> samplePDF = sample.pdf;
            // const vfloat<VMM::VectorSize> value =  weight * samplePDF;

            const embree::Vec3<embree::vfloat<VMM::VectorSize> > localDirection =
                embree::frame(vmm._meanDirections[tmp.quot]).inverse() * embree::Vec3<embree::vfloat<VMM::VectorSize> >(sampleDirection);
            // const embree::Vec2< embree::vfloat<VMM::VectorSize> > localDirection2D = Map3DTo2D< embree::Vec3< embree::vfloat<VMM::VectorSize> >,  embree::Vec2<
            // embree::vfloat<VMM::VectorSize> >, embree::vfloat<VMM::VectorSize> >(localDirection);
            const Vector2 localDirection2D = Map3DTo2D<Vector3, Vector2, float>(Vector3(localDirection.x[tmp.rem], localDirection.y[tmp.rem], localDirection.z[tmp.rem]));

            const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[tmp.quot] * weight;
            local2D[n].x = localDirection2D.x;
            local2D[n].y = localDirection2D.y;
            local2D[n].z = assignedWeight[tmp.rem];

            sumWeights += assignedWeight[tmp.rem];
#ifdef OPENPGL_ZERO_MEAN
            mean.x += 0.0f;
            mean.y += 0.0f;
#else
            mean.x += assignedWeight[tmp.rem] * localDirection2D.x;
            mean.y += assignedWeight[tmp.rem] * localDirection2D.y;
#endif
            covarianceStats.x += assignedWeight[tmp.rem] * localDirection2D.x * localDirection2D.x;
            covarianceStats.y += assignedWeight[tmp.rem] * localDirection2D.y * localDirection2D.y;
            covarianceStats.z += assignedWeight[tmp.rem] * localDirection2D.x * localDirection2D.y;
        }
    }
    mean /= sumWeights;

    splitInfo.mean = mean;
    splitInfo.covariance.x = covarianceStats.x / sumWeights - mean.x * mean.x;
    splitInfo.covariance.y = covarianceStats.y / sumWeights - mean.y * mean.y;
    splitInfo.covariance.z = covarianceStats.z / sumWeights - mean.x * mean.y;

    float D = embree::sqrt((splitInfo.covariance.x - splitInfo.covariance.y) * (splitInfo.covariance.x - splitInfo.covariance.y) +
                           (splitInfo.covariance.z * splitInfo.covariance.z * 4.0f)) *
              0.5f;
    splitInfo.eigenValue0 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue0 += D;

    splitInfo.eigenValue1 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue1 -= D;

    splitInfo.eigenVector0.x = -splitInfo.covariance.z;
    splitInfo.eigenVector0.y = splitInfo.covariance.x - splitInfo.eigenValue0;

    splitInfo.eigenVector1.x = splitInfo.covariance.z;
    splitInfo.eigenVector1.y = splitInfo.covariance.x - splitInfo.eigenValue1;

    splitInfo.eigenVector0 /= embree::sqrt(splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y);
    splitInfo.eigenVector1 /= embree::sqrt(splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y);
#ifdef OPENPGL_SHOW_PRINT_OUTS
    std::cout << "split: " << "\tmean: " << splitInfo.mean.x << ", \t " << splitInfo.mean.y << "\t covariance: " << splitInfo.covariance.x << ", \t " << splitInfo.covariance.y
              << ", \t " << splitInfo.covariance.z << std::endl;
    std::cout << "eigen: " << "\tevalue0: " << splitInfo.eigenValue0 << "\teVec0: " << splitInfo.eigenVector0.x << ", \t " << splitInfo.eigenVector0.y
              << "\tevalue1: " << splitInfo.eigenValue1 << "\teVec1: " << splitInfo.eigenVector1.x << ", \t " << splitInfo.eigenVector1.y << std::endl;
#endif
    return splitInfo;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::CalucalteWeightsEstimates(const VMM &vmm, ComponentSplitStatistics &splitStats, const SampleData *data,
                                                                                        const size_t &numData) const
{
    OPENPGL_ASSERT(vmm._numComponents == splitStats.numComponents);

    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        splitStats.weightsEstimates[k] = zeros;
        splitStats.weights2ndmomentEstimates[k] = zeros;
        splitStats.numWeightsEstimatesSamples[k] = zeros;
    }

    for (size_t n = 0; n < numData; n++)
    {
        const SampleData sample = data[n];
        pgl_vec3f direction = sample.direction;
        const openpgl::Vector3 sampleDirection(direction.x, direction.y, direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            const embree::vfloat<VMM::VectorSize> weight = sample.weight;
            for (size_t k = 0; k < cnt; k++)
            {
                const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight;
                splitStats.numWeightsEstimatesSamples[k] += 1.0f;
                auto oldWeightsEstimates = splitStats.weightsEstimates[k];
                splitStats.weightsEstimates[k] += (assignedWeight - splitStats.weightsEstimates[k]) / splitStats.numWeightsEstimatesSamples[k];
                splitStats.weights2ndmomentEstimates[k] += (assignedWeight * assignedWeight - splitStats.weights2ndmomentEstimates[k]) / splitStats.numWeightsEstimatesSamples[k];
                splitStats.weightsVarianceEstimates[k] += (assignedWeight - splitStats.weightsEstimates[k]) * (assignedWeight - oldWeightsEstimates);
            }
        }
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate,
                                                                                    const SampleData *data, const size_t &numData, bool updateWeightsEstimates, bool onlyConsiderFireflySamples) const
{
    // std::cout << "UpdateSplitStatistics" << std::endl;

    OPENPGL_ASSERT(vmm._numComponents == splitStats.numComponents);

    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    // size_t validDataCount = 0.0f;

    const embree::vint<VMM::VectorSize> stFF((int32_t)EFirefly);
    const embree::vint<VMM::VectorSize> stMM((int32_t)EMultiModal);

    for (size_t k = 0; k < cnt; k++)
    {
        splitStats.weights[k] = zeros;
        splitStats.weightedMeans[k].x = zeros;
        splitStats.weightedMeans[k].y = zeros;
        splitStats.weightedMeans[k].z = zeros;
        splitStats.splitType[k] = stMM;
    }

    for (size_t n = 0; n < numData; n++)
    {
        const SampleData sample = data[n];
        pgl_vec3f direction = sample.direction;
        const openpgl::Vector3 sampleDirection(direction.x, direction.y, direction.z);
        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            const embree::vfloat<VMM::VectorSize> weight = sample.weight;
            const embree::vfloat<VMM::VectorSize> samplePDF = sample.pdf;
            const embree::vfloat<VMM::VectorSize> value = weight * samplePDF;
            // std::cout << "data[" << n << "]: " << "value: " << value << "\t samplePDF: " << samplePDF;
            softAssign.pdf = std::max(softAssign.pdf, FLT_EPSILON);

            for (size_t k = 0; k < cnt; k++)
            {
                const embree::vfloat<VMM::VectorSize> weightMean = splitStats.weightsEstimates[k];
                const embree::vfloat<VMM::VectorSize> weightStd =
                    select(splitStats.weights2ndmomentEstimates[k] > 0, embree::sqrt(splitStats.weights2ndmomentEstimates[k] - weightMean * weightMean), 0.f);

                const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight;
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));

                // Checking based on the assigned mean and std if the current sample is a firefly or not
                // We use different splitting methods if a split is triggered by a firefly and not by a multi modal distribution
                splitStats.splitType[k] = select((assignedWeight > weightMean + 3.f * weightStd), stFF, splitStats.splitType[k]);
                
                embree::vfloat<VMM::VectorSize> assignedWeightTmp = assignedWeight;
                if(onlyConsiderFireflySamples)
                    assignedWeightTmp = select((assignedWeight > weightMean + 3.f * weightStd), assignedWeight, zeros);
                splitStats.weights[k] += assignedWeightTmp;
                splitStats.weightedMeans[k].x += assignedWeightTmp * direction.x;
                splitStats.weightedMeans[k].y += assignedWeightTmp * direction.y;
                splitStats.weightedMeans[k].z += assignedWeightTmp * direction.z;

                // splitStats.weights[k] += assignedWeight;
                // splitStats.weightedMeans[k].x += assignedWeight * direction.x;
                // splitStats.weightedMeans[k].y += assignedWeight * direction.y;
                // splitStats.weightedMeans[k].z += assignedWeight * direction.z;

                embree::vfloat<VMM::VectorSize> vmfPDF = softAssign.assignments[k] * softAssign.pdf;
                embree::vfloat<VMM::VectorSize> partialValuePDF = vmfPDF * value;
                partialValuePDF /= (mcEstimate * softAssign.pdf);
                // partialValuePDF /= vmm._weights[k] * mcEstimate;
                // std::cout << "\tweights: " << vmm._weights[k] << "\t assign: " << softAssign.assignments[k] << "\t pdf: " << softAssign.pdf << std::endl;
                // std::cout << "\tpvPDF: " << partialValuePDF << "\t vmfPDF: " << vmfPDF << std::endl;

                embree::vfloat<VMM::VectorSize> chiSquareEst = value * value * vmfPDF;
                chiSquareEst /= mcEstimate * mcEstimate * softAssign.pdf * softAssign.pdf;
                // chiSquareEst *= chiSquareEst;
                chiSquareEst -= 2.0f * partialValuePDF;
                chiSquareEst += vmfPDF;
                chiSquareEst /= samplePDF;

                chiSquareEst = select(softAssign.assignments[k] > FLT_EPSILON, chiSquareEst, zeros);

                splitStats.sumAssignedSamples[k] += softAssign.assignments[k];
                // incremental updated of the MC chiSquare estimate
                splitStats.numSamples[k] += 1.0f;
                splitStats.chiSquareMCEstimates[k] += (chiSquareEst - splitStats.chiSquareMCEstimates[k]) / splitStats.numSamples[k];
                splitStats.chiSquareMCEstimate2ndMoments[k] += (chiSquareEst * chiSquareEst - splitStats.chiSquareMCEstimate2ndMoments[k]) / splitStats.numSamples[k];
                if (updateWeightsEstimates)
                {
                    splitStats.numWeightsEstimatesSamples[k] += 1.0f;
                    auto oldWeightsEstimates = splitStats.weightsEstimates[k];
                    splitStats.weightsEstimates[k] += (softAssign.assignments[k] * weight - splitStats.weightsEstimates[k]) / splitStats.numWeightsEstimatesSamples[k];
                    splitStats.weights2ndmomentEstimates[k] +=
                        ((softAssign.assignments[k] * weight) * (softAssign.assignments[k] * weight) - splitStats.weights2ndmomentEstimates[k]) /
                        splitStats.numWeightsEstimatesSamples[k];
                    splitStats.weightsVarianceEstimates[k] +=
                        ((softAssign.assignments[k] * weight) - splitStats.weightsEstimates[k]) * ((softAssign.assignments[k] * weight) - oldWeightsEstimates);
                }
                const embree::Vec3<embree::vfloat<VMM::VectorSize> > localDirection =
                    embree::frame(vmm._meanDirections[k]).inverse() * embree::Vec3<embree::vfloat<VMM::VectorSize> >(sampleDirection);
                const embree::Vec2<embree::vfloat<VMM::VectorSize> > localDirection2D(localDirection.x, localDirection.y);
                // const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight;
                //  const vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight * weight;

                splitStats.sumWeights[k] += assignedWeight;
                //                const vfloat<VMM::VectorSize> incWeight = select(splitStats.sumWeights[k] > 0.0f, assignedWeight / splitStats.sumWeights[k], zeros);

#ifdef OPENPGL_ZERO_MEAN
                splitStats.splitMeans[k] += embree::Vec2<embree::vfloat<VMM::VectorSize> >(0.0f);
                splitStats.splitWeightedSampleCovariances[k].x += assignedWeight * (localDirection2D.x * localDirection2D.x);
                splitStats.splitWeightedSampleCovariances[k].y += assignedWeight * (localDirection2D.y * localDirection2D.y);
                splitStats.splitWeightedSampleCovariances[k].z += assignedWeight * (localDirection2D.x * localDirection2D.y);
#else
                const Vec2<vfloat<VMM::VectorSize> > previousSplitMeans = splitStats.splitMeans[k];
                splitStats.splitMeans[k] += incWeight * (localDirection2D - splitStats.splitMeans[k]);
                splitStats.splitWeightedSampleCovariances[k].x +=
                    assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.x - splitStats.splitMeans[k].x));
                splitStats.splitWeightedSampleCovariances[k].y +=
                    assignedWeight * ((localDirection2D.y - previousSplitMeans.y) * (localDirection2D.y - splitStats.splitMeans[k].y));
                splitStats.splitWeightedSampleCovariances[k].z +=
                    assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.y - splitStats.splitMeans[k].y));
#endif
                OPENPGL_ASSERT(embree::all(embree::isvalid(assignedWeight)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));
                // splitStats.sumWeights[k] += assignedWeight;
            }
            // validDataCount++;
            // std::cout << std::endl;
        }
    }
    /*
    for (int k = 0; k<vmm.getNumComponents(); k++) {
        const div_t tmpK = div(k, static_cast<int>(VMM::VectorSize));
        std::cout << "SplitType: idx = " << k << "\t type = " << (((SplitType)splitStats.splitType[tmpK.quot][tmpK.rem])==EFirefly ? "FireFly" : "MultiModal" )<< std::endl;
    }
    */
    // splitStats.numSamplesOld += validDataCount;
    // splitStats.mcEstimate += mcEstimate;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::PartialUpdateSplitStatistics(const VMM &vmm, const PartialFittingMask &mask, ComponentSplitStatistics &splitStats,
                                                                                           const float &mcEstimate, const SampleData *data, const size_t &numData) const
{
    // std::cout << "UpdateSplitStatistics" << std::endl;

    OPENPGL_ASSERT(vmm._numComponents == splitStats.numComponents);

    typename VMM::SoftAssignment softAssign;
    const embree::vfloat<VMM::VectorSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    // size_t validDataCount = 0.0f;
    const embree::vint<VMM::VectorSize> stFF((int32_t)EFirefly);
    const embree::vint<VMM::VectorSize> stMM((int32_t)EMultiModal);

    for (size_t k = 0; k < cnt; k++)
    {
        splitStats.weights[k] = select(mask.mask[k], zeros, splitStats.weights[k]);
        splitStats.weightedMeans[k].x = select(mask.mask[k], zeros, splitStats.weightedMeans[k].x);
        splitStats.weightedMeans[k].y = select(mask.mask[k], zeros, splitStats.weightedMeans[k].y);
        splitStats.weightedMeans[k].z = select(mask.mask[k], zeros, splitStats.weightedMeans[k].z);
        splitStats.splitType[k] = select(mask.mask[k], stMM, splitStats.splitType[k]);
    }

    for (size_t n = 0; n < numData; n++)
    {
        const SampleData sample = data[n];
        const pgl_vec3f direction = sample.direction;
        const Vector3 sampleDirection(direction.x, direction.y, direction.z);

        if (vmm.softAssignment(sampleDirection, softAssign))
        {
            softAssign.pdf = std::max(softAssign.pdf, FLT_EPSILON);
            const embree::vfloat<VMM::VectorSize> weight = sample.weight;
            const embree::vfloat<VMM::VectorSize> samplePDF = sample.pdf;
            const embree::vfloat<VMM::VectorSize> value = weight * samplePDF;
            // std::cout << "data[" << n << "]: " << "value: " << value << "\t samplePDF: " << samplePDF;
            for (size_t k = 0; k < cnt; k++)
            {
                const embree::vfloat<VMM::VectorSize> weightMean = splitStats.weightsEstimates[k];
                const embree::vfloat<VMM::VectorSize> weightStd =
                    select(splitStats.weights2ndmomentEstimates[k] > 0, embree::sqrt(splitStats.weights2ndmomentEstimates[k] - weightMean * weightMean), 0.f);

                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));

                // splitStats.splitType[k] = select((assignedWeight > weightMean + 3.f * weightStd), stFF, splitStats.splitType[k]);

                splitStats.weights[k] = select(mask.mask[k], splitStats.weights[k] + softAssign.assignments[k] * weight, splitStats.weights[k]);
                splitStats.weightedMeans[k].x =
                    select(mask.mask[k], splitStats.weightedMeans[k].x + softAssign.assignments[k] * weight * direction.x, splitStats.weightedMeans[k].x);
                splitStats.weightedMeans[k].y =
                    select(mask.mask[k], splitStats.weightedMeans[k].y + softAssign.assignments[k] * weight * direction.y, splitStats.weightedMeans[k].y);
                splitStats.weightedMeans[k].z =
                    select(mask.mask[k], splitStats.weightedMeans[k].z + softAssign.assignments[k] * weight * direction.z, splitStats.weightedMeans[k].z);

                embree::vfloat<VMM::VectorSize> vmfPDF = softAssign.assignments[k] * softAssign.pdf;
                embree::vfloat<VMM::VectorSize> partialValuePDF = vmfPDF * value;
                partialValuePDF /= (mcEstimate * softAssign.pdf);
                // partialValuePDF /= vmm._weights[k] * mcEstimate;
                // std::cout << "\tweights: " << vmm._weights[k] << "\t assign: " << softAssign.assignments[k] << "\t pdf: " << softAssign.pdf << std::endl;
                // std::cout << "\tpvPDF: " << partialValuePDF << "\t vmfPDF: " << vmfPDF << std::endl;

                embree::vfloat<VMM::VectorSize> chiSquareEst = value * value * vmfPDF;
                chiSquareEst /= mcEstimate * mcEstimate * softAssign.pdf * softAssign.pdf;
                // chiSquareEst *= chiSquareEst;
                chiSquareEst -= 2.0f * partialValuePDF;
                chiSquareEst += vmfPDF;
                chiSquareEst /= samplePDF;

                chiSquareEst = select(softAssign.assignments[k] > FLT_EPSILON, chiSquareEst, zeros);

                splitStats.sumAssignedSamples[k] = select(mask.mask[k], splitStats.sumAssignedSamples[k] + softAssign.assignments[k], splitStats.sumAssignedSamples[k]);
                // incremental updated of the MC chiSquare estimate
                splitStats.numSamples[k] = select(mask.mask[k], splitStats.numSamples[k] + 1.0f, splitStats.numSamples[k]);

                embree::vfloat<VMM::VectorSize> chiSquareEstOld = splitStats.chiSquareMCEstimates[k];
                splitStats.chiSquareMCEstimates[k] =
                    select(mask.mask[k], splitStats.chiSquareMCEstimates[k] + (chiSquareEst - splitStats.chiSquareMCEstimates[k]) / splitStats.numSamples[k],
                           splitStats.chiSquareMCEstimates[k]);
                // splitStats.chiSquareMCVariances[k] += ((chiSquareEst - chiSquareEstOld) * (chiSquareEst - splitStats.chiSquareMCEstimates[k]));// / splitStats.numSamples[k];
                splitStats.chiSquareMCEstimate2ndMoments[k] =
                    select(mask.mask[k],
                           splitStats.chiSquareMCEstimate2ndMoments[k] + (chiSquareEst * chiSquareEst - splitStats.chiSquareMCEstimate2ndMoments[k]) / splitStats.numSamples[k],
                           splitStats.chiSquareMCEstimate2ndMoments[k]);

                const embree::Vec3<embree::vfloat<VMM::VectorSize> > localDirection =
                    embree::frame(vmm._meanDirections[k]).inverse() * embree::Vec3<embree::vfloat<VMM::VectorSize> >(sampleDirection);
                const embree::Vec2<embree::vfloat<VMM::VectorSize> > localDirection2D(localDirection.x, localDirection.y);
                const embree::vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight;
                // const vfloat<VMM::VectorSize> assignedWeight = softAssign.assignments[k] * weight * weight;

                splitStats.numWeightsEstimatesSamples[k] = select(mask.mask[k], splitStats.numWeightsEstimatesSamples[k] + 1.0f, splitStats.numWeightsEstimatesSamples[k]);
                auto oldWeightsEstimates = splitStats.weightsEstimates[k];
                splitStats.weightsEstimates[k] =
                    select(mask.mask[k], splitStats.weightsEstimates[k] + (assignedWeight - splitStats.weightsEstimates[k]) / splitStats.numWeightsEstimatesSamples[k],
                           splitStats.weightsEstimates[k]);
                // splitStats.chiSquareMCVariances[k] += ((chiSquareEst - chiSquareEstOld) * (chiSquareEst - splitStats.chiSquareMCEstimates[k]));// / splitStats.numSamples[k];
                splitStats.weights2ndmomentEstimates[k] =
                    select(mask.mask[k],
                           splitStats.chiSquareMCEstimate2ndMoments[k] +
                               (assignedWeight * assignedWeight - splitStats.chiSquareMCEstimate2ndMoments[k]) / splitStats.numWeightsEstimatesSamples[k],
                           splitStats.chiSquareMCEstimate2ndMoments[k]);

                splitStats.weightsVarianceEstimates[k] =
                    select(mask.mask[k], splitStats.weightsVarianceEstimates[k] + ((assignedWeight - splitStats.weightsEstimates[k]) * (assignedWeight - oldWeightsEstimates)),
                           splitStats.weightsVarianceEstimates[k]);

                splitStats.sumWeights[k] += assignedWeight;
                //                const vfloat<VMM::VectorSize> incWeight = select(splitStats.sumWeights[k] > 0.0f, assignedWeight / splitStats.sumWeights[k], zeros);

#ifdef OPENPGL_ZERO_MEAN
                splitStats.splitMeans[k] += embree::Vec2<embree::vfloat<VMM::VectorSize> >(0.0f);
                splitStats.splitWeightedSampleCovariances[k].x =
                    select(mask.mask[k], splitStats.splitWeightedSampleCovariances[k].x + assignedWeight * (localDirection2D.x * localDirection2D.x),
                           splitStats.splitWeightedSampleCovariances[k].x);
                splitStats.splitWeightedSampleCovariances[k].y =
                    select(mask.mask[k], splitStats.splitWeightedSampleCovariances[k].y + assignedWeight * (localDirection2D.y * localDirection2D.y),
                           splitStats.splitWeightedSampleCovariances[k].y);
                splitStats.splitWeightedSampleCovariances[k].z =
                    select(mask.mask[k], splitStats.splitWeightedSampleCovariances[k].z + assignedWeight * (localDirection2D.x * localDirection2D.y),
                           splitStats.splitWeightedSampleCovariances[k].z);
#else
                const Vec2<vfloat<VMM::VectorSize> > previousSplitMeans = splitStats.splitMeans[k];
                splitStats.splitMeans[k] = select(mask.mask[k], splitStats.splitMeans[k] + incWeight * (localDirection2D - splitStats.splitMeans[k]), splitStats.splitMeans[k]);
                splitStats.splitWeightedSampleCovariances[k].x =
                    select(mask.mask[k],
                           splitStats.splitWeightedSampleCovariances[k].x +
                               assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.x - splitStats.splitMeans[k].x)),
                           splitStats.splitWeightedSampleCovariances[k].x);
                splitStats.splitWeightedSampleCovariances[k].y =
                    select(mask.mask[k],
                           splitStats.splitWeightedSampleCovariances[k].y +
                               assignedWeight * ((localDirection2D.y - previousSplitMeans.y) * (localDirection2D.y - splitStats.splitMeans[k].y)),
                           splitStats.splitWeightedSampleCovariances[k].y);
                splitStats.splitWeightedSampleCovariances[k].z =
                    select(mask.mask[k],
                           splitStats.splitWeightedSampleCovariances[k].z +
                               assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.y - splitStats.splitMeans[k].y)),
                           splitStats.splitWeightedSampleCovariances[k].z);
#endif
                OPENPGL_ASSERT(embree::all(embree::isvalid(assignedWeight)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitMeans[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].x)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].y)));
                OPENPGL_ASSERT(embree::all(embree::isvalid(splitStats.splitWeightedSampleCovariances[k].z)));
                // splitStats.sumWeights[k] += assignedWeight;
            }
            // validDataCount++;
            // std::cout << std::endl;
        }
    }
    // splitStats.numSamplesOld += validDataCount;
    // splitStats.mcEstimate += mcEstimate;
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitComponent(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats,
                                                                             const size_t idx) const
{
    ComponentSplitinfoV2 splitInfo;
    const div_t tmpK = div(idx, static_cast<int>(VMM::VectorSize));

    // the number of samples that got assinged to the component that should be split
    float numAssignedSamples = splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem];

    float inv_sumWeights = embree::rcp(splitStats.sumWeights[tmpK.quot][tmpK.rem]);
    OPENPGL_ASSERT(embree::isvalid(inv_sumWeights));
    splitInfo.mean = Vector2(splitStats.splitMeans[tmpK.quot].x[tmpK.rem], splitStats.splitMeans[tmpK.quot].y[tmpK.rem]);

    splitInfo.covariance.x = splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.y = splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.z = splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] * inv_sumWeights;

    float D = embree::sqrt((splitInfo.covariance.x - splitInfo.covariance.y) * (splitInfo.covariance.x - splitInfo.covariance.y) +
                           (splitInfo.covariance.z * splitInfo.covariance.z * 4.0f)) *
              0.5f;
    splitInfo.eigenValue0 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue0 += D;

    splitInfo.eigenValue1 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue1 -= D;

    splitInfo.eigenVector0.x = -splitInfo.covariance.z;
    splitInfo.eigenVector0.y = splitInfo.covariance.x - splitInfo.eigenValue0;

    splitInfo.eigenVector1.x = splitInfo.covariance.z;
    splitInfo.eigenVector1.y = splitInfo.covariance.x - splitInfo.eigenValue1;

    splitInfo.eigenVector0 /= embree::sqrt(splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y);
    splitInfo.eigenVector1 /= embree::sqrt(splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y);
    /* */
    // std::cout << "D: " << D << std::endl;
    // std::cout << "sumWeights: " << splitStats.sumWeights[tmpK.quot][tmpK.rem] << "\t inSumWeights: " << inv_sumWeights << std::endl;
    // std::cout << "splitMean: " << splitInfo.mean << "\t splitCovariance: " << splitInfo.covariance << std::endl;

    // std::cout << "splitCovariancesRaw: " << splitStats.splitCovariances[tmpK.quot].x[tmpK.rem] << "\t" << splitStats.splitCovariances[tmpK.quot].y[tmpK.rem] << "\t" <<
    // splitStats.splitCovariances[tmpK.quot].z[tmpK.rem] << std::endl;
    //    std::cout << "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl;
    //    std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t eigenVector1: " << splitInfo.eigenVector1 << std::endl;
    /**/

    float weight = vmm._weights[tmpK.quot][tmpK.rem];
    float meanCosine = vmm._meanCosines[tmpK.quot][tmpK.rem];
    float kappa = vmm._kappas[tmpK.quot][tmpK.rem];

    if (kappa >= OPENPGL_MAX_KAPPA * 0.9)
    {
        return false;
    }

    Vector3 meanDirection = Vector3(vmm._meanDirections[tmpK.quot].x[tmpK.rem], vmm._meanDirections[tmpK.quot].y[tmpK.rem], vmm._meanDirections[tmpK.quot].z[tmpK.rem]);

    float newWeight0 = weight * 0.5f;
    float newWeight1 = newWeight0;

    Vector3 meanDirection0 = meanDirection;
    Vector3 meanDirection1 = meanDirection;

    float newMeanCosine0 = meanCosine;
    float newMeanCosine1 = meanCosine * meanCosine;

    if (D > 1e-8f)
    {
        Vector2 meanDir2D0 = splitInfo.mean + (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f);
        meanDirection0 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D0);
        newMeanCosine0 = meanCosine / std::abs(dot(meanDirection, meanDirection0));
        // std::cout << "D = " << D << "\t eigenVector0 = "<< splitInfo.eigenVector0 << "\t eigenValue0 = "<< splitInfo.eigenValue0 << std::endl;
        // std::cout << "splitInfo.mean = " << splitInfo.mean << "\t meanDir2D0 = " << meanDir2D0 << "\t meanDir3D0 = "<< Map2DTo3D<Vector3, Vector2, float>(meanDir2D0) << "\t tmp
        // = " << (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f)<< std::endl;

        // TODO: further investigate:
        // newMeanCosine0 = meanCosine / dot(meanDirection, meanDirection0);

        OPENPGL_ASSERT(meanCosine >= 0.f);
        OPENPGL_ASSERT(std::abs(dot(meanDirection, meanDirection0)) > 0.f);
        OPENPGL_ASSERT(newMeanCosine0 >= 0.f);
        // ensure that the new mean cosine is in a valid range (i.e., < 1.0 and < the mean cosine of max kappa)
        newMeanCosine0 = std::min(newMeanCosine0, KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA));
        newMeanCosine1 = newMeanCosine0;

        Vector2 meanDir2D1 = splitInfo.mean - (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f);
        meanDirection1 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D1);

        if (dot(meanDirection0, meanDirection1) >= 0.99f)
        {
            // std::cout << "eigenVector0 = " << splitInfo.eigenVector0 << "\t eigenValue0 = " << splitInfo.eigenValue0 << std::endl;
            // std::cout << "cos = " << dot(meanDirection0, meanDirection1) << std::endl;
            // std::cout << "meanCosine = " << meanCosine << "\t newMeanCosine0 = " << newMeanCosine0 << "\t newMeanCosine1 = " << newMeanCosine1 << std::endl;
            // std::cout << "meanDirection = " << meanDirection << "\t meanDirection0 = " << meanDirection0 << "\t meanDirection1 = " << meanDirection1 << std::endl;
            //  STEP ONE FIX
            std::cout << "Unsucessfull split" << std::endl;
            return false;
        }
#ifdef OPENPGL_SHOW_PRINT_OUTS
        // std::cout << "meanCosine: " << meanCosine << "\t kappa: " << kappa << "\t newMeanCosine: " << newMeanCosine0 << " \t newKkappa: " <<  newKkappa0 << std::endl;
        // std::cout << "localMeanDirection0: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D0) << "\t meanDirection0: " << meanDirection0 << "\t meanCosine: " << meanCosine << "
        // \t costheta0: " <<  dot(meanDirection, meanDirection0) << std::endl; std::cout << "localMeanDirection1: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D1) << "\t
        // meanDirection1: " << meanDirection1 << "\t meanCosine: " << meanCosine << " \t costheta1: " <<  dot(meanDirection, meanDirection1) << std::endl; std::cout <<
        // "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl; std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t
        // eigenVector1: " << splitInfo.eigenVector1 << std::endl;
        std::cout << "D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;
        // std::cout << "kappa: " << kappa <<  " \t newKkappa: " <<  newKkappa0  << " \t costheta0: " <<  dot(meanDirection, meanDirection0) << "\t angle: " <<
        // std::acos(dot(meanDirection, meanDirection0)) * 180.0f / M_PI_F<< std::endl;
#endif
    }
    else
    {
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << "!!!!   D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;

        std::cout << "sampleCovariance: [" << splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] << ",\t"
                  << splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] << ",\t" << splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] << "]"
                  << std::endl;
        std::cout << "sumWeights: " << splitStats.sumWeights[tmpK.quot][tmpK.rem] << std::endl;
        std::cout << "weight: " << weight << "\t meanCosine: " << meanCosine << std::endl;
#endif
        if (numAssignedSamples < 2.0f)
        {
            splitStats.chiSquareMCEstimates[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.chiSquareMCEstimate2ndMoments[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.weightsEstimates[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.weights2ndmomentEstimates[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.weightsVarianceEstimates[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.numWeightsEstimatesSamples[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.numSamples[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.sumWeights[tmpK.quot][tmpK.rem] = 0.0f;
            splitStats.splitMeans[tmpK.quot].x[tmpK.rem] = 0.0f;
            splitStats.splitMeans[tmpK.quot].y[tmpK.rem] = 0.0f;
            splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] = 0.0f;
            splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] = 0.0f;
            splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] = 0.0f;
            return false;
        }
    }

    // std::cout << "meanDirection = " << meanDirection << "\t meanDirection0 = " << meanDirection0 << "\t meanDirection1 = " << meanDirection1 << "\t cos0 = " <<
    // dot(meanDirection, meanDirection0) << "\t cos1 = " << dot(meanDirection, meanDirection1) << "\t cos = " << dot(meanDirection1, meanDirection0) << std::endl; std::cout <<
    // "kappa = " << vmm._kappas[tmpK.quot][tmpK.rem] << "\t meanCosine = " << meanCosine << "\t newMeanCosine0 = " << newMeanCosine0 << "\t newMeanCosine1 = " << newMeanCosine1 <<
    // std::endl;

    size_t K = vmm._numComponents;

    const div_t tmpI = tmpK;
    const div_t tmpJ = div(K, static_cast<int>(VMM::VectorSize));
    //    VMM vmmOrg = vmm;
    vmm.splitComponent(idx, K, newWeight0, newWeight1, meanDirection0, meanDirection1, newMeanCosine0, newMeanCosine1);
    // std::cout << "kappa0 = " << vmm._kappas[tmpI.quot][tmpI.rem] << "\t kappa1 = " << vmm._kappas[tmpJ.quot][tmpJ.rem] << std::endl;

    /*
        VMM vmmSplit = vmm;
        VMM vmmMerge = vmmSplit;
        vmmMerge.mergeComponents(idx, K);
        std::cout << "VMMOrg:   " << std::endl <<  vmmOrg.toString() << std::endl;
        std::cout << "vmmMerge: " << std::endl <<  vmmMerge.toString() << std::endl;
        std::cout << "vmmSplit: " << std::endl <<  vmmSplit.toString() << std::endl;

        SufficientStatistics suffStatsOrg = suffStats;
    */
    suffStats.splitComponentsStats(idx, K, meanDirection0, meanDirection1, newMeanCosine0, newMeanCosine1);
    /*    SufficientStatistics suffStatsSplit = suffStats;
        SufficientStatistics suffStatsMerge = suffStatsSplit;
        suffStatsMerge.mergeComponentStats(idx, K);

        std::cout << "suffStatsOrg:   " << std::endl <<  suffStatsOrg.toString() << std::endl;
        std::cout << "suffStatsMerge: " << std::endl <<  suffStatsMerge.toString() << std::endl;
        std::cout << "suffStatsSplit: " << std::endl <<  suffStatsSplit.toString() << std::endl;
    */
    // reseting the split statistics for the two new components
    splitStats.chiSquareMCEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.chiSquareMCEstimate2ndMoments[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weightsEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weights2ndmomentEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weightsVarianceEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numWeightsEstimatesSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumWeights[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem] = 0.0f;

    splitStats.chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.chiSquareMCEstimate2ndMoments[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weightsEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weights2ndmomentEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weightsVarianceEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numWeightsEstimatesSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumWeights[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem] = 0.0f;

    splitStats.numComponents = K + 1;

    return true;
}

/**
 * Spilts or better adds a commenent that represents a firelfy observed during the last training iteration.
 */
template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitComponentFireFly(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats,
                                                                                    const size_t idx, const Configuration &cfg) const
{
    OPENPGL_ASSERT(vmm.isValid());
    ComponentSplitinfoV2 splitInfo;
    const div_t tmpK = div(idx, static_cast<int>(VMM::VectorSize));

    if (splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem] < 1.0f || splitStats.weights[tmpK.quot][tmpK.rem] < FLT_EPSILON)
    {
        return false;
    }
    float weight = vmm._weights[tmpK.quot][tmpK.rem];
    float meanCosine = vmm._meanCosines[tmpK.quot][tmpK.rem];
    float kappa = vmm._kappas[tmpK.quot][tmpK.rem];

    // the old component mean
    Vector3 meanDirection = Vector3(vmm._meanDirections[tmpK.quot].x[tmpK.rem], vmm._meanDirections[tmpK.quot].y[tmpK.rem], vmm._meanDirections[tmpK.quot].z[tmpK.rem]);

    // the new firefly component mean
    Vector3 newMeanDirection =
        Vector3(splitStats.weightedMeans[tmpK.quot].x[tmpK.rem], splitStats.weightedMeans[tmpK.quot].y[tmpK.rem], splitStats.weightedMeans[tmpK.quot].z[tmpK.rem]);
    OPENPGL_ASSERT(splitStats.weights[tmpK.quot][tmpK.rem] > 0.f);
    newMeanDirection /= splitStats.weights[tmpK.quot][tmpK.rem];
    OPENPGL_ASSERT(embree::isvalid(newMeanDirection.x));
    OPENPGL_ASSERT(embree::isvalid(newMeanDirection.y));
    OPENPGL_ASSERT(embree::isvalid(newMeanDirection.z));
    float newMeanCosine = embree::length(newMeanDirection);
    OPENPGL_ASSERT(newMeanCosine > 0.f);
    newMeanDirection /= newMeanCosine;

    // the fraction calcualtes the ratio the energy/weight changed due to adding the firefly
    float frac = splitStats.weights[tmpK.quot][tmpK.rem] / (suffStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] * suffStats.inv_norm);
    float fracOld = frac;
    frac = std::min(0.9f, frac);

    //    std::cout << "firefly sumWeight: " << splitStats.weights[tmpK.quot][tmpK.rem] << std::endl;
    //    std::cout << "frac: "<< frac << "\t fracOld = " << fracOld << "\t split.weights = "<< splitStats.weights[tmpK.quot][tmpK.rem] << "\t suffStats = " <<
    //    suffStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] * suffStats.inv_norm <<  std::endl; std::cout << "meanCosine0: "<< vmm._meanCosines[tmpK.quot][tmpK.rem] << "\t
    //    meanCosine1: " << newMeanCosine << std::endl;

    //    std::cout << "oldMeanDirection: " << vmm._meanDirections[tmpK.quot].x[tmpK.rem] << "\t" << vmm._meanDirections[tmpK.quot].y[tmpK.rem] << "\t"<<
    //    vmm._meanDirections[tmpK.quot].z[tmpK.rem] <<  std::endl; std::cout << "newMeanDirection: " << newMeanDirection.x << "\t" << newMeanDirection.y << "\t"<<
    //    newMeanDirection.z <<  std::endl;
    // the weight of the old component is the approximated weight without the firefly
    float newWeight0 = (1.f - frac) * weight;
    // the weight of the firefly component is the approximated weight of the firely
    float newWeight1 = frac * weight;

    //    std::cout << "oldWeight = "<< (suffStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] / suffStats.numSamples) << "\t newWeight0: " << newWeight0 << "\t newWeight1: " <<
    //    newWeight1 << std::endl;

    float partialNumSamples = newWeight1 * suffStats.overallNumSamples;
    // newMeanCosine = ( cfg.meanCosinePrior * cfg.meanCosinePriorStrength + newMeanCosine * partialNumSamples ) / ( cfg.meanCosinePriorStrength + partialNumSamples );
    newMeanCosine = embree::min(cfg.maxMeanCosine, newMeanCosine);

    // Vector3 meanDirection0 =
    size_t K = vmm._numComponents;

    const div_t tmpI = tmpK;
    const div_t tmpJ = div(K, static_cast<int>(VMM::VectorSize));

    // vmm.splitComponent(idx, K, newWeight0, newWeight1, meanDirection, newMeanDirection, meanCosine, newMeanCosine);
    vmm._weights[tmpI.quot][tmpI.rem] = newWeight0;

    vmm._weights[tmpJ.quot][tmpJ.rem] = newWeight1;
    vmm._meanDirections[tmpJ.quot].x[tmpJ.rem] = newMeanDirection.x;
    vmm._meanDirections[tmpJ.quot].y[tmpJ.rem] = newMeanDirection.y;
    vmm._meanDirections[tmpJ.quot].z[tmpJ.rem] = newMeanDirection.z;
    vmm._meanCosines[tmpJ.quot][tmpJ.rem] = newMeanCosine;
    vmm._kappas[tmpJ.quot][tmpJ.rem] = MeanCosineToKappa<float>(newMeanCosine);
    vmm._distances[tmpJ.quot][tmpJ.rem] = vmm._distances[tmpI.quot][tmpI.rem];
    vmm._numComponents++;
    vmm._calculateNormalization();

    // suffStats.splitComponentsStats(idx, K, meanDirection0, meanDirection1, newMeanCosine0, newMeanCosine1);
    suffStats.numComponents += 1;

    suffStats.sumOfWeightedStats[tmpJ.quot][tmpJ.rem] = suffStats.sumOfWeightedStats[tmpI.quot][tmpI.rem];
    suffStats.sumOfWeightedStats[tmpI.quot][tmpI.rem] *= 1.f - frac;
    suffStats.sumOfWeightedStats[tmpJ.quot][tmpJ.rem] *= frac;

    suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem] *= 1.f - frac;
    suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem] *= 1.f - frac;
    suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem] *= 1.f - frac;

    newMeanDirection *= newMeanCosine * suffStats.sumOfWeightedStats[tmpJ.quot][tmpJ.rem];
    suffStats.sumOfWeightedDirections[tmpJ.quot].x[tmpJ.rem] = newMeanDirection.x;
    suffStats.sumOfWeightedDirections[tmpJ.quot].y[tmpJ.rem] = newMeanDirection.y;
    suffStats.sumOfWeightedDirections[tmpJ.quot].z[tmpJ.rem] = newMeanDirection.z;
    // sumOfWeightedDirections[VMM::NumVectors]
    // suffStats.numComponents += 1;
    //  reseting the split statistics for the two new components

    splitStats.chiSquareMCEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.chiSquareMCEstimate2ndMoments[tmpI.quot][tmpI.rem] = 0.0f;

    splitStats.weightsEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weights2ndmomentEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weightsVarianceEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numWeightsEstimatesSamples[tmpI.quot][tmpI.rem] = 0.0f;

    splitStats.sumAssignedSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumWeights[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem] = 0.0f;
    splitStats.weights[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weightedMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.weightedMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.weightedMeans[tmpI.quot].z[tmpI.rem] = 0.0f;

    splitStats.chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.chiSquareMCEstimate2ndMoments[tmpJ.quot][tmpJ.rem] = 0.0f;

    splitStats.weightsEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weights2ndmomentEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weightsVarianceEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numWeightsEstimatesSamples[tmpJ.quot][tmpJ.rem] = 0.0f;

    splitStats.sumAssignedSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumWeights[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem] = 0.0f;
    splitStats.weights[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weightedMeans[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.weightedMeans[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.weightedMeans[tmpJ.quot].z[tmpJ.rem] = 0.0f;

    splitStats.numComponents = K + 1;
    OPENPGL_ASSERT(splitStats.isValid());
    OPENPGL_ASSERT(vmm.isValid());
    return true;
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitComponentIntoThree(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatistics &suffStats,
                                                                                      const size_t idx) const
{
    ComponentSplitinfoV2 splitInfo;
    const div_t tmpK = div(idx, static_cast<int>(VMM::VectorSize));

    float numAssignedSamples = splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem];

    float inv_sumWeights = rcp(splitStats.sumWeights[tmpK.quot][tmpK.rem]);
    splitInfo.mean = Vector2(splitStats.splitMeans[tmpK.quot].x[tmpK.rem], splitStats.splitMeans[tmpK.quot].y[tmpK.rem]);

    splitInfo.covariance.x = splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.y = splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.z = splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] * inv_sumWeights;

    float D = embree::sqrt((splitInfo.covariance.x - splitInfo.covariance.y) * (splitInfo.covariance.x - splitInfo.covariance.y) +
                           (splitInfo.covariance.z * splitInfo.covariance.z * 4.0f)) *
              0.5f;
    splitInfo.eigenValue0 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue0 += D;

    splitInfo.eigenValue1 = (splitInfo.covariance.x + splitInfo.covariance.y) * 0.5;
    splitInfo.eigenValue1 -= D;

    splitInfo.eigenVector0.x = -splitInfo.covariance.z;
    splitInfo.eigenVector0.y = splitInfo.covariance.x - splitInfo.eigenValue0;

    splitInfo.eigenVector1.x = splitInfo.covariance.z;
    splitInfo.eigenVector1.y = splitInfo.covariance.x - splitInfo.eigenValue1;

    splitInfo.eigenVector0 /= embree::sqrt(splitInfo.eigenVector0.x * splitInfo.eigenVector0.x + splitInfo.eigenVector0.y * splitInfo.eigenVector0.y);
    splitInfo.eigenVector1 /= embree::sqrt(splitInfo.eigenVector1.x * splitInfo.eigenVector1.x + splitInfo.eigenVector1.y * splitInfo.eigenVector1.y);
    /* */
    // std::cout << "D: " << D << std::endl;
    // std::cout << "sumWeights: " << splitStats.sumWeights[tmpK.quot][tmpK.rem] << "\t inSumWeights: " << inv_sumWeights << std::endl;
    // std::cout << "splitMean: " << splitInfo.mean << "\t splitCovariance: " << splitInfo.covariance << std::endl;

    // std::cout << "splitCovariancesRaw: " << splitStats.splitCovariances[tmpK.quot].x[tmpK.rem] << "\t" << splitStats.splitCovariances[tmpK.quot].y[tmpK.rem] << "\t" <<
    // splitStats.splitCovariances[tmpK.quot].z[tmpK.rem] << std::endl;
    //    std::cout << "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl;
    //    std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t eigenVector1: " << splitInfo.eigenVector1 << std::endl;
    /**/

    float weight = vmm._weights[tmpK.quot][tmpK.rem];
    float meanCosine = vmm._meanCosines[tmpK.quot][tmpK.rem];
    float kappa = vmm._kappas[tmpK.quot][tmpK.rem];

    if (kappa >= OPENPGL_MAX_KAPPA * 0.9)
    {
        return false;
    }

    Vector3 meanDirection = Vector3(vmm._meanDirections[tmpK.quot].x[tmpK.rem], vmm._meanDirections[tmpK.quot].y[tmpK.rem], vmm._meanDirections[tmpK.quot].z[tmpK.rem]);

    float distance = vmm._distances[tmpK.quot][tmpK.rem];

    float newWeight0 = weight * embree::rcp(3.0f);
    float newWeight1 = newWeight0;
    float newWeight2 = newWeight0;

    Vector3 meanDirection0 = meanDirection;
    Vector3 meanDirection1 = meanDirection;

    float newMeanCosine0 = meanCosine;
    float newMeanCosine1 = meanCosine * meanCosine;

    float newKkappa0 = MeanCosineToKappa<float>(newMeanCosine0);
    float newKkappa1 = MeanCosineToKappa<float>(newMeanCosine1);

    if (D > 1e-8f)
    {
        Vector2 meanDir2D0 = splitInfo.mean + (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 1.0f);
        meanDirection0 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D0);
        newMeanCosine0 = meanCosine / dot(meanDirection, meanDirection0);
        // ensure that the new mean cosine is in a valid range (i.e., < 1.0 and < the mean cosine of max kappa)
        newMeanCosine0 = std::min(newMeanCosine0, KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA));
        newMeanCosine1 = newMeanCosine0;
        newKkappa0 = MeanCosineToKappa<float>(newMeanCosine0);
        newKkappa1 = newKkappa0;

        Vector2 meanDir2D1 = splitInfo.mean - (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 1.0f);
        meanDirection1 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D1);
        // float meanCosine1 = meanCosine / meanDirection0.z;
        // float kappa1 = MeanCosineToKappa<float> (meanCosine1);
#ifdef OPENPGL_SHOW_PRINT_OUTS
        // std::cout << "meanCosine: " << meanCosine << "\t kappa: " << kappa << "\t newMeanCosine: " << newMeanCosine0 << " \t newKkappa: " <<  newKkappa0 << std::endl;
        // std::cout << "localMeanDirection0: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D0) << "\t meanDirection0: " << meanDirection0 << "\t meanCosine: " << meanCosine << "
        // \t costheta0: " <<  dot(meanDirection, meanDirection0) << std::endl; std::cout << "localMeanDirection1: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D1) << "\t
        // meanDirection1: " << meanDirection1 << "\t meanCosine: " << meanCosine << " \t costheta1: " <<  dot(meanDirection, meanDirection1) << std::endl; std::cout <<
        // "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl; std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t
        // eigenVector1: " << splitInfo.eigenVector1 << std::endl;
        std::cout << "D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;
        std::cout << "kappa: " << kappa << " \t newKkappa: " << newKkappa0 << " \t costheta0: " << dot(meanDirection, meanDirection0)
                  << "\t angle: " << std::acos(dot(meanDirection, meanDirection0)) * 180.0f / M_PI_F << std::endl;
#endif
    }
    else
    {
#ifdef OPENPGL_SHOW_PRINT_OUTS
        std::cout << "!!!!   D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples << std::endl;

        std::cout << "sampleCovariance: [" << splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] << ",\t"
                  << splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] << ",\t" << splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] << "]"
                  << std::endl;
        std::cout << "sumWeights: " << splitStats.sumWeights[tmpK.quot][tmpK.rem] << std::endl;
        std::cout << "weight: " << weight << "\t meanCosine: " << meanCosine << std::endl;
#endif
        if (numAssignedSamples < 2.0f)
        {
            return false;
        }
    }
    size_t K = vmm._numComponents;
    // vmm.swapComponents(K-1, idx);
    // suffStats.swapComponentStats(K-1, idx);
    // const div_t tmpI = div(K-1, static_cast<int>(VMM::VectorSize));
    const div_t tmpI = tmpK;
    const div_t tmpJ = div(K, static_cast<int>(VMM::VectorSize));
    const div_t tmpL = div(K + 1, static_cast<int>(VMM::VectorSize));

    vmm._weights[tmpI.quot][tmpI.rem] = newWeight0;
    vmm._meanCosines[tmpI.quot][tmpI.rem] = newMeanCosine0;
    vmm._kappas[tmpI.quot][tmpI.rem] = newKkappa0;
    vmm._meanDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x;
    vmm._meanDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y;
    vmm._meanDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z;
    vmm._distances[tmpI.quot][tmpI.rem] = distance;

    vmm._weights[tmpJ.quot][tmpJ.rem] = newWeight1;
    vmm._meanCosines[tmpJ.quot][tmpJ.rem] = newMeanCosine1;
    vmm._kappas[tmpJ.quot][tmpJ.rem] = newKkappa1;
    vmm._meanDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x;
    vmm._meanDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y;
    vmm._meanDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z;
    vmm._distances[tmpJ.quot][tmpJ.rem] = distance;

    vmm._weights[tmpL.quot][tmpL.rem] = newWeight2;
    vmm._meanCosines[tmpL.quot][tmpL.rem] = meanCosine;
    vmm._kappas[tmpL.quot][tmpL.rem] = kappa;
    vmm._meanDirections[tmpL.quot].x[tmpL.rem] = meanDirection.x;
    vmm._meanDirections[tmpL.quot].y[tmpL.rem] = meanDirection.y;
    vmm._meanDirections[tmpL.quot].z[tmpL.rem] = meanDirection.z;
    vmm._distances[tmpL.quot][tmpL.rem] = distance;

    vmm._numComponents = K + 2;
    vmm._calculateNormalization();

    float sumStatsWeight = suffStats.sumOfWeightedStats[tmpK.quot][tmpK.rem];
    sumStatsWeight /= 3.0f;

    suffStats.sumOfWeightedStats[tmpI.quot][tmpI.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x * newMeanCosine0 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y * newMeanCosine0 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z * newMeanCosine0 * sumStatsWeight;

    suffStats.sumOfWeightedStats[tmpJ.quot][tmpJ.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x * newMeanCosine1 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y * newMeanCosine1 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z * newMeanCosine1 * sumStatsWeight;

    suffStats.sumOfWeightedStats[tmpL.quot][tmpL.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpL.quot].x[tmpL.rem] = meanDirection.x * meanCosine * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpL.quot].y[tmpL.rem] = meanDirection.y * meanCosine * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpL.quot].z[tmpL.rem] = meanDirection.z * meanCosine * sumStatsWeight;

    suffStats.numComponents = K + 2;

    OPENPGL_ASSERT(!std::isnan(suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem]) && std::isfinite(suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem]));
    OPENPGL_ASSERT(!std::isnan(suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem]) && std::isfinite(suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem]));
    OPENPGL_ASSERT(!std::isnan(suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem]) && std::isfinite(suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem]));

    // reseting the split statistics for the two new components
    splitStats.chiSquareMCEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.chiSquareMCEstimate2ndMoments[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weightsEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weights2ndmomentEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.weightsVarianceEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numWeightsEstimatesSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumWeights[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem] = 0.0f;

    splitStats.chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.chiSquareMCEstimate2ndMoments[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weightsEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weights2ndmomentEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.weightsVarianceEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numWeightsEstimatesSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumWeights[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem] = 0.0f;

    splitStats.chiSquareMCEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.chiSquareMCEstimate2ndMoments[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.weightsEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.weights2ndmomentEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.weightsVarianceEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.numWeightsEstimatesSamples[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.numSamples[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.sumWeights[tmpL.quot][tmpL.rem] = 0.0f;
    splitStats.splitMeans[tmpL.quot].x[tmpL.rem] = 0.0f;
    splitStats.splitMeans[tmpL.quot].y[tmpL.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpL.quot].x[tmpL.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpL.quot].y[tmpL.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpL.quot].z[tmpL.rem] = 0.0f;

    splitStats.numComponents = K + 2;

    return true;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::serialize(std::ostream &stream) const
{
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimates);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimate2ndMoments);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weightsEstimates);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weights2ndmomentEstimates);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weightsVarianceEstimates);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numWeightsEstimatesSamples);
    serializeVec2Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitMeans);
    serializeVec3Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitWeightedSampleCovariances);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numSamples);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumWeights);
    serializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumAssignedSamples);
    serializeIntVectors<VMM::NumVectors, VMM::VectorSize>(stream, splitType);
    stream.write(reinterpret_cast<const char *>(&numComponents), sizeof(size_t));
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::deserialize(std::istream &stream)
{
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimates);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, chiSquareMCEstimate2ndMoments);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weightsEstimates);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weights2ndmomentEstimates);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, weightsVarianceEstimates);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numWeightsEstimatesSamples);
    deserializeVec2Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitMeans);
    deserializeVec3Vectors<VMM::NumVectors, VMM::VectorSize>(stream, splitWeightedSampleCovariances);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, numSamples);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumWeights);
    deserializeFloatVectors<VMM::NumVectors, VMM::VectorSize>(stream, sumAssignedSamples);
    deserializeIntVectors<VMM::NumVectors, VMM::VectorSize>(stream, splitType);
    stream.read(reinterpret_cast<char *>(&numComponents), sizeof(size_t));
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::isValid() const
{
    bool valid = true;

    embree::vbool<VMM::VectorSize> validVec(true);
    const int cnt = (VMM::MaxComponents + VMM::VectorSize - 1) / VMM::VectorSize;
    for (size_t k = 0; k < cnt; k++)
    {
        validVec &= embree::isvalid(splitMeans[k].x);
        validVec &= embree::isvalid(splitMeans[k].y);
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(splitWeightedSampleCovariances[k].x);
        validVec &= embree::isvalid(splitWeightedSampleCovariances[k].y);
        validVec &= embree::isvalid(splitWeightedSampleCovariances[k].z);
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(chiSquareMCEstimates[k]);
        validVec &= chiSquareMCEstimates[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(chiSquareMCEstimate2ndMoments[k]);
        validVec &= chiSquareMCEstimate2ndMoments[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(weightsEstimates[k]);
        validVec &= weightsEstimates[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(weights2ndmomentEstimates[k]);
        validVec &= weights2ndmomentEstimates[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(weightsVarianceEstimates[k]);
        validVec &= weightsVarianceEstimates[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(numWeightsEstimatesSamples[k]);
        validVec &= numWeightsEstimatesSamples[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(sumWeights[k]);
        validVec &= sumWeights[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(sumAssignedSamples[k]);
        validVec &= sumAssignedSamples[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));

        validVec &= embree::isvalid(numSamples[k]);
        validVec &= numSamples[k] >= 0.0f;
        OPENPGL_ASSERT(embree::any(validVec));
    }

    valid = valid && embree::any(validVec);
    OPENPGL_ASSERT(valid);
    valid = valid && numComponents > 0;
    valid = valid && numComponents <= VMM::MaxComponents;
    OPENPGL_ASSERT(valid);

    return valid;
}

template <class TVMMFactory>
bool VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::operator==(const ComponentSplitStatistics &b) const
{
    bool equal = true;
    if (numComponents != b.numComponents)
    {
        equal = false;
    }

    for (int k = 0; k < VMM::NumVectors; k++)
    {
        if (embree::any(chiSquareMCEstimates[k] != b.chiSquareMCEstimates[k]) || embree::any(chiSquareMCEstimate2ndMoments[k] != b.chiSquareMCEstimate2ndMoments[k]) ||
            embree::any(weightsEstimates[k] != b.weightsEstimates[k]) || embree::any(weights2ndmomentEstimates[k] != b.weights2ndmomentEstimates[k]) ||
            embree::any(weightsVarianceEstimates[k] != b.weightsVarianceEstimates[k]) || embree::any(numWeightsEstimatesSamples[k] != b.numWeightsEstimatesSamples[k]) ||
            embree::any(splitMeans[k].x != b.splitMeans[k].x) || embree::any(splitMeans[k].y != b.splitMeans[k].y) ||
            embree::any(splitWeightedSampleCovariances[k].x != b.splitWeightedSampleCovariances[k].x) ||
            embree::any(splitWeightedSampleCovariances[k].y != b.splitWeightedSampleCovariances[k].y) ||
            embree::any(splitWeightedSampleCovariances[k].z != b.splitWeightedSampleCovariances[k].z) || embree::any(numSamples[k] != b.numSamples[k]) ||
            embree::any(sumWeights[k] != b.sumWeights[k]) || embree::any(sumAssignedSamples[k] != b.sumAssignedSamples[k]))
        {
            equal = false;
        }
    }

    return equal;
}

template <class TVMMFactory>
Vector2 VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitMean(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return Vector2(splitMeans[tmp.quot].x[tmp.rem], splitMeans[tmp.quot].y[tmp.rem]);
}

template <class TVMMFactory>
Vector3 VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitCovariance(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    Vector3 covariance(splitWeightedSampleCovariances[tmp.quot].x[tmp.rem], splitWeightedSampleCovariances[tmp.quot].y[tmp.rem],
                       splitWeightedSampleCovariances[tmp.quot].z[tmp.rem]);
    covariance /= sumWeights[tmp.quot][tmp.rem];
    return covariance;
}

template <class TVMMFactory>
typename VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitType VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitType(
    const size_t &idx, const VMM &vmm, const VMM &previousVMM, const ComponentSplitStatistics &previousSplitStats) const
{
    // const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    // float frac = splitStats.weights[tmpK.quot][tmpK.rem] / (suffStats.sumOfWeightedStats[tmpK.quot][tmpK.rem] * suffStats.inv_norm);
    SplitType splitType = EMultiModal;
    SplitType splitTypeV2 = getSplitType(idx);
    float weightOld = previousVMM.getComponentWeight(idx);
    float weightNew = vmm.getComponentWeight(idx);
    float relativeChange = weightNew / weightOld;

    float weightOldV2 = previousSplitStats.getWeightsEst(idx);
    float weightNewV2 = getWeightsEst(idx);
    float relativeWeightChangeV2 = weightNewV2 / weightOldV2;

    float relVarianceOldV2 = previousSplitStats.getRelVarianceEst(idx);
    float relVarianceNewV2 = getRelVarianceEst(idx);
    float relativeVarianceChangeV2 = relVarianceNewV2 / relVarianceOldV2;

    float varianceOldV2 = previousSplitStats.getVarianceEst(idx);
    float varianceNewV2 = getVarianceEst(idx);
    float varianceChangeV2 = varianceNewV2 / varianceOldV2;
    // if (weightOld > 0.f && !embree::isvalid(relativeChange))
    //     int i = 0;
    // if (weightOld > 0.f)
    // std::cout << "getSplitType: idx = " << idx << "\t relativeChange = " << relativeChange << std::endl;
    if (weightOld <= 0.f)
    {
        std::cout << "getSplitType: relativeChange = " << relativeChange << std::endl;
    }

    // if (weightOld > 0.f && relativeChange > 1.25f)
    // if (weightOld > 0.f && relativeChangeV2 > 1.25f)
    // if (weightOld > 0.f && relativeVarianceChangeV2 > 2.0f)
    if (weightOld > 0.f && (varianceChangeV2 > 1.5f || relativeWeightChangeV2 > 1.5f))
    {
        splitType = EFirefly;
    }
    std::cout << "getSplitType: idx = " << idx << "\t relativeChange = " << relativeChange << "\t relativeWeightChangeV2 = " << relativeWeightChangeV2
              << "\t varianceChangeV2 = " << varianceChangeV2 << "\t relativeVarianceChangeV2 = " << relativeVarianceChangeV2
              << "\t type = " << (splitType == EFirefly ? "FireFly" : "MultiModal") << "\t typeV2 = " << (splitTypeV2 == EFirefly ? "FireFly" : "MultiModal") << std::endl;

    // TODO fix EMultiModal
    // splitType = EFirefly;
    // splitType = EMultiModal;
    return splitType;
}

template <class TVMMFactory>
typename VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitType VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitType(
    const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return (VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitType)splitType[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::mergeComponentStats(const size_t &idxI, const size_t &idxJ, const float &weightI,
                                                                                                            const Vector3 &meanDirectionI, const float &weightJ,
                                                                                                            const Vector3 &meanDirectionJ, const float &weightK,
                                                                                                            const Vector3 &meanDirectionK)
{
    // TODO: check if numSamples or sumWeights are zero for one of the merge components
    // std::cout << "mergeComponentStats: " << "\tidxI: " << idxI << "\tidxJ: " << idxJ << "\tweightI: " << weightI << "\tmeanDirectionI: " << meanDirectionI<< "\tweightJ: " <<
    // weightJ << "\tmeanDirectionJ: " << meanDirectionJ<< "\tweightK: " << weightK << "\tmeanDirectionK: " << meanDirectionK << std::endl;

    // EM algorithms for Gaussian mixtures with split-and-merge operation
    const div_t tmpI = div(idxI, static_cast<int>(VMM::VectorSize));
    const div_t tmpJ = div(idxJ, static_cast<int>(VMM::VectorSize));

    const div_t tmpL = div(numComponents - 1, VMM::VectorSize);

    auto transformK = embree::frame(meanDirectionK);
    auto inv_transformK = transformK.inverse();
#ifdef OPENPGL_ZERO_MEAN
    Vector3 meanDirectionI3D = meanDirectionI;
    Vector3 meanDirectionJ3D = meanDirectionJ;
#else
    auto transformI = embree::frame(meanDirectionI);
    auto transformJ = embree::frame(meanDirectionJ);
    Vector2 meanDirection2DI = Vector2(splitMeans[tmpI.quot].x[tmpI.rem], splitMeans[tmpI.quot].y[tmpI.rem]);
    Vector2 meanDirection2DJ = Vector2(splitMeans[tmpJ.quot].x[tmpJ.rem], splitMeans[tmpJ.quot].y[tmpJ.rem]);

    Vector3 meanDirectionI3D = transformI * Map2DTo3D<Vector3, Vector2, float>(meanDirection2DI);
    Vector3 meanDirectionJ3D = transformJ * Map2DTo3D<Vector3, Vector2, float>(meanDirection2DJ);

#endif

    Vector2 meanDirection2DItoK = Map3DTo2D<Vector3, Vector2, float>(inv_transformK * meanDirectionI3D);
    Vector2 meanDirection2DJtoK = Map3DTo2D<Vector3, Vector2, float>(inv_transformK * meanDirectionJ3D);

#ifdef APPLY_PATCH
    const float inv_weightK = (weightK > FLT_EPSILON) ? embree::rcp(weightK) : 1.f;
#else
    const float inv_weightK = (weightK > 0.f) ? embree::rcp(weightK) : 1.f;
#endif
    const float sumWeightsI = sumWeights[tmpI.quot][tmpI.rem];
    const float sumWeightsJ = sumWeights[tmpJ.quot][tmpJ.rem];
    const float sumWeightsK = sumWeightsI + sumWeightsJ;

    // std::cout << "\tsumWeightsI: " << sumWeightsI << "\tsumWeightsJ: " << sumWeightsJ << "\tsumWeightsK: " << sumWeightsK << std::endl;
    // std::cout << "\tnumSamplesI: " << numSamples[tmpI.quot][tmpI.rem] << "\tsumWeightsJ: " << numSamples[tmpJ.quot][tmpJ.rem] << std::endl;

#ifdef APPLY_PATCH
    const Vector3 covarianceI = (sumWeightsI > FLT_EPSILON) ? Vector3(splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem], splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem],
                                                                      splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem]) *
                                                                  embree::rcp(sumWeightsI)
                                                            : Vector3(0.f);
    const Vector3 covarianceJ = (sumWeightsJ > FLT_EPSILON) ? Vector3(splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem], splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem],
                                                                      splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem]) *
                                                                  embree::rcp(sumWeightsJ)
                                                            : Vector3(0.f);
#else
    const Vector3 covarianceI = (sumWeightsI > 0.f) ? Vector3(splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem], splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem],
                                                              splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem]) *
                                                          embree::rcp(sumWeightsI)
                                                    : Vector3(0.f);
    const Vector3 covarianceJ = (sumWeightsJ > 0.f) ? Vector3(splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem], splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem],
                                                              splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem]) *
                                                          embree::rcp(sumWeightsJ)
                                                    : Vector3(0.f);
#endif

#ifdef OPENPGL_ZERO_MEAN
    const Vector2 meanDirectionK2D(0.f);
#else
    const Vector2 meanDirectionK2D = inv_weightK * (weightI * meanDirection2DItoK + weightJ * meanDirection2DJtoK);
#endif
    Vector3 meanII = Vector3(meanDirection2DItoK.x * meanDirection2DItoK.x, meanDirection2DItoK.y * meanDirection2DItoK.y, meanDirection2DItoK.x * meanDirection2DItoK.y);
    Vector3 meanJJ = Vector3(meanDirection2DJtoK.x * meanDirection2DJtoK.x, meanDirection2DJtoK.y * meanDirection2DJtoK.y, meanDirection2DJtoK.x * meanDirection2DJtoK.y);
    Vector3 covarianceK = (weightI * covarianceI + weightI * meanII + weightJ * covarianceJ + weightJ * meanJJ);
    covarianceK *= inv_weightK;
#ifndef OPENPGL_ZERO_MEAN
    Vector3 meanKK = Vector3(meanDirectionK2D.x * meanDirectionK2D.x, meanDirectionK2D.y * meanDirectionK2D.y, meanDirectionK2D.x * meanDirectionK2D.y);
    covarianceK -= meanKK;
#endif
    const Vector3 sampleCovarianceK = covarianceK * sumWeightsK;
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.x));
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.y));
    OPENPGL_ASSERT(embree::isvalid(sampleCovarianceK.z));

    // merge additional stats
    const float sumAssignedSamplesK = sumAssignedSamples[tmpI.quot][tmpI.rem] + sumAssignedSamples[tmpJ.quot][tmpJ.rem];
    const float numSamplesK = inv_weightK * (weightI * numSamples[tmpI.quot][tmpI.rem] + weightJ * numSamples[tmpJ.quot][tmpJ.rem]);
    const float chiSquareMCEstimatesK = chiSquareMCEstimates[tmpI.quot][tmpI.rem] + chiSquareMCEstimates[tmpJ.quot][tmpJ.rem];
    const float chiSquareMCEstimate2ndMomentsK = chiSquareMCEstimate2ndMoments[tmpI.quot][tmpI.rem] + chiSquareMCEstimate2ndMoments[tmpJ.quot][tmpJ.rem];

    const float weightsEstimateK = inv_weightK * (weightI * weightsEstimates[tmpI.quot][tmpI.rem] + weightJ * weightsEstimates[tmpJ.quot][tmpJ.rem]);
    const float weights2ndmomentEstimateK = inv_weightK * (weightI * weights2ndmomentEstimates[tmpI.quot][tmpI.rem] + weightJ * weights2ndmomentEstimates[tmpJ.quot][tmpJ.rem]);
    const float weightsVarianceEstimatesK = inv_weightK * (weightI * weightsVarianceEstimates[tmpI.quot][tmpI.rem] + weightJ * weightsVarianceEstimates[tmpJ.quot][tmpJ.rem]);
    const float numWeightsEstimatesSamplesK = inv_weightK * (weightI * numWeightsEstimatesSamples[tmpI.quot][tmpI.rem] + weightJ * numWeightsEstimatesSamples[tmpJ.quot][tmpJ.rem]);

    // insert stats of the merged components a the ith positions
#ifdef OPENPGL_ZERO_MEAN
    splitMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
#else
    splitMeans[tmpI.quot].x[tmpI.rem] = meanDirectionK2D.x;
    splitMeans[tmpI.quot].y[tmpI.rem] = meanDirectionK2D.y;
#endif
    splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem] = sampleCovarianceK.x;
    splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem] = sampleCovarianceK.y;
    splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem] = sampleCovarianceK.z;

    sumWeights[tmpI.quot][tmpI.rem] = sumWeightsK;
    numSamples[tmpI.quot][tmpI.rem] = numSamplesK;
    sumAssignedSamples[tmpI.quot][tmpI.rem] = sumAssignedSamplesK;
    chiSquareMCEstimates[tmpI.quot][tmpI.rem] = chiSquareMCEstimatesK;
    chiSquareMCEstimate2ndMoments[tmpI.quot][tmpI.rem] = chiSquareMCEstimate2ndMomentsK;

    weightsEstimates[tmpI.quot][tmpI.rem] = weightsEstimateK;
    weights2ndmomentEstimates[tmpI.quot][tmpI.rem] = weights2ndmomentEstimateK;
    weightsVarianceEstimates[tmpI.quot][tmpI.rem] = weightsVarianceEstimatesK;
    numWeightsEstimatesSamples[tmpI.quot][tmpI.rem] = numWeightsEstimatesSamplesK;

    // replace stats of the last and jth component
    splitMeans[tmpJ.quot].x[tmpJ.rem] = splitMeans[tmpL.quot].x[tmpL.rem];
    splitMeans[tmpJ.quot].y[tmpJ.rem] = splitMeans[tmpL.quot].y[tmpL.rem];
    splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem] = splitWeightedSampleCovariances[tmpL.quot].x[tmpL.rem];
    splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem] = splitWeightedSampleCovariances[tmpL.quot].y[tmpL.rem];
    splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem] = splitWeightedSampleCovariances[tmpL.quot].z[tmpL.rem];
    sumWeights[tmpJ.quot][tmpJ.rem] = sumWeights[tmpL.quot][tmpL.rem];
    numSamples[tmpJ.quot][tmpJ.rem] = numSamples[tmpL.quot][tmpL.rem];
    sumAssignedSamples[tmpJ.quot][tmpJ.rem] = sumAssignedSamples[tmpL.quot][tmpL.rem];
    chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] = chiSquareMCEstimates[tmpL.quot][tmpL.rem];
    chiSquareMCEstimate2ndMoments[tmpJ.quot][tmpJ.rem] = chiSquareMCEstimate2ndMoments[tmpL.quot][tmpL.rem];
    weightsEstimates[tmpJ.quot][tmpJ.rem] = weightsEstimates[tmpL.quot][tmpL.rem];
    weights2ndmomentEstimates[tmpJ.quot][tmpJ.rem] = weights2ndmomentEstimates[tmpL.quot][tmpL.rem];
    weightsVarianceEstimates[tmpJ.quot][tmpJ.rem] = weightsVarianceEstimates[tmpL.quot][tmpL.rem];
    numWeightsEstimatesSamples[tmpJ.quot][tmpJ.rem] = numWeightsEstimatesSamples[tmpL.quot][tmpL.rem];

    // reset stats of last component
    splitMeans[tmpL.quot].x[tmpL.rem] = 0.0f;
    splitMeans[tmpL.quot].y[tmpL.rem] = 0.0f;
    splitWeightedSampleCovariances[tmpL.quot].x[tmpL.rem] = 0.0f;
    splitWeightedSampleCovariances[tmpL.quot].y[tmpL.rem] = 0.0f;
    splitWeightedSampleCovariances[tmpL.quot].z[tmpL.rem] = 0.0f;
    sumWeights[tmpL.quot][tmpL.rem] = 0.0f;
    numSamples[tmpL.quot][tmpL.rem] = 0.0f;
    sumAssignedSamples[tmpL.quot][tmpL.rem] = 0.0f;
    chiSquareMCEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    chiSquareMCEstimate2ndMoments[tmpL.quot][tmpL.rem] = 0.0f;
    weightsEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    weights2ndmomentEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    weightsVarianceEstimates[tmpL.quot][tmpL.rem] = 0.0f;
    numWeightsEstimatesSamples[tmpL.quot][tmpL.rem] = 0.0f;

    numComponents--;
}

template <class TVMMFactory>
std::vector<typename VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitCandidate>
VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSplitCandidates(const float splitThreshold, const bool useConfidence) const
{
    std::vector<SplitCandidate> splitCandidates;
    for (size_t k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        if (chiSquareMCEstimates[tmp.quot][tmp.rem] > splitThreshold)
        {
            /*
            if (useConfidence)
            {
                float chiSquareVar = chiSquareMCEstimate2ndMoments[tmp.quot][tmp.rem] - chiSquareMCEstimates[tmp.quot][tmp.rem] * chiSquareMCEstimates[tmp.quot][tmp.rem];
                chiSquareVar /= sumAssignedSamples[tmp.quot][tmp.rem];
                float chiSquareStd = sqrt(chiSquareVar);
                float confidence = chiSquareMCEstimates[tmp.quot][tmp.rem] / (chiSquareMCEstimates[tmp.quot][tmp.rem] + chiSquareStd);
                std::cout << "chiSquare = " << chiSquareMCEstimates[tmp.quot][tmp.rem] << "\t chiSquareVar = " << chiSquareVar << "\t chiSquareStd = " << chiSquareStd
                          << "\t sumAssignedSamples = " << sumAssignedSamples[tmp.quot][tmp.rem] << "\t confidence = " << confidence << std::endl;
            }
            */
            SplitCandidate sc;
            sc.chiSquareEst = chiSquareMCEstimates[tmp.quot][tmp.rem];
            sc.componentIndex = k;
            splitCandidates.push_back(sc);
        }
    }
    if (splitCandidates.size() > 1)
    {
        std::sort(splitCandidates.begin(), splitCandidates.end(), [](SplitCandidate a, SplitCandidate b) {
            return a > b;
        });
    }
    return splitCandidates;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::clear(const size_t &_numComponents)
{
    const embree::vfloat<VMM::VectorSize> zeros(0.f);

    this->numComponents = _numComponents;
    // const int cnt = (this->numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    // for (size_t k =0; k < cnt; k++)
    for (size_t k = 0; k < VMM::NumVectors; k++)
    {
        chiSquareMCEstimates[k] = zeros;
        chiSquareMCEstimate2ndMoments[k] = zeros;
        weightsEstimates[k] = zeros;
        weights2ndmomentEstimates[k] = zeros;
        weightsVarianceEstimates[k] = zeros;
        numWeightsEstimatesSamples[k] = zeros;

        splitWeightedSampleCovariances[k].x = zeros;
        splitWeightedSampleCovariances[k].y = zeros;
        splitWeightedSampleCovariances[k].z = zeros;

        splitMeans[k].x = zeros;
        splitMeans[k].y = zeros;

        numSamples[k] = zeros;
        sumWeights[k] = zeros;
        sumAssignedSamples[k] = zeros;

        weights[k] = zeros;
        weightedMeans[k].x = zeros;
        weightedMeans[k].y = zeros;
        weightedMeans[k].z = zeros;
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::clearMasked(const size_t &_numComponents, const PartialFittingMask &mask)
{
    const embree::vfloat<VMM::VectorSize> zeros(0.f);

    this->numComponents = _numComponents;
    const int cnt = (this->numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        chiSquareMCEstimates[k] = select(mask.mask[k], zeros, chiSquareMCEstimates[k]);
        chiSquareMCEstimate2ndMoments[k] = select(mask.mask[k], zeros, chiSquareMCEstimate2ndMoments[k]);
        weightsEstimates[k] = select(mask.mask[k], zeros, weightsEstimates[k]);
        weights2ndmomentEstimates[k] = select(mask.mask[k], zeros, weights2ndmomentEstimates[k]);
        weightsVarianceEstimates[k] = select(mask.mask[k], zeros, weightsVarianceEstimates[k]);
        numWeightsEstimatesSamples[k] = select(mask.mask[k], zeros, numWeightsEstimatesSamples[k]);

        splitWeightedSampleCovariances[k].x = select(mask.mask[k], zeros, splitWeightedSampleCovariances[k].x);
        splitWeightedSampleCovariances[k].y = select(mask.mask[k], zeros, splitWeightedSampleCovariances[k].y);
        splitWeightedSampleCovariances[k].z = select(mask.mask[k], zeros, splitWeightedSampleCovariances[k].z);

        splitMeans[k].x = select(mask.mask[k], zeros, splitMeans[k].x);
        splitMeans[k].y = select(mask.mask[k], zeros, splitMeans[k].y);

        numSamples[k] = select(mask.mask[k], zeros, numSamples[k]);
        sumWeights[k] = select(mask.mask[k], zeros, sumWeights[k]);
        sumAssignedSamples[k] = select(mask.mask[k], zeros, sumAssignedSamples[k]);

        weights[k] = select(mask.mask[k], zeros, weights[k]);
        weightedMeans[k].x = select(mask.mask[k], zeros, weightedMeans[k].x);
        weightedMeans[k].y = select(mask.mask[k], zeros, weightedMeans[k].y);
        weightedMeans[k].z = select(mask.mask[k], zeros, weightedMeans[k].z);
    }
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::decay(const float &alpha)
{
    const int cnt = (this->numComponents + VMM::VectorSize - 1) / VMM::VectorSize;

    for (size_t k = 0; k < cnt; k++)
    {
        splitWeightedSampleCovariances[k].x *= alpha;
        splitWeightedSampleCovariances[k].y *= alpha;
        splitWeightedSampleCovariances[k].z *= alpha;

        numSamples[k] *= alpha;
        sumWeights[k] *= alpha;
        sumAssignedSamples[k] *= alpha;

        weightsVarianceEstimates[k] *= alpha;
        numWeightsEstimatesSamples[k] *= alpha;
    }
}

template <class TVMMFactory>
size_t VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getHighestChiSquareIdx() const
{
    size_t maxIdx = 0;
    float maxChiSquareValue = chiSquareMCEstimates[0][0];
    for (size_t k = 1; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        if (chiSquareMCEstimates[tmp.quot][tmp.rem] > maxChiSquareValue)
        {
            maxChiSquareValue = chiSquareMCEstimates[tmp.quot][tmp.rem];
            maxIdx = k;
        }
    }
    return maxIdx;
}

template <class TVMMFactory>
typename VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::SplitCandidate
VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getHighestValidChiSquareSplitComponent(const VMM &vmm, const VMM &previousVMM,
                                                                                                                          const ComponentSplitStatistics &previousSplitStats,
                                                                                                                          const bool *alreadySplitted,
                                                                                                                          const bool useConfidence) const
{
    SplitCandidate candidate;
    candidate.componentIndex = VMM::MaxComponents;
    candidate.splitType = ENone;
    float maxChiSquareValue = 0.f;
    for (size_t k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        if (!alreadySplitted[k] && chiSquareMCEstimates[tmp.quot][tmp.rem] > maxChiSquareValue && vmm._kappas[tmp.quot][tmp.rem] < OPENPGL_MAX_KAPPA * 0.9)
        {
            /*
            if (useConfidence)
            {
                float chiSquareVar = chiSquareMCEstimate2ndMoments[tmp.quot][tmp.rem] - chiSquareMCEstimates[tmp.quot][tmp.rem] * chiSquareMCEstimates[tmp.quot][tmp.rem];
                chiSquareVar /= sumAssignedSamples[tmp.quot][tmp.rem];
                float chiSquareStd = sqrt(chiSquareVar);
                float confidence = chiSquareMCEstimates[tmp.quot][tmp.rem] / (chiSquareMCEstimates[tmp.quot][tmp.rem] + chiSquareStd);
                std::cout << "chiSquare = " << chiSquareMCEstimates[tmp.quot][tmp.rem] << "\t chiSquareVar = " << chiSquareVar << "\t chiSquareStd = " << chiSquareStd
                          << "\t sumAssignedSamples = " << sumAssignedSamples[tmp.quot][tmp.rem] << "\t confidence = " << confidence << std::endl;
            }
            */
            maxChiSquareValue = chiSquareMCEstimates[tmp.quot][tmp.rem];
            candidate.chiSquareEst = maxChiSquareValue;
            candidate.componentIndex = k;
        }
    }
    if (candidate.componentIndex < VMM::MaxComponents)
    {
        // candidate.splitType = getSplitType(candidate.componentIndex, vmm, previousVMM, previousSplitStats);
        candidate.splitType = getSplitType(candidate.componentIndex);
        // candidate.splitType = EFirefly;
    }
    // std::cout << "SplitCandiate: idx = " << candidate.componentIndex << "\t type = " << (candidate.splitType==EFirefly ? "FireFly" : "MultiModal" )<< std::endl;
    return candidate;
}

template <class TVMMFactory>
void VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::clearAll()
{
    this->clear(VMM::MaxComponents);
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getChiSquareEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return chiSquareMCEstimates[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getChiSquare2ndMomentEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return chiSquareMCEstimate2ndMoments[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getWeightsEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return weightsEstimates[tmp.quot][tmp.rem];
}
template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getRelVarianceEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    // return std::sqrt(std::abs(weights2ndmomentEstimates[tmp.quot][tmp.rem] - weightsEstimates[tmp.quot][tmp.rem]*weightsEstimates[tmp.quot][tmp.rem])) /
    // weightsEstimates[tmp.quot][tmp.rem];
    return (std::abs(weights2ndmomentEstimates[tmp.quot][tmp.rem] - weightsEstimates[tmp.quot][tmp.rem] * weightsEstimates[tmp.quot][tmp.rem])) /
           (weightsEstimates[tmp.quot][tmp.rem] * weightsEstimates[tmp.quot][tmp.rem]);
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getVarianceEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    // return std::sqrt(std::abs(weights2ndmomentEstimates[tmp.quot][tmp.rem] - weightsEstimates[tmp.quot][tmp.rem]*weightsEstimates[tmp.quot][tmp.rem])) /
    // weightsEstimates[tmp.quot][tmp.rem];
    return (std::abs(weights2ndmomentEstimates[tmp.quot][tmp.rem] - weightsEstimates[tmp.quot][tmp.rem] * weightsEstimates[tmp.quot][tmp.rem]));
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getWeights2ndMomentEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return weights2ndmomentEstimates[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getWeightsVarianceEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return weightsVarianceEstimates[tmp.quot][tmp.rem] / numWeightsEstimatesSamples[tmp.quot][tmp.rem];
}

template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getSumChiSquareEst() const
{
    float sumChiSquareEst = 0.0f;

    for (int k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        sumChiSquareEst += chiSquareMCEstimates[tmp.quot][tmp.rem];
    }
    return sumChiSquareEst;
}
/*
template <class TVMMFactory>
float VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::getChiSquareVar(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VMM::VectorSize));
    return chiSquareMCVariances[tmp.quot][tmp.rem] - chiSquareMCEstimates[tmp.quot][tmp.rem] * chiSquareMCEstimates[tmp.quot][tmp.rem];
    // return chiSquareMCVariances[tmp.quot][tmp.rem] / numSamples[tmp.quot][tmp.rem];
}
*/
template <class TVMMFactory>
std::string VonMisesFisherChiSquareComponentSplitterV2<TVMMFactory>::ComponentSplitStatistics::toString() const
{
    std::stringstream ss;
    ss << "ComponentSplitStatistics:" << std::endl;
    ss << "numComponents: " << numComponents << std::endl;
    float sumChiSquareEst = 0.0f;
    // for ( int k = 0; k < numComponents; k++)
    for (int k = 0; k < VMM::MaxComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VMM::VectorSize));
        ss << "\t stats[" << k << "]: " << "chiSquareEst: " << chiSquareMCEstimates[tmp.quot][tmp.rem];
        ss << std::endl;
        ss << "\t" << "mean: [" << splitMeans[tmp.quot].x[tmp.rem] << ",\t" << splitMeans[tmp.quot].y[tmp.rem] << "]";
        ss << "\t samplevar: [" << splitWeightedSampleCovariances[tmp.quot].x[tmp.rem] << ",\t" << splitWeightedSampleCovariances[tmp.quot].y[tmp.rem] << ",\t"
           << splitWeightedSampleCovariances[tmp.quot].z[tmp.rem] << "]";
        if (sumWeights[tmp.quot][tmp.rem] > 0.f)
        {
            ss << "\t covar: [" << splitWeightedSampleCovariances[tmp.quot].x[tmp.rem] / sumWeights[tmp.quot][tmp.rem] << ",\t"
               << splitWeightedSampleCovariances[tmp.quot].y[tmp.rem] / sumWeights[tmp.quot][tmp.rem] << ",\t"
               << splitWeightedSampleCovariances[tmp.quot].z[tmp.rem] / sumWeights[tmp.quot][tmp.rem] << "]";
        }
        else
        {
            ss << "\t covar: [" << 0.0f << ",\t" << 0.0f << ",\t" << 0.0f << "]";
        }
        ss << std::endl;

        ss << "\t" << "numSamples: " << numSamples[tmp.quot][tmp.rem] << "\t sumWeights: " << sumWeights[tmp.quot][tmp.rem]
           << "\t sumAssignedSamples: " << sumAssignedSamples[tmp.quot][tmp.rem];
        ss << std::endl;

        sumChiSquareEst += chiSquareMCEstimates[tmp.quot][tmp.rem];
    }
    ss << "sumChiSquareEst: " << sumChiSquareEst << std::endl;
    return ss.str();
}

}  // namespace openpgl
