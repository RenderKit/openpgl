// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "VMM.h"
#include "../data/DirectionalSampleData.h"
#include "WeightedEMVMMFactory.h"

#include <algorithm>
#include <vector>


#define RKGUIDE_USE_LOGMAP
//#define RKGUIDE_ZERO_MEAN


namespace rkguide
{


struct ComponentSplitinfo
{
    Vector2 mean {0.0f};
    Vector3 covariance{0.0f};

    float eigenValue0 {0.0f};
    float eigenValue1 {0.0f};

    Vector2 eigenVector0 {0.0f};
    Vector2 eigenVector1 {0.0f};

    std::string toString() const;
};

template<int VecSize, int maxComponents>
struct VonMisesFisherChiSquareComponentSplitter
{

public:
    typedef VonMisesFisherMixture<VecSize, maxComponents> VMM;
    typedef std::integral_constant<size_t, (maxComponents + (VecSize -1)) / VecSize> NumVectors;
    typedef typename WeightedEMVonMisesFisherFactory<VecSize, maxComponents>::SufficientStatisitcs SufficientStatisitcs;
    typedef typename WeightedEMVonMisesFisherFactory<VecSize, maxComponents>::PartialFittingMask PartialFittingMask;
    typedef WeightedEMVonMisesFisherFactory<VecSize, maxComponents> VMMFactory;

struct SplitCandidate
{
    size_t componentIndex;
    float chiSquareEst;

    bool operator<( const SplitCandidate &sc ) const
    {
        return chiSquareEst < sc.chiSquareEst;
    }

    bool operator>( const SplitCandidate &sc ) const
    {
        return chiSquareEst > sc.chiSquareEst;
    }
};

struct ComponentSplitStatistics
{
    vfloat<VecSize> chiSquareMCEstimates[NumVectors::value];
    Vec2<vfloat<VecSize> > splitMeans[NumVectors::value];
    Vec3<vfloat<VecSize> > splitWeightedSampleCovariances[NumVectors::value];

    vfloat<VecSize> numSamples[NumVectors::value];
    vfloat<VecSize> sumWeights[NumVectors::value];

    vfloat<VecSize> sumAssignedSamples[NumVectors::value];

    size_t numComponents{0};
    //float mcEstimate{0.0f};
    //float numSamplesOld{0.0f};

    void clear(const size_t &_numComponents);
    void clearAll();

    float getChiSquareEst(const size_t &idx) const;
    float getSumChiSquareEst() const;
    size_t getHighestChiSquareIdx() const;

    std::vector<SplitCandidate> getSplitCandidates() const;

    void decay( const float &alpha );
    std::string toString() const;
};


void PerformSplitting(VMM &vmm, const float &splitThreshold, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData, const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit, const int &maxSplittingItr = -1) const;

void PerformRecursiveSplitting(VMM &vmm, typename VMMFactory::SufficientStatisitcs &suffStats, const float &splitThreshold, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData, const typename VMMFactory::Configuration factoryCfg) const;

void PerformSplittingIteration(VMM &vmm, const float &splitThreshold) const;

void CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const;

void UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const;

bool SplitComponent(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatisitcs &suffStats, const size_t idx) const;

ComponentSplitinfo GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const DirectionalSampleData *data, const size_t &numData, Vector3 *local2D) const;


};

#ifndef RKGUIDE_USE_LOGMAP

template<typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec2Type Map3DTo2D(const Vec3Type &vec3D)
{
    return Vec2Type(vec3D.x, vec3D.y);
}

template<typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec3Type Map2DTo3D(const Vec2Type &vec2D)
{
    Vec3Type vec3D = Vec3Type(0.0f);
    vec3D.x = vec2D.x;
    vec3D.y = vec2D.y;
    vec3D.z = embree::sqrt(1.0f - vec2D.x*vec2D.x - vec2D.y*vec2D.y);
    return vec3D;
}

#else

//logMapping https://ronnybergmann.net/mvirt/manifolds/Sn/log.html
template<typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec2Type Map3DTo2D(const Vec3Type &vec3D)
{
    Vec2Type vec2D(0.0f);

    //RKGUIDE_ASSERT((vec3D.z <= 1.0f &&  vec3D.z >= -1.0f));
    ScalarType alpha = embree::fastapprox::acos(vec3D.z);
    ScalarType inv_sinc = alpha / embree::fastapprox::sin(alpha);

    vec2D.x = select(alpha > 0.0f, vec3D.x * inv_sinc, vec2D.x);
    vec2D.y = select(alpha > 0.0f, vec3D.y * inv_sinc, vec2D.y);
    return vec2D;
}

//expMapping https://ronnybergmann.net/mvirt/manifolds/Sn/exp.html
template<typename Vec3Type, typename Vec2Type, typename ScalarType>
inline Vec3Type Map2DTo3D(const Vec2Type &vec2D)
{
    Vec3Type vec3D = Vec3Type(0.0f);
    ScalarType length = embree::sqrt(vec2D.x*vec2D.x + vec2D.y*vec2D.y);
    RKGUIDE_ASSERT(length < M_PI);
    ScalarType sinc = embree::fastapprox::sin(length) / length;

    vec3D.x = select(length > 0.0f, vec2D.x * sinc, vec3D.x);
    vec3D.y = select(length > 0.0f, vec2D.y * sinc, vec3D.y);
    vec3D.z = embree::cos(length);

    return vec3D;
}


#endif

std::string ComponentSplitinfo::toString() const
{
    std::stringstream ss;
    ss << "ComponentSplitinfo:" << std::endl;
    ss << "mean: " << mean << std::endl;
    ss << "covariance: " << covariance << std::endl;
    ss << "eigenValue0: " << eigenValue0 << std::endl;
    ss << "eigenValue1: " << eigenValue1 << std::endl;
    ss << "eigenVector0: " << eigenVector0 << std::endl;
    ss << "eigenVector1: " << eigenVector1 << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const
{
    splitStats.clear(vmm._numComponents);
    this->UpdateSplitStatistics(vmm, splitStats, mcEstimate, data, numData);
}


template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::PerformSplitting(VMM &vmm, const float &splitThreshold, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData, const typename VMMFactory::Configuration factoryCfg, const bool &doPartialRefit, const int &maxSplittingItr) const
{
    PartialFittingMask mask;
    ComponentSplitStatistics splitStatistics;
    SufficientStatisitcs suffStatistics;

    bool stopSplitting = false;

    size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;
    //std::cout << "vmm: " << vmm.toString() << std::endl;

    while ( vmm._numComponents < maxComponents && !stopSplitting)
    //for (size_t j =0; j<1; j++)
    {
        stopSplitting = true;
        splitStatistics.clearAll();
        this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);

        std::vector<SplitCandidate> splitComps = splitStatistics.getSplitCandidates();

        mask.resetToFalse();
        const size_t numComp = vmm._numComponents;
        for (size_t k = 0; k < numComp; k++)
        {
            if (splitComps[k].chiSquareEst > splitThreshold && vmm._numComponents  < maxComponents)
            {
                //std::cout << "split[" << k << "]: idx:" << splitComps[k].componentIndex << "\t chi2: " << splitComps[k].chiSquareEst << std::endl;

                bool splitSucess = SplitComponent(vmm, splitStatistics, suffStatistics, splitComps[k].componentIndex);
                mask.setToTrue(splitComps[k].componentIndex);
                mask.setToTrue(vmm._numComponents-1);
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
        //std::cout << "mask: " << mask.toString() << std::endl;
        //std::cout << "vmmSplit: " << vmm.toString() << std::endl;
        //std::cout << "factoryCfg: " << factoryCfg.toString() << std::endl;
        //std::cout << "suffStatistics: " << suffStatistics.toString() << std::endl;
        if (doPartialRefit)
        {
            vmmFactory.partialUpdateMixture(vmm, mask, suffStatistics, data, numData, factoryCfg, vmmFitStats);
            //std::cout << "vmmpartialUpdate: " << vmm.toString() << std::endl;
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

template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::PerformRecursiveSplitting(VMM &vmm, typename VMMFactory::SufficientStatisitcs &suffStatistics, const float &splitThreshold, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData, const typename VMMFactory::Configuration factoryCfg) const
{
    PartialFittingMask mask;
    ComponentSplitStatistics splitStatistics;
    SufficientStatisitcs tempSuffStatistics;

    //bool stopSplitting = false;

    size_t splitItr = 0;

    VMMFactory vmmFactory;
    typename VMMFactory::FittingStatistics vmmFitStats;
    //std::cout << "vmm: " << vmm.toString() << std::endl;
    int numSplits = -1;
    while ( vmm._numComponents < maxComponents && numSplits != 0)
    //for (size_t j =0; j<1; j++)
    {
        numSplits = 0;
        splitStatistics.clearAll();
        this->CalculateSplitStatistics(vmm, splitStatistics, mcEstimate, data, numData);

        std::vector<SplitCandidate> splitComps = splitStatistics.getSplitCandidates();

        mask.resetToFalse();
        const size_t numComp = vmm._numComponents;
        for (size_t k = 0; k < numComp; k++)
        {
            if (splitComps[k].chiSquareEst > splitThreshold && vmm._numComponents  < maxComponents)
            {
                bool splitSucess = SplitComponent(vmm, splitStatistics, tempSuffStatistics, splitComps[k].componentIndex);
                mask.setToTrue(splitComps[k].componentIndex);
                mask.setToTrue(vmm._numComponents-1);
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
            tempSuffStatistics.clear(vmm._numComponents);
            vmmFactory.partialUpdateMixture(vmm, mask, tempSuffStatistics, data, numData, factoryCfg, vmmFitStats);
            //std::cout << "tempSuffStatistics" << std::endl << tempSuffStatistics.toString() << std::endl;
            suffStatistics.numComponents = vmm._numComponents;
            suffStatistics.maskedReplace(mask, tempSuffStatistics);
        }
        //std::cout << "vmmpartialUpdate: " << vmm.toString() << std::endl;
        splitItr++;
    }
}

template<int VecSize, int maxComponents>
ComponentSplitinfo VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const DirectionalSampleData *data, const size_t &numData, Vector3 *local2D) const
{
    typename VMM::SoftAssignment softAssign;
    const vfloat<VecSize> zeros(0.f);
    //const int cnt = (vmm._numComponents + VecSize-1) / VecSize;
    //size_t validDataCount = 0.0f;

    ComponentSplitinfo splitInfo;

    Vector2 mean(0.0f);
    Vector3 covarianceStats(0.0f);
    float sumWeights = 0.0f;

    for (size_t n = 0; n < numData; n++)
    {
        const DirectionalSampleData sample = data[n];
        if (vmm.softAssignment(sample.direction, softAssign) )
        {
            const div_t tmp = div(idx, static_cast<int>(VecSize));

            const vfloat<VecSize> weight = sample.weight;
            const vfloat<VecSize> samplePDF = sample.pdf;
            //const vfloat<VecSize> value =  weight * samplePDF;


            const Vec3< vfloat<VecSize> > localDirection = embree::frame( vmm._meanDirections[tmp.quot] ).inverse() * Vec3< vfloat<VecSize> > (sample.direction);

            const Vec2< vfloat<VecSize> > localDirection2D = Map3DTo2D< Vec3< vfloat<VecSize> >,  Vec2< vfloat<VecSize> >, vfloat<VecSize> >(localDirection);

            const vfloat<VecSize> assignedWeight = softAssign.assignments[tmp.quot] * weight;
            local2D[n].x = localDirection2D.x[tmp.rem];
            local2D[n].y = localDirection2D.y[tmp.rem];
            local2D[n].z = assignedWeight[tmp.rem];

            sumWeights += assignedWeight[tmp.rem];
#ifdef RKGUIDE_ZERO_MEAN
            mean.x += 0.0f;
            mean.y += 0.0f;
#else
            mean.x += assignedWeight[tmp.rem] * localDirection2D.x[tmp.rem];
            mean.y += assignedWeight[tmp.rem] * localDirection2D.y[tmp.rem];
#endif
            covarianceStats.x += assignedWeight[tmp.rem] * localDirection2D.x[tmp.rem] * localDirection2D.x[tmp.rem];
            covarianceStats.y += assignedWeight[tmp.rem] * localDirection2D.y[tmp.rem] * localDirection2D.y[tmp.rem];
            covarianceStats.z += assignedWeight[tmp.rem] * localDirection2D.x[tmp.rem] * localDirection2D.y[tmp.rem];
        }
    }
    mean /= sumWeights;

    splitInfo.mean = mean;
    splitInfo.covariance.x = covarianceStats.x / sumWeights - mean.x*mean.x;
    splitInfo.covariance.y = covarianceStats.y / sumWeights - mean.y*mean.y;
    splitInfo.covariance.z = covarianceStats.z / sumWeights - mean.x*mean.y;

    float D = embree::sqrt((splitInfo.covariance.x - splitInfo.covariance.y) * (splitInfo.covariance.x - splitInfo.covariance.y) + (splitInfo.covariance.z * splitInfo.covariance.z *4.0f)) * 0.5f;
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

    std::cout << "split: " << "\tmean: " << splitInfo.mean.x << ", \t " << splitInfo.mean.y<< "\t covariance: " << splitInfo.covariance.x << ", \t "  << splitInfo.covariance.y << ", \t "  << splitInfo.covariance.z << std::endl;
    std::cout << "eigen: " << "\tevalue0: " << splitInfo.eigenValue0 << "\teVec0: " << splitInfo.eigenVector0.x << ", \t " << splitInfo.eigenVector0.y << "\tevalue1: " << splitInfo.eigenValue1 << "\teVec1: " << splitInfo.eigenVector1.x << ", \t " << splitInfo.eigenVector1.y << std::endl;

    return splitInfo;
}


template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const
{
    //std::cout << "UpdateSplitStatistics" << std::endl;

    typename VMM::SoftAssignment softAssign;
    const vfloat<VecSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VecSize-1) / VecSize;
    size_t validDataCount = 0.0f;

    for (size_t n = 0; n < numData; n++)
    {
        const DirectionalSampleData sample = data[n];
        if (vmm.softAssignment(sample.direction, softAssign) )
        {
            const vfloat<VecSize> weight = sample.weight;
            const vfloat<VecSize> samplePDF = sample.pdf;
            const vfloat<VecSize> value =  weight * samplePDF;
            //std::cout << "data[" << n << "]: " << "value: " << value << "\t samplePDF: " << samplePDF;
            for (size_t k =0; k < cnt; k++)
            {
                vfloat<VecSize> vmfPDF = softAssign.assignments[k] * softAssign.pdf;
                vfloat<VecSize> partialValuePDF = vmfPDF * value;
                partialValuePDF /= (mcEstimate * softAssign.pdf);
                //partialValuePDF /= vmm._weights[k] * mcEstimate;
                //std::cout << "\tweights: " << vmm._weights[k] << "\t assign: " << softAssign.assignments[k] << "\t pdf: " << softAssign.pdf << std::endl;
                //std::cout << "\tpvPDF: " << partialValuePDF << "\t vmfPDF: " << vmfPDF << std::endl;

                vfloat<VecSize> chiSquareEst = value *value * vmfPDF;
                chiSquareEst /= mcEstimate * mcEstimate * softAssign.pdf * softAssign.pdf;
                //chiSquareEst *= chiSquareEst;
                chiSquareEst -= 2.0f * partialValuePDF;
                chiSquareEst += vmfPDF;
                chiSquareEst /= samplePDF;

                chiSquareEst = select(softAssign.assignments[k] > 0.f , chiSquareEst, zeros);

                splitStats.sumAssignedSamples[k] += softAssign.assignments[k];
                // incremental updated of the MC chiSquare estimate
                splitStats.numSamples[k] += 1.0f;
                splitStats.chiSquareMCEstimates[k] += (chiSquareEst - splitStats.chiSquareMCEstimates[k]) / splitStats.numSamples[k];

                const Vec3< vfloat<VecSize> > localDirection = embree::frame( vmm._meanDirections[k] ).inverse() * Vec3< vfloat<VecSize> > (sample.direction);
                const Vec2< vfloat<VecSize> > localDirection2D(localDirection.x, localDirection.y);
                const vfloat<VecSize> assignedWeight = softAssign.assignments[k] * weight;
                //const vfloat<VecSize> assignedWeight = softAssign.assignments[k] * weight * weight;

                splitStats.sumWeights[k] += assignedWeight;
                const vfloat<VecSize> incWeight = select(splitStats.sumWeights[k] > 0.0f, assignedWeight / splitStats.sumWeights[k], zeros);

#ifdef RKGUIDE_ZERO_MEAN
                splitStats.splitMeans[k] += Vec2< vfloat<VecSize> >(0.0f);
                splitStats.splitWeightedSampleCovariances[k].x += assignedWeight * (localDirection2D.x * localDirection2D.x);
                splitStats.splitWeightedSampleCovariances[k].y += assignedWeight * (localDirection2D.y * localDirection2D.y);
                splitStats.splitWeightedSampleCovariances[k].z += assignedWeight * (localDirection2D.x * localDirection2D.y);
#else
                const Vec2< vfloat<VecSize> > previousSplitMeans = splitStats.splitMeans[k];
                splitStats.splitMeans[k] += incWeight * ( localDirection2D - splitStats.splitMeans[k]);
                splitStats.splitWeightedSampleCovariances[k].x += assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.x - splitStats.splitMeans[k].x));
                splitStats.splitWeightedSampleCovariances[k].y += assignedWeight * ((localDirection2D.y - previousSplitMeans.y) * (localDirection2D.y - splitStats.splitMeans[k].y));
                splitStats.splitWeightedSampleCovariances[k].z += assignedWeight * ((localDirection2D.x - previousSplitMeans.x) * (localDirection2D.y - splitStats.splitMeans[k].y));

#endif
                //splitStats.sumWeights[k] += assignedWeight;
            }
            validDataCount++;
            //std::cout << std::endl;
        }
    }
    //splitStats.numSamplesOld += validDataCount;
    //splitStats.mcEstimate += mcEstimate;
}


template<int VecSize, int maxComponents>
bool VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::SplitComponent(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatisitcs &suffStats, const size_t idx) const
{
    ComponentSplitinfo splitInfo;
    const div_t tmpK = div(idx, static_cast<int>(VecSize));


    float numAssignedSamples = splitStats.sumAssignedSamples[tmpK.quot][tmpK.rem];

    float inv_sumWeights = rcp(splitStats.sumWeights[tmpK.quot][tmpK.rem]);
    splitInfo.mean = Vector2(splitStats.splitMeans[tmpK.quot].x[tmpK.rem], splitStats.splitMeans[tmpK.quot].y[tmpK.rem]);

    splitInfo.covariance.x = splitStats.splitWeightedSampleCovariances[tmpK.quot].x[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.y = splitStats.splitWeightedSampleCovariances[tmpK.quot].y[tmpK.rem] * inv_sumWeights;
    splitInfo.covariance.z = splitStats.splitWeightedSampleCovariances[tmpK.quot].z[tmpK.rem] * inv_sumWeights;

    float D = embree::sqrt((splitInfo.covariance.x - splitInfo.covariance.y) * (splitInfo.covariance.x - splitInfo.covariance.y) + (splitInfo.covariance.z * splitInfo.covariance.z *4.0f)) * 0.5f;
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
    //std::cout << "D: " << D << std::endl;
    //std::cout << "sumWeights: " << splitStats.sumWeights[tmpK.quot][tmpK.rem] << "\t inSumWeights: " << inv_sumWeights << std::endl;
    //std::cout << "splitMean: " << splitInfo.mean << "\t splitCovariance: " << splitInfo.covariance << std::endl;

    //std::cout << "splitCovariancesRaw: " << splitStats.splitCovariances[tmpK.quot].x[tmpK.rem] << "\t" << splitStats.splitCovariances[tmpK.quot].y[tmpK.rem] << "\t" << splitStats.splitCovariances[tmpK.quot].z[tmpK.rem] << std::endl;
//    std::cout << "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl;
//    std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t eigenVector1: " << splitInfo.eigenVector1 << std::endl;
/**/

    float sumStatsWeight = suffStats.sumOfWeightedStats[tmpK.quot][tmpK.rem];
    float weight = vmm._weights[tmpK.quot][tmpK.rem];
    float meanCosine = vmm._meanCosines[tmpK.quot][tmpK.rem];
    float kappa = vmm._kappas[tmpK.quot][tmpK.rem];

    if (kappa >= RKGUIDE_MAX_KAPPA * 0.9)
    {
        return false;
    }

    Vector3 meanDirection = Vector3(vmm._meanDirections[tmpK.quot].x[tmpK.rem],
                                    vmm._meanDirections[tmpK.quot].y[tmpK.rem],
                                    vmm._meanDirections[tmpK.quot].z[tmpK.rem]);

    float newWeight0 = weight * 0.5f;
    float newWeight1 = newWeight0;

    Vector3 meanDirection0 = meanDirection;
    Vector3 meanDirection1 = meanDirection;

    float newMeanCosine0 = meanCosine;
    float newMeanCosine1 = meanCosine*meanCosine;

    float newKkappa0 = MeanCosineToKappa<float> (newMeanCosine0);
    float newKkappa1 = MeanCosineToKappa<float> (newMeanCosine1);

    if (D > 1e-8f)
    {
        Vector2 meanDir2D0 = splitInfo.mean + (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f);
        meanDirection0 =  embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D0);
        newMeanCosine0 = meanCosine / dot(meanDirection, meanDirection0);
        // ensure that the new mean cosine is in a valid range (i.e., < 1.0 and < the mean cosine of max kappa)
        newMeanCosine0 = std::min(newMeanCosine0, KappaToMeanCosine<float>(RKGUIDE_MAX_KAPPA));
        newMeanCosine1 = newMeanCosine0;
        newKkappa0 = MeanCosineToKappa<float> (newMeanCosine0);
        newKkappa1 = newKkappa0;

        Vector2 meanDir2D1 = splitInfo.mean - (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5f);
        meanDirection1 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2, float>(meanDir2D1);
        //float meanCosine1 = meanCosine / meanDirection0.z;
        //float kappa1 = MeanCosineToKappa<float> (meanCosine1);
        std::cout << "meanCosine: " << meanCosine << "\t kappa: " << kappa << "\t newMeanCosine: " << newMeanCosine0 << " \t newKkappa: " <<  newKkappa0 << std::endl;
        std::cout << "localMeanDirection0: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D0) << "\t meanDirection0: " << meanDirection0 << "\t meanCosine: " << meanCosine << " \t costheta0: " <<  dot(meanDirection, meanDirection0) << std::endl;
        std::cout << "localMeanDirection1: " << Map2DTo3D<Vector3, Vector2, float>(meanDir2D1) << "\t meanDirection1: " << meanDirection1 << "\t meanCosine: " << meanCosine << " \t costheta1: " <<  dot(meanDirection, meanDirection1) << std::endl;
        std::cout << "eigenValue0: " << splitInfo.eigenValue0 << "\t eigenVector0: " << splitInfo.eigenVector0 << std::endl;
        std::cout << "eigenValue1: " << splitInfo.eigenValue1 << "\t eigenVector1: " << splitInfo.eigenVector1 << std::endl;
        std::cout << "D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples <<std::endl;
    }
    else
    {
        std::cout << "!!!!   D: " << D << "\t idx: " << idx << " \t assignedSamples: " << numAssignedSamples <<std::endl;
        std::cout << "weight: " << weight << "\t meanCosine: " << meanCosine <<std::endl;
        if( numAssignedSamples <2.0f)
        {
            return false;
        }
    }
    size_t K = vmm._numComponents;
    //vmm.swapComponents(K-1, idx);
    //suffStats.swapComponentStats(K-1, idx);
    //const div_t tmpI = div(K-1, static_cast<int>(VecSize));
    const div_t tmpI = tmpK;
    const div_t tmpJ = div(K, static_cast<int>(VecSize));

    vmm._weights[tmpI.quot][tmpI.rem] = newWeight0;
    vmm._meanCosines[tmpI.quot][tmpI.rem] = newMeanCosine0;
    vmm._kappas[tmpI.quot][tmpI.rem] = newKkappa0;
    vmm._meanDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x;
    vmm._meanDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y;
    vmm._meanDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z;

    vmm._weights[tmpJ.quot][tmpJ.rem] = newWeight1;
    vmm._meanCosines[tmpJ.quot][tmpJ.rem] = newMeanCosine1;
    vmm._kappas[tmpJ.quot][tmpJ.rem] = newKkappa1;
    vmm._meanDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x;
    vmm._meanDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y;
    vmm._meanDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z;

    vmm._numComponents = K + 1;
    vmm._calculateNormalization();

    sumStatsWeight /= 2.0f;

    suffStats.sumOfWeightedStats[tmpI.quot][tmpI.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x * newMeanCosine0 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y * newMeanCosine0 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z * newMeanCosine0 * sumStatsWeight;

    suffStats.sumOfWeightedStats[tmpJ.quot][tmpJ.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x * newMeanCosine1 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y * newMeanCosine1 * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z * newMeanCosine1 * sumStatsWeight;
    suffStats.numComponents = K + 1;


    // reseting the split statistics for the two new components
    splitStats.chiSquareMCEstimates[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.numSamples[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.sumWeights[tmpI.quot][tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].x[tmpI.rem] = 0.0f;
    splitStats.splitMeans[tmpI.quot].y[tmpI.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].x[tmpI.rem] = 0.0;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].y[tmpI.rem] = 0.0;
    splitStats.splitWeightedSampleCovariances[tmpI.quot].z[tmpI.rem] = 0.0;

    splitStats.chiSquareMCEstimates[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumAssignedSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.numSamples[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.sumWeights[tmpJ.quot][tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].x[tmpJ.rem] = 0.0f;
    splitStats.splitMeans[tmpJ.quot].y[tmpJ.rem] = 0.0f;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].x[tmpJ.rem] = 0.0;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].y[tmpJ.rem] = 0.0;
    splitStats.splitWeightedSampleCovariances[tmpJ.quot].z[tmpJ.rem] = 0.0;

    splitStats.numComponents = K +1;

    return true;
}

template<int VecSize, int maxComponents>
std::vector<typename VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::SplitCandidate > VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::getSplitCandidates() const
{
    std::vector<SplitCandidate> splitCandidates;
    for (size_t k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        SplitCandidate sc;
        sc.chiSquareEst = chiSquareMCEstimates[tmp.quot][tmp.rem];
        sc.componentIndex = k;
        splitCandidates.push_back(sc);
    }

    std::sort(splitCandidates.begin(), splitCandidates.end(), [](SplitCandidate a, SplitCandidate b) {return a > b; });
    return splitCandidates;
}

template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::clear(const size_t &_numComponents)
{
    const vfloat<VecSize> zeros(0.f);

    this->numComponents = _numComponents;
    const int cnt = (this->numComponents + VecSize-1) / VecSize;

    for (size_t k =0; k < cnt; k++)
    {
        chiSquareMCEstimates[k] = zeros;
        splitWeightedSampleCovariances[k].x = zeros;
        splitWeightedSampleCovariances[k].y = zeros;
        splitWeightedSampleCovariances[k].z = zeros;

        splitMeans[k].x = zeros;
        splitMeans[k].y = zeros;

        numSamples[k] = zeros;
        sumWeights[k] = zeros;
        sumAssignedSamples[k] = zeros;
    }

    //mcEstimate = 0.0f;
    //numSamplesOld = 0.0f;
}


template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::decay(const float &alpha)
{

    const int cnt = (this->numComponents + VecSize-1) / VecSize;

    for (size_t k =0; k < cnt; k++)
    {
        splitWeightedSampleCovariances[k].x *= alpha;
        splitWeightedSampleCovariances[k].y *= alpha;
        splitWeightedSampleCovariances[k].z *= alpha;

        numSamples[k] *= alpha;
        sumWeights[k] *= alpha;
        sumAssignedSamples[k] *= alpha;
    }

    //mcEstimate = 0.0f;
    //numSamplesOld = 0.0f;
}

template<int VecSize, int maxComponents>
size_t VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::getHighestChiSquareIdx() const
{
    size_t maxIdx = 0;
    float maxChiSquareValue = chiSquareMCEstimates[0][0];
    for (size_t k = 1; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        if (chiSquareMCEstimates[tmp.quot][tmp.rem] > maxChiSquareValue)
        {
            maxChiSquareValue = chiSquareMCEstimates[tmp.quot][tmp.rem];
            maxIdx = k;
        }
    }
    return maxIdx;
}

template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::clearAll()
{
    this->clear(maxComponents);
}

template<int VecSize, int maxComponents>
float VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::getChiSquareEst(const size_t &idx) const
{
    const div_t tmp = div(idx, static_cast<int>(VecSize));
    return chiSquareMCEstimates[tmp.quot][tmp.rem];
}

template<int VecSize, int maxComponents>
float VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::getSumChiSquareEst() const
{
    float sumChiSquareEst = 0.0f;

    for ( int k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        sumChiSquareEst += chiSquareMCEstimates[tmp.quot][tmp.rem];
    }
    return sumChiSquareEst;
}

template<int VecSize, int maxComponents>
std::string VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::toString() const
{
    std::stringstream ss;
    ss << "ComponentSplitStatistics:" << std::endl;
    ss << "numComponents: " << numComponents << std::endl;
    float sumChiSquareEst = 0.0f;
    for ( int k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        ss << "\t stats[" << k << "]: " << "chiSquareEst: " << chiSquareMCEstimates[tmp.quot][tmp.rem];

        ss << std::endl;
        ss << "\t" << "mean: ["  << splitMeans[tmp.quot].x[tmp.rem] << ",\t" << splitMeans[tmp.quot].y[tmp.rem] << "]";
        ss << std::endl;

        ss << "\t covar: [" << splitWeightedSampleCovariances[tmp.quot].x[tmp.rem] / sumWeights[tmp.quot][tmp.rem] << ",\t" << splitWeightedSampleCovariances[tmp.quot].y[tmp.rem]  / sumWeights[tmp.quot][tmp.rem]<< ",\t" << splitWeightedSampleCovariances[tmp.quot].z[tmp.rem]  / sumWeights[tmp.quot][tmp.rem] << "]";
        ss << std::endl;

        ss << "\t" << "numSamples: " << numSamples[tmp.quot][tmp.rem] << "\t sumWeights: " << sumWeights[tmp.quot][tmp.rem] << "\t sumAssignedSamples: " << sumAssignedSamples[tmp.quot][tmp.rem];
        ss << std::endl;

        sumChiSquareEst += chiSquareMCEstimates[tmp.quot][tmp.rem];
    }
    ss << "sumChiSquareEst: " << sumChiSquareEst << std::endl;
    //ss << "mcEstimate: " << mcEstimate << std::endl;
    //ss << "numSamples: " << numSamplesOld << std::endl;
    return ss.str();
}

}