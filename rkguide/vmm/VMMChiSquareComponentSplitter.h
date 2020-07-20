// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "VMM.h"
#include "../data/DirectionalSampleData.h"
#include "WeightedEMVMMFactory.h"

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

struct ComponentSplitStatistics
{

    vfloat<VecSize> chiSquareMCEstimates[NumVectors::value];
    Vec2<vfloat<VecSize> > splitMeans[NumVectors::value];
    Vec3<vfloat<VecSize> > splitCovariances[NumVectors::value];
    vfloat<VecSize> sumWeights[NumVectors::value];

    size_t numComponents{0};
    float mcEstimate{0.0f};
    float numSamples{0.0f};

    void clear(const size_t &_numComponents);
    void clearAll();

    float getSumChiSquareEst() const;
    size_t getHighestChiSquareIdx() const;

    std::string toString() const;
};

void CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const;

void UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const;

void SplitComponent(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatisitcs &suffStats, const size_t idx) const;

ComponentSplitinfo GetProjectedLocalDirections(const VMM &vmm, const size_t &idx, const DirectionalSampleData *data, const size_t &numData, Vector3 *local2D) const;


};

template<typename Vec3Type, typename Vec2Type>
inline Vec2Type Map3DTo2D(const Vec3Type &vec3D)
{
    return Vec2Type(vec3D.x, vec3D.y);
}

template<typename Vec3Type, typename Vec2Type>
inline Vec3Type Map2DTo3D(const Vec2Type &vec2D)
{
    Vec3Type vec3D = Vec3Type(0.0f);
    vec3D.x = vec2D.x;
    vec3D.y = vec2D.y;
    vec3D.z = embree::sqrt(1.0f - vec2D.x*vec2D.x - vec2D.y*vec2D.y);
    std::cout << "Map2DTo3D: length: " << embree::length(vec3D) << std::endl; 
    return vec3D;
}

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
            const vfloat<VecSize> value =  weight * samplePDF;
            //std::cout << "data[" << n << "]: " << "value: " << value << "\t samplePDF: " << samplePDF;

            const Vec3< vfloat<VecSize> > localDirection = embree::frame( vmm._meanDirections[tmp.quot] ).inverse() * Vec3< vfloat<VecSize> > (sample.direction);
            const Vec2< vfloat<VecSize> > localDirection2D(localDirection.x, localDirection.y);
            const vfloat<VecSize> assignedWeight = softAssign.assignments[tmp.quot] * weight;
            local2D[n].x = localDirection2D.x[tmp.rem];
            local2D[n].y = localDirection2D.y[tmp.rem];
            local2D[n].z = assignedWeight[tmp.rem];

            sumWeights += assignedWeight[tmp.rem];

            mean.x += assignedWeight[tmp.rem] * localDirection2D.x[tmp.rem];
            mean.y += assignedWeight[tmp.rem] * localDirection2D.y[tmp.rem];

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

    //std::cout << "split: " << "\tmean: " << mean.x << ", \t " << mean.y<< "\t covariance: " << covariance.x << ", \t "  << covariance.y << ", \t "  << covariance.z << std::endl;
    //std::cout << "eigen: " << "\tevalue0: " << eigenValue0 << "\teVec0: " << eigenVector0.x << ", \t " << eigenVector0.y << "\tevalue1: " << eigenValue1 << "\teVec1: " << eigenVector1.x << ", \t " << eigenVector1.y << std::endl;

    return splitInfo;
}


template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const
{
    std::cout << "UpdateSplitStatistics" << std::endl;
    
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

                //chiSquareEst *=vmm._weights[k];
                splitStats.chiSquareMCEstimates[k] += select(softAssign.assignments[k] > 0.f , chiSquareEst, zeros);

                const Vec3< vfloat<VecSize> > localDirection = embree::frame( vmm._meanDirections[k] ).inverse() * Vec3< vfloat<VecSize> > (sample.direction);
                const Vec2< vfloat<VecSize> > localDirection2D(localDirection.x, localDirection.y);
                const vfloat<VecSize> assignedWeight = softAssign.assignments[k] * weight;
                splitStats.splitMeans[k] += localDirection2D * assignedWeight;
                splitStats.splitCovariances[k].x += localDirection2D.x * localDirection2D.x * assignedWeight;
                splitStats.splitCovariances[k].y += localDirection2D.y * localDirection2D.y * assignedWeight;
                splitStats.splitCovariances[k].z += localDirection2D.x * localDirection2D.y * assignedWeight;
                splitStats.sumWeights[k] += assignedWeight;
            }
            validDataCount++;
            //std::cout << std::endl;
        }
    }
    splitStats.numSamples += validDataCount;
    splitStats.mcEstimate += mcEstimate;

    size_t ccount = 0;
    for (size_t k =0; k < cnt; k++)
    {
        const Vec2< vfloat<VecSize> > splitMean = splitStats.splitMeans[k] / splitStats.sumWeights[k];
        Vec3< vfloat<VecSize> > covariance;
        covariance.x = (splitStats.splitCovariances[k].x / splitStats.sumWeights[k]) - (splitMean.x * splitMean.x);
        covariance.y = (splitStats.splitCovariances[k].y / splitStats.sumWeights[k]) - (splitMean.y * splitMean.y);
        covariance.z = (splitStats.splitCovariances[k].z / splitStats.sumWeights[k]) - (splitMean.x * splitMean.y);

        for (size_t j = 0; j < VecSize; j++)
        {
            if (ccount < splitStats.numComponents)
            {
                std::cout << "split[" << ccount << "]: " << "\tmean: " << splitMean.x[j] << ", \t " << splitMean.y[j] << "\t covariance: " << covariance.x[j] << ", \t "  << covariance.y[j] << ", \t "  << covariance.z[j] << std::endl;
                ccount++;
            }
        }
    }

}

template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::SplitComponent(VMM &vmm, ComponentSplitStatistics &splitStats, SufficientStatisitcs &suffStats, const size_t idx) const
{
    ComponentSplitinfo splitInfo;
    const div_t tmp = div(idx, static_cast<int>(VecSize));

    float inv_sumWeights = rcp(splitStats.sumWeights[tmp.quot][tmp.rem]);
    splitInfo.mean = Vector2(splitStats.splitMeans[tmp.quot].x[tmp.rem], splitStats.splitMeans[tmp.quot].y[tmp.rem]);
    splitInfo.mean *= inv_sumWeights;

    splitInfo.covariance.x = splitStats.splitCovariances[tmp.quot].x[tmp.rem] * inv_sumWeights - splitInfo.mean.x*splitInfo.mean.x;
    splitInfo.covariance.y = splitStats.splitCovariances[tmp.quot].y[tmp.rem] * inv_sumWeights - splitInfo.mean.y*splitInfo.mean.y;
    splitInfo.covariance.z = splitStats.splitCovariances[tmp.quot].z[tmp.rem] * inv_sumWeights - splitInfo.mean.x*splitInfo.mean.y;

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


    float sumStatsWeight = suffStats.sumOfWeightedStats[tmp.quot][tmp.rem];
    float weight = vmm._weights[tmp.quot][tmp.rem];
    float meanCosine = vmm._meanCosines[tmp.quot][tmp.rem];
    Vector3 meanDirection = Vector3(vmm._meanDirections[tmp.quot].x[tmp.rem],
                                    vmm._meanDirections[tmp.quot].y[tmp.rem],
                                    vmm._meanDirections[tmp.quot].z[tmp.rem]);

    float newWeight = weight * 0.5f;

    Vector2 meanDir2D0 = /*splitInfo.mean + */ (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5);
    Vector3 meanDirection0 =  embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2>(meanDir2D0);
    float newMeanCosine = meanCosine / dot(meanDirection, meanDirection0);
    float newKkappa = MeanCosineToKappa<float> (newMeanCosine);

    //float weight1 = weight * 0.5f;
    Vector2 meanDir2D1 = /*splitInfo.mean + */ - (splitInfo.eigenVector0 * splitInfo.eigenValue0 * 0.5);
    Vector3 meanDirection1 = embree::frame(meanDirection) * Map2DTo3D<Vector3, Vector2>(meanDir2D1);
    //float meanCosine1 = meanCosine / meanDirection0.z;
    //float kappa1 = MeanCosineToKappa<float> (meanCosine1);

    size_t K = vmm._numComponents;
    vmm.swapComponents(K-1, idx);
    suffStats.swapComponentStats(K-1, idx);
    const div_t tmpI = div(K-1, static_cast<int>(VecSize));
    const div_t tmpJ = div(K, static_cast<int>(VecSize));

    vmm._weights[tmpI.quot][tmpI.rem] = newWeight;
    vmm._meanCosines[tmpI.quot][tmpI.rem] = newMeanCosine;
    vmm._kappas[tmpI.quot][tmpI.rem] = newKkappa;
    vmm._meanDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x;
    vmm._meanDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y;
    vmm._meanDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z;

    vmm._weights[tmpJ.quot][tmpJ.rem] = newWeight;
    vmm._meanCosines[tmpJ.quot][tmpJ.rem] = newMeanCosine;
    vmm._kappas[tmpJ.quot][tmpJ.rem] = newKkappa;
    vmm._meanDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x;
    vmm._meanDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y;
    vmm._meanDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z;

    vmm._numComponents = K + 1;
    vmm._calculateNormalization();

    sumStatsWeight /= 2.0f;

    suffStats.sumOfWeightedStats[tmpI.quot][tmpI.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].x[tmpI.rem] = meanDirection0.x * newMeanCosine * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].y[tmpI.rem] = meanDirection0.y * newMeanCosine * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpI.quot].z[tmpI.rem] = meanDirection0.z * newMeanCosine * sumStatsWeight;

    suffStats.sumOfWeightedStats[tmpJ.quot][tmpJ.rem] = sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].x[tmpJ.rem] = meanDirection1.x * newMeanCosine * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].y[tmpJ.rem] = meanDirection1.y * newMeanCosine * sumStatsWeight;
    suffStats.sumOfWeightedDirections[tmpJ.quot].z[tmpJ.rem] = meanDirection1.z * newMeanCosine * sumStatsWeight;
    suffStats.numComponents = K + 1;

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
        splitCovariances[k].x = zeros;
        splitCovariances[k].y = zeros;
        splitCovariances[k].z = zeros;

        splitMeans[k].x = zeros;
        splitMeans[k].y = zeros;

        sumWeights[k] = zeros;
    }

    mcEstimate = 0.0f;
    numSamples = 0.0f;
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
float VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::getSumChiSquareEst() const
{
    float sumChiSquareEst = 0.0f;

    for ( int k = 0; k < numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        sumChiSquareEst += chiSquareMCEstimates[tmp.quot][tmp.rem];
    }
    return sumChiSquareEst  / (float) numSamples;
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
        ss << "\t stats[" << k << "]: " << "chiSquareEst: " << chiSquareMCEstimates[tmp.quot][tmp.rem] / (float) numSamples;
        /*
        ss << "\t kappa: " <<  _kappas[tmp.quot][tmp.rem];
        ss << "\t meanDirection: [" <<  _meanDirections[tmp.quot].x[tmp.rem] << "\t" <<  _meanDirections[tmp.quot].y[tmp.rem] << "\t" <<  _meanDirections[tmp.quot].z[tmp.rem] << "]";
        ss << "\t normalization: " <<  _normalizations[tmp.quot][tmp.rem];
        ss << "\t eMinus2Kappa: " <<  _eMinus2Kappa[tmp.quot][tmp.rem];
        ss << "\t meanCosine: " <<  _meanCosines[tmp.quot][tmp.rem];
        */
        ss << std::endl;
        sumChiSquareEst += chiSquareMCEstimates[tmp.quot][tmp.rem];
    }
    ss << "sumChiSquareEst: " << sumChiSquareEst  / (float) numSamples << std::endl;
    ss << "mcEstimate: " << mcEstimate << std::endl;
    ss << "numSamples: " << numSamples << std::endl;
    return ss.str();
}

}