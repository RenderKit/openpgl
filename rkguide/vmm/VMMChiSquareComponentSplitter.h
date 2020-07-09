// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "VMM.h"
#include "../data/DirectionalSampleData.h"

namespace rkguide
{

template<int VecSize, int maxComponents>
struct VonMisesFisherChiSquareComponentSplitter
{

public:
    typedef VonMisesFisherMixture<VecSize, maxComponents> VMM;
    typedef std::integral_constant<size_t, (maxComponents + (VecSize -1)) / VecSize> NumVectors;

struct ComponentSplitStatistics
{

    vfloat<VecSize> chiSquareMCEstimates[maxComponents];
    Vec3<vfloat<VecSize> > covariances[maxComponents];
    size_t numComponents{0};
    float mcEstimate{0.0f};
    float numSamples{0.0f};

    void clear(const size_t &_numComponents);
    void clearAll();

    std::string toString() const;
};

void CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const;

void UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const;

};

template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::CalculateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const
{
    splitStats.clear(vmm._numComponents);
    this->UpdateSplitStatistics(vmm, splitStats, mcEstimate, data, numData);
}

template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::UpdateSplitStatistics(const VMM &vmm, ComponentSplitStatistics &splitStats, const float &mcEstimate, const DirectionalSampleData *data, const size_t &numData) const
{
    typename VMM::SoftAssignment softAssign;
    const vfloat<VecSize> zeros(0.f);
    const int cnt = (splitStats.numComponents + VecSize-1) / VecSize;
    size_t validDataCount = 0.0f;

    for (size_t n = 0; n < numData; n++)
    {
        const DirectionalSampleData sample = data[n];
        if (vmm.softAssignment(sample.direction, softAssign) )
        {
            const vfloat<VecSize> samplePDF = sample.pdf;
            const vfloat<VecSize> value = sample.weight * sample.pdf;
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

                chiSquareEst *=vmm._weights[k];
                splitStats.chiSquareMCEstimates[k] += select(softAssign.assignments[k] > 0.f , chiSquareEst, zeros);
            }
            validDataCount++;
            //std::cout << std::endl;
        }
    }
    splitStats.numSamples += validDataCount;
    splitStats.mcEstimate += mcEstimate;
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
        covariances[k].x = zeros;
        covariances[k].y = zeros;
        covariances[k].z = zeros;
    }

    mcEstimate = 0.0f;
    numSamples = 0.0f;
}

template<int VecSize, int maxComponents>
void VonMisesFisherChiSquareComponentSplitter<VecSize, maxComponents>::ComponentSplitStatistics::clearAll()
{
    this->clear(maxComponents);
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