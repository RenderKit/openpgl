// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "VMM.h"

//using namespace embree;

namespace openpgl
{

template<int VecSize, int maxComponents>
struct ParallaxAwareVonMisesFisherMixture: public VonMisesFisherMixture<VecSize, maxComponents>
{

public:

    ParallaxAwareVonMisesFisherMixture() = default;

    typedef VonMisesFisherMixture<VecSize, maxComponents> VMM;

    embree::vfloat<VecSize> _distances[VMM::NumVectors];

    Point3 _pivotPosition {0.0f, 0.0f, 0.0f};

    void serialize(std::ostream& stream) const override;

    void deserialize(std::istream& stream) override;

    void mergeComponents(const size_t &idx0, const size_t &idx1) override;

    void splitComponent(const size_t &idx0, const size_t &idx1, const float &weight0, const float &weight1, const Vector3 &meanDirection0, const Vector3 &meanDirection1, const float &meanCosine0, const float &meanCosine1) override;

    void performRelativeParallaxShift( const Vector3 &shiftDirection);

    float getComponentDistance(const size_t &idx) const;

    void setComponentDistance(const size_t &idx, const float &distance);

    bool isValid() const override;

    std::string toString() const;
private:

};

template<int VecSize, int maxComponents>
std::string ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::toString() const{
    std::stringstream ss;
    ss.precision(5);
    ss << "ParallaxAwareVonMisesFisherMixture:" << std::endl;
    ss << "maxComponents: " << maxComponents << std::endl;
    ss << "VecSize: " << VecSize << std::endl;
    ss << "numVectors: " << VMM::NumVectors << std::endl;
    ss << "---------------------- "  << std::endl;
    ss << "numComponents: " << this->_numComponents << std::endl;
    float sumWeights = 0.0f;
    //for ( int k = 0; k < this->_numComponents; k++)
    for ( int k = 0; k < maxComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        ss << "vmm[" << k << "]: " << "weight: " << this->_weights[tmp.quot][tmp.rem];
        ss << "\t kappa: " <<  this->_kappas[tmp.quot][tmp.rem];
        ss << "\t meanDirection: [" <<  this->_meanDirections[tmp.quot].x[tmp.rem] << "\t" <<  this->_meanDirections[tmp.quot].y[tmp.rem] << "\t" <<  this->_meanDirections[tmp.quot].z[tmp.rem] << "]";
        ss << "\t length: " <<  embree::length(Vector3(this->_meanDirections[tmp.quot].x[tmp.rem], this->_meanDirections[tmp.quot].y[tmp.rem], this->_meanDirections[tmp.quot].z[tmp.rem]));
        ss << "\t normalization: " <<  this->_normalizations[tmp.quot][tmp.rem];
        ss << "\t eMinus2Kappa: " <<  this->_eMinus2Kappa[tmp.quot][tmp.rem];
        ss << "\t meanCosine: " <<  this->_meanCosines[tmp.quot][tmp.rem];
        ss << "\t distance: " <<  _distances[tmp.quot][tmp.rem];
        ss << std::endl;
        sumWeights += this->_weights[tmp.quot][tmp.rem];
    }

    ss << "pivot: " << _pivotPosition << std::endl;
    ss << "sumWeights: " << sumWeights << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::splitComponent(const size_t &idx0, const size_t &idx1, const float &weight0, const float &weight1, const Vector3 &meanDirection0, const Vector3 &meanDirection1, const float &meanCosine0, const float &meanCosine1)
{
    const div_t tmpIdx0 = div( idx0, VecSize);
    const div_t tmpIdx1 = div( idx1, VecSize);

    _distances[tmpIdx1.quot][tmpIdx1.rem] = _distances[tmpIdx0.quot][tmpIdx0.rem];

    VonMisesFisherMixture<VecSize, maxComponents>::splitComponent(idx0, idx1, weight0, weight1, meanDirection0, meanDirection1, meanCosine0, meanCosine1);
}


template<int VecSize, int maxComponents>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::mergeComponents(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div( idx0, VecSize);
    const div_t tmpIdx1 = div( idx1, VecSize);

    const div_t tmpIdx2 = div( this->_numComponents -1, VecSize);
    if (idx0 != idx1)
    {
        const float weight0 = this->_weights[tmpIdx0.quot][tmpIdx0.rem];
        const float weight1 = this->_weights[tmpIdx1.quot][tmpIdx1.rem];

        const float distance0 = _distances[tmpIdx0.quot][tmpIdx0.rem];
        const float distance1 = _distances[tmpIdx1.quot][tmpIdx1.rem];

        float newDistance = weight0 * distance0 + weight1 * distance1;
        newDistance /= (weight0 + weight1);
        _distances[tmpIdx0.quot][tmpIdx0.rem] = newDistance;
        _distances[tmpIdx1.quot][tmpIdx1.rem] = _distances[tmpIdx2.quot][tmpIdx2.rem];
        _distances[tmpIdx2.quot][tmpIdx2.rem] = 0.0f;
    }

    VonMisesFisherMixture<VecSize, maxComponents>::mergeComponents(idx0, idx1);
}

template<int VecSize, int maxComponents>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::serialize(std::ostream& stream) const
{
    VonMisesFisherMixture<VecSize, maxComponents>::serialize(stream);
    for(uint32_t k=0;k<VMM::NumVectors;k++){
        stream.write(reinterpret_cast<const char*>(&_distances[k]), sizeof(embree::vfloat<VecSize>));
    }
    stream.write(reinterpret_cast<const char*>(&_pivotPosition), sizeof(Point3));
}

template<int VecSize, int maxComponents>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::deserialize(std::istream& stream)
{
    VonMisesFisherMixture<VecSize, maxComponents>::deserialize(stream);
    for(uint32_t k=0;k<VMM::NumVectors;k++){
        stream.read(reinterpret_cast<char*>(&_distances[k]), sizeof(embree::vfloat<VecSize>));
    }
    stream.read(reinterpret_cast<char*>(&_pivotPosition), sizeof(Point3));
}

template<int VecSize, int maxComponents>
bool ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::isValid() const
{
    bool valid = VonMisesFisherMixture<VecSize, maxComponents>::isValid();

    for(size_t k = 0; k < this->_numComponents; k++){
        const div_t tmpK = div( k, VecSize );
        valid = valid && embree::isvalid(_distances[tmpK.quot][tmpK.rem]);
        valid = valid && _distances[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);
    }
    for(size_t k = this->_numComponents; k < maxComponents; k++){
        const div_t tmpK = div( k, VecSize );
        valid = valid && embree::isvalid(_distances[tmpK.quot][tmpK.rem]);
        valid = valid && _distances[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);
    }
    OPENPGL_ASSERT(valid);
    return valid;
}

template<int VecSize, int maxComponents>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::setComponentDistance(const size_t &idx, const float &distance)
{
    const div_t tmpIdx = div( idx, VecSize);
    _distances[tmpIdx.quot][tmpIdx.rem]= distance;
}

template<int VecSize, int maxComponents>
float ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::getComponentDistance(const size_t &idx) const
{
    const div_t tmpIdx = div( idx, VecSize);
    return _distances[tmpIdx.quot][tmpIdx.rem];
}

template<int VecSize, int maxComponents>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents>::performRelativeParallaxShift( const Vector3 &shiftDirection )
{
    const embree::vfloat<VecSize> ones(1.0f);
    const embree::vfloat<VecSize> zeros(0.0f);

    const int cnt = (this->_numComponents+VMM::VectorSize-1) / VMM::VectorSize;
    //const int rem = this->_numComponents % VMM::VectorSize;

    const embree::Vec3<embree::vfloat<VecSize> > shiftDirectionVec(shiftDirection);
    embree::Vec3<embree::vfloat<VecSize> > parallaxCorrectedMeanDirections;
    embree::vfloat<VecSize> lengths;
    for(uint32_t k=0;k<cnt;k++){
        parallaxCorrectedMeanDirections = this->_meanDirections[k] * _distances[k] + shiftDirectionVec;
        lengths = embree::length(parallaxCorrectedMeanDirections);
        parallaxCorrectedMeanDirections /= lengths;
        this->_meanDirections[k].x = select((_distances[k] > 0.0f) & embree::isfinite<VMM::VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.x, this->_meanDirections[k].x);
        this->_meanDirections[k].y = select((_distances[k] > 0.0f) & embree::isfinite<VMM::VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.y, this->_meanDirections[k].y);
        this->_meanDirections[k].z = select((_distances[k] > 0.0f) & embree::isfinite<VMM::VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.z, this->_meanDirections[k].z);
        _distances[k] = select(_distances[k] > 0.0f, lengths, zeros);
    }

    _pivotPosition -= shiftDirection;
}

}
