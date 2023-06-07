// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"

#include <embreeSrc/common/simd/simd.h>

#include <embreeSrc/common/math/vec2.h>
#include <embreeSrc/common/math/vec3.h>
#include <embreeSrc/common/math/linearspace3.h>
#include <embreeSrc/common/math/transcendental.h>

#include <math.h>

#include <algorithm>
#include <sstream>

#include <fstream>
#include <iostream>

namespace openpgl
{

template<typename Type>
Type MeanCosineToKappa(const Type &meanCosine);

template<typename Type>
Type KappaToMeanCosine(const Type &kappa);

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
struct ParallaxAwareVonMisesFisherMixture
{

public:
    enum {
        ParallaxCompensation = UseParallaxCompensation
    };
    
    enum{
        MaxComponents = maxComponents,
        VectorSize = VecSize,
        NumVectors = (maxComponents + (VecSize -1)) / VecSize
    };
public:
    struct SoftAssignment{
        embree::vfloat<VecSize> assignments[NumVectors];
        size_t size;
        float pdf;

        std::string toString() const;
        bool isValid() const;
    };
public:

    ParallaxAwareVonMisesFisherMixture() = default;

    // VMM attributes
    embree::vfloat<VecSize> _weights[NumVectors];
    embree::vfloat<VecSize> _kappas[NumVectors];
    embree::Vec3<embree::vfloat<VecSize> > _meanDirections[NumVectors];

    embree::vfloat<VecSize> _normalizations[NumVectors];
    embree::vfloat<VecSize> _eMinus2Kappa[NumVectors];
    embree::vfloat<VecSize> _meanCosines[NumVectors];

    size_t _numComponents{maxComponents};

    // Parallax-aware attributes
    embree::vfloat<VecSize> _distances[NumVectors];
    Point3 _pivotPosition {0.0f, 0.0f, 0.0f};

    embree::vfloat<VecSize> _volumeScatterProbabilityWeights[NumVectors];

    void serialize(std::ostream& stream) const;

    void deserialize(std::istream& stream);

    void uniformInit( float kappa );

    bool softAssignment( Vector3 direction, SoftAssignment &assignment ) const;

    float pdf( Vector3 direction ) const;

    Vector3 sample( const Vector2 sample ) const;

    void mergeComponents(const size_t &idx0, const size_t &idx1);

    void splitComponent(const size_t &idx0, const size_t &idx1, const float &weight0, const float &weight1, const Vector3 &meanDirection0, const Vector3 &meanDirection1, const float &meanCosine0, const float &meanCosine1);

    void performRelativeParallaxShift( const Vector3 &shiftDirection);

    // Product and convolution functions
    void convole(const float &meanCosine);

    float product(const float &weight, const Vector3 &meanDirection, const float &kappa);

    float product(const float &weight, const Vector3 &meanDirection, const float &kappa, const float &normalization);

    // Mixture component methods
    void swapComponents(const size_t &idx0, const size_t &idx1);

    void clearComponent(const size_t &idx);

    // Getter methods for the PAVMM attributes
    size_t getNumComponents() const;

    void setNumComponents(const size_t &numComponents);

    Vector3 getComponentMeanDirection(const size_t &idx) const;

    void setComponentMeanDirection(const size_t idx, const Vector3 &meanDirection);

    float getComponentWeight(const size_t &idx) const;

    void setComponentWeight(const size_t idx, const float &weight);

    float getComponentKappa(const size_t &idx) const;

    void setComponentKappa(const size_t idx, const float &kappa);

    float getComponentDistance(const size_t &idx) const;

    void setComponentDistance(const size_t &idx, const float &distance);

    bool isValid() const;

    float volumeScatterProbability( const Vector3 &direction ) const;

    std::string toString() const;

    void _calculateNormalization();

    void _calculateMeanCosines();

    void _normalizeWeights();

};

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
size_t ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::getNumComponents() const
{
    return _numComponents;
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::setNumComponents(const size_t &numComponents)
{
    _numComponents = numComponents;
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
std::string ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::SoftAssignment::toString() const{
    std::stringstream ss;
    ss << "SoftAssignment:" << std::endl;
    ss << "size: " << size << std::endl;
    ss << "pdf: " << pdf << std::endl;
    for ( int k = 0; k < size; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        ss << "assign[" << k << "]: " << assignments[tmp.quot][tmp.rem];
        ss << std::endl;
    }
    return ss.str();
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
bool ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::SoftAssignment::isValid() const{
    bool valid = true;

    valid = valid && size > 0;
    valid = valid && size <= maxComponents;
    OPENPGL_ASSERT(valid);

    valid = valid && pdf >= 0;    
    valid = valid && embree::isvalid(pdf);
    OPENPGL_ASSERT(valid);

    for ( int k = 0; k < size; k++)
    {
        const div_t tmpK = div(k, static_cast<int>(VecSize));
        valid = valid && assignments[tmpK.quot][tmpK.rem] >= 0.0f;
        valid = valid && embree::isvalid(assignments[tmpK.quot][tmpK.rem]);
        OPENPGL_ASSERT(valid);
    }
    OPENPGL_ASSERT(valid);
    return valid;
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
std::string ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::toString() const{
    std::stringstream ss;
    ss.precision(5);
    ss << "ParallaxAwareVonMisesFisherMixture:" << std::endl;
    ss << "maxComponents: " << maxComponents << std::endl;
    ss << "VecSize: " << VecSize << std::endl;
    ss << "numVectors: " << NumVectors << std::endl;
    ss << "---------------------- "  << std::endl;
    ss << "numComponents: " << this->_numComponents << std::endl;
    float sumWeights = 0.0f;
    for ( int k = 0; k < this->_numComponents; k++)
    //for ( int k = 0; k < maxComponents; k++)
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
        ss << "\t volumeScatterProbabilityWeight: " <<  _volumeScatterProbabilityWeights[tmp.quot][tmp.rem];
        ss << std::endl;
        sumWeights += this->_weights[tmp.quot][tmp.rem];
    }

    ss << "pivot: " << _pivotPosition << std::endl;
    ss << "sumWeights: " << sumWeights << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::splitComponent(const size_t &idx0, const size_t &idx1, const float &weight0, const float &weight1, const Vector3 &meanDirection0, const Vector3 &meanDirection1, const float &meanCosine0, const float &meanCosine1)
{
    OPENPGL_ASSERT(meanCosine0 > 0.0f && meanCosine0 <=1.0f);
    OPENPGL_ASSERT(meanCosine1 > 0.0f && meanCosine1 <=1.0f);

    const div_t tmpIdx0 = div(idx0, static_cast<int>(VectorSize));
    const div_t tmpIdx1 = div(idx1, static_cast<int>(VectorSize));
    
    // splitting PAVMM
    _distances[tmpIdx1.quot][tmpIdx1.rem] = _distances[tmpIdx0.quot][tmpIdx0.rem];

    // splitting VMM
    _weights[tmpIdx0.quot][tmpIdx0.rem] = weight0;
    _meanCosines[tmpIdx0.quot][tmpIdx0.rem] = meanCosine0;
    _kappas[tmpIdx0.quot][tmpIdx0.rem] = MeanCosineToKappa<float> (meanCosine0);
    _meanDirections[tmpIdx0.quot].x[tmpIdx0.rem] = meanDirection0.x;
    _meanDirections[tmpIdx0.quot].y[tmpIdx0.rem] = meanDirection0.y;
    _meanDirections[tmpIdx0.quot].z[tmpIdx0.rem] = meanDirection0.z;

    _weights[tmpIdx1.quot][tmpIdx1.rem] = weight1;
    _meanCosines[tmpIdx1.quot][tmpIdx1.rem] = meanCosine1;
    _kappas[tmpIdx1.quot][tmpIdx1.rem] = MeanCosineToKappa<float> (meanCosine1);
    _meanDirections[tmpIdx1.quot].x[tmpIdx1.rem] = meanDirection1.x;
    _meanDirections[tmpIdx1.quot].y[tmpIdx1.rem] = meanDirection1.y;
    _meanDirections[tmpIdx1.quot].z[tmpIdx1.rem] = meanDirection1.z;

    _volumeScatterProbabilityWeights[tmpIdx1.quot][tmpIdx1.rem] = _volumeScatterProbabilityWeights[tmpIdx0.quot][tmpIdx0.rem];

    if (idx1 == _numComponents){
        _numComponents++;
    }
    _calculateNormalization();
}


template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::mergeComponents(const size_t &idx0, const size_t &idx1)
{
    const div_t tmpIdx0 = div( idx0, VecSize);
    const div_t tmpIdx1 = div( idx1, VecSize);

    //const div_t tmpIdx2 = div( this->_numComponents -1, VecSize);
    if (idx0 != idx1)
    {
        const float weight0 = _weights[tmpIdx0.quot][tmpIdx0.rem];
        const float weight1 = _weights[tmpIdx1.quot][tmpIdx1.rem];

        const float meanCosine0 = _meanCosines[tmpIdx0.quot][tmpIdx0.rem];
        const float meanCosine1 = _meanCosines[tmpIdx1.quot][tmpIdx1.rem];


        float kappa = 0.0f;
        float norm = ONE_OVER_FOUR_PI;
        float eMin2Kappa = 1.0f;

        float weight = weight0 + weight1;

        float meanDirectionX = weight0 * meanCosine0 * _meanDirections[tmpIdx0.quot].x[tmpIdx0.rem]
            + weight1 * meanCosine1 * _meanDirections[tmpIdx1.quot].x[tmpIdx1.rem];
        float meanDirectionY = weight0 * meanCosine0 * _meanDirections[tmpIdx0.quot].y[tmpIdx0.rem]
            + weight1 * meanCosine1 * _meanDirections[tmpIdx1.quot].y[tmpIdx1.rem];
        float meanDirectionZ = weight0 * meanCosine0 * _meanDirections[tmpIdx0.quot].z[tmpIdx0.rem]
            + weight1 * meanCosine1 * _meanDirections[tmpIdx1.quot].z[tmpIdx1.rem];


        //std::cout << "mergeComponents: cosTheta: " << _meanDirections[tmpIdx0.quot].x[tmpIdx0.rem] *_meanDirections[tmpIdx1.quot].x[tmpIdx1.rem] +
        //                                                _meanDirections[tmpIdx0.quot].y[tmpIdx0.rem] *_meanDirections[tmpIdx1.quot].y[tmpIdx1.rem] +
        //                                                _meanDirections[tmpIdx0.quot].z[tmpIdx0.rem] *_meanDirections[tmpIdx1.quot].z[tmpIdx1.rem] << std::endl;

        meanDirectionX /= weight;
        meanDirectionY /= weight;
        meanDirectionZ /= weight;

        float meanCosine = meanDirectionX * meanDirectionX + meanDirectionY * meanDirectionY
            + meanDirectionZ * meanDirectionZ;

        if ( meanCosine > 0.0f )
        {
            meanCosine = std::sqrt(meanCosine);

            kappa = MeanCosineToKappa<float>(meanCosine);
            kappa = kappa < 1e-3f ? 0.f : kappa;
            //eMin2Kappa = math::fastexp( -2.0f * kappa );
            eMin2Kappa = embree::exp( -2.0f * kappa );
            norm = kappa / ( 2.0f * M_PI * ( 1.0f - eMin2Kappa ) );

            meanDirectionX /= meanCosine;
            meanDirectionY /= meanCosine;
            meanDirectionZ /= meanCosine;
        }
        else
        {
            meanDirectionX = _meanDirections[tmpIdx0.quot].x[tmpIdx0.rem];
            meanDirectionY = _meanDirections[tmpIdx0.quot].y[tmpIdx0.rem];
            meanDirectionZ = _meanDirections[tmpIdx0.quot].z[tmpIdx0.rem];
        }

        _weights[tmpIdx0.quot][tmpIdx0.rem] = weight;
        _kappas[tmpIdx0.quot][tmpIdx0.rem] = kappa;
        _meanCosines[tmpIdx0.quot][tmpIdx0.rem] = meanCosine;

        _normalizations[tmpIdx0.quot][tmpIdx0.rem] = norm;
        _eMinus2Kappa[tmpIdx0.quot][tmpIdx0.rem] = eMin2Kappa;

        _meanDirections[tmpIdx0.quot].x[tmpIdx0.rem] = meanDirectionX;
        _meanDirections[tmpIdx0.quot].y[tmpIdx0.rem] = meanDirectionY;
        _meanDirections[tmpIdx0.quot].z[tmpIdx0.rem] = meanDirectionZ;

        const float distance0 = _distances[tmpIdx0.quot][tmpIdx0.rem];
        const float distance1 = _distances[tmpIdx1.quot][tmpIdx1.rem];

        float newDistance = weight0 * distance0 + weight1 * distance1;
        newDistance /= (weight0 + weight1);
        _distances[tmpIdx0.quot][tmpIdx0.rem] = newDistance;

        const float volumeScatterProbability0 = _volumeScatterProbabilityWeights[tmpIdx0.quot][tmpIdx0.rem];
        const float volumeScatterProbability1 = _volumeScatterProbabilityWeights[tmpIdx1.quot][tmpIdx1.rem];

        float newVolumeScatterProbability = weight0 * volumeScatterProbability0 + weight1 * volumeScatterProbability1;
        newVolumeScatterProbability /= (weight0 + weight1);
        _volumeScatterProbabilityWeights[tmpIdx0.quot][tmpIdx0.rem] = newVolumeScatterProbability;

        //std::cout << "mergeComponents: weight: " << weight << "\tkappa: " << kappa << "\tmeanDirection: " << meanDirectionX << "\t" << meanDirectionY << "\t" << meanDirectionZ << std::endl;
        swapComponents( idx1, _numComponents -1 );
        clearComponent( _numComponents -1 );
        _numComponents -= 1;
    }
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::swapComponents( const size_t &idx0, const size_t &idx1 )
{
    const div_t tmpIdx0 = div( idx0, VecSize);
    const div_t tmpIdx1 = div( idx1, VecSize);

    if (idx0 != idx1)
    {
        std::swap(_weights[tmpIdx0.quot][tmpIdx0.rem], _weights[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_kappas[tmpIdx0.quot][tmpIdx0.rem], _kappas[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_eMinus2Kappa[tmpIdx0.quot][tmpIdx0.rem], _eMinus2Kappa[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_meanCosines[tmpIdx0.quot][tmpIdx0.rem], _meanCosines[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_normalizations[tmpIdx0.quot][tmpIdx0.rem], _normalizations[tmpIdx1.quot][tmpIdx1.rem]);

        std::swap(_meanDirections[tmpIdx0.quot].x[tmpIdx0.rem], _meanDirections[tmpIdx1.quot].x[tmpIdx1.rem]);
        std::swap(_meanDirections[tmpIdx0.quot].y[tmpIdx0.rem], _meanDirections[tmpIdx1.quot].y[tmpIdx1.rem]);
        std::swap(_meanDirections[tmpIdx0.quot].z[tmpIdx0.rem], _meanDirections[tmpIdx1.quot].z[tmpIdx1.rem]);

        std::swap(_distances[tmpIdx0.quot][tmpIdx0.rem], _distances[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_volumeScatterProbabilityWeights[tmpIdx0.quot][tmpIdx0.rem], _volumeScatterProbabilityWeights[tmpIdx1.quot][tmpIdx1.rem]);
    }
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::clearComponent( const size_t &idx )
{
    const div_t tmpIdx = div( idx, VecSize);

    _weights[tmpIdx.quot][tmpIdx.rem]= 0.f;
    _kappas[tmpIdx.quot][tmpIdx.rem] = 0.f;
    _eMinus2Kappa[tmpIdx.quot][tmpIdx.rem] = 1.f;
    _meanCosines[tmpIdx.quot][tmpIdx.rem] = 0.f;
    _normalizations[tmpIdx.quot][tmpIdx.rem] = ONE_OVER_FOUR_PI;

    _meanDirections[tmpIdx.quot].x[tmpIdx.rem] = 0.f;
    _meanDirections[tmpIdx.quot].y[tmpIdx.rem] = 0.f;
    _meanDirections[tmpIdx.quot].z[tmpIdx.rem] = 1.f;

    _distances[tmpIdx.quot][tmpIdx.rem] = 0.0f;
    _volumeScatterProbabilityWeights[tmpIdx.quot][tmpIdx.rem] = 0.0f;
    
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::serialize(std::ostream& stream) const
{

    for(uint32_t k=0;k<NumVectors;k++){
        stream.write(reinterpret_cast<const char*>(&_weights[k]), sizeof(embree::vfloat<VecSize>));
        stream.write(reinterpret_cast<const char*>(&_kappas[k]), sizeof(embree::vfloat<VecSize>));
        stream.write(reinterpret_cast<const char*>(&_meanDirections[k]), sizeof(embree::Vec3<embree::vfloat<VecSize> >));

        stream.write(reinterpret_cast<const char*>(&_normalizations[k]), sizeof(embree::vfloat<VecSize>));
        stream.write(reinterpret_cast<const char*>(&_eMinus2Kappa[k]), sizeof(embree::vfloat<VecSize>));
        stream.write(reinterpret_cast<const char*>(&_meanCosines[k]), sizeof(embree::vfloat<VecSize>));
        stream.write(reinterpret_cast<const char*>(&_distances[k]), sizeof(embree::vfloat<VecSize>));
        stream.write(reinterpret_cast<const char*>(&_volumeScatterProbabilityWeights[k]), sizeof(embree::vfloat<VecSize>));
    }
    stream.write(reinterpret_cast<const char*>(&_numComponents), sizeof(_numComponents));
    stream.write(reinterpret_cast<const char*>(&_pivotPosition), sizeof(Point3));
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::deserialize(std::istream& stream)
{

    for(uint32_t k=0;k<NumVectors;k++){
        stream.read(reinterpret_cast<char*>(&_weights[k]), sizeof(embree::vfloat<VecSize>));
        stream.read(reinterpret_cast<char*>(&_kappas[k]), sizeof(embree::vfloat<VecSize>));
        stream.read(reinterpret_cast<char*>(&_meanDirections[k]), sizeof(embree::Vec3<embree::vfloat<VecSize> >));

        stream.read(reinterpret_cast<char*>(&_normalizations[k]), sizeof(embree::vfloat<VecSize>));
        stream.read(reinterpret_cast<char*>(&_eMinus2Kappa[k]), sizeof(embree::vfloat<VecSize>));
        stream.read(reinterpret_cast<char*>(&_meanCosines[k]), sizeof(embree::vfloat<VecSize>));
        stream.read(reinterpret_cast<char*>(&_distances[k]), sizeof(embree::vfloat<VecSize>));
        stream.read(reinterpret_cast<char*>(&_volumeScatterProbabilityWeights[k]), sizeof(embree::vfloat<VecSize>));
    }
    stream.read(reinterpret_cast<char*>(&_numComponents), sizeof(_numComponents));
    stream.read(reinterpret_cast<char*>(&_pivotPosition), sizeof(Point3));
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
bool ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::isValid() const
{
        bool valid = true;
    float sumWeights = 0.0f;

    for(size_t k = 0; k < _numComponents; k++){
        const div_t tmpK = div( k, VecSize );
        sumWeights += _weights[tmpK.quot][tmpK.rem];

        valid = valid && embree::isvalid(_weights[tmpK.quot][tmpK.rem]);
        valid = valid && _weights[tmpK.quot][tmpK.rem] >= 0.0f;
        valid = valid && _weights[tmpK.quot][tmpK.rem] <= 1.0f + 1e-6f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_kappas[tmpK.quot][tmpK.rem]);
        valid = valid && _kappas[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_meanCosines[tmpK.quot][tmpK.rem]);
        valid = valid && _meanCosines[tmpK.quot][tmpK.rem] >= 0.0f;
        valid = valid && _meanCosines[tmpK.quot][tmpK.rem] <= 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_meanDirections[tmpK.quot].x[tmpK.rem]);
        valid = valid && _meanDirections[tmpK.quot].x[tmpK.rem] >= -1.0f;
        valid = valid && _meanDirections[tmpK.quot].x[tmpK.rem] <= 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_meanDirections[tmpK.quot].y[tmpK.rem]);
        valid = valid && _meanDirections[tmpK.quot].y[tmpK.rem] >= -1.0f;
        valid = valid && _meanDirections[tmpK.quot].y[tmpK.rem] <= 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_meanDirections[tmpK.quot].z[tmpK.rem]);
        valid = valid && _meanDirections[tmpK.quot].z[tmpK.rem] >= -1.0f;
        valid = valid && _meanDirections[tmpK.quot].z[tmpK.rem] <= 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_normalizations[tmpK.quot][tmpK.rem]);
        valid = valid && _normalizations[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_eMinus2Kappa[tmpK.quot][tmpK.rem]);
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_distances[tmpK.quot][tmpK.rem]);
        valid = valid && _distances[tmpK.quot][tmpK.rem] >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_volumeScatterProbabilityWeights[tmpK.quot][tmpK.rem]);
        valid = valid && _volumeScatterProbabilityWeights[tmpK.quot][tmpK.rem] >= 0.0f;
        valid = valid && _volumeScatterProbabilityWeights[tmpK.quot][tmpK.rem] <= 1.0f;
        OPENPGL_ASSERT(valid);
    }

    // check unused componets
    for(int k = _numComponents; k < MaxComponents; k++){
        const div_t tmpK = div( k, VecSize );
        valid = valid && embree::isvalid(_weights[tmpK.quot][tmpK.rem]);
        valid = valid && _weights[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_kappas[tmpK.quot][tmpK.rem]);
        valid = valid && _kappas[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_meanDirections[tmpK.quot].x[tmpK.rem]);
        valid = valid && _meanDirections[tmpK.quot].x[tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_meanDirections[tmpK.quot].y[tmpK.rem]);
        valid = valid && _meanDirections[tmpK.quot].y[tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_meanDirections[tmpK.quot].z[tmpK.rem]);
        valid = valid && _meanDirections[tmpK.quot].z[tmpK.rem] == 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_meanCosines[tmpK.quot][tmpK.rem]);
        valid = valid && _meanCosines[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_normalizations[tmpK.quot][tmpK.rem]);
        valid = valid && std::fabs(_normalizations[tmpK.quot][tmpK.rem] - ONE_OVER_FOUR_PI) < 1e-6f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_eMinus2Kappa[tmpK.quot][tmpK.rem]);
        valid = valid && _eMinus2Kappa[tmpK.quot][tmpK.rem] == 1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_distances[tmpK.quot][tmpK.rem]);
        valid = valid && _distances[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(_volumeScatterProbabilityWeights[tmpK.quot][tmpK.rem]);
        valid = valid && _volumeScatterProbabilityWeights[tmpK.quot][tmpK.rem] == 0.0f;
        OPENPGL_ASSERT(valid);
    }

    return valid;
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::setComponentWeight(const size_t idx, const float &weight)
{
    const div_t tmpIdx = div( idx, VecSize);

    _weights[tmpIdx.quot][tmpIdx.rem]= weight;
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::setComponentKappa(const size_t idx, const float &kappa)
{
    const div_t tmpIdx = div( idx, VecSize);

    _kappas[tmpIdx.quot][tmpIdx.rem]= kappa;
    _calculateNormalization();
    _calculateMeanCosines();
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::setComponentMeanDirection(const size_t idx, const Vector3 &meanDirection)
{
    const div_t tmpIdx = div( idx, VecSize);

    _meanDirections[tmpIdx.quot].x[tmpIdx.rem] = meanDirection.x;
    _meanDirections[tmpIdx.quot].y[tmpIdx.rem] = meanDirection.y;
    _meanDirections[tmpIdx.quot].z[tmpIdx.rem] = meanDirection.z;
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
Vector3 ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::getComponentMeanDirection(const size_t &idx) const
{
    const div_t tmpIdx = div( idx, VecSize);
    return Vector3( _meanDirections[tmpIdx.quot].x[tmpIdx.rem],  _meanDirections[tmpIdx.quot].y[tmpIdx.rem],  _meanDirections[tmpIdx.quot].z[tmpIdx.rem]);
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
float ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::getComponentWeight(const size_t &idx) const
{
    const div_t tmpIdx = div( idx, VecSize);
    return _weights[tmpIdx.quot][tmpIdx.rem];
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
float ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::getComponentKappa(const size_t &idx) const
{
    const div_t tmpIdx = div( idx, VecSize);
    return _kappas[tmpIdx.quot][tmpIdx.rem];
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::setComponentDistance(const size_t &idx, const float &distance)
{
    const div_t tmpIdx = div( idx, VecSize);
    _distances[tmpIdx.quot][tmpIdx.rem]= distance;
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
float ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::getComponentDistance(const size_t &idx) const
{
    const div_t tmpIdx = div( idx, VecSize);
    return _distances[tmpIdx.quot][tmpIdx.rem];
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::convole(const float &_meanCosine)
{
    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const embree::vfloat<VecSize> meanCosine = _meanCosine;

    for(int k = 0; k < cnt;k++)
    {
        _meanCosines[k] *= meanCosine;
        _kappas[k] = MeanCosineToKappa< embree::vfloat<VecSize> >(meanCosine);
    }
    _calculateNormalization();
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
float ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::product(const float &_weight, const Vector3 &_meanDirection, const float &_kappa)
{

    float _normalization = ONE_OVER_FOUR_PI;
    //float _eMinus2Kappa = embree::fastapprox::exp< float >(-2.0f * _kappa);
    // TODO: use faster exp
    float _eMinus2Kappa = std::exp(-2.0f * _kappa);

    if ( _kappa > 0.0f)
    {
        _normalization = _kappa/(2.0f*M_PI*(1.0f-_eMinus2Kappa));
    }
    return this->product(_weight, _meanDirection, _kappa, _normalization);
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
float ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::product(const float &_weight, const Vector3 &_meanDirection, const float &_kappa, const float &_normalization)
{
    const embree::vfloat<VecSize> twoPi(2.0f*M_PI);
    const embree::vfloat<VecSize> ones(1.0f);
    const embree::vfloat<VecSize> minusTwos(-2.0f);
    const embree::vfloat<VecSize> zeros(0.0f);
    const embree::vfloat<VecSize> zeroKappaNorm(ONE_OVER_FOUR_PI);

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const embree::vfloat<VecSize> weight = _weight;
    const embree::vfloat<VecSize> kappa = _kappa;
    const embree::vfloat<VecSize> normalization = _normalization;

    const embree::Vec3<embree::vfloat<VecSize> > meanDirection = _meanDirection;

    embree::vfloat<VecSize> productIntegralVec(0.f);

    for(int k = 0; k < cnt;k++)
    {
        embree::Vec3<embree::vfloat<VecSize> > newMeanDirection = _kappas[k] * _meanDirections[k] + kappa * meanDirection;
        embree::vfloat<VecSize> newKappa = embree::sqrt(embree::dot( newMeanDirection, newMeanDirection));
        auto checkNewKappa = (newKappa > 1e-3f);
        newKappa = select( checkNewKappa,  newKappa, zeros);

        // TODO: update meanCosine
        newMeanDirection.x = select( checkNewKappa,  newMeanDirection.x / newKappa,  _meanDirections[k].x);
        newMeanDirection.y = select( checkNewKappa,  newMeanDirection.y / newKappa,  _meanDirections[k].y);
        newMeanDirection.z = select( checkNewKappa,  newMeanDirection.z / newKappa,  _meanDirections[k].z);

        embree::vfloat<VecSize> newEMinus2Kappa = embree::fastapprox::exp(minusTwos * newKappa);
        embree::vfloat<VecSize> newNormalization = newKappa / (twoPi * ( ones - newEMinus2Kappa ));
        newNormalization = select( checkNewKappa, newNormalization, zeroKappaNorm);

        embree::vfloat<VecSize> scale = ( _normalizations[k] * normalization ) / newNormalization;

        embree::vfloat<VecSize> cosTheta0 = embree::dot( _meanDirections[k], newMeanDirection );
        embree::vfloat<VecSize> cosTheta1 = embree::dot( meanDirection, newMeanDirection );

        //std::cout << "cosTheta0: " << cosTheta0 <<"\tcosTheta1: " << cosTheta1 << std::endl;
        //std::cout << "_kappas[k]: " << _kappas[k] <<"\tkappa: " << kappa << std::endl;
        //std::cout << "tmp: " <<  _kappas[k] * (cosTheta0 - ones) + kappa * (cosTheta1 - ones) << std::endl;
        embree::vfloat<VecSize> eval = embree::fastapprox::exp( _kappas[k] * (cosTheta0 - ones) + kappa * (cosTheta1 - ones) );
        //std::cout << "scale: " << scale <<"\teval: " << eval << std::endl;
        scale *= eval;
        scale *= _weights[k] * weight;

        _weights[k] = scale;
        _kappas[k] = newKappa;
        _meanDirections[k] = newMeanDirection;
        _normalizations[k] = newNormalization;
        _eMinus2Kappa[k] = newEMinus2Kappa;

        productIntegralVec += _weights[k];
    }

    float productIntegral = embree::reduce_add(productIntegralVec);
    for(int k = 0; k < cnt;k++)
    {
        _weights[k] /= productIntegral;
    }
    return productIntegral;
    //_normalizeWeights();
}



template<int VecSize, int maxComponents, bool UseParallaxCompensation>
float ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::pdf( Vector3 direction ) const{
    const int cnt = (_numComponents+VecSize-1) / VecSize;

    embree::vfloat<VecSize> pdf = {0.0f};
    embree::Vec3< embree::vfloat<VecSize> > vec3Direction(direction[0], direction[1], direction[2]);

    const embree::vfloat<VecSize> ones(1.0f);
    const embree::vfloat<VecSize> zeros(0.0f);

    for(int k = 0; k < cnt;k++)
    {
        const embree::vfloat<VecSize> cosTheta = embree::dot(vec3Direction, _meanDirections[k]);
        const embree::vfloat<VecSize> cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
        const embree::vfloat<VecSize> eval = _normalizations[k] * embree::fastapprox::exp< embree::vfloat<VecSize> >( _kappas[k] * cosThetaMinusOne );
        pdf += _weights[k] * eval;
    }

    return reduce_add(pdf);
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
bool ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::softAssignment( Vector3 direction, typename ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::SoftAssignment &softAssign ) const{

    const int cnt = (_numComponents+VecSize-1) / VecSize;

    embree::vfloat<VecSize> pdf = {0.0f};
    embree::Vec3< embree::vfloat<VecSize> > vec3Direction(direction[0], direction[1], direction[2]);

    const embree::vfloat<VecSize> ones(1.0f);
    const embree::vfloat<VecSize> zeros(0.0f);

    for(int k = 0; k < cnt;k++)
    {
        const embree::vfloat<VecSize> cosTheta = embree::dot(vec3Direction, _meanDirections[k]);
        const embree::vfloat<VecSize> cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
        const embree::vfloat<VecSize> eval = _normalizations[k] * embree::fastapprox::exp< embree::vfloat<VecSize> >( _kappas[k] * cosThetaMinusOne );
        softAssign.assignments[k] =  _weights[k] * eval;
        OPENPGL_ASSERT(embree::isvalid(softAssign.assignments[k]));
        pdf += softAssign.assignments[k];
    }
    OPENPGL_ASSERT(embree::isvalid(pdf));
    softAssign.pdf = embree::reduce_add(pdf);
    softAssign.size = _numComponents;

    if ( softAssign.pdf <= 1e-16f)
    {
        return false;
    }

    embree::vfloat<VecSize> inv_pdf = embree::rcp(softAssign.pdf);
    OPENPGL_ASSERT(embree::isvalid(inv_pdf));
    for(int k = 0; k < cnt;k++)
    {
        softAssign.assignments[k] *= inv_pdf;
    }

    return true;
}


template<int VecSize, int maxComponents, bool UseParallaxCompensation>
Vector3 ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::sample( const Vector2 sample ) const{

    uint32_t selectedVector {0};
    uint32_t selectedComponent {0};
    // First, identify component we want to sample

    Vector2 _sample = sample;
    float searched = _sample[1];
    float sumWeights = 0.0f;
    float cdf = 0.0f;
    //int k0 = 0;
    //int k1 = 0;
    // find comp

    const div_t tmp = div( _numComponents-1, VecSize);

    while(true){
        cdf = reduce_add(_weights[selectedVector]);
        if(sumWeights+cdf >= searched || selectedVector+1 >= (tmp.quot+1)){
            break;
        }else{
            sumWeights+=cdf;
            selectedVector++;
        }
    }

    int maxSelectedComponent = selectedVector == tmp.quot ? tmp.rem +1 : VecSize;

    while(true){
        cdf = _weights[selectedVector][selectedComponent];
        if(sumWeights+cdf >= searched || selectedComponent+1 >= maxSelectedComponent){
            break;
        }else{
            sumWeights+=cdf;
            selectedComponent++;
        }
    }

    _sample[1] = std::min(1 - FLT_EPSILON, (searched - sumWeights) / cdf);

     embree::Vec3<float> sampledDirection(0.f, 0.f, 1.f);

    // Second, sample selected component
    const float sKappa = _kappas[selectedVector][selectedComponent];
    const float sEMinus2Kappa = _eMinus2Kappa[selectedVector][selectedComponent];
    const embree::Vec3<float> meanDirection( _meanDirections[selectedVector].x[selectedComponent],  _meanDirections[selectedVector].y[selectedComponent],  _meanDirections[selectedVector].z[selectedComponent]);

    if (sKappa == 0.0f)
    {
        sampledDirection = squareToUniformSphere( _sample );
    }
    else
    {
        float cosTheta = 1.f + (embree::fastapprox::log<float>(1.0f + ((sEMinus2Kappa-1.f) * _sample[0]))) / sKappa;
        //float cosTheta = 1.f + (std::log1p((sEMinus2Kappa-1.f) * _sample[0])) / sKappa;

        // safeguard for numerical imprecisions (if sample[0] is 0.999999999)
        cosTheta = std::min(1.0f, std::max(cosTheta, -1.f));

        const float sinTheta = std::sqrt(1.f-cosTheta*cosTheta);

        const float phi = 2.f * M_PI * _sample[1];

        float sinPhi, cosPhi;
        sincosf(phi, &sinPhi, &cosPhi);
        sampledDirection = openpgl::sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
    }

    return embree::frame( meanDirection ) * sampledDirection;
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::performRelativeParallaxShift( const Vector3 &shiftDirection )
{
    const embree::vfloat<VecSize> ones(1.0f);
    const embree::vfloat<VecSize> zeros(0.0f);

    const int cnt = (this->_numComponents+VectorSize-1) / VectorSize;
    //const int rem = this->_numComponents % VMM::VectorSize;

    const embree::Vec3<embree::vfloat<VecSize> > shiftDirectionVec(shiftDirection);
    embree::Vec3<embree::vfloat<VecSize> > parallaxCorrectedMeanDirections;
    embree::vfloat<VecSize> lengths;
    for(uint32_t k=0;k<cnt;k++){
        parallaxCorrectedMeanDirections = this->_meanDirections[k] * _distances[k] + shiftDirectionVec;
        lengths = embree::length(parallaxCorrectedMeanDirections);
        parallaxCorrectedMeanDirections /= lengths;
        this->_meanDirections[k].x = select((_distances[k] > 0.0f) & embree::isfinite<VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.x, this->_meanDirections[k].x);
        this->_meanDirections[k].y = select((_distances[k] > 0.0f) & embree::isfinite<VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.y, this->_meanDirections[k].y);
        this->_meanDirections[k].z = select((_distances[k] > 0.0f) & embree::isfinite<VectorSize>(_distances[k]), parallaxCorrectedMeanDirections.z, this->_meanDirections[k].z);
        _distances[k] = select(_distances[k] > 0.0f, lengths, zeros);
    }

    _pivotPosition -= shiftDirection;
}



template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::_normalizeWeights( ) {

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    embree::vfloat<VecSize> sumWeights = 0.0f;
    for(int k = 0; k < cnt;k++){
        sumWeights += _weights[k];
    }

    embree::vfloat<VecSize> inv_sumWeights = 1.0f / reduce_add(sumWeights);
    for(int k = 0; k < cnt;k++){
        _weights[k] *= inv_sumWeights;
    }

}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::_calculateNormalization( ) {
    const embree::vfloat<VecSize> zeroKappaNorm(ONE_OVER_FOUR_PI);

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const embree::vfloat<VecSize> minusTwo(-2.0f);
    for(int k = 0; k < cnt;k++){
        _eMinus2Kappa[k] = embree::fastapprox::exp< embree::vfloat<VecSize> >(minusTwo*_kappas[k]);
        const embree::vfloat<VecSize> norm = _kappas[k]/(2.0f*M_PI*(1.0f-_eMinus2Kappa[k]));
        _normalizations[k] = select(_kappas[k] > 0.f, norm, zeroKappaNorm);
    }

}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
void ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::_calculateMeanCosines( ) {

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const embree::vfloat<VecSize> zeros(0.0f);
    const embree::vfloat<VecSize> ones(1.0f);
    for(int k = 0; k < cnt;k++){
        embree::vfloat<VecSize>  tanh = ones - 2.0f / ( embree::fastapprox::exp( 2.0f * _kappas[k] ) - ones );
        embree::vfloat<VecSize>  meanCosine = ones /tanh - ones / _kappas[k];
        //std::cout << "meanCosine: " << meanCosine << std::endl;
        _meanCosines[k] = select(_kappas[k] > 0.f, meanCosine, zeros);
    }
}

template<int VecSize, int maxComponents, bool UseParallaxCompensation>
float ParallaxAwareVonMisesFisherMixture<VecSize, maxComponents,UseParallaxCompensation>::volumeScatterProbability( const Vector3 &direction ) const{
    const int cnt = (_numComponents+VecSize-1) / VecSize;

    embree::vfloat<VecSize> volumeScatterProbability = {0.0f};
    embree::vfloat<VecSize> pdf = {0.0f};
    embree::Vec3< embree::vfloat<VecSize> > vec3Direction(direction[0], direction[1], direction[2]);

    const embree::vfloat<VecSize> ones(1.0f);
    const embree::vfloat<VecSize> zeros(0.0f);

    for(int k = 0; k < cnt;k++)
    {
        const embree::vfloat<VecSize> cosTheta = embree::dot(vec3Direction, _meanDirections[k]);
        const embree::vfloat<VecSize> cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
        const embree::vfloat<VecSize> eval = _weights[k] * _normalizations[k] * embree::fastapprox::exp< embree::vfloat<VecSize> >( _kappas[k] * cosThetaMinusOne );
        pdf += eval;
        volumeScatterProbability += _volumeScatterProbabilityWeights[k] * eval;
    }

    return reduce_add(volumeScatterProbability) / reduce_add(pdf);
}

template<typename Type>
inline Type KappaToMeanCosine(const Type &kappa)
{
    const Type ones( 1.0f);
    const Type zeros( 0.0f);

    Type meanCosine = ones / embree::tanh( kappa) - ones / kappa;
    return embree::select( kappa > 0.f, meanCosine, zeros);
}


template<typename Type>
inline Type MeanCosineToKappa(const Type &meanCosine)
{
    const Type ones( 1.0f);
    const Type dim( 3.0f);
    const Type meanCosine2 = meanCosine * meanCosine;
    return ( meanCosine * dim - meanCosine * meanCosine2) / ( ones - meanCosine2 );
}

}
