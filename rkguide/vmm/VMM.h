// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

#include <embree/common/simd/simd.h>

#include <embree/common/math/vec2.h>
#include <embree/common/math/vec3.h>
#include <embree/common/math/linearspace3.h>
#include <embree/common/math/transcendental.h>

#include <math.h>

#include <algorithm>
#include <sstream>

using namespace embree;

namespace rkguide
{

template<typename Type>
Type MeanCosineToKappa(const Type &meanCosine);

template<typename Type>
Type KappaToMeanCosine(const Type &kappa);

template<int VecSize, int maxComponents>
struct VonMisesFisherMixture
{

public:
    typedef std::integral_constant<size_t, (maxComponents + (VecSize -1)) / VecSize> NumVectors;
private:
    //const static int numVectors {(maxComponents+VecSize-1)/VecSize};

public:

    struct SoftAssignment{
        vfloat<VecSize> assignments[NumVectors::value];
        size_t size;
        float pdf;
    };


public:

    VonMisesFisherMixture() = default;
    VonMisesFisherMixture( const VonMisesFisherMixture &a);

    vfloat<VecSize> _weights[NumVectors::value];
    vfloat<VecSize> _kappas[NumVectors::value];
    Vec3<vfloat<VecSize> > _meanDirections[NumVectors::value];

    vfloat<VecSize> _normalizations[NumVectors::value];
    vfloat<VecSize> _eMinus2Kappa[NumVectors::value];
    vfloat<VecSize> _meanCosines[NumVectors::value];

    size_t _numComponents{maxComponents};

    void uniformInit( float kappa );
    float pdf( Vec3<float> direction ) const;

    bool softAssignment( Vec3<float> direction, SoftAssignment &assignment ) const;

    Vec3<float> sample( const Vec2<float> sample ) const;

    std::string toString() const;


    void swapComponents(const size_t &idx0, const size_t &idx1);

    void mergeComponents(const size_t &idx0, const size_t &idx1);

    void clearComponent(const size_t &idx);

    void convole(const float &meanCosine);

    float product(const float &weight, const Vector3 &meanDirection, const float &kappa);

    float product(const float &weight, const Vector3 &meanDirection, const float &kappa, const float &normalization);

//private:
    void _calculateNormalization();

    void _calculateMeanCosines();
};

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::convole(const float &_meanCosine)
{
    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const vfloat<VecSize> meanCosine = _meanCosine;

    for(int k = 0; k < cnt;k++)
    {
        _meanCosines[k] *= meanCosine;
        _kappas[k] = MeanCosineToKappa< vfloat<VecSize> >(meanCosine);
    }
    _calculateNormalization();
}

template<int VecSize, int maxComponents>
float VonMisesFisherMixture<VecSize, maxComponents>::product(const float &_weight, const Vector3 &_meanDirection, const float &_kappa)
{

    float _normalization = 1.0f / (4.0f*M_PI);
    //float _eMinus2Kappa = embree::fastapprox::exp< float >(-2.0f * _kappa);
    // TODO: use faster exp
    float _eMinus2Kappa = std::exp(-2.0f * _kappa);

    if ( _kappa > 0.0f)
    {
        _normalization = _kappa/(2.0f*M_PI*(1.0f-_eMinus2Kappa));
    }
    return this->product(_weight, _meanDirection, _kappa, _normalization);
}

template<int VecSize, int maxComponents>
float VonMisesFisherMixture<VecSize, maxComponents>::product(const float &_weight, const Vector3 &_meanDirection, const float &_kappa, const float &_normalization)
{
    const vfloat<VecSize> twoPi(2.0f*M_PI);
    const vfloat<VecSize> ones(1.0f);
    const vfloat<VecSize> minusTwos(-2.0f);
    const vfloat<VecSize> zeros(0.0f);
    const vfloat<VecSize> zeroKappaNorm(1.0f/(4.0f*M_PI));

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const vfloat<VecSize> weight = _weight;
    const vfloat<VecSize> kappa = _kappa;
    const vfloat<VecSize> normalization = _normalization;

    const Vec3<vfloat<VecSize> > meanDirection = _meanDirection;

    vfloat<VecSize> productIntegralVec(0.f);

    for(int k = 0; k < cnt;k++)
    {
        Vec3<vfloat<VecSize> > newMeanDirection = _kappas[k] * _meanDirections[k] + kappa * meanDirection;
        vfloat<VecSize> newKappa = embree::sqrt(embree::dot( newMeanDirection, newMeanDirection));
        auto checkNewKappa = (newKappa > 1e-3f);
        newKappa = select( checkNewKappa,  newKappa, zeros);

        // TODO: update meanCosine
        newMeanDirection.x = select( checkNewKappa,  newMeanDirection.x / newKappa,  _meanDirections[k].x);
        newMeanDirection.y = select( checkNewKappa,  newMeanDirection.y / newKappa,  _meanDirections[k].y);
        newMeanDirection.z = select( checkNewKappa,  newMeanDirection.z / newKappa,  _meanDirections[k].z);

        vfloat<VecSize> newEMinus2Kappa = embree::fastapprox::exp(minusTwos * newKappa);
        vfloat<VecSize> newNormalization = newKappa / (twoPi * ( ones - newEMinus2Kappa ));
        newNormalization = select( checkNewKappa, newNormalization, zeroKappaNorm);

        vfloat<VecSize> scale = ( _normalizations[k] * normalization ) / newNormalization;

        vfloat<VecSize> cosTheta0 = embree::dot( _meanDirections[k], newMeanDirection );
        vfloat<VecSize> cosTheta1 = embree::dot( meanDirection, newMeanDirection );

        //std::cout << "cosTheta0: " << cosTheta0 <<"\tcosTheta1: " << cosTheta1 << std::endl;
        //std::cout << "_kappas[k]: " << _kappas[k] <<"\tkappa: " << kappa << std::endl;
        //std::cout << "tmp: " <<  _kappas[k] * (cosTheta0 - ones) + kappa * (cosTheta1 - ones) << std::endl;
        vfloat<VecSize> eval = embree::fastapprox::exp( _kappas[k] * (cosTheta0 - ones) + kappa * (cosTheta1 - ones) );
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

    float productIntegral = reduce_add(productIntegralVec);
    for(int k = 0; k < cnt;k++)
    {
        _weights[k] /= productIntegral;
    }
    return productIntegral;
    //_normalizeWeights();
}


template<int VecSize, int maxComponents>
VonMisesFisherMixture<VecSize, maxComponents>::VonMisesFisherMixture( const VonMisesFisherMixture &a)
{
    for (size_t k = 0; k < NumVectors::value; k++)
    {
        _weights[k]= a._weights[k];
        _kappas[k]=  a._kappas[k];
        _eMinus2Kappa[k]=  a._eMinus2Kappa[k];
        _meanCosines[k] =  a._meanCosines[k];
        _normalizations[k] =  a._normalizations[k];

        _meanDirections[k] =  a._meanDirections[k];

    }
}

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::clearComponent( const size_t &idx )
{
    const div_t tmpIdx = div( idx, VecSize);

    _weights[tmpIdx.quot][tmpIdx.rem]= 0.f;
    _kappas[tmpIdx.quot][tmpIdx.rem] = 0.f;
    _eMinus2Kappa[tmpIdx.quot][tmpIdx.rem] = 0.f;
    _meanCosines[tmpIdx.quot][tmpIdx.rem] = 0.f;
    _normalizations[tmpIdx.quot][tmpIdx.rem] = 0.f;

    _meanDirections[tmpIdx.quot].x[tmpIdx.rem] = 0.f;
    _meanDirections[tmpIdx.quot].y[tmpIdx.rem] = 0.f;
    _meanDirections[tmpIdx.quot].z[tmpIdx.rem] = 1.f;
}

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::swapComponents( const size_t &idx0, const size_t &idx1 )
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
    }
}

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::mergeComponents( const size_t &idx0, const size_t &idx1 )
{
    const div_t tmpIdx0 = div( idx0, VecSize);
    const div_t tmpIdx1 = div( idx1, VecSize);

    if (idx0 != idx1)
    {
        const float weight0 = _weights[tmpIdx0.quot][tmpIdx0.rem];
        const float weight1 = _weights[tmpIdx1.quot][tmpIdx1.rem];

        const float meanCosine0 = _meanCosines[tmpIdx0.quot][tmpIdx0.rem];
        const float meanCosine1 = _meanCosines[tmpIdx1.quot][tmpIdx1.rem];


        float kappa = 0.0f;
        float norm = 1.0f/(4.0f*M_PI);
        float eMin2Kappa = 1.0f;

        float weight = weight0 + weight1;

        float meanDirectionX = weight0 * meanCosine0 * _meanDirections[tmpIdx0.quot].x[tmpIdx0.rem]
            + weight1 * meanCosine1 * _meanDirections[tmpIdx1.quot].x[tmpIdx1.rem];
        float meanDirectionY = weight0 * meanCosine0 * _meanDirections[tmpIdx0.quot].y[tmpIdx0.rem]
            + weight1 * meanCosine1 * _meanDirections[tmpIdx1.quot].y[tmpIdx1.rem];
        float meanDirectionZ = weight0 * meanCosine0 * _meanDirections[tmpIdx0.quot].z[tmpIdx0.rem]
            + weight1 * meanCosine1 * _meanDirections[tmpIdx1.quot].z[tmpIdx1.rem];


        std::cout << "mergeComponents: cosTheta: " << _meanDirections[tmpIdx0.quot].x[tmpIdx0.rem] *_meanDirections[tmpIdx1.quot].x[tmpIdx1.rem] +
                                                        _meanDirections[tmpIdx0.quot].y[tmpIdx0.rem] *_meanDirections[tmpIdx1.quot].y[tmpIdx1.rem] +
                                                        _meanDirections[tmpIdx0.quot].z[tmpIdx0.rem] *_meanDirections[tmpIdx1.quot].z[tmpIdx1.rem] << std::endl;

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
        _meanCosines[tmpIdx0.quot][tmpIdx0.rem] = meanCosine;
        _kappas[tmpIdx0.quot][tmpIdx0.rem] = kappa;

        _normalizations[tmpIdx0.quot][tmpIdx0.rem] = norm;
        _eMinus2Kappa[tmpIdx0.quot][tmpIdx0.rem] = eMin2Kappa;

        _meanDirections[tmpIdx0.quot].x[tmpIdx0.rem] = meanDirectionX;
        _meanDirections[tmpIdx0.quot].y[tmpIdx0.rem] = meanDirectionY;
        _meanDirections[tmpIdx0.quot].z[tmpIdx0.rem] = meanDirectionZ;

        std::cout << "mergeComponents: weight: " << weight << "\tkappa: " << kappa << "\tmeanDirection: " << meanDirectionX << "\t" << meanDirectionY << "\t" << meanDirectionZ << std::endl;
        swapComponents( idx1, _numComponents -1 );
        clearComponent( _numComponents -1 );
        _numComponents -= 1;
    }
}


template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::uniformInit(float kappa){
    const int cnt = (_numComponents+VecSize-1) / VecSize;

    for(int k = 0; k < cnt;k++){
        _kappas[k] = vfloat<VecSize>(kappa);
        _weights[k] = vfloat<VecSize>(1.0f/float(maxComponents));
        _meanDirections[k] = Vec3< vfloat<VecSize> >(0.0, 0.0, 1.0);
    }
    _calculateNormalization();
    _calculateMeanCosines();
}

template<int VecSize, int maxComponents>
float VonMisesFisherMixture<VecSize, maxComponents>::pdf( Vec3<float> direction ) const{
    const int cnt = (_numComponents+VecSize-1) / VecSize;

    vfloat<VecSize> pdf = {0.0f};
    Vec3< vfloat<VecSize> > vec3Direction(direction[0], direction[1], direction[2]);

    const vfloat<VecSize> ones(1.0f);
    const vfloat<VecSize> zeros(0.0f);

    for(int k = 0; k < cnt;k++)
    {
        const vfloat<VecSize> cosTheta = embree::dot(vec3Direction, _meanDirections[k]);
        const vfloat<VecSize> cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
        const vfloat<VecSize> eval = _normalizations[k] * embree::fastapprox::exp< vfloat<VecSize> >( _kappas[k] * cosThetaMinusOne );
        pdf += _weights[k] * eval;
    }

    return reduce_add(pdf);
}

template<int VecSize, int maxComponents>
bool VonMisesFisherMixture<VecSize, maxComponents>::softAssignment( Vec3<float> direction, typename VonMisesFisherMixture<VecSize, maxComponents>::SoftAssignment &softAssign ) const{

    const int cnt = (_numComponents+VecSize-1) / VecSize;

    vfloat<VecSize> pdf = {0.0f};
    Vec3< vfloat<VecSize> > vec3Direction(direction[0], direction[1], direction[2]);

    const vfloat<VecSize> ones(1.0f);
    const vfloat<VecSize> zeros(0.0f);

    for(int k = 0; k < cnt;k++)
    {
        const vfloat<VecSize> cosTheta = embree::dot(vec3Direction, _meanDirections[k]);
        const vfloat<VecSize> cosThetaMinusOne = embree::min(cosTheta - ones, zeros);
        const vfloat<VecSize> eval = _normalizations[k] * embree::fastapprox::exp< vfloat<VecSize> >( _kappas[k] * cosThetaMinusOne );
        softAssign.assignments[k] =  _weights[k] * eval;
        pdf += softAssign.assignments[k];
    }

    softAssign.pdf = reduce_add(pdf);
    softAssign.size = _numComponents;

    if ( softAssign.pdf <= 0.0f)
    {
        return false;
    }

    vfloat<VecSize> inv_pdf = rcp(softAssign.pdf);
    for(int k = 0; k < cnt;k++)
    {
        softAssign.assignments[k] *= inv_pdf;
    }

    return true;
}


template<int VecSize, int maxComponents>
Vec3<float> VonMisesFisherMixture<VecSize, maxComponents>::sample( const Vec2<float> sample ) const{

    uint32_t selectedVector {0};
    uint32_t selectedComponent {0};
    // First, identify component we want to sample

    Vector2 _sample = sample;
    float searched = _sample[1];
    float sumWeights = 0.0f;
    float cdf = 0.0;
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

     Vec3<float> sampledDirection(0.f, 0.f, 1.f);

    // Second, sample selected component
    const float sKappa = _kappas[selectedVector][selectedComponent];
    const float sEMinus2Kappa = _eMinus2Kappa[selectedVector][selectedComponent];
    const Vec3<float> meanDirection( _meanDirections[selectedVector].x[selectedComponent],  _meanDirections[selectedVector].y[selectedComponent],  _meanDirections[selectedVector].z[selectedComponent]);

    if (sKappa == 0.0f)
    {
        sampledDirection = squareToUniformSphere( _sample );
    }
    else
    {
        float cosTheta = 1.f + (std::log1p((sEMinus2Kappa-1.f) * _sample[0])) / sKappa;

        // safeguard for numerical imprecisions (if sample[0] is 0.999999999)
        cosTheta = std::min(1.0f, std::max(cosTheta, -1.f));

        const float sinTheta = std::sqrt(1.f-cosTheta*cosTheta);

        const float phi = 2.f * M_PI * _sample[1];

        float sinPhi, cosPhi;
        sincosf(phi, &sinPhi, &cosPhi);
        sampledDirection = rkguide::sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
    }

    return embree::frame( meanDirection ) * sampledDirection;
}

template<int VecSize, int maxComponents>
std::string VonMisesFisherMixture<VecSize, maxComponents>::toString() const{
    std::stringstream ss;
    ss << "VonMisesFisherMixture:" << std::endl;
    ss << "maxComponents: " << maxComponents << std::endl;
    ss << "VecSize: " << VecSize << std::endl;
    ss << "numVectors: " << NumVectors::value << std::endl;
    ss << "---------------------- "  << std::endl;
    ss << "numComponents: " << _numComponents << std::endl;
    float sumWeights = 0.0f;
    for ( int k = 0; k < _numComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        ss << "vmm[" << k << "]: " << "weight: " << _weights[tmp.quot][tmp.rem];
        ss << "\t kappa: " <<  _kappas[tmp.quot][tmp.rem];
        ss << "\t meanDirection: [" <<  _meanDirections[tmp.quot].x[tmp.rem] << "\t" <<  _meanDirections[tmp.quot].y[tmp.rem] << "\t" <<  _meanDirections[tmp.quot].z[tmp.rem] << "]";
        ss << "\t normalization: " <<  _normalizations[tmp.quot][tmp.rem];
        ss << "\t eMinus2Kappa: " <<  _eMinus2Kappa[tmp.quot][tmp.rem];
        ss << "\t meanCosine: " <<  _meanCosines[tmp.quot][tmp.rem];
        ss << std::endl;
        sumWeights += _weights[tmp.quot][tmp.rem];
    }
    ss << "sumWeights: " << sumWeights << std::endl;
    return ss.str();
}

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::_calculateNormalization( ) {
    const vfloat<VecSize> zeroKappaNorm(1.0f/(4.0f*M_PI));

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const vfloat<VecSize> minusTwo(-2.0f);
    for(int k = 0; k < cnt;k++){
        _eMinus2Kappa[k] = embree::fastapprox::exp< vfloat<VecSize> >(minusTwo*_kappas[k]);
        const vfloat<VecSize> norm = _kappas[k]/(2.0f*M_PI*(1.0f-_eMinus2Kappa[k]));
        _normalizations[k] = select(_kappas[k] > 0.f, norm, zeroKappaNorm);
    }

}

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::_calculateMeanCosines( ) {

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const vfloat<VecSize> zeros(0.0f);
    const vfloat<VecSize> ones(1.0f);
    for(int k = 0; k < cnt;k++){
        vfloat<VecSize>  tanh = ones - 2.0f / ( embree::fastapprox::exp( 2.0f * _kappas[k] ) - ones );
        vfloat<VecSize>  meanCosine = ones /tanh - ones / _kappas[k];
        //std::cout << "meanCosine: " << meanCosine << std::endl;
        _meanCosines[k] = select(_kappas[k] > 0.f, meanCosine, zeros);
    }
}


template<typename Type>
inline Type KappaToMeanCosine(const Type &kappa)
{
    const Type ones( 1.0f);
    const Type zeros( 0.0f);

    Type meanCosine = ones / embree::tanh( kappa) - ones / kappa;
    return select( kappa > 0.f, meanCosine, zeros);
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