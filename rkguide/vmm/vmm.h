// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

#include <embree/common/simd/simd.h>

#include <embree/common/math/vec2.h>
#include <embree/common/math/vec3.h>
#include <embree/common/math/linearspace3.h>
#include <embree/common/math/transcendental.h>

#include <algorithm>
#include <sstream>

using namespace embree;

namespace rkguide
{

template<int VecSize, int maxComponents>
struct VonMisesFisherMixture
{

public:
    typedef std::integral_constant<size_t, (maxComponents + (VecSize -1)) / VecSize> NumVectors;
private:
    //const static int numVectors {(maxComponents+VecSize-1)/VecSize};

public:

    float _numComponents{maxComponents};

    vfloat<VecSize> _weights[NumVectors::value];
    vfloat<VecSize> _kappas[NumVectors::value];
    Vec3<vfloat<VecSize> > _meanDirections[NumVectors::value];

    vfloat<VecSize> _normalization[NumVectors::value];
    vfloat<VecSize> _eMinus2Kappa[NumVectors::value];
    vfloat<VecSize> _meanCosines[NumVectors::value];

    void uniformInit( float kappa );
    float pdf( Vec3<float> direction ) const;

    Vec3<float> sample( const Vec2<float> sample ) const;

    std::string toString() const;


    void switchComponents(const size_t &idx0, const size_t &idx1);

    void mergeComponents(const size_t &idx0, const size_t &idx1);
//private:
    void _calculateNormalization();

    void _calculateMeanCosines();
};

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::switchComponents( const size_t &idx0, const size_t &idx1 )
{
    const div_t tmpIdx0 = div( idx0, VecSize);
    const div_t tmpIdx1 = div( idx1, VecSize);

    if (idx0 != idx1)
    {
        std::swap(_weights[tmpIdx0.quot][tmpIdx0.rem], _weights[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_kappas[tmpIdx0.quot][tmpIdx0.rem], _kappas[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_eMinus2Kappa[tmpIdx0.quot][tmpIdx0.rem], _eMinus2Kappa[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_meanCosines[tmpIdx0.quot][tmpIdx0.rem], _meanCosines[tmpIdx1.quot][tmpIdx1.rem]);
        std::swap(_normalization[tmpIdx0.quot][tmpIdx0.rem], _normalization[tmpIdx1.quot][tmpIdx1.rem]);

        std::swap(_meanDirections[tmpIdx0.quot].x[tmpIdx0.rem], _meanDirections[tmpIdx1.quot].x[tmpIdx1.rem]);
        std::swap(_meanDirections[tmpIdx0.quot].y[tmpIdx0.rem], _meanDirections[tmpIdx1.quot].y[tmpIdx1.rem]);
        std::swap(_meanDirections[tmpIdx0.quot].z[tmpIdx0.rem], _meanDirections[tmpIdx1.quot].z[tmpIdx1.rem]);

    }
}


/*
template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::mergeComponents( const size_t &idx0, const size_t &idx1 ) 
{
    const div_t tmpIdx0 = div( idx0, VecSize);
    const div_t tmpIdx1 = div( idx1, VecSize);

    if (idx0 != idx1)
    {
        const float weigth0 = _weights[tmpIdx0.quot][tmpIdx0.rem];
        const float weigth1 = _weights[tmpIdx1.quot][tmpIdx1.rem];

        const meanCosine0 = _meanCosines[tmpIdx0.quot][tmpIdx0.rem];
        const meanCosine1 = _meanCosines[tmpIdx1.quot][tmpIdx1.rem];


        if ( meanCosine > 0.0f )
        {
            meanCosine = std::sqrt(meanCosine);
        }


        switchComponents( idx1, _numComponents -1 );
        _numComponents -= 1;
    }
}

*/

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::uniformInit(float kappa){
    const int cnt = (_numComponents+VecSize-1) / VecSize;

    for(int k = 0; k < cnt;k++){
        _kappas[k] = vfloat<VecSize>(kappa);
        _weights[k] = vfloat<VecSize>(1.0f/float(maxComponents));
        _meanDirections[k] = Vec3< vfloat<VecSize> >(0.0, 0.0, 1.0);
    }
    _calculateNormalization();
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
        const vfloat<VecSize> eval = _normalization[k] * embree::fastapprox::exp< vfloat<VecSize> >( _kappas[k] * cosThetaMinusOne );
        pdf += _weights[k] * eval;
    }
    /*
    float t =0.0f;
    for (int i=0; i < VecSize; i++)
    {
        t += pdf[i];
    }
    */
    return reduce_add(pdf);
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
    for ( int k = 0; k < maxComponents; k++)
    {
        const div_t tmp = div(k, static_cast<int>(VecSize));
        ss << "vmm[" << k << "]: " << "weight: " << _weights[tmp.quot][tmp.rem];
        ss << "\t kappa: " <<  _kappas[tmp.quot][tmp.rem];
        ss << "\t meanDirection: [" <<  _meanDirections[tmp.quot].x[tmp.rem] << "\t" <<  _meanDirections[tmp.quot].y[tmp.rem] << "\t" <<  _meanDirections[tmp.quot].z[tmp.rem] << "]";
        ss << "\t normalization: " <<  _normalization[tmp.quot][tmp.rem];
        ss << "\t eMinus2Kappa: " <<  _eMinus2Kappa[tmp.quot][tmp.rem];
        ss << std::endl;
    }
    return ss.str();
}

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::_calculateNormalization( ) {
    const vfloat<VecSize> zeroKappaNorm(1.0f/(4.0f*M_PI));

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const vfloat<VecSize> minusTwo(-2.0f);
    for(int k = 0; k < cnt;k++){
        const vfloat<VecSize> norm = _kappas[k]/(2.0f*M_PI*(1.0f-_eMinus2Kappa[k]));
        _eMinus2Kappa[k] = embree::fastapprox::exp< vfloat<VecSize> >(minusTwo*_kappas[k]);
        _normalization[k] = select(_kappas[k] > 0.f, norm, zeroKappaNorm);
    }

}

template<int VecSize, int maxComponents>
void VonMisesFisherMixture<VecSize, maxComponents>::_calculateMeanCosines( ) {
    const vfloat<VecSize> zeroKappaNorm(1.0f/(4.0f*M_PI));

    const int cnt = (_numComponents+VecSize-1) / VecSize;
    const vfloat<VecSize> minusTwo(-2.0f);
    for(int k = 0; k < cnt;k++){
        const vfloat<VecSize> norm = _kappas[k]/(2.0f*M_PI*(1.0f-_eMinus2Kappa[k]));
        _eMinus2Kappa[k] = embree::fastapprox::exp< vfloat<VecSize> >(minusTwo*_kappas[k]);
        _normalization[k] = select(_kappas[k] > 0.f, norm, zeroKappaNorm);
    }
}

}