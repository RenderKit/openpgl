// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"
#include "VMM.h"


namespace openpgl
{

template<class TVMMDistribution>
struct VonMisesFisherFactory
{
    VonMisesFisherFactory();
    typedef TVMMDistribution VMM;

    void InitUniformVMM( VMM &vmm, const int &numComponents, const float &kappa) const;

private:
    void _initUniformDirections();

    embree::Vec3< embree::vfloat<VMM::VectorSize> >_uniformDirections[VMM::MaxComponents][VMM::NumVectors];

};


template<class TVMMDistribution>
VonMisesFisherFactory<TVMMDistribution>::VonMisesFisherFactory( )
{
    _initUniformDirections();
}

template<class TVMMDistribution>
void VonMisesFisherFactory<TVMMDistribution>::InitUniformVMM( VMM &vmm, const int &numComponents, const float &kappa) const
{
    vmm._numComponents = numComponents;
    const size_t nComp = vmm._numComponents;
    const float weight = 1.f / float(vmm._numComponents);

    size_t n = 0;
    for ( int i = 0; i < VMM::NumVectors ; i ++)
    {
        vmm._meanDirections[i] = _uniformDirections[nComp-1][i];
        for (int j = 0; j < VMM::VectorSize; j++){
            if ( n < nComp)
            {
                vmm._kappas[i][j] = kappa;
                vmm._weights[i][j] = weight;
            }
            else
            {
                vmm._kappas[i][j] = 0.0f;
                vmm._weights[i][j] = 0.0f;
                vmm._normalizations[i][j] = ONE_OVER_FOUR_PI;
                vmm._eMinus2Kappa[i][j] = 1.0f;
                vmm._meanCosines[i][j] = 0.0f;
            }
            n++;
        }
    }

    vmm._calculateNormalization();
    vmm._calculateMeanCosines();

}


template<class TVMMDistribution>
void VonMisesFisherFactory<TVMMDistribution>::_initUniformDirections( )
{
    const float gr = 1.618033988749895f;

    for(uint32_t l=0; l < VMM::MaxComponents; l++){

        /// distributes samples l+1 uniform samples over the sphere
        /// based on "Spherical Fibonacci Point Sets for Illumination Integrals"
        uint32_t n = 0;
        for(uint32_t k=0; k < VMM::NumVectors; k++){

            for(uint32_t i=0; i< VMM::VectorSize; i++){

                if(n<l+1){
                    float phi = 2.0f*M_PI*((float)n / gr);
                    float z = 1.0f - ((2.0f*n + 1.0f) / float(l+1));
                    float theta = std::acos(z);

                    Vector3 mu = sphericalDirection(theta, phi);
                    _uniformDirections[l][k].x[i] = mu[0];
                    _uniformDirections[l][k].y[i] = mu[1];
                    _uniformDirections[l][k].z[i] = mu[2];
                }else{
                    _uniformDirections[l][k].x[i] = 0.0f;
                    _uniformDirections[l][k].y[i] = 0.0f;
                    _uniformDirections[l][k].z[i] = 1.0f;
                }
                n++;
            }

        }
    }
}

}