
// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __cplusplus
#include <cstdint>
#include <cstdlib>
#else
#include <stdint.h>
#include <stdlib.h>
#endif

#include "common.h"

/**
 * @brief 
 * 
 */
struct PGLSampleData
{
    enum Flags
    {
        /// point does not represent any real scene intersection point
        ESplatted = 1<<0, 
        /// point does not represent any real scene intersection point
        EInsideVolume = 1<<1 
    };

    /// the position of the sample (i.e., at which energy arrives)
    pgl_point3f position;

    /// a vector pointing into the direction the energy (e.g., radiance or importance)
    /// comes from 
    pgl_vec3f direction; 

    /// a scalar representation of the incident radiance divide by @ref pdf
    float weight;

    /// the PDF used for sampling @ref direction at @ref position   
    float pdf;

    /// the distance to the source of the incomming radiance  
    float distance; 

    ///
    uint32_t flags;
};

/**
 * @brief 
 * 
 */
struct PGLPathSegmentData
{
    /// The starting position of the path segment (on a surface or inside a volume)
    pgl_point3f position {0.f, 0.f, 0.f};
    /// The direction to the starting position of the next path segment
    pgl_vec3f directionIn {0.f, 1.f, 0.f};
    /// The direction to the starting position of the previous path segment
    pgl_vec3f directionOut {1.f, 0.f, 0.f};
    /// The surface normal at the current @ref position. Inside volumes @ref normal can be an arbritrary valid direction.
    pgl_vec3f normal {0.f, 0.f, 1.f};

    //float distance {0.0f};

    /// If the corrent segment starts inside a volume
    bool volumeScatter{false};

    /// The PDF of sampling @ref directionIn at @ref position towards the next segment
    float pdfDirectionIn {1.0f};

    /// If the scattering interaction at @ref position whas a delta Dirac (e.g., perfect mirror or glass).
    bool isDelta {false};

    /// The BSDF*cos or phase function evaluation at @ref position for the out-going direction @ref directionOut and the in-comming direction @ref directionIn divided by the sampling PDF @ref pdfDirectionIn.
    pgl_vec3f scatteringWeight{1.0f, 1.0f, 1.0f};

    /// The transmittance from the current @ref position to the @ref position of the next segment divided by the corresponding distance sampling PDF.
    pgl_vec3f transmittanceWeight{1.0f, 1.0f, 1.0f};

    /// the direct contribution (i.e., emission) at @ref position into @ref directionOut
    pgl_vec3f directContribution{0.0f, 0.0f, 0.0f};
    
    /// The MIS weight which would be applied to @ref directContribution (e.g., miWeight = bsdfPDF^2/(bsdfPDF^2+neePDF^2)).
    float miWeight {1.0f};

    /// The contribution which is scattered at @ref position into @ref directionOut. There can be multiple source us such contribution (sub-surface-scattering (dipole) or scattered NEE).
    pgl_vec3f scatteredContribution{0.0f, 0.0f, 0.0f};

    /// The probability to survive Russian roulette
    float russianRouletteProbability{1.0f};

    // BSDF information
    /// The refractive index (eta) of the material at @ref position.
    float eta {1.0f};

    /// The roughness of the material (e.g., GGX roughness on surfaces and mean cosine in volumes). 
    float roughness {1.0f};

    /// The pointer to the Region
    const void* regionPtr{nullptr};
};