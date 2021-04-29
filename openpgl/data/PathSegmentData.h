// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
//#include "../include/openpgl/data.h"

namespace openpgl
{

inline float OPENPGL_SPECTRUM_TO_FLOAT(Vector3 spectrum)
{
    return (spectrum[0] + spectrum[1] + spectrum[2] ) / 3.0f;
}

//using PathSegmentData = PGLPathSegmentData;

struct PathSegmentData
{
    /// The starting position of the path segment (on a surface or inside a volume)
    Point3 position;
    /// The direction to the starting position of the next path segment
    Vector3 directionIn;
    /// The direction to the starting position of the previous path segment
    Vector3 directionOut;
    /// The surface normal at the current @ref position. Inside volumes @ref normal can be an arbritrary valid direction.
    Vector3 normal;

    //float distance {0.0f};

    /// If the corrent segment starts inside a volume
    bool volumeScatter{false};

    // The PDF of sampling @ref directionIn at @ref position towards the next segment
    float pdfDirectionIn {1.0f};

    /// If the scattering interaction at @ref position whas a delta Dirac (e.g., perfect mirror or glass).
    bool isDelta {false};

    /// The BSDF*cos or phase function evaluation at @ref position for the out-going direction @ref directionOut and the in-comming direction @ref directionIn divided by the sampling PDF @ref pdfDirectionIn.
    Vector3 scatteringWeight{1.0f};

    /// The transmittance from the current @ref position to the @ref position of the next segment divided by the corresponding distance sampling PDF.
    Vector3 transmittanceWeight{1.0f};

    /// the direct contribution (i.e., emission) at @ref position into @ref directionOut
    Vector3 directContribution{0.0f};
    
    /// The MIS weight which would be applied to @ref directContribution when sampling this contribution using NEE at the previous path segment.
    float miWeight {1.0f};

    /// The contribution which is scattered at @ref position into @ref directionOut. There can be multiple source us such contribution (sub-surface-scattering (dipole) or scattered NEE).
    Vector3 scatteredContribution{0.0f};

    /// The probability to survive Russian roulette
    float russianRouletteProbability{1.0f};

    // BSDF information
    /// The refractive index (eta) of the material at @ref position.
    float eta {1.0f};

    /// The roughness of the material (e.g., GGX roughness on surfaces and mean cosine in volumes). 
    float roughness {1.0f};

    /// The pointer to the Region
    const void* regionPtr{nullptr};

    PathSegmentData() = default;

    PathSegmentData(const Point3 &_pos, const Vector3 &_normal, const Vector3 &_outDir):
        position(_pos), directionOut(_outDir), normal(_normal){}
};

}