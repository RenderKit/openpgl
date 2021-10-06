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


    std::string toString() const
    {
        std::stringstream ss;

        ss << "PathSegmentData: " 
            << "pos = " << position 
            << "\t dirIn = " << directionIn 
            << "\t dirOut = " << directionOut
            << "\t normal = " << normal
            << "\t volume = " << volumeScatter
            << "\t pdf = " << pdfDirectionIn 
            << "\t delta = " << isDelta 
            << "\t scatteringWeight = " << scatteringWeight
            << "\t transmittanceWeight = " << transmittanceWeight
            << "\t directContribution = " << directContribution
            << "\t miWeight = " << miWeight
            << "\t scatteredContribution = " << scatteredContribution
            << "\t russianRouletteProbability = " << russianRouletteProbability
            << "\t eta = " << eta 
            << "\t rough = " << roughness;
        ss << std::endl;
        
        return ss.str();   
    }
};


inline bool isValid(const PathSegmentData& psd)
    {
        bool valid = true;
        valid = valid && embree::isvalid(psd.position.x);
        valid = valid && embree::isvalid(psd.position.y);
        valid = valid && embree::isvalid(psd.position.z);
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.directionIn.x);
        valid = valid && psd.directionIn.x <= 1.0f && psd.directionIn.x >= -1.0f;
        valid = valid && embree::isvalid(psd.directionIn.y);
        valid = valid && psd.directionIn.y <= 1.0f && psd.directionIn.y >= -1.0f;
        valid = valid && embree::isvalid(psd.directionIn.z);
        valid = valid && psd.directionIn.z <= 1.0f && psd.directionIn.z >= -1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.directionOut.x);
        valid = valid && psd.directionOut.x <= 1.0f && psd.directionOut.x >= -1.0f;
        valid = valid && embree::isvalid(psd.directionOut.y);
        valid = valid && psd.directionOut.y <= 1.0f && psd.directionOut.y >= -1.0f;
        valid = valid && embree::isvalid(psd.directionOut.z);
        valid = valid && psd.directionOut.z <= 1.0f && psd.directionOut.z >= -1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.normal.x);
        valid = valid && psd.normal.x <= 1.0f && psd.normal.x >= -1.0f;
        valid = valid && embree::isvalid(psd.normal.y);
        valid = valid && psd.normal.y <= 1.0f && psd.normal.y >= -1.0f;
        valid = valid && embree::isvalid(psd.normal.z);
        valid = valid && psd.normal.z <= 1.0f && psd.normal.z >= -1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.pdfDirectionIn);
        valid = valid && psd.pdfDirectionIn >=0.f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.scatteringWeight.x);
        valid = valid && psd.scatteringWeight.x >= 0.0f;
        valid = valid && embree::isvalid(psd.scatteringWeight.y);
        valid = valid && psd.scatteringWeight.y >= 0.0f;
        valid = valid && embree::isvalid(psd.scatteringWeight.z);
        valid = valid && psd.scatteringWeight.z >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.transmittanceWeight.x);
        valid = valid && psd.transmittanceWeight.x >= 0.0f;
        valid = valid && embree::isvalid(psd.transmittanceWeight.y);
        valid = valid && psd.transmittanceWeight.y >= 0.0f;
        valid = valid && embree::isvalid(psd.transmittanceWeight.z);
        valid = valid && psd.transmittanceWeight.z >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.directContribution.x);
        valid = valid && psd.directContribution.x >= 0.0f;
        valid = valid && embree::isvalid(psd.directContribution.y);
        valid = valid && psd.directContribution.y >= 0.0f;
        valid = valid && embree::isvalid(psd.directContribution.z);
        valid = valid && psd.directContribution.z >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.scatteredContribution.x);
        valid = valid && psd.scatteredContribution.x >= 0.0f;
        valid = valid && embree::isvalid(psd.scatteredContribution.y);
        valid = valid && psd.scatteredContribution.y >= 0.0f;
        valid = valid && embree::isvalid(psd.scatteredContribution.z);
        valid = valid && psd.scatteredContribution.z >= 0.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.miWeight);
        valid = valid && psd.miWeight >=0.f && psd.miWeight <=1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.russianRouletteProbability);
        valid = valid && psd.miWeight >=0.f && psd.russianRouletteProbability <=1.0f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.eta);
        valid = valid && psd.eta >=0.f;
        OPENPGL_ASSERT(valid);

        valid = valid && embree::isvalid(psd.roughness);
        valid = valid && psd.roughness >=0.f && psd.roughness <=1.f;
        OPENPGL_ASSERT(valid);
        
        return valid;
    }

}