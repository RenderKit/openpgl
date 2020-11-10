// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

namespace rkguide
{

inline float RKGUIDE_SPECTRUM_TO_FLOAT(Vector3 spectrum)
{
    return (spectrum[0] + spectrum[1] + spectrum[2] ) / 3.0f;
    //return spectrum.max();
}

struct PathSegmentData
{
    // world space information
    Point3 position;
    Vector3 directionIn;
    Vector3 directionOut;
    Vector3 normal;
    float distance {0.0f};

    float pdfDirectionIn {1.0f};
    bool isDelta {false};

    // BSDF or phase function evaluation
    // divided by the sampling PDF
    Vector3 scatteringWeight{1.0f};

    // local space information
    //float cosThetaIncomming {1.0f};
    //float cosThetaOutgoing {1.0f};

    // Emission
    Vector3 directContribution{0.0f};
    // MIS information
    float miWeight {1.0f};

    // information (sub-surface-scattering (dipole) or scattered NEE)
    Vector3 scatteredContribution{0.0f};

    float russianRouletteProbability{1.0f};

    // BSDF information
    float eta {1.0f};
    float roughness {1.0f};

    const void* regionPtr{nullptr};

};

}