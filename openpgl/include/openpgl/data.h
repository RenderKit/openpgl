
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

struct PGLDirectionalSampleData
{
    enum Flags
    {
        ESplatted = 1<<0, // point does not represent any real scene intersection point
        EInsideVolume = 1<<1 // point does not represent any real scene intersection point
    };

    pgl_point3f position;
    pgl_vec3f direction;
    float weight;
    float pdf;
    float distance;
    uint32_t flags;
};

struct PGLPathSegmentData
{
    // world space information
    pgl_point3f position;
    pgl_vec3f directionIn;
    pgl_vec3f directionOut;
    pgl_vec3f normal;
    //float distance {0.0f};

    bool volumeScatter{false};

    float pdfDirectionIn {1.0f};
    bool isDelta {false};

    // BSDF or phase function evaluation
    // divided by the sampling PDF
    pgl_vec3f scatteringWeight{1.0f};

    pgl_vec3f transmittanceWeight{1.0f};

    // local space information
    //float cosThetaIncomming {1.0f};
    //float cosThetaOutgoing {1.0f};

    // Emission
    pgl_vec3f directContribution{0.0f};
    // MIS information
    float miWeight {1.0f};

    // information (sub-surface-scattering (dipole) or scattered NEE)
    pgl_vec3f scatteredContribution{0.0f};

    float russianRouletteProbability{1.0f};

    // BSDF information
    float eta {1.0f};
    float roughness {1.0f};

    const void* regionPtr{nullptr};
};