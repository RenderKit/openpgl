// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "../include/openpgl/data.h"

namespace openpgl
{

inline float OPENPGL_SPECTRUM_TO_FLOAT(Vector3 spectrum)
{
    return (spectrum[0] + spectrum[1] + spectrum[2] ) / 3.0f;
}

using PathSegmentData = PGLPathSegmentData;

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