// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "Common.h"

namespace openpgl
{
namespace cpp
{
namespace util
{
    OPENPGL_INLINE float StandardThroughputBasedRussianRoulette(const pgl_vec3f &throughput, const float minSurvivalProbability = 0.0f, const float maxSurvivalProbability = 1.f)
    {
        return std::max(minSurvivalProbability, std::min(maxSurvivalProbability, Max(throughput)));
    }

    OPENPGL_INLINE float GuidedRussianRoulette(const pgl_vec3f &throughput, const pgl_vec3f &adjoint, const pgl_vec3f &referenceEstimate, const float minSurvivalProbability = 0.0f, const float maxSurvivalProbability = 1.f)
    {
        float survivalProb = 1.0f;
        // Guided/Adjoint-driven Russian Roulette using the weight window as described by
        // Vorba and Krivanek in "Adjoint-Driven Russian Roulette and Splitting in Light Transport Simulation"
        if(IsValid(adjoint) && !IsZero(adjoint)){
            const float s = 5.0f;
            // weight window center
            pgl_vec3f Cww = referenceEstimate / adjoint;
            Cww.x = adjoint.x > 0.f ? Cww.x: 0.f;
            Cww.y = adjoint.y > 0.f ? Cww.y: 0.f;
            Cww.z = adjoint.z > 0.f ? Cww.z: 0.f;
            OPENPGL_ASSERT(IsValid(Cww));
            // weight window lower bound
            pgl_vec3f min = 2.0f * Cww / (1.0f + s);
            // weight window upper bound
            float fbeta = Average(throughput);
            float fmin = Average(min);
            fmin = fmin > 0 ? fmin : fbeta;
            survivalProb = fbeta / fmin;
        }
        OPENPGL_ASSERT(std::isfinite(survivalProb) && !std::isnan(survivalProb));
        survivalProb = std::max(minSurvivalProbability, survivalProb);
        return std::min(survivalProb, maxSurvivalProbability);
    }
}
}
}