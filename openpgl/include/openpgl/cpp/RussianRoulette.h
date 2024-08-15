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
/**
 * @brief Calculating the Russian Roulette survival probability using the common throughput-based RR heuristic.
 *
 * This function implements the throughput-based RR heuristic presented by Arvo and Kirk [AK90]. This heuristic assumes that a path will
 * only have a low contribution to the final pixel estimate when it undergoes dark (i.e., low albedo) scattering events. Therefore, the survival
 * probability will be low. On the other hand, if the prefix path undergoes several bright (i.e., high albedo) scattering events, it is assumed
 * that the potential future contribution is also high. The @ref throughput used for RR at a given path vertex is the LTE evaluated for the prefix
 * path till the given vertex times the local BSDF eval divided by the sampling PDF for the continuation direction.
 * It is worth noting that if BSDF importance sampling is used, the LTE evaluation divided by the PDF becomes a low-variance
 * estimator for the local directional albedo leading to a similar RR strategy as presented by Jensen [Jen01].
 *
 * @param throughput The Monte-Carlo throughput weight of the prefix-path (i.e., LTE evaluation divided by PDF and previous survival probabilities).
 *
 * @param minSurvivalProbability The minimal returned survival probability (default = 0.0).
 *
 * @param maxSurvivalProbability The maximum returned survival probability (default = 1.0).
 */
OPENPGL_INLINE float StandardThroughputBasedRussianRoulette(const pgl_vec3f &throughput, const float minSurvivalProbability = 0.0f, const float maxSurvivalProbability = 1.f)
{
    return std::max(minSurvivalProbability, std::min(maxSurvivalProbability, Max(throughput)));
}

/**
 * @brief Calculating the Russian Roulette survival probability using the Guided/Adjoint-driven RR strategy.
 *
 * This function implements the Guided/Adjoint-driven RR strategy derived from the zero-variance sampling theory,
 * which Vorba et al. [Vorba2016] and Herholz et al. [Vorba2019] use to guide toward the optimal RR decision for surface-based
 * and volumetric light transport simulations. While the standard throughput-based RR heuristic only relies on the throughput weight of the
 * prefix path to estimate the potential contribution and its importance to the estimate of the pixel value, this guided strategy also takes
 * estimates of the expected pixel value ( @ref contributionEstimate ) and the expected contribution of the suffix path ( @ref adjoint ) into account.
 *
 * @param throughput The Monte-Carlo throughput weight of the prefix-path (i.e., LTE evaluation divided by PDF and previous survival probabilities).
 *
 * @param adjoint An estimate of the incoming, outgoing, or in-scattered radiance collected when continuing the suffix path.
 *
 * @param contributionEstimate An estimate of the final pixel value (i.e., the expected value of the random work).
 *
 * @param minSurvivalProbability The minimal returned survival probability (default = 0.0).
 *
 * @param maxSurvivalProbability The maximum returned survival probability (default = 1.0).
 */
OPENPGL_INLINE float GuidedRussianRoulette(const pgl_vec3f &throughput, const pgl_vec3f &adjoint, const pgl_vec3f &contributionEstimate, const float minSurvivalProbability = 0.0f,
                                           const float maxSurvivalProbability = 1.f)
{
    float survivalProb = 1.0f;
    // Guided/Adjoint-driven Russian Roulette using the weight window as described by
    // Vorba and Krivanek in "Adjoint-Driven Russian Roulette and Splitting in Light Transport Simulation"
    if (IsValid(adjoint) && !IsZero(adjoint))
    {
        const float s = 5.0f;
        // weight window center
        pgl_vec3f Cww = contributionEstimate / adjoint;
        Cww.x = adjoint.x > 0.f ? Cww.x : 0.f;
        Cww.y = adjoint.y > 0.f ? Cww.y : 0.f;
        Cww.z = adjoint.z > 0.f ? Cww.z : 0.f;
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
}  // namespace util
}  // namespace cpp
}  // namespace openpgl