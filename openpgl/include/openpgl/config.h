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

#ifdef __cplusplus
extern "C" {
#endif

#include "types.h"

enum PGL_DEVICE_TYPE
{
    PGL_DEVICE_TYPE_CPU_4,
    PGL_DEVICE_TYPE_CPU_8,
    PGL_DEVICE_TYPE_CPU_16,
};

struct PGLKDTreeArguments
{
    bool knnLookup {true};
    size_t minSamples {100};
    size_t maxSamples {32000};
    size_t maxDepth{32};
};

struct PGLVMMFactoryArguments
{
    PGLVMMFactoryArguments(bool parallaxCompensation = true)
    {
        this->parallaxCompensation = parallaxCompensation;
    }
    
    // weighted EM arguments
    size_t initK {PGL_VMM_MAX_COMPONENTS/2};
    float initKappa {0.5f};

    size_t maxK {PGL_VMM_MAX_COMPONENTS};
    size_t maxEMIterrations {100};

    float maxKappa {PGL_VMM_MAX_KAPPA};
    //float maxMeanCosine { KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA)};
    float convergenceThreshold {0.005f};

    // MAP prior parameters
    // weight prior
    float weightPrior{0.01f};

    // concentration/meanCosine prior
    float meanCosinePriorStrength {0.2f};
    float meanCosinePrior {0.0f};

    // adaptive split and merge arguments
    bool useSplitAndMerge {true};

    float splittingThreshold { 0.5f };
    float mergingThreshold { 0.025f };

    bool partialReFit { true };
    int maxSplitItr { 1 };

    int minSamplesForSplitting { 32000/8 };
    int minSamplesForPartialRefitting { 32000/8 };
    int minSamplesForMerging { 32000/4 };

    bool parallaxCompensation {true};
};

enum PGLDQTLeafEstimator
{
    REJECTION_SAMPLING = 0,
    PER_LEAF
};

enum PGLDQTSplitMetric
{
    MEAN = 0,
    SECOND_MOMENT
};

struct PGLDQTFactoryArguments
{
    PGLDQTLeafEstimator leafEstimator { PGLDQTLeafEstimator::REJECTION_SAMPLING };
    PGLDQTSplitMetric splitMetric { PGLDQTSplitMetric::MEAN };
    float splitThreshold { 0.01f };
    float footprintFactor { 1 };
    uint32_t maxLevels { 12 };
};

struct PGLFieldArguments
{
    PGL_SPATIAL_STRUCTURE_TYPE spatialStructureType;
    void *spatialSturctureArguments;
    PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalDistributionType;
    void *directionalDistributionArguments;
    // for debugging
    bool useParallaxCompensation;
    bool deterministic {false};
};



OPENPGL_CORE_INTERFACE void pglFieldArgumentsSetDefaults(PGLFieldArguments &fieldArguments, const PGL_SPATIAL_STRUCTURE_TYPE spatialType, const PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalType);


#ifdef __cplusplus
}  // extern "C"
#endif