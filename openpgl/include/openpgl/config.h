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


enum PGL_SPATIAL_STRUCTURE_TYPE
{
    PGL_SPATIAL_STRUCTURE_KDTREE = 0
};

enum PGL_DIRECTIONAL_DISTRIBUTION_TYPE
{
    PGL_DIRECTIONAL_DISTRIBUTION_VMM = 0,
    PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM,
    PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE
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
    // weighted EM arguments
    size_t initK {PGL_VECTOR_SIZE};

    size_t maxK {PGL_VMM_MAX_COMPONENTS};
    size_t maxEMIterrations {100};

    float maxKappa {PGL_VMM_MAX_KAPPA};
    //float maxMeanCosine { KappaToMeanCosine<float>(OPENPGL_MAX_KAPPA)};
    float convergenceThreshold {0.0025f};

    // MAP prior parameters
    // weight prior
    float weightPrior{0.1f};

    // concentration/meanCosine prior
    float meanCosinePriorStrength {0.1f};
    float meanCosinePrior {0.0f};

    // adaptive split and merge arguments
    bool useSplitAndMerge {true};

    float splittingThreshold { 0.75 };
    float mergingThreshold { 0.00625 };

    bool partialReFit { false };
    int maxSplitItr { 1 };

    int minSamplesForSplitting { 0 };
    int minSamplesForPartialRefitting { 0 };
    int minSamplesForMerging { 0 };
};


struct PGLFieldArguments
{
    PGL_SPATIAL_STRUCTURE_TYPE spatialStructureType;
    void *spatialSturctureArguments;
    PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalDistributionType;
    void *directionalDistributionArguments;
    // for debugging
    bool useParallaxCompensation;
};


#ifdef __cplusplus
}  // extern "C"
#endif