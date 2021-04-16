// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openpgl_common.h"

#include "data/DirectionalSampleData.h"
#include "data/PathSegmentDataStorage.h"
#include "data/PathSegmentData.h"

#include "data/SampleDataStorage.h"
#include "field/SurfaceVolumeFieldParallaxAwareVMM.h"
#include "vmm/ParallaxAwareVMM.h"
#include "vmm/AdaptiveSplitandMergeFactory.h"
#include "vmm/VMMSurfaceSamplingDistribution.h"
#include "vmm/VMMVolumeSamplingDistribution.h"
#include "sampler/Sampler.h"

namespace openpgl
{
    typedef SurfaceVolumeFieldParallaxAwareVMM<4, 32, SampleDataStorage::SampleDataContainer> GuidingField;
    typedef GuidingField::RegionType GuidingRegion;
    typedef GuidingField::DistributionType GuidingDistribution;
    typedef VMMBSDFSamplingDistribution<GuidingDistribution> GuidedSurfaceSamplingDistribution;
    typedef VMMPhaseFunctionSamplingDistribution<GuidingDistribution> GuidedVolumeSamplingDistribution;
    typedef GuidingField::Settings GuidingFieldProperties;
    typedef SamplerC GuidingSampler;
}