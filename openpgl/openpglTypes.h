// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openpgl_common.h"

#include "data/SampleData.h"
#include "data/PathSegmentDataStorage.h"
#include "data/PathSegmentData.h"

#include "data/SampleDataStorage.h"
//#include "field/SurfaceVolumeFieldParallaxAwareVMM.h"
#include "field/SurfaceVolumeField2.h"
#include "directional/ISurfaceSamplingDistribution.h"
#include "directional/IVolumeSamplingDistribution.h"
#include "directional/vmm/ParallaxAwareVMM.h"
#include "directional/vmm/AdaptiveSplitandMergeFactory.h"
#include "directional/vmm/VMMSurfaceSamplingDistribution.h"
#include "directional/vmm/VMMVolumeSamplingDistribution.h"
#include "sampler/Sampler.h"

#include "spatial/kdtree/KDTreeBuilder.h"

namespace openpgl
{
    //typedef  SpatialStructureBuilder;
    typedef AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<4, 32>> DirectionalDistriubtionFactory;

    //typedef SurfaceVolumeField2<DirectionalDistriubtionFactory, KDTreePartitionBuilder, SampleDataStorage::SampleDataContainer> GuidingField;
    typedef SurfaceVolumeField2<DirectionalDistriubtionFactory, KDTreePartitionBuilder> GuidingField;
    
    typedef GuidingField::RegionType GuidingRegion;
    typedef GuidingField::DirectionalDistribution GuidingDistribution;
    typedef VMMBSDFSamplingDistribution<GuidingDistribution> GuidedSurfaceSamplingDistribution;
    typedef VMMPhaseFunctionSamplingDistribution<GuidingDistribution> GuidedVolumeSamplingDistribution;
    typedef GuidingField::Settings GuidingFieldProperties;
    typedef SamplerC GuidingSampler;
}