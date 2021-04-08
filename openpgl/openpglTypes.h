// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openpgl.h"

#include "data/DirectionalSampleData.h"
#include "data/PathSegmentDataStorage.h"
#include "data/PathSegmentData.h"

#include "data/SampleDataStorage.h"
#include "field/SurfaceVolumeFieldParallaxAwareVMM.h"
#include "vmm/ParallaxAwareVMM.h"
#include "vmm/AdaptiveSplitandMergeFactory.h"

namespace openpgl
{

typedef SurfaceVolumeFieldParallaxAwareVMM<4, 32, SampleDataStorage::SampleDataContainer> GuidingField;
typedef GuidingField::RegionType GuidingRegion;
typedef GuidingField::DistributionType GuidingDistribution;
typedef GuidingField::Settings GuidingFieldProperties;
}