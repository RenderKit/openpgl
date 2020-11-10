// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "../data/SampleStatistics.h"
namespace rkguide
{
    template <typename TDistribution , typename TTrainingStatistics>
    struct Region
    {
        TDistribution distribution;
        BBox regionBounds;
        TTrainingStatistics trainingStatistics;
        SampleStatistics sampleStatistics;
        bool splitFlag {false};
        bool valid{true};

        inline const BBox &getRegionBounds() const
        {
            return regionBounds;
        }

        inline const BBox &getSampleBounds() const
        {
            return sampleStatistics.sampleBounds;
        }


        TDistribution getDistribution(Point3 samplePosition, const bool &useParallaxComp) const
        {
            TDistribution pDistribution = distribution;
            if(useParallaxComp)
            {
                Point3 pivotPosition = pDistribution._pivotPosition;
                pDistribution.performRelativeParallaxShift(pivotPosition - samplePosition);
            }
            return pDistribution;
        }

    };
}