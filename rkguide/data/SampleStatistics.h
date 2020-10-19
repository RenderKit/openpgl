// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

namespace rkguide
{
    struct SampleStatistics
    {
        Point3 mean{0.0f};
        Vector3 variance {0.0f};
        size_t numSamples {0};

        BBox sampleBound;

        inline void clear()
        {
            mean = Point3(0.0f);
            variance = Vector3(0.0f);
            numSamples = 0.0f;
            //sampleBound
        }

        inline void addSample( const Point3 sample)
        {
            numSamples++;
            float incWeight = rcp(float(numSamples));
            RKGUIDE_ASSERT(numSamples > 0.0f);
            RKGUIDE_ASSERT(isvalid(incWeight));
            RKGUIDE_ASSERT(incWeight >=0.0f);
            //mean += (sample - oldMean) * incWeight;
            const Point3 oldMean = mean * incWeight;
            mean += sample;
            const Point3 newMean = mean * incWeight;
            variance += (sample - oldMean) * (sample - newMean);
            sampleBound.extend( sample );
        }

        inline Point3 getMean() const
        {
            return mean / float(numSamples);
        }

        inline Vector3 getVaraince() const
        {
            RKGUIDE_ASSERT( numSamples > 0);
            return variance / float(numSamples);
        }

        inline void decay( const float &a)
        {
            numSamples *= a;
            variance *= a;
        }

        inline float getNumSamples() const
        {
            return numSamples;
        }

        void merge( const SampleStatistics &a)
        {
            mean += a.mean;
            variance += a.variance;
            numSamples += a.numSamples;
        }

        inline bool isValid() const
        {
            bool valid = true;
            valid &= numSamples >=0.0f;

            valid &= isvalid(mean.x);
            valid &= isvalid(mean.y);
            valid &= isvalid(mean.z);

            valid &= isvalid(variance.x);
            valid &= isvalid(variance.y);
            valid &= isvalid(variance.z);

            return valid;
        }

    };

}