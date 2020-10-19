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

        BBox sampleBound{rkguide::Vector3(std::numeric_limits<float>::max()), rkguide::Vector3(-std::numeric_limits<float>::max())};

        inline void clear()
        {
            mean = Point3(0.0f);
            variance = Vector3(0.0f);
            numSamples = 0.0f;
            sampleBound.lower = rkguide::Vector3(std::numeric_limits<float>::max());
            sampleBound.upper = rkguide::Vector3(-std::numeric_limits<float>::max());
            //sampleBound
        }

        inline void addSample( const Point3 sample)
        {
            numSamples++;
            float incWeight = rcp(float(numSamples));
            RKGUIDE_ASSERT(numSamples > 0.0f);
            RKGUIDE_ASSERT(isvalid(incWeight));
            RKGUIDE_ASSERT(incWeight >=0.0f);

            const Point3 oldMean = mean;
            mean += (sample - oldMean) * incWeight;

            //mean += sample;
            //const Point3 newMean = mean * incWeight;
            variance += ((sample - oldMean) * (sample - mean));
            sampleBound.extend( sample );
        }

        inline Point3 getMean() const
        {
            return mean; //  / float(numSamples);
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
            
            mean = mean*(float)numSamples + a.mean*(float)a.numSamples;
            variance = variance + a.variance;
            numSamples += a.numSamples;
            mean /= float(numSamples);
            
            //mean += a.mean;
            //variance += a.variance;
            //numSamples += a.numSamples;
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

        constexpr SampleStatistics operator()(const SampleStatistics &a, const SampleStatistics &b) const{
            SampleStatistics merged = a;
            merged.merge(b);
            return merged;
        }

        std::string toString() const
        {
            std::stringstream ss;
            ss.precision(5);
            ss << "SampleStatistics:" << std::endl;
            ss << "numSamples: " << numSamples << std::endl;
            ss << "mean: " << mean[0] << ",\t"<< mean[1] << ",\t"<< mean[2]  << std::endl;
            ss << "variance: " << variance[0] << ",\t"<< variance[1] << ",\t"<< variance[2]  << std::endl;
            
            //ss << "maxComponents: " << maxComponents << std::endl;
            //ss << "maxComponents: " << maxComponents << std::endl;
            return ss.str();
        }

    };

}