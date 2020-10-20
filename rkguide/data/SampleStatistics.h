// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

namespace rkguide
{
    struct SampleStatistics
    {
        Point3 mean{0.0f};
        Vector3 sampleVariance {0.0f};
        size_t numSamples {0};

        BBox sampleBound{rkguide::Vector3(std::numeric_limits<float>::max()), rkguide::Vector3(-std::numeric_limits<float>::max())};

        inline void clear()
        {
            mean = Point3(0.0f);
            sampleVariance = Vector3(0.0f);
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
            sampleVariance += ((sample - oldMean) * (sample - mean));
            sampleBound.extend( sample );
        }

        inline Point3 getMean() const
        {
            return mean; //  / float(numSamples);
        }

        inline Vector3 getVaraince() const
        {
            RKGUIDE_ASSERT( numSamples > 0);
            return sampleVariance / float(numSamples);
        }

        inline void decay( const float &a)
        {
            numSamples *= a;
            sampleVariance *= a;
        }

        inline float getNumSamples() const
        {
            return numSamples;
        }

        void split(const uint8_t &splitDim, const float &splitPos, const float &decay, const bool &splitLower)
        {
            RKGUIDE_ASSERT(decay >0.0f && decay <= 1.0f) ;

            if(numSamples > 0)
            {
                const float variance = sampleVariance[splitDim] / numSamples;
                const float stdDerivation = std::sqrt(variance);

                float const newVariance = variance - variance / 4.0f;
                sampleVariance[splitDim] = newVariance * numSamples;
                if(splitLower)
                {
                    sampleBound.lower[splitDim] = std::max(splitPos, sampleBound.lower[splitDim]);
                    mean[splitDim] = std::min(sampleBound.upper[splitDim], mean[splitDim] + stdDerivation / 2.0f);
                    RKGUIDE_ASSERT(mean[splitDim] >= sampleBound.lower[splitDim]);
                    //mean[splitDim] += stdDerivation / 2.0f;
                }
                else
                {
                    sampleBound.upper[splitDim] = std::min(splitPos, sampleBound.upper[splitDim]);
                    mean[splitDim] = std::max(sampleBound.lower[splitDim], mean[splitDim] - stdDerivation / 2.0f);
                    RKGUIDE_ASSERT(mean[splitDim] <= sampleBound.upper[splitDim]);
                    //mean[splitDim] -= stdDerivation / 2.0f;
                }

                numSamples *= decay;
                sampleVariance *= decay;
            }
        }

        void merge( const SampleStatistics &b)
        {
            const Point3 meanA = mean;
            const Point3 meanB = b.mean;

            const Vector3 sampleVarianceA = sampleVariance;
            const Vector3 sampleVarianceB = b.sampleVariance;

            const size_t numSamplesA = numSamples;
            const size_t numSamplesB = b.numSamples;

            mean = meanA*(float)numSamplesA + meanB*(float)numSamplesB;
            numSamples += numSamplesB;
            mean /= float(numSamples);

            sampleVariance = (sampleVarianceA + numSamplesA*meanA*meanA + sampleVarianceB + numSamplesB*meanB*meanB) - numSamples * mean*mean;
            sampleBound.extend(b.sampleBound);
        }

        inline bool isValid() const
        {
            bool valid = true;
            valid &= numSamples >=0.0f;

            valid &= isvalid(mean.x);
            valid &= isvalid(mean.y);
            valid &= isvalid(mean.z);

            valid &= isvalid(sampleVariance.x);
            valid &= isvalid(sampleVariance.y);
            valid &= isvalid(sampleVariance.z);

            return valid;
        }

        constexpr SampleStatistics operator()(const SampleStatistics &a, const SampleStatistics &b) const{
            SampleStatistics merged = a;
            merged.merge(b);
            return merged;
        }

        std::string toString() const
        {
            Vector3 variance = getVaraince();
            std::stringstream ss;
            ss.precision(5);
            ss << "SampleStatistics:" << std::endl;
            ss << "numSamples: " << numSamples << std::endl;
            ss << "mean: " << mean[0] << ",\t"<< mean[1] << ",\t"<< mean[2]  << std::endl;
            ss << "variance: " << variance[0] << ",\t"<< variance[1] << ",\t"<< variance[2]  << std::endl;
            ss << "sampleBound: [" << sampleBound.lower[0] << ",\t"<< sampleBound.lower[1] << ",\t"<< sampleBound.lower[2] << "] \t [" << sampleBound.upper[0] << ",\t"<< sampleBound.upper[1] << ",\t"<< sampleBound.upper[2] << "] "<< std::endl;
            //ss << "maxComponents: " << maxComponents << std::endl;
            //ss << "maxComponents: " << maxComponents << std::endl;
            return ss.str();
        }

    };

}