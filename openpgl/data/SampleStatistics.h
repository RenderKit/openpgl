// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

#define INTERGER_V2

namespace openpgl
{
    struct SampleStatistics
    {
        Point3 mean{0.0f};
        Vector3 sampleVariance {0.0f};
        float numSamples {0};
        float numInvalidSamples {0.0f};

        BBox sampleBounds{openpgl::Vector3(std::numeric_limits<float>::max()), openpgl::Vector3(-std::numeric_limits<float>::max())};

        inline void clear()
        {
            mean = Point3(0.0f);
            sampleVariance = Vector3(0.0f);
            numSamples = 0.0f;
            numInvalidSamples = 0.0f;
            sampleBounds.lower = openpgl::Vector3(std::numeric_limits<float>::max());
            sampleBounds.upper = openpgl::Vector3(-std::numeric_limits<float>::max());
        }

        inline void addSample( const Point3 sample)
        {
            numSamples++;
            float incWeight = embree::rcp(float(numSamples));
            OPENPGL_ASSERT(numSamples > 0.0f);
            OPENPGL_ASSERT(embree::isvalid(incWeight));
            OPENPGL_ASSERT(incWeight >=0.0f);

            const Point3 oldMean = mean;
            mean += (sample - oldMean) * incWeight;

            //mean += sample;
            //const Point3 newMean = mean * incWeight;
            sampleVariance += ((sample - oldMean) * (sample - mean));
            sampleBounds.extend( sample );
        }

        inline Point3 getMean() const
        {
            return mean; //  / float(numSamples);
        }

        inline Vector3 getVariance() const
        {
            OPENPGL_ASSERT( numSamples > 0.f);
            return sampleVariance / float(numSamples);
        }

        inline void decay( const float &a)
        {
            numSamples *= a;
            sampleVariance *= a;
            numInvalidSamples *= a;
        }

        inline float getNumSamples() const
        {
            return numSamples;
        }

        inline void addNumInvalidSamples( const int numInvalidSamples)
        {
            this->numInvalidSamples += numInvalidSamples;
        }

        inline float getNumInvalidSamples() const
        {
            return numInvalidSamples;
        }

        void split(const uint8_t &splitDim, const float &splitPos, const float &decay, const bool &splitLower)
        {
            OPENPGL_ASSERT(decay >0.0f && decay <= 1.0f) ;

            if(numSamples > 0.f)
            {
                const float variance = sampleVariance[splitDim] / numSamples;
                const float stdDerivation = std::sqrt(variance);

                float const newVariance = variance - variance / 4.0f;
                sampleVariance[splitDim] = newVariance * numSamples;
                if(splitLower)
                {
                    sampleBounds.lower[splitDim] = std::max(splitPos, sampleBounds.lower[splitDim]);
                    mean[splitDim] = std::min(sampleBounds.upper[splitDim], mean[splitDim] + stdDerivation / 2.0f);
                    // TODO: there are rare ocasions where this can happen (boarder of the head scene)
                    // find a way to handle these
                    //OPENPGL_ASSERT(mean[splitDim] >= sampleBounds.lower[splitDim]);
                    //mean[splitDim] += stdDerivation / 2.0f;
                }
                else
                {
                    sampleBounds.upper[splitDim] = std::min(splitPos, sampleBounds.upper[splitDim]);
                    mean[splitDim] = std::max(sampleBounds.lower[splitDim], mean[splitDim] - stdDerivation / 2.0f);
                    //OPENPGL_ASSERT(mean[splitDim] <= sampleBounds.upper[splitDim]);
                    //mean[splitDim] -= stdDerivation / 2.0f;
                }

                numSamples *= decay;
                numInvalidSamples *= decay;
                sampleVariance *= decay;
            }
        }

        void merge( const SampleStatistics &b)
        {
            const Point3 meanA = mean;
            const Point3 meanB = b.mean;

            const Vector3 sampleVarianceA = sampleVariance;
            const Vector3 sampleVarianceB = b.sampleVariance;

            const float numSamplesA = numSamples;
            const float numSamplesB = b.numSamples;

            mean = meanA*(float)numSamplesA + meanB*(float)numSamplesB;
            numSamples += numSamplesB;
            mean /= float(numSamples);

            sampleVariance = (sampleVarianceA + numSamplesA*meanA*meanA + sampleVarianceB + numSamplesB*meanB*meanB) - numSamples * mean*mean;
            sampleBounds.extend(b.sampleBounds);

            numInvalidSamples += b.numInvalidSamples;
        }

        inline bool isValid() const
        {
            bool valid = true;
            valid = valid && numSamples >=0.0f;
            valid = valid && numInvalidSamples >=0.0f;

            valid = valid && embree::isvalid(mean.x);
            valid = valid && embree::isvalid(mean.y);
            valid = valid && embree::isvalid(mean.z);

            valid = valid && embree::isvalid(sampleVariance.x);
            valid = valid && embree::isvalid(sampleVariance.y);
            valid = valid && embree::isvalid(sampleVariance.z);

            return valid;
        }

        SampleStatistics operator()(const SampleStatistics &a, const SampleStatistics &b) const{
            SampleStatistics merged = a;
            merged.merge(b);
            return merged;
        }

        std::string toString() const
        {
            Vector3 variance = getVariance();
            std::stringstream ss;
            ss.precision(5);
            ss << "SampleStatistics:" << std::endl;
            ss << "numSamples: " << numSamples << std::endl;
            ss << "numInvalidSamples: " << numInvalidSamples << std::endl;
            ss << "mean: " << mean[0] << ",\t"<< mean[1] << ",\t"<< mean[2]  << std::endl;
            ss << "variance: " << variance[0] << ",\t"<< variance[1] << ",\t"<< variance[2]  << std::endl;
            ss << "sampleBounds: [" << sampleBounds.lower[0] << ",\t"<< sampleBounds.lower[1] << ",\t"<< sampleBounds.lower[2] << "] \t [" << sampleBounds.upper[0] << ",\t"<< sampleBounds.upper[1] << ",\t"<< sampleBounds.upper[2] << "] "<< std::endl;
            //ss << "maxComponents: " << maxComponents << std::endl;
            //ss << "maxComponents: " << maxComponents << std::endl;
            return ss.str();
        }

        void serialize(std::ostream& stream) const
        {
            stream.write(reinterpret_cast<const char*>(&mean), sizeof(Point3));
            stream.write(reinterpret_cast<const char*>(&sampleVariance), sizeof(Vector3));
            stream.write(reinterpret_cast<const char*>(&numSamples), sizeof(float));
            stream.write(reinterpret_cast<const char*>(&numInvalidSamples), sizeof(float));
            stream.write(reinterpret_cast<const char*>(&sampleBounds), sizeof(BBox));
        }

        void deserialize(std::istream& stream)
        {
            stream.read(reinterpret_cast<char*>(&mean), sizeof(Point3));
            stream.read(reinterpret_cast<char*>(&sampleVariance), sizeof(Vector3));
            stream.read(reinterpret_cast<char*>(&numSamples), sizeof(float));
            stream.read(reinterpret_cast<char*>(&numInvalidSamples), sizeof(float));
            stream.read(reinterpret_cast<char*>(&sampleBounds), sizeof(BBox));
        }

        bool operator==(const SampleStatistics& b) const { 
            bool equal = true;
            if(mean.x != b.mean.x || mean.y != b.mean.y ||
                mean.z != b.mean.z || sampleVariance.x != b.sampleVariance.x || 
                sampleVariance.y != b.sampleVariance.y || sampleVariance.z != b.sampleVariance.z ||
                numSamples != b.numSamples || sampleBounds.lower.x != b.sampleBounds.lower.x || 
                sampleBounds.lower.y != b.sampleBounds.lower.y || sampleBounds.lower.z != b.sampleBounds.lower.z ||
                sampleBounds.upper.x != b.sampleBounds.upper.x || sampleBounds.upper.y != b.sampleBounds.upper.y || 
                sampleBounds.upper.z != b.sampleBounds.upper.z)
                {
                    equal = false;
                    //std::cout << std::fixed;
                    //std::cout << std::setprecision(12);
                    //std::cout << "SampleStatistics: NOT-EQUAL" << std::endl;
                    //std::cout << "SampleStatisticsLeft:  numSamples = " << numSamples << "\t mean = " << mean.x << "\t" << mean.y << "\t" << mean.z << "\t sampleVariance = "<< sampleVariance.x << "\t" << sampleVariance.y << "\t" << sampleVariance.z << std::endl;
                    //std::cout << "SampleStatisticsRight: numSamples = " << b.numSamples << "\t mean = " << b.mean.x << "\t" << b.mean.y << "\t" << b.mean.z << "\t sampleVariance = "<< b.sampleVariance.x << "\t" << b.sampleVariance.y << "\t" << b.sampleVariance.z << std::endl;
                }
            return equal;
        }

    };

    #define  INTEGER_BINS 4096.f
    struct IntegerSampleStatistics
    {
        Point3i mean{0};
        Vector3i variance {0};
        uint32_t numSamples {0};
        BBoxi sampleBounds{openpgl::Vector3i(std::numeric_limits<int>::max()), openpgl::Vector3i(-std::numeric_limits<int>::max())};
        Vector3 sampleBoundsMin {0};
        Vector3 sampleBoundsMax {0};
        Vector3 sampleBoundsExtend {0};
        Vector3 invSampleBoundsExtend {0};
#ifdef INTERGER_V2
        Vector3 sampleBoundsCenter {0};
        Vector3 sampleBoundsHalfExtend {0};
        Vector3 invSampleBoundsHalfExtend {0};
#endif      
        IntegerSampleStatistics(){
            mean = Point3i(0);
            variance = Vector3i(0);
            numSamples = 0;
            sampleBounds = BBoxi(openpgl::Vector3i(std::numeric_limits<int>::max()), openpgl::Vector3i(-std::numeric_limits<int>::max()));
            sampleBoundsMin = Vector3(0);
            sampleBoundsMax = Vector3(0);
            sampleBoundsExtend = Vector3(0);
            invSampleBoundsExtend = Vector3(0);
#ifdef INTERGER_V2
            sampleBoundsCenter = Vector3(0);
            sampleBoundsHalfExtend = Vector3(0);
            invSampleBoundsHalfExtend = Vector3(0);
#endif
        }
        
        IntegerSampleStatistics(const BBox& bounds){
            mean = Point3i(0);
            variance = Vector3i(0);
            numSamples = 0;
            sampleBounds = BBoxi(openpgl::Vector3i(std::numeric_limits<int>::max()), openpgl::Vector3i(-std::numeric_limits<int>::max()));
            sampleBoundsMin = bounds.lower;
            sampleBoundsMax = bounds.upper;
            sampleBoundsExtend = bounds.upper - bounds.lower;
            invSampleBoundsExtend = embree::rcp(sampleBoundsExtend);
#ifdef INTERGER_V2
            sampleBoundsHalfExtend = sampleBoundsExtend * 0.5f;
            invSampleBoundsHalfExtend = embree::rcp(sampleBoundsHalfExtend);
            sampleBoundsCenter = sampleBoundsMin + sampleBoundsHalfExtend;
#endif
        }


        void clear() {
            mean = Point3i(0);
            variance = Vector3i(0);
            numSamples = 0;

            sampleBounds = BBoxi(openpgl::Vector3i(std::numeric_limits<int>::max()), openpgl::Vector3i(-std::numeric_limits<int>::max()));
            sampleBoundsMin = Vector3(0);
            sampleBoundsMax = Vector3(0);
            sampleBoundsExtend = Vector3(0);
            invSampleBoundsExtend = Vector3(0);
#ifdef INTERGER_V2
            sampleBoundsCenter = Vector3(0);
            sampleBoundsHalfExtend = Vector3(0);
            invSampleBoundsHalfExtend = Vector3(0);
#endif
        }

        inline void addSample( const Point3 sample)
        {
            numSamples++;
#ifdef INTERGER_V2
            Point3 tmpSample = ((sample - sampleBoundsCenter) * invSampleBoundsHalfExtend); //* INTEGER_BINS;
#else
            Point3 tmpSample = ((sample - sampleBoundsMin) * invSampleBoundsExtend); //* INTEGER_BINS;
#endif
            Vector3 tmpVariance = (tmpSample * tmpSample) * INTEGER_BINS;
            tmpSample *= INTEGER_BINS;

            Point3i iSample(tmpSample.x, tmpSample.y, tmpSample.z);
            mean += iSample;
            variance += Vector3i(tmpVariance.x, tmpVariance.y, tmpVariance.z);

            sampleBounds.extend(iSample);
        }

        void merge( const IntegerSampleStatistics &b)
        {
            mean += b.mean;
            variance += b.variance;
            numSamples += b.numSamples;
            sampleBounds.extend(b.sampleBounds);
        }

        static IntegerSampleStatistics merge(const IntegerSampleStatistics &a, const IntegerSampleStatistics &b)
        {
            IntegerSampleStatistics stats = a;
            stats.mean += b.mean;
            stats.variance += b.variance;
            stats.numSamples += b.numSamples;
            stats.sampleBounds.extend(b.sampleBounds);
            return stats;
        }

        SampleStatistics getSampleStatistics() const {
            
            SampleStatistics sampleStats;
            if (numSamples > 0) {
                float invNumSamples = 1.f / float(numSamples);
                Point3 sampleMean = (Point3(mean.x, mean.y, mean.z) / INTEGER_BINS) * invNumSamples;
                Vector3 sampleVariance = (Vector3(variance.x, variance.y, variance.z) / (INTEGER_BINS) ) * invNumSamples;
                sampleVariance -= sampleMean * sampleMean;
                sampleVariance = Vector3(std::fabs(sampleVariance.x), std::fabs(sampleVariance.y), std::fabs(sampleVariance.z));
#ifdef INTERGER_V2
                sampleMean = sampleMean * sampleBoundsHalfExtend;
                sampleMean += sampleBoundsCenter;
                sampleVariance = sampleVariance * (sampleBoundsHalfExtend * sampleBoundsHalfExtend);
#else
                sampleMean = sampleMean * sampleBoundsExtend;
                sampleMean += sampleBoundsMin;
                sampleVariance = sampleVariance * (sampleBoundsExtend * sampleBoundsExtend);
#endif

                sampleStats.mean = sampleMean;
                sampleStats.numSamples = numSamples;
                sampleStats.sampleVariance = sampleVariance * float(numSamples);
                Point3 sampleBoundLower = Point3(float(sampleBounds.lower.x) - 0.0f, float(sampleBounds.lower.y) - 0.0f, float(sampleBounds.lower.z) - 0.0f) / INTEGER_BINS;
#ifdef INTERGER_V2
                sampleBoundLower = sampleBoundLower * sampleBoundsHalfExtend;
                sampleBoundLower = sampleBoundLower + sampleBoundsCenter;
#else
                sampleBoundLower = sampleBoundLower * sampleBoundsExtend;
                sampleBoundLower = sampleBoundLower + sampleBoundsMin;
#endif
                
                Point3 sampleBoundUpper = Point3(float(sampleBounds.upper.x) + 0.0f, float(sampleBounds.upper.y) + 0.0f, float(sampleBounds.upper.z) + 0.0f) / INTEGER_BINS;
#ifdef INTERGER_V2
                sampleBoundUpper = sampleBoundUpper * sampleBoundsHalfExtend;
                sampleBoundUpper = sampleBoundUpper + sampleBoundsCenter;
#else
                sampleBoundUpper = sampleBoundUpper * sampleBoundsExtend;
                sampleBoundUpper = sampleBoundUpper + sampleBoundsMin;
#endif
                sampleStats.sampleBounds.lower = sampleBoundLower;
                sampleStats.sampleBounds.upper = sampleBoundUpper;
            }
            return sampleStats;
        }

    };

}