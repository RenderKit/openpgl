// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "../data/SampleStatistics.h"
namespace openpgl
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

        /*
        void getDistribution(TDistribution &pDistribution, Point3 samplePosition, const bool &useParallaxComp) const
        {
            pDistribution = distribution;
            if(useParallaxComp)
            {
                Point3 pivotPosition = pDistribution._pivotPosition;
                pDistribution.performRelativeParallaxShift(pivotPosition - samplePosition);
            }
            //return pDistribution;
        }
        */

        void serialize(std::ostream& stream) const
        {
            distribution.serialize(stream);
            stream.write(reinterpret_cast<const char*>(&regionBounds), sizeof(BBox));
            trainingStatistics.serialize(stream);
            sampleStatistics.serialize(stream);
            stream.write(reinterpret_cast<const char*>(&splitFlag), sizeof(bool));
            stream.write(reinterpret_cast<const char*>(&valid), sizeof(bool));
        }

        void deserialize(std::istream& stream)
        {
            distribution.deserialize(stream);
            stream.read(reinterpret_cast<char*>(&regionBounds), sizeof(BBox));
            trainingStatistics.deserialize(stream);
            sampleStatistics.deserialize(stream);
            stream.read(reinterpret_cast<char*>(&splitFlag), sizeof(bool));
            stream.read(reinterpret_cast<char*>(&valid), sizeof(bool));
        }

        bool isValid() const
        {
            bool valid = true;
            valid &= distribution.isValid();
            valid &= trainingStatistics.isValid();
//            valid &= sampleStatistics.isValid();
            return valid;
        }

        void splatSample(SampleData &sample/*, const BBox &sceneBounds*/, const Point2 &sample2D) const
        {
            const Vector3 boundsExtents = (regionBounds.upper - regionBounds.lower) * 0.5f;
            const Vector3 sampleDisplacement = boundsExtents * squareToUniformSphere(sample2D);
            const Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
            Vector3 sampleDirection(sample.direction.x, sample.direction.y, sample.direction.z);
            
            Point3 splattedPosition = samplePosition + sampleDisplacement;
            if (!embree::inside(regionBounds, splattedPosition))
            {
                Point3 sourcePosition = samplePosition + sampleDirection * sample.distance;
// code if we want to ensure that the sample is not spatted outside the scene bounds
/*
                for (int i = 0; i < 3; i++)
                {
                    if (splattedPosition[i] < sceneBounds.lower[i])
                    {
                        splattedPosition[i] = sceneBounds.lower[i];
                    }
                    if (splattedPosition[i] > sceneBounds.upper[i])
                    {
                        splattedPosition[i] = sceneBounds.upper[i];
                    }
                }
*/
                sample.position.x = splattedPosition[0];
                sample.position.y = splattedPosition[1];
                sample.position.z = splattedPosition[2];

                sampleDirection = sourcePosition - splattedPosition;
                sample.distance = embree::length(sampleDirection);
                sampleDirection = sampleDirection / sample.distance;

                sample.direction.x = sampleDirection[0];
                sample.direction.y = sampleDirection[1];
                sample.direction.z = sampleDirection[2];

                sample.flags |= SampleData::ESplatted;
            }
        }

        std::string toString() const
        {
            std::stringstream ss;
            ss.precision(5);
            ss << "Region:" << std::endl;
            ss << "\t regionBounds: " << regionBounds << std::endl;
            ss << "\t distribution: " << distribution.toString() << std::endl;
            ss << "\t trainingStatistics: " << trainingStatistics.toString() << std::endl;
            ss << "\t sampleStatistics: " << sampleStatistics.toString() << std::endl;
            ss << "\t splitFlag: " << splitFlag << std::endl;
            ss << "\t valid: " << valid << std::endl;
            return ss.str();
        }

    };
}