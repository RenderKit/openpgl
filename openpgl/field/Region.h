// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
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