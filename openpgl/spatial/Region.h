// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "../data/SampleStatistics.h"
#include "IRegion.h"

namespace openpgl
{
    template <typename TDistribution , typename TTrainingStatistics>
    struct Region: public IRegion
    {
        TDistribution distribution;
        BBox regionBounds;
        TTrainingStatistics trainingStatistics;
        SampleStatistics sampleStatistics;
        bool splitFlag {false};
        //bool valid{true};

        inline const BBox &getRegionBounds() const
        {
            return regionBounds;
        }

        inline const BBox &getSampleBounds() const
        {
            return sampleStatistics.sampleBounds;
        }

/*
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
*/

        const TDistribution* getDistribution(Point3 samplePosition) const
        {
           return &distribution;
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
            stream.write(reinterpret_cast<const char*>(&valid), sizeof(valid));
            distribution.serialize(stream);
            stream.write(reinterpret_cast<const char*>(&regionBounds), sizeof(regionBounds));
            trainingStatistics.serialize(stream);
            sampleStatistics.serialize(stream);
            stream.write(reinterpret_cast<const char*>(&splitFlag), sizeof(splitFlag));
        }

        void deserialize(std::istream& stream)
        {
            stream.read(reinterpret_cast<char*>(&valid), sizeof(valid));
            distribution.deserialize(stream);
            stream.read(reinterpret_cast<char*>(&regionBounds), sizeof(regionBounds));
            trainingStatistics.deserialize(stream);
            sampleStatistics.deserialize(stream);
            stream.read(reinterpret_cast<char*>(&splitFlag), sizeof(splitFlag));
        }

        bool isValid() const
        {
            bool valid = true;
            valid = valid && distribution.isValid();
            valid = valid && trainingStatistics.isValid();
//            valid = valid && sampleStatistics.isValid();
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

        bool operator==(const Region& b) const {
            bool equal = true;
            if (!sampleStatistics.operator==(b.sampleStatistics) || splitFlag != b.splitFlag)
            {
                equal = false;
            }

            if (!distribution.operator==(b.distribution))
            {
                equal = false;
            }

            if (!trainingStatistics.operator==(b.trainingStatistics))
            {
                equal = false;
            }

            if (!(regionBounds == b.regionBounds))
            {
                equal = false;
            }

            return equal;
        }

    };
}