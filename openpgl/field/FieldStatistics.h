// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "../spatial/kdtree/KDTreeStatistics.h"
#include "../directional/DirectionalDistributionStatistics.h"

#include <string>
#include <sstream>

namespace openpgl
{

struct FieldStatistics{
    using SpatialStatistics = KDTreeStatistics;

    size_t numCacheRegions {0};
    size_t numCacheRegionsReserved {0};
    size_t sizePerCacheRegions {0};
    size_t sizeAllCacheRegionsUsed {0};
    size_t sizeAllCacheRegionsReserved {0};

    float timeLastUpdate {0.f};
    float timeLastUpdateCopySamples {0.f};
    float timeLastUpdateSpatialStructureUpdate {0.f};
    float timeLastUpdateDirectionalDistriubtionUpdate {0.f};

    SpatialStatistics spatialStructureStatistics;
    DirectionalDistributionStatistics directionalDistributionStatistics;

    std::string headerCSVString() const{
        std::stringstream ss;
        ss << "FieldStatistics:" << " , ";
        ss << "numCacheRegions" << " , ";
        ss << "numCacheRegionsReserved" << " , ";
        ss << "sizePerCacheRegions(bs)" << " , ";
        ss << "sizeAllCacheRegionsUsed(Mbs)" << " , ";
        ss << "sizeAllCacheRegionsReserved(Mbs)" << " , ";

        ss << "timeUpdate(ms)" << " , ";
        ss << "timeCopySamples(ms)" << " , ";
        ss << "timeSpatialStructureUpdate(ms)" << " , ";
        ss << "timeDirectionalDistriubtionUpdate(ms)" << " , ";

        ss << spatialStructureStatistics.headerCSVString();
        ss << directionalDistributionStatistics.headerCSVString();

        return ss.str();
    }

    std::string toCSVString() const{
        std::stringstream ss;
        ss << " " << " , ";
        ss << numCacheRegions << " , ";
        ss << numCacheRegionsReserved << " , ";
        ss << sizePerCacheRegions << " , ";
        ss << float(sizeAllCacheRegionsUsed) / 1024  / 1024 << " , ";
        ss << float(sizeAllCacheRegionsReserved) / 1024  / 1024 << " , ";

        ss << timeLastUpdate << " , ";
        ss << timeLastUpdateCopySamples << " , ";
        ss << timeLastUpdateSpatialStructureUpdate << " , ";
        ss << timeLastUpdateDirectionalDistriubtionUpdate << " , ";

        ss << spatialStructureStatistics.toCSVString();
        ss << directionalDistributionStatistics.toCSVString();

        return ss.str();
    }

    std::string toString() const{
        std::stringstream ss;
        ss << "FieldStatistics:" << std::endl;
        ss << "\t" << "numCacheRegions = " << numCacheRegions << std::endl;
        ss << "\t" << "numCacheRegionsReserved = " << numCacheRegionsReserved << std::endl;
        ss << "\t" << "sizePerCacheRegions = " << sizePerCacheRegions << " bs" << std::endl;
        ss << "\t" << "sizeAllCacheRegionsUsed = " << float(sizeAllCacheRegionsUsed) / 1024  / 1024 << " Mbs" << std::endl;
        ss << "\t" << "sizeAllCacheRegionsReserved = " << float(sizeAllCacheRegionsReserved) / 1024 / 1024 << " Mbs" << std::endl;
        
        ss << "\t" << "timeUpdate = " << timeLastUpdate << " ms" << std::endl;
        ss << "\t" << "timeCopySamples = " << timeLastUpdateCopySamples << " ms" << std::endl;
        ss << "\t" << "timeSpatialStructureUpdate = " << timeLastUpdateSpatialStructureUpdate << " ms" << std::endl;
        ss << "\t" << "timeDirectionalDistriubtionUpdate= " << timeLastUpdateDirectionalDistriubtionUpdate << " ms" << std::endl;

        ss << spatialStructureStatistics.toString();
        ss << directionalDistributionStatistics.toString();

        return ss.str();
    }
};

}
