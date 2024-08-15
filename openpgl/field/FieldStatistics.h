// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sstream>
#include <string>

#include "../directional/DirectionalDistributionStatistics.h"
#include "../openpgl_common.h"
#include "../spatial/kdtree/KDTreeStatistics.h"

namespace openpgl
{

struct FieldStatistics
{
    using SpatialStatistics = KDTreeStatistics;

    size_t numCacheRegions{0};
    size_t numCacheRegionsReserved{0};
    size_t sizePerCacheRegions{0};
    size_t sizeAllCacheRegionsUsed{0};
    size_t sizeAllCacheRegionsReserved{0};

    float timeLastUpdate{0.f};
    float timeLastUpdateCopySamples{0.f};
    float timeLastUpdateSpatialStructureUpdate{0.f};
    float timeLastUpdateDirectionalDistriubtionUpdate{0.f};

    SpatialStatistics spatialStructureStatistics;
    DirectionalDistributionStatistics directionalDistributionStatistics;

    std::string headerCSVString() const
    {
        const std::string separator = " , ";
        std::stringstream ss;
        ss << "FieldStatistics:" << separator;
        ss << "numCacheRegions" << separator;
        ss << "numCacheRegionsReserved" << separator;
        ss << "sizePerCacheRegions(bs)" << separator;
        ss << "sizeAllCacheRegionsUsed(Mbs)" << separator;
        ss << "sizeAllCacheRegionsReserved(Mbs)" << separator;

        ss << "timeUpdate(ms)" << separator;
        ss << "timeCopySamples(ms)" << separator;
        ss << "timeSpatialStructureUpdate(ms)" << separator;
        ss << "timeDirectionalDistriubtionUpdate(ms)" << separator;

        ss << spatialStructureStatistics.headerCSVString();
        ss << directionalDistributionStatistics.headerCSVString();

        return ss.str();
    }

    std::string toCSVString() const
    {
        const std::string separator = " , ";
        std::stringstream ss;
        ss << " " << separator;
        ss << numCacheRegions << separator;
        ss << numCacheRegionsReserved << separator;
        ss << sizePerCacheRegions << separator;
        ss << float(sizeAllCacheRegionsUsed) / 1024 / 1024 << separator;
        ss << float(sizeAllCacheRegionsReserved) / 1024 / 1024 << separator;

        ss << timeLastUpdate << separator;
        ss << timeLastUpdateCopySamples << separator;
        ss << timeLastUpdateSpatialStructureUpdate << separator;
        ss << timeLastUpdateDirectionalDistriubtionUpdate << separator;

        ss << spatialStructureStatistics.toCSVString();
        ss << directionalDistributionStatistics.toCSVString();

        return ss.str();
    }

    std::string toString() const
    {
        const std::string tab = "\t";
        std::stringstream ss;
        ss << "FieldStatistics:" << std::endl;
        ss << tab << "numCacheRegions = " << numCacheRegions << std::endl;
        ss << tab << "numCacheRegionsReserved = " << numCacheRegionsReserved << std::endl;
        ss << tab << "sizePerCacheRegions = " << sizePerCacheRegions << " bs" << std::endl;
        ss << tab << "sizeAllCacheRegionsUsed = " << float(sizeAllCacheRegionsUsed) / 1024 / 1024 << " Mbs" << std::endl;
        ss << tab << "sizeAllCacheRegionsReserved = " << float(sizeAllCacheRegionsReserved) / 1024 / 1024 << " Mbs" << std::endl;

        ss << tab << "timeUpdate = " << timeLastUpdate << " ms" << std::endl;
        ss << tab << "timeCopySamples = " << timeLastUpdateCopySamples << " ms" << std::endl;
        ss << tab << "timeSpatialStructureUpdate = " << timeLastUpdateSpatialStructureUpdate << " ms" << std::endl;
        ss << tab << "timeDirectionalDistriubtionUpdate= " << timeLastUpdateDirectionalDistriubtionUpdate << " ms" << std::endl;

        ss << spatialStructureStatistics.toString();
        ss << directionalDistributionStatistics.toString();

        return ss.str();
    }
};

}  // namespace openpgl
