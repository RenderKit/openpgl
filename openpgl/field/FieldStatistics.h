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

    SpatialStatistics spatialStructureStatistics;
    DirectionalDistributionStatistics directionalDistributionStatistics;

    std::string headerCSVString() const{
        std::stringstream ss;
        ss << "FieldStatistics:" << " , ";
        ss << "numCacheRegions" << " , ";
        ss << "numCacheRegionsReserved" << " , ";
        ss << "sizePerCacheRegions" << " , ";
        ss << "sizeAllCacheRegionsUsed" << " , ";
        ss << "sizeAllCacheRegionsReserved" << " , ";

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
        ss << sizeAllCacheRegionsUsed << " , ";
        ss << sizeAllCacheRegionsReserved << " , ";

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

        ss << spatialStructureStatistics.toString();
        ss << directionalDistributionStatistics.toString();

        return ss.str();
    }
};

}
