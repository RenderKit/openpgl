// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

#include <fstream>
#include <string>
#include <iostream>

namespace openpgl
{

struct DirectionalDistributionStatistics{
    size_t sizePerDistribution {0};
    float minNumberOfComponents {0};
    float maxNumberOfComponents {0};
    float averageNumberOfComponents {0};
    float secondMomentNumberOfComponents {0};

    std::string headerCSVString() const {
        std::stringstream ss;
        ss << "DirectionalDistributionStatistics:" << " , ";
        ss << "sizePerDistribution" << " , ";
        ss << "minNumberOfComponents" << " , ";
        ss << "maxNumberOfComponents" << " , ";
        ss << "averageNumberOfComponents" << " , ";
        ss << "secondMomentNumberOfComponents" << " , ";
        return ss.str();
    }

    std::string toCSVString() const {
        std::stringstream ss;
        ss << " " << " , ";
        ss << sizePerDistribution << " , ";
        ss << minNumberOfComponents << " , ";
        ss << maxNumberOfComponents << " , ";
        ss << averageNumberOfComponents << " , ";
        ss << secondMomentNumberOfComponents << " , ";
        return ss.str();
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "DirectionalDistributionStatistics: " << std::endl;
        ss << "\t" << "sizePerDistribution              = " << sizePerDistribution << " bs"<<std::endl;
        ss << "\t" << "minNumberOfComponents            = " << minNumberOfComponents << std::endl;
        ss << "\t" << "maxNumberOfComponents            = " << maxNumberOfComponents << std::endl;
        ss << "\t" << "averageNumberOfComponents        = " << averageNumberOfComponents << std::endl;
        ss << "\t" << "secondMomentNumberOfComponents   = " << secondMomentNumberOfComponents << std::endl;
        return ss.str();
    }
};
}
