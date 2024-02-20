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
        const std::string separator = " , ";
        std::stringstream ss;
        ss << "DirectionalDistributionStatistics:" << separator;
        ss << "sizePerDistribution" << separator;
        ss << "minNumberOfComponents" << separator;
        ss << "maxNumberOfComponents" << separator;
        ss << "averageNumberOfComponents" << separator;
        ss << "secondMomentNumberOfComponents" << separator;
        return ss.str();
    }

    std::string toCSVString() const {
        const std::string separator = " , ";
        std::stringstream ss;
        ss << " " << separator;
        ss << sizePerDistribution << separator;
        ss << minNumberOfComponents << separator;
        ss << maxNumberOfComponents << separator;
        ss << averageNumberOfComponents << separator;
        ss << secondMomentNumberOfComponents << separator;
        return ss.str();
    }

    std::string toString() const {
        const std::string tab = "\t";
        std::stringstream ss;
        ss << "DirectionalDistributionStatistics: " << std::endl;
        ss << tab << "sizePerDistribution              = " << sizePerDistribution << " bs"<<std::endl;
        ss << tab << "minNumberOfComponents            = " << minNumberOfComponents << std::endl;
        ss << tab << "maxNumberOfComponents            = " << maxNumberOfComponents << std::endl;
        ss << tab << "averageNumberOfComponents        = " << averageNumberOfComponents << std::endl;
        ss << tab << "secondMomentNumberOfComponents   = " << secondMomentNumberOfComponents << std::endl;
        return ss.str();
    }
};
}
