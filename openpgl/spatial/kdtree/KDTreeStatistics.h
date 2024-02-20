// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"

#include <string>
#include <sstream>

namespace openpgl
{

struct KDTreeStatistics{
    size_t maxDepth {0};
    size_t numberOfNodes {0};
    size_t numberOfReservedNodes{0};
    size_t sizePerNode {0};
    size_t sizeAllNodesReserved {0};
    size_t sizeAllNodesUsed{0};

    std::string headerCSVString() const {
        const std::string separator = " , ";
        std::stringstream ss;
        ss << "KDTreeStatistics:" << separator;
        ss << "numberOfNodes" << separator;
        ss << "numberOfReservedNodes" << separator;
        ss << "maxDepth" << separator;
        ss << "sizePerNode" << separator;
        ss << "sizeAllNodesUsed" << separator;
        ss << "sizeAllNodesReserved" << separator;
        return ss.str();
    }

    std::string toCSVString() const {
        const std::string separator = " , ";
        std::stringstream ss;
        ss << " " << separator;
        ss << numberOfNodes << separator;
        ss << numberOfReservedNodes << separator;
        ss << maxDepth << separator;
        ss << sizePerNode << separator;
        ss << sizeAllNodesUsed << separator;
        ss << sizeAllNodesReserved << separator;
        return ss.str();
    }

    std::string toString() const {
        const std::string tab = "\t";
        std::stringstream ss;
        ss << "KDTreeStatistics: " << std::endl;
        ss << tab << "numberOfNodes            = " << numberOfNodes << std::endl;
        ss << tab << "numberOfReservedNodes    = " << numberOfReservedNodes << std::endl;
        ss << tab << "maxDepth                 = " << maxDepth << std::endl;
        ss << tab << "sizePerNode              = " << sizePerNode << " bs"<<std::endl;
        ss << tab << "sizeAllNodesUsed         = " << float(sizeAllNodesUsed) / 1024 << " kbs"<< std::endl;
        ss << tab << "sizeAllNodesReserved     = " << float(sizeAllNodesReserved) / 1024 << " kbs"<< std::endl;
        return ss.str();
    }
};
}
