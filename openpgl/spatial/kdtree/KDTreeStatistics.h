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
        std::stringstream ss;
        ss << "KDTreeStatistics:" << " , ";
        ss << "numberOfNodes" << " , ";
        ss << "numberOfReservedNodes" << " , ";
        ss << "maxDepth" << " , ";
        ss << "sizePerNode" << " , ";
        ss << "sizeAllNodesUsed" << " , ";
        ss << "sizeAllNodesReserved" << " , ";
        return ss.str();
    }

    std::string toCSVString() const {
        std::stringstream ss;
        ss << " " << " , ";
        ss << numberOfNodes << " , ";
        ss << numberOfReservedNodes << " , ";
        ss << maxDepth << " , ";
        ss << sizePerNode << " , ";
        ss << sizeAllNodesUsed << " , ";
        ss << sizeAllNodesReserved << " , ";
        return ss.str();
    }

    std::string toString() const {
        std::stringstream ss;
        ss << "KDTreeStatistics: " << std::endl;
        ss << "\t" << "numberOfNodes            = " << numberOfNodes << std::endl;
        ss << "\t" << "numberOfReservedNodes    = " << numberOfReservedNodes << std::endl;
        ss << "\t" << "maxDepth                 = " << maxDepth << std::endl;
        ss << "\t" << "sizePerNode              = " << sizePerNode << " bs"<<std::endl;
        ss << "\t" << "sizeAllNodesUsed         = " << float(sizeAllNodesUsed) / 1024 << " kbs"<< std::endl;
        ss << "\t" << "sizeAllNodesReserved     = " << float(sizeAllNodesReserved) / 1024 << " kbs"<< std::endl;
        return ss.str();
    }
};
}
