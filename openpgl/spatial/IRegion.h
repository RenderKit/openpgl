// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "data/SampleData.h"

namespace openpgl
{
    
    struct IRegion
    {
        virtual ~IRegion(){};

        virtual Vector3 getOutgoingRadiance(const Vector3 dir) const = 0;

        public:
        bool valid{true};
    };
}