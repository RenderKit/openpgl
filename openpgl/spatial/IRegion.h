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
        virtual void splatSample(SampleData &sample, const Point2 &sample2D) const = 0 ;

        public:
        bool valid{true};
    };
}