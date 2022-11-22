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

        public:
        bool valid{true};
    };
}