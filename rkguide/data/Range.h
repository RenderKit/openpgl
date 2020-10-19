// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

namespace rkguide
{
    struct Range
    {
        size_t start{0};
        size_t end{0};

        inline size_t size() const
        {
            return (end - start) + 1;
        }
    };
}