// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

namespace rkguide
{
    template< typename TContainer>
    struct Range
    {
        typename TContainer::iterator start;
        typename TContainer::iterator end;

        inline size_t size() const
        {
            return std::distance(start, end);
        }
    };
}