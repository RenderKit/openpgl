// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

namespace openpgl
{
    struct Range
    {
        size_t m_begin;
        size_t m_end;

        Range() = default;
        Range(size_t begin, size_t end) : m_begin(begin), m_end(end) {}

        inline size_t size() const
        {
            return m_end - m_begin;
        }
    };
}