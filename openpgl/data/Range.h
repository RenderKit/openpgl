// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

namespace openpgl
{
    struct Range
    {
        size_t m_begin {0};
        size_t m_end {0};

        Range() = default;
        Range(size_t begin, size_t end) : m_begin(begin), m_end(end) {}

        inline size_t size() const
        {
            if(int(m_end) - int(m_begin) < 0)
                std::cout << "WTF" << std::endl;
            return m_end - m_begin;
        }

        inline void reset()
        {
            m_begin = 0;
            m_end = 0;
        }

        void serialize(std::ostream &os) const {
            os.write(reinterpret_cast<const char*>(&m_begin), sizeof(m_begin));
            os.write(reinterpret_cast<const char*>(&m_end), sizeof(m_end));
        }

        void deserialize(std::istream& is)
        {
            is.read(reinterpret_cast<char*>(&m_begin), sizeof(m_begin));
            is.read(reinterpret_cast<char*>(&m_end), sizeof(m_end));
        }

        bool operator==(const Range& b) const {
            bool equal = true;
            if(m_begin != b.m_begin || m_end != b.m_end)
            {
                equal = false;
            }
            return equal;
        }
    };
}