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

#ifdef OPENPGL_RADIANCE_CACHES
        size_t m_is_begin {0};
        size_t m_is_end {0};
#endif

        Range() = default;
        Range(size_t begin, size_t end) : m_begin(begin), m_end(end) {}

        inline size_t size() const
        {
            OPENPGL_ASSERT(int(m_end) - int(m_begin) >= 0);
            return m_end - m_begin;
        }
#ifdef OPENPGL_RADIANCE_CACHES
        inline size_t sizeZeroValueSamples() const
        {
            OPENPGL_ASSERT(int(m_is_end) - int(m_is_begin) >= 0);
            return m_is_end - m_is_begin;
        }
#endif
        inline void reset()
        {
            m_begin = 0;
            m_end = 0;
#ifdef OPENPGL_RADIANCE_CACHES
            m_is_begin = 0;
            m_is_end = 0;
#endif
        }

        void serialize(std::ostream &os) const {
            os.write(reinterpret_cast<const char*>(&m_begin), sizeof(m_begin));
            os.write(reinterpret_cast<const char*>(&m_end), sizeof(m_end));
#ifdef OPENPGL_RADIANCE_CACHES
            os.write(reinterpret_cast<const char*>(&m_is_begin), sizeof(m_is_begin));
            os.write(reinterpret_cast<const char*>(&m_is_end), sizeof(m_is_end));
#endif
        }

        void deserialize(std::istream& is)
        {
            is.read(reinterpret_cast<char*>(&m_begin), sizeof(m_begin));
            is.read(reinterpret_cast<char*>(&m_end), sizeof(m_end));
#ifdef OPENPGL_RADIANCE_CACHES
            is.read(reinterpret_cast<char*>(&m_is_begin), sizeof(m_is_begin));
            is.read(reinterpret_cast<char*>(&m_is_end), sizeof(m_is_end));
#endif
        }

        bool operator==(const Range& b) const {
            bool equal = true;
            if(m_begin != b.m_begin || m_end != b.m_end
#ifdef OPENPGL_RADIANCE_CACHES
                || m_is_begin != b.m_is_begin || m_is_end != b.m_is_end
#endif
            )
            {
                equal = false;
            }
            return equal;
        }

        bool isValid() const {
            bool valid = true;
            valid = valid && m_end >= m_begin;
            return valid;
        }
    };
}