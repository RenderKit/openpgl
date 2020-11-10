// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

namespace rkguide
{
    template< typename TContainer>
    struct Range
    {
        typedef TContainer Container;
        typedef typename TContainer::value_type DataType;

        typename TContainer::iterator m_start;
        typename TContainer::iterator m_end;

        Range() = default;

        Range(TContainer &container)
        {
            m_start = container.begin();
            m_end = container.end();
        }

        inline size_t size() const
        {
            return std::distance(m_start, m_end);
        }

        typename TContainer::iterator begin() const
        {
            return m_start;
        }

        typename TContainer::iterator end() const
        {
            return m_end;
        }

        DataType& operator[](ptrdiff_t offset) {
            return *(m_start+offset);
        }

        const DataType& operator[](ptrdiff_t offset) const {
            return *(m_start+offset);
        }
    };
}