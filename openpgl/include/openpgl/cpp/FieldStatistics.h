// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "../openpgl.h"

namespace openpgl
{
namespace cpp
{

/**
 * @brief Class that stores some statistics about the spatial and directional structures of a guiding Field.
 *
 * This class can be used to print information about a guiding Field. It supports printing
 * the information as plain string or in CSV string format.
 */
struct FieldStatistics
{
    FieldStatistics(PGLFieldStatistics fieldStatisticsHandle);

    ~FieldStatistics();

    /**
     * @brief Returns all statistics of the guiding Field as plain string.
     *
     * @return std::string
     */
    std::string ToString() const;

    /**
     * @brief Returns a CSV string containing the headers of the columns for the CSV format.
     *
     * @return std::string
     */
    std::string HeaderCSVString() const;

    /**
     * @brief Returns all statistics as one line of CSV values.
     *
     * @return std::string
     */
    std::string ToCSVString() const;

   private:
    PGLFieldStatistics m_fieldStatisticsHandle{nullptr};
};

////////////////////////////////////////////////////////////
/// Implementation
////////////////////////////////////////////////////////////

OPENPGL_INLINE FieldStatistics::FieldStatistics(PGLFieldStatistics fieldStatisticsHandle)
{
    m_fieldStatisticsHandle = fieldStatisticsHandle;
}

OPENPGL_INLINE FieldStatistics::~FieldStatistics()
{
    OPENPGL_ASSERT(m_fieldStatisticsHandle);
    pglReleaseFieldStatistics(m_fieldStatisticsHandle);
    m_fieldStatisticsHandle = nullptr;
}

OPENPGL_INLINE std::string FieldStatistics::ToString() const
{
    OPENPGL_ASSERT(m_fieldStatisticsHandle);
    PGLString pglString = pglFieldStatisticsToString(m_fieldStatisticsHandle);
    std::string str = "";
    if (pglString.m_str)
        str = std::string(pglString.m_str);

    pglReleaseString(pglString);

    return str;
}

OPENPGL_INLINE std::string FieldStatistics::HeaderCSVString() const
{
    OPENPGL_ASSERT(m_fieldStatisticsHandle);
    PGLString pglString = pglFieldStatisticsHeaderCSVString(m_fieldStatisticsHandle);
    std::string str = "";
    if (pglString.m_str)
        str = std::string(pglString.m_str);

    pglReleaseString(pglString);

    return str;
}

OPENPGL_INLINE std::string FieldStatistics::ToCSVString() const
{
    OPENPGL_ASSERT(m_fieldStatisticsHandle);
    PGLString pglString = pglFieldStatisticsToCSVString(m_fieldStatisticsHandle);
    std::string str = "";
    if (pglString.m_str)
        str = std::string(pglString.m_str);

    pglReleaseString(pglString);

    return str;
}
}  // namespace cpp
}  // namespace openpgl
