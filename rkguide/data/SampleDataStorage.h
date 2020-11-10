// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

#include "DirectionalSampleData.h"
#include "../sampler/Sampler.h"

#include <tbb/concurrent_vector.h>

namespace rkguide
{

struct SampleDataStorage
{
    typedef tbb::concurrent_vector<DirectionalSampleData> SampleDataContainer;
    SampleDataContainer m_container;
    bool m_splatSamples {false};
    BBox m_sceneBounds;

    void setSceneBounds(const BBox &sceneBounds)
    {
        m_sceneBounds = sceneBounds;
    }

    template<typename TRegion>
    inline void addSample(DirectionalSampleData sample, const TRegion *region, Sampler *sampler)
    {
        if (m_splatSamples && region)
        {
            splatSample(sample, region->getSampleBounds(), sampler->next2D());
        }
        m_container.push_back(sample);
    }

    template<typename TRegion>
    inline void addSamples(const std::vector<std::pair<DirectionalSampleData, const void*> > &samples,  Sampler *sampler)
    {
        for (auto& sample : samples)
        {
            addSample(sample.first, (const TRegion*)sample.second, sampler);
        }
    }

/*
    template<typename TRegion, class ... Args>
    inline void emplace_back(const TRegion *region, Sampler *sampler, Args&&... args)
    {
        if (m_splatSamples && region)
        {

        }
        m_container.emplace_back(std::forward<Args>(args)...);
    }
*/
    inline void reserve(const size_t &size)
    {
        m_container.reserve(size);
    }

    inline size_t size() const
    {
        return m_container.size();
    }

    inline void clear()
    {
        m_container.clear();
    }

    void sort()
    {
        std::sort(m_container.begin(), m_container.end());
    }

    private:

    void splatSample(DirectionalSampleData &sample, const BBox &splattingBounds, const Point2 &sample2D) const
    {
        const Vector3 boundsExtents = (splattingBounds.upper - splattingBounds.lower) * 0.5f;
        //float maxBoundsExtent = (boundsExtents.x > boundsExtents.y) ?   ((boundsExtents.x > boundsExtents.z) ? boundsExtents.x : boundsExtents.z):
        //                                                                ((boundsExtents.y > boundsExtents.z) ? boundsExtents.y : boundsExtents.z);
        const Vector3 sampleDisplacement = boundsExtents * squareToUniformSphere(sample2D);

        Point3 splattedPosition = sample.position + sampleDisplacement;
        if (!embree::inside(splattingBounds, splattedPosition))
        {
            Point3 sourcePosition = sample.position + sample.direction * sample.distance;
            for (int i = 0; i < 3; i++)
            {
                if (splattedPosition[i] < m_sceneBounds.lower[i])
                {
                    splattedPosition[i] = m_sceneBounds.lower[i];
                }
                if (splattedPosition[i] > m_sceneBounds.upper[i])
                {
                    splattedPosition[i] = m_sceneBounds.upper[i];
                }
            }

            sample.position = splattedPosition;
            sample.direction = sourcePosition - splattedPosition;
            sample.distance = embree::length(sample.direction);
            sample.direction /= sample.distance;

            sample.flags |= DirectionalSampleData::ESplatted;
        }
    }
};

}