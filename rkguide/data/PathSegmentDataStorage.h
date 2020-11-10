// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "PathSegmentData.h"
#include "DirectionalSampleData.h"

namespace rkguide
{

struct PathSegmentDataStorage
{
    std::vector<PathSegmentData> m_segmentStorage;

    void reserve(const size_t &size)
    {
        m_segmentStorage.reserve(size);
        m_sampleStorage.reserve(size);
        //m_regionPtrStorage.reserve(size);
    }

    size_t size()
    {
        return m_segmentStorage.size();
    }

    void clear()
    {
        m_segmentStorage.clear();
    }

    void push_back(const PathSegmentData &psData)
    {
        m_segmentStorage.push_back(psData);
    }


    uint32_t prepareSamples(const bool useNEEMiWeights = false)
    {
        m_sampleStorage.clear();
        //m_regionPtrStorage.clear();

        const float minPDF {0.1f};
        const rkguide::Vector3 maxThroughput {10.0f};

        size_t numSegments = m_segmentStorage.size();

        float lastDistance = 0.0f;
        for (int i=numSegments-2; i>=0; --i)
        {
            const rkguide::PathSegmentData &currentPathSegment = m_segmentStorage[i];

            float distance = currentPathSegment.distance + lastDistance;
            if(!currentPathSegment.isDelta && currentPathSegment.roughness >=0.3f)
            {
                lastDistance = 0.0f;
            }
            else
            {
                lastDistance = distance;
                if(currentPathSegment.eta!= 1.0f)
                {
                    float cosThetaI = embree::dot(currentPathSegment.normal, currentPathSegment.directionOut);
                    float cosThetaO = embree::dot(currentPathSegment.normal, currentPathSegment.directionIn);
                    lastDistance *= std::fabs(cosThetaI/(cosThetaO*currentPathSegment.eta));
                }
            }

            // we only collect samples on non delta surfaces and which are generated
            // using a non-direct (delta) sampling method
            if (!currentPathSegment.isDelta && currentPathSegment.roughness > 0.01f)
            {

                // prepare the current pos, direction, distance, pdf at the current
                // path vertex
                rkguide::Point3 pos = currentPathSegment.position;
                rkguide::Vector3 dir = currentPathSegment.directionIn;
                float pdf = std::max(minPDF,currentPathSegment.pdfDirectionIn);
                uint32_t flags{0};
                const void* regionPtr = (const void*)currentPathSegment.regionPtr;

                // evalaute the incident radiance the incident
                rkguide::Vector3 throughput {1.0f};
                rkguide::Vector3 contribution {0.0f};
                for (size_t j = i+1; j < numSegments; ++j)
                {
                    const rkguide::PathSegmentData &nextPathSegment = m_segmentStorage[j];
                    rkguide::Vector3 clampedThroughput = embree::min(throughput, maxThroughput);
                    contribution += clampedThroughput * nextPathSegment.scatteredContribution;
                    if(j == i+1 && !useNEEMiWeights)
                    {
                        contribution += clampedThroughput * nextPathSegment.directContribution;
                    }
                    else
                    {
                        contribution += clampedThroughput * nextPathSegment.miWeight * nextPathSegment.directContribution;
                    }
                    throughput = throughput * nextPathSegment.scatteringWeight;
                    throughput /= nextPathSegment.russianRouletteProbability;
                }

                if (contribution[0] > 0.0f || contribution[1] > 0.0f || contribution[2] > 0.0f )
                {
                    //if (considerNEE) SLog(EInfo, "Sample 2[%d]: pos: %f, %f, %f \t dir: %f, %f, %f \t pdf: %f \t distance: %f \t con: %f, %f, %f ", 
                    //    i, pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], pdf, distance, contribution[0], contribution[1], contribution[2]);
                    m_sampleStorage.emplace_back(DirectionalSampleData(pos, dir, RKGUIDE_SPECTRUM_TO_FLOAT(contribution)/pdf,
                                            pdf, distance, flags), regionPtr);
                    //m_regionPtrStorage.emplace_back(regionPtr);
                }

            }
        }
        //pathSegmentDataStorage->clear();
        return m_sampleStorage.size();
    }

    const std::vector<std::pair<DirectionalSampleData, const void*>>& getSamples()const
    {
        return m_sampleStorage;
    }


private:
    std::vector<std::pair<DirectionalSampleData, const void*>> m_sampleStorage;
    //std::vector<const void*> m_regionPtrStorage;

};
}