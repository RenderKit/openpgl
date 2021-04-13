// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "PathSegmentData.h"
#include "DirectionalSampleData.h"

namespace openpgl
{

struct PathSegmentDataStorage
{
    std::vector<PathSegmentData> m_segmentStorage;

    void reserve(const size_t &size)
    {
        m_segmentStorage.reserve(size);
        m_sampleStorage.reserve(size);
        m_sampleStorage2.reserve(size);
        //m_regionPtrStorage.reserve(size);
    }

    size_t size()
    {
        return m_segmentStorage.size();
    }

    void clear()
    {
        m_segmentStorage.clear();
        m_sampleStorage.clear();
        m_sampleStorage2.clear();
    }

    PathSegmentData *create_back(const Point3 &pos, const Vector3 &normal, const Vector3 &outDir)
    {
       m_segmentStorage.emplace_back(pos, normal, outDir);
       return &m_segmentStorage.back();
    }

    PathSegmentData *next()
    {
       m_segmentStorage.emplace_back();
       return &m_segmentStorage.back();
    }

    void push_back(const PathSegmentData &psData)
    {
        m_segmentStorage.push_back(psData);
    }


    uint32_t prepareSamples(const bool useNEEMiWeights = false, const bool guideDirectLight = false)
    {
        //m_sampleStorage.clear();
        //m_regionPtrStorage.clear();

        const float minPDF {0.1f};
        const openpgl::Vector3 maxThroughput {10.0f};

        size_t numSegments = m_segmentStorage.size();

        float lastDistance = 0.0f;
        for (int i=numSegments-2; i>=0; --i)
        {
            const openpgl::PathSegmentData &currentPathSegment = m_segmentStorage[i];
            float currentDistance = embree::length(m_segmentStorage[i+1].position - currentPathSegment.position);
            float distance = currentDistance + lastDistance;
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
                openpgl::Point3 pos = currentPathSegment.position;
                // using the direction directly is numerically more stable than recalcuating
                // it using position of the next segment when the distance is small. 
                openpgl::Vector3 dir = currentPathSegment.directionIn;
                float pdf = std::max(minPDF,currentPathSegment.pdfDirectionIn);
                uint32_t flags{0};
                const void* regionPtr = (const void*)currentPathSegment.regionPtr;
                bool insideVolume = currentPathSegment.volumeScatter;
                if(insideVolume)
                {
                    flags |= DirectionalSampleData::EInsideVolume;
                }
                // evalaute the incident radiance the incident
                openpgl::Vector3 throughput {1.0f};
                openpgl::Vector3 contribution {0.0f};
                for (size_t j = i+1; j < numSegments; ++j)
                {
                    const openpgl::PathSegmentData &nextPathSegment = m_segmentStorage[j];
                    throughput = throughput * nextPathSegment.transmittanceWeight;
                    openpgl::Vector3 clampedThroughput = embree::min(throughput, maxThroughput);
                    contribution += clampedThroughput * nextPathSegment.scatteredContribution;
                    OPENPGL_ASSERT(embree::isvalid(contribution));
                    if(j == i+1 && !useNEEMiWeights)
                    {
                        if(guideDirectLight)
                        {
                            contribution += clampedThroughput * nextPathSegment.directContribution;
                            OPENPGL_ASSERT(embree::isvalid(contribution));
                        }
                    }
                    else
                    {
                        if(j>i+1 || guideDirectLight)
                        {
                            contribution += clampedThroughput * nextPathSegment.miWeight * nextPathSegment.directContribution;
                            OPENPGL_ASSERT(embree::isvalid(contribution));
                        }
                    }
                    throughput = throughput * nextPathSegment.scatteringWeight;
                    throughput /= nextPathSegment.russianRouletteProbability;
                }

				OPENPGL_ASSERT(embree::isvalid(contribution));
                if (contribution[0] > 0.0f || contribution[1] > 0.0f || contribution[2] > 0.0f )
                {
                    //if (considerNEE) SLog(EInfo, "Sample 2[%d]: pos: %f, %f, %f \t dir: %f, %f, %f \t pdf: %f \t distance: %f \t con: %f, %f, %f ", 
                    //    i, pos[0], pos[1], pos[2], dir[0], dir[1], dir[2], pdf, distance, contribution[0], contribution[1], contribution[2]);
					OPENPGL_ASSERT(embree::isvalid(distance));
                    if(distance>0){
                        const float weight = OPENPGL_SPECTRUM_TO_FLOAT(contribution)/pdf;
                        OPENPGL_ASSERT(embree::isvalid(weight));
                        DirectionalSampleData dsd;
                        dsd.position.x = pos[0];
                        dsd.position.y = pos[1];
                        dsd.position.z = pos[2];
                        dsd.direction.x = dir[0];
                        dsd.direction.y = dir[1];
                        dsd.direction.z = dir[2];
                        dsd.weight = weight;
                        dsd.pdf = pdf;
                        dsd.distance = distance;
                        dsd.flags = flags;
                        m_sampleStorage.emplace_back(dsd, regionPtr);
                        m_sampleStorage2.emplace_back(dsd);
                    }
                    else
                    {
                        std::cout << "PathSegmentDataStorage::prepareSamples(): !(distance>0)" << std::endl;
                    }
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

    const std::vector<DirectionalSampleData>& getSamples2()const
    {
        return m_sampleStorage2;
    }

    void addSample(const DirectionalSampleData &sampleData, const void *regionPtr)
    {
        OPENPGL_ASSERT(isValid(sampleData));
        OPENPGL_ASSERT(sampleData.distance > 0);
        OPENPGL_ASSERT(embree::isvalid(sampleData.distance));
        m_sampleStorage.push_back(std::pair<DirectionalSampleData, const void*>(sampleData, regionPtr));
    }


    void addSample2(const DirectionalSampleData &sampleData)
    {
        OPENPGL_ASSERT(isValid(sampleData));
        OPENPGL_ASSERT(sampleData.distance > 0);
        OPENPGL_ASSERT(embree::isvalid(sampleData.distance));
        m_sampleStorage2.push_back(sampleData);
    }

private:
    std::vector<std::pair<DirectionalSampleData, const void*>> m_sampleStorage;
    std::vector<DirectionalSampleData> m_sampleStorage2;
    //std::vector<const void*> m_regionPtrStorage;

};
}