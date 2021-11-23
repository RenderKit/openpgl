// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "PathSegmentData.h"
#include "SampleData.h"
#include "../sampler/Sampler.h"
#include "../spatial/Region.h"

namespace openpgl
{
struct PathSegmentDataStorage
{
    PathSegmentDataStorage() = default;
    ~PathSegmentDataStorage() = default;
    std::vector<PathSegmentData> m_segmentStorage;

    void reserve(const size_t &size)
    {
        m_segmentStorage.reserve(size);
        m_sampleStorage.reserve(size);
    }

    size_t size()
    {
        return m_segmentStorage.size();
    }

    void clear()
    {
        m_segmentStorage.clear();
        m_sampleStorage.clear();
    }

    PathSegmentData *next()
    {
       m_segmentStorage.emplace_back();
       return &m_segmentStorage.back();
    }

    void addSegment(const PGLPathSegmentData& segment)
    {
        m_segmentStorage.push_back(segment);
    }

    void push_back(const PathSegmentData &psData)
    {
        m_segmentStorage.push_back(psData);
    }


    size_t prepareSamples(const bool splatSamples, Sampler* sampler, const bool useNEEMiWeights = false, const bool guideDirectLight = false)
    {
        const float minPDF {0.1f};
        const openpgl::Vector3 maxThroughput {10.0f};

        size_t numSegments = m_segmentStorage.size();

        float lastDistance = 0.0f;
        for (int i=numSegments-2; i>=0; --i)
        {
            const openpgl::PathSegmentData &currentPathSegment = m_segmentStorage[i];
            float currentDistance = embree::length(openpgl::Point3(m_segmentStorage[i+1].position.x, m_segmentStorage[i+1].position.y, m_segmentStorage[i+1].position.z) - openpgl::Point3(currentPathSegment.position.x, currentPathSegment.position.y, currentPathSegment.position.z));
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
                    float cosThetaI = embree::dot(openpgl::Vector3(currentPathSegment.normal.x, currentPathSegment.normal.y, currentPathSegment.normal.z), openpgl::Vector3(currentPathSegment.directionOut.x, currentPathSegment.directionOut.y, currentPathSegment.directionOut.z));
                    float cosThetaO = embree::dot(openpgl::Vector3(currentPathSegment.normal.x, currentPathSegment.normal.y, currentPathSegment.normal.z), openpgl::Vector3(currentPathSegment.directionIn.x, currentPathSegment.directionIn.y, currentPathSegment.directionIn.z));
                    lastDistance *= std::fabs(cosThetaI/(cosThetaO*currentPathSegment.eta));
                }
            }

            // we only collect samples on non delta surfaces and which are generated
            // using a non-direct (delta) sampling method
            if (!currentPathSegment.isDelta && currentPathSegment.roughness > 0.01f)
            {

                // prepare the current pos, direction, distance, pdf at the current
                // path vertex
                openpgl::Point3 pos = openpgl::Point3(currentPathSegment.position.x, currentPathSegment.position.y, currentPathSegment.position.z);
                // using the direction directly is numerically more stable than recalcuating
                // it using position of the next segment when the distance is small. 
                openpgl::Vector3 dir = openpgl::Vector3(currentPathSegment.directionIn.x, currentPathSegment.directionIn.y, currentPathSegment.directionIn.z);
                float pdf = std::max(minPDF,currentPathSegment.pdfDirectionIn);
                uint32_t flags{0};
                const IRegion* regionPtr = (const IRegion*)currentPathSegment.regionPtr;
                //OPENPGL_ASSERT(regionPtr != nullptr);
                bool insideVolume = currentPathSegment.volumeScatter;
                if(insideVolume)
                {
                    flags |= SampleData::EInsideVolume;
                }
                
                // evalaute the incident radiance the incident
                openpgl::Vector3 throughput {1.0f};
                openpgl::Vector3 contribution {0.0f};
                for (size_t j = i+1; j < numSegments; ++j)
                {
                    const openpgl::PathSegmentData &nextPathSegment = m_segmentStorage[j];

                    //throughput = throughput * openpgl::Vector3(nextPathSegment.transmittanceWeight.x, nextPathSegment.transmittanceWeight.y, nextPathSegment.transmittanceWeight.z);
                    throughput = throughput * openpgl::Vector3(m_segmentStorage[j-1].transmittanceWeight.x, m_segmentStorage[j-1].transmittanceWeight.y, m_segmentStorage[j-1].transmittanceWeight.z);
                    OPENPGL_ASSERT(embree::isvalid(throughput));
                    OPENPGL_ASSERT(throughput[0] >= 0.f && throughput[1] >= 0.f && throughput[2] >= 0.f)
                    openpgl::Vector3 clampedThroughput = embree::min(throughput, maxThroughput);
                    contribution += clampedThroughput * openpgl::Vector3(nextPathSegment.scatteredContribution.x, nextPathSegment.scatteredContribution.y, nextPathSegment.scatteredContribution.z);
                    OPENPGL_ASSERT(embree::isvalid(contribution));
                    OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);
                    if(j == i+1 && !useNEEMiWeights)
                    {
                        if(guideDirectLight)
                        {
                            contribution += clampedThroughput * openpgl::Vector3(nextPathSegment.directContribution.x, nextPathSegment.directContribution.y, nextPathSegment.directContribution.z);
                            OPENPGL_ASSERT(embree::isvalid(contribution));
                            OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);
                        }
                    }
                    else
                    {
                        if(j>i+1 || guideDirectLight)
                        {
                            contribution += clampedThroughput * nextPathSegment.miWeight * openpgl::Vector3(nextPathSegment.directContribution.x, nextPathSegment.directContribution.y, nextPathSegment.directContribution.z);
                            OPENPGL_ASSERT(embree::isvalid(contribution));
                            OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);
                        }
                    }
                    throughput = throughput * openpgl::Vector3(nextPathSegment.scatteringWeight.x, nextPathSegment.scatteringWeight.y, nextPathSegment.scatteringWeight.z);
                    throughput /= nextPathSegment.russianRouletteProbability;

                    OPENPGL_ASSERT(embree::isvalid(throughput));
                    OPENPGL_ASSERT(throughput[0] >= 0.f && throughput[1] >= 0.f && throughput[2] >= 0.f)
                }

				OPENPGL_ASSERT(embree::isvalid(contribution));
                OPENPGL_ASSERT(contribution[0] >= 0.f && contribution[1] >= 0.f && contribution[2] >= 0.f);
                if (contribution[0] > 0.0f || contribution[1] > 0.0f || contribution[2] > 0.0f )
                {
					OPENPGL_ASSERT(embree::isvalid(distance));
                    if(distance>0){
                        const float weight = OPENPGL_SPECTRUM_TO_FLOAT(contribution)/pdf;
                        OPENPGL_ASSERT(embree::isvalid(weight));
                        OPENPGL_ASSERT(weight >= 0.f);
                        SampleData dsd;
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
                        //m_sampleStorage.emplace_back(dsd, regionPtr);
                        if(sampler && regionPtr !=nullptr && splatSamples)
                        {
                            regionPtr->splatSample(dsd,sampler->next2D()); 
                        }
                        m_sampleStorage.emplace_back(dsd);
                    }
                    else
                    {
                        std::cout << "PathSegmentDataStorage::prepareSamples(): !(distance>0)" << std::endl;
                    }
                }

            }
        }
        //std::cout << std::endl;
        //std::cout << this->toString() << std::endl;
        return m_sampleStorage.size();
    }

    const std::vector<SampleData>& getSamples()const
    {
        return m_sampleStorage;
    }

    void addSample(const SampleData &sampleData)
    {
        OPENPGL_ASSERT(isValid(sampleData));
        OPENPGL_ASSERT(sampleData.distance > 0);
        OPENPGL_ASSERT(embree::isvalid(sampleData.distance));
        m_sampleStorage.push_back(sampleData);
    }

    bool samplesValid() const
    {
        bool valid = true;
        for ( int s = 0; s < m_sampleStorage.size(); s++)
        {
            SampleData sample = m_sampleStorage[s];
            valid = valid && isValid(sample);
            OPENPGL_ASSERT(valid);
        }
        return valid;
    }

    bool segmentsValid() const
    {
        bool valid = true;
        for ( int s = 0; s < m_segmentStorage.size(); s++)
        {
            PathSegmentData psd = m_segmentStorage[s];
            valid = valid && isValid(psd);
            OPENPGL_ASSERT(valid);
        }
        return valid;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss << "PathSegmentDataStorage:" << std::endl;
        ss << "segment storage: size = "<< m_segmentStorage.size() << std::endl;
        for ( int s = 0; s < m_segmentStorage.size(); s++)
        {
            PathSegmentData psd = m_segmentStorage[s];
            ss << "seg[" << s << "]: " << openpgl::toString(psd) ;
			ss << "\t valid = " << isValid(psd);
            ss << std::endl;
        }

        for ( int s = 0; s < m_sampleStorage.size(); s++)
        {
            SampleData sample = m_sampleStorage[s];
            ss << "sample[" << s << "]: " << openpgl::toString(sample);
            ss << "\t valid = " << isValid(sample);
            ss << std::endl;
        }

        return ss.str();
    }

private:
    std::vector<SampleData> m_sampleStorage;
};
}