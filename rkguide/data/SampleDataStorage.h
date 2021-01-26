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
    SampleDataContainer m_surfaceContainer;
    SampleDataContainer m_volumeContainer;
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
        if(sample.isInsideVolume())
        {
            m_volumeContainer.push_back(sample);
        }
        else
        {
            m_surfaceContainer.push_back(sample);
        }
    }

    template<typename TRegion>
    inline void addSamples(const std::vector<std::pair<DirectionalSampleData, const void*> > &samples,  Sampler *sampler)
    {
        for (auto& sample : samples)
        {
            addSample(sample.first, (const TRegion*)sample.second, sampler);
        }
    }

    inline void addSample2(DirectionalSampleData sample)
    {
        if(sample.isInsideVolume())
        {
            m_volumeContainer.push_back(sample);
        }
        else
        {
            m_surfaceContainer.push_back(sample);
        }
    }

    inline void addSamples2(const std::vector<std::pair<DirectionalSampleData, const void*> > &samples)
    {
        for (auto& sample : samples)
        {
            addSample2(sample.first);
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
    inline void reserveSurface(const size_t &size)
    {
        m_surfaceContainer.reserve(size);
    }

    inline size_t sizeSurface() const
    {
        return m_surfaceContainer.size();
    }

    inline void clearSurface()
    {
        m_surfaceContainer.clear();
    }

    void sortSurface()
    {
        std::sort(m_surfaceContainer.begin(), m_surfaceContainer.end());
    }

    inline void reserveVolume(const size_t &size)
    {
        m_volumeContainer.reserve(size);
    }

    inline size_t sizeVolume() const
    {
        return m_volumeContainer.size();
    }

    inline void clearVolume()
    {
        m_volumeContainer.clear();
    }

    void sortVolume()
    {
        std::sort(m_volumeContainer.begin(), m_volumeContainer.end());
    }


    void exportSurfaceSamplesToObj(std::string objFileName, bool pointsOnly = true)
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());
        exportSamplesToObj(objFile, m_surfaceContainer, pointsOnly);
        objFile.close();
    }

    void exportVolumeSamplesToObj(std::string objFileName, bool pointsOnly = true)
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());
        exportSamplesToObj(objFile, m_volumeContainer, pointsOnly);
        objFile.close();
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

    void exportSamplesToObj(std::ofstream &objFile, const SampleDataContainer &sampleContainer, bool pointsOnly = true)
    {
        std::vector<DirectionalSampleData> subSampledData;
        subSampledData.reserve(sampleContainer.size());
        for (size_t i =0; i < sampleContainer.size(); i++)
        {
            //int idx = sampler->next1D() * nData;
            subSampledData.push_back(sampleContainer[i]);
        }

/*
            Properties props("independent");
            props.setInteger("sampleCount", numExportedSamples);
            ref<Sampler> sampler = static_cast<Sampler *>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), props));

            std::vector<DData> subSampledData;
            subSampledData.reserve(numExportedSamples);
            for (size_t i =0; i < numExportedSamples; i++)
            {
                int idx = sampler->next1D() * nData;
                subSampledData.push_back(data[idx]);
            }

            std::ofstream objFile;
            objFile.open(objFileName.c_str());
*/
            for (auto& sample : subSampledData)
            {
                objFile << "v " << sample.position[0] << "\t" << sample.position[1] << "\t"<< sample.position[2] << std::endl;
                if (!pointsOnly)
                {
                    Vector3 dir = sample.direction;
                    Point3 pos2 = sample.position + dir * sample.distance;
                    objFile << "v " << pos2[0] << "\t" << pos2[1] << "\t"<< pos2[2] << std::endl;
                    objFile << "v " << sample.position[0] << "\t" << sample.position[1] << "\t"<< sample.position[2] << std::endl;
                }
            }

            for (auto& sample : subSampledData)
            {
                Vector3 dir = sample.direction;
                //dir *= sample.distance;
                objFile << "vn " << dir[0] << "\t" << dir[1] << "\t"<< dir[2] << std::endl;
                if (!pointsOnly)
                {
                    objFile << "vn " << dir[0] << "\t" << dir[1] << "\t"<< dir[2] << std::endl;
                    objFile << "vn " << dir[0] << "\t" << dir[1] << "\t"<< dir[2] << std::endl;
                }
            }

            if (!pointsOnly)
            {
                for (int i = 0; i < subSampledData.size(); i++)
                {
                    objFile << "f " << i*3+1 << "\t" << i*3+2 << "\t"<< i*3+3 << std::endl;
                }
            }
            //objFile.close();
    }


};

}