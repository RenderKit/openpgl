// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

#include "DirectionalSampleData.h"
#include "../sampler/Sampler.h"

#include <tbb/concurrent_vector.h>

namespace openpgl
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
        if(isInsideVolume(sample))
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
        if(isInsideVolume(sample))
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
        std::sort(m_surfaceContainer.begin(), m_surfaceContainer.end(), DirectionalSampleDataLess);
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
        std::sort(m_volumeContainer.begin(), m_volumeContainer.end(), DirectionalSampleDataLess);
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


    void serialize(std::ostream& stream) const
    {
        size_t num_surface_samples = m_surfaceContainer.size();
        stream.write(reinterpret_cast<const char*>(&num_surface_samples), sizeof(size_t));
        for ( size_t n = 0; n < num_surface_samples; n++)
        {
            DirectionalSampleData dsd = m_surfaceContainer[n];
            stream.write(reinterpret_cast<const char*>(&dsd), sizeof(DirectionalSampleData));
        }

        size_t num_volume_samples = m_volumeContainer.size();
        stream.write(reinterpret_cast<const char*>(&num_volume_samples), sizeof(size_t));
        for ( size_t n = 0; n < num_volume_samples; n++)
        {
            DirectionalSampleData dsd = m_volumeContainer[n];
            stream.write(reinterpret_cast<const char*>(&dsd), sizeof(DirectionalSampleData));
        }
        stream.write(reinterpret_cast<const char*>(&m_splatSamples), sizeof(bool));
        stream.write(reinterpret_cast<const char*>(&m_sceneBounds), sizeof(BBox));
    }

    void deserialize(std::istream& stream)
    {
        size_t num_surface_samples;
        stream.read(reinterpret_cast<char*>(&num_surface_samples), sizeof(size_t));
        m_surfaceContainer.reserve(num_surface_samples);
        for ( size_t n = 0; n < num_surface_samples; n++)
        {
            DirectionalSampleData dsd;
            stream.read(reinterpret_cast<char*>(&dsd), sizeof(DirectionalSampleData));
            m_surfaceContainer.push_back(dsd);
        }

        size_t num_volume_samples;
        stream.read(reinterpret_cast<char*>(&num_volume_samples), sizeof(size_t));
        m_volumeContainer.reserve(num_volume_samples);
        for ( size_t n = 0; n < num_volume_samples; n++)
        {
            DirectionalSampleData dsd;
            stream.read(reinterpret_cast<char*>(&dsd), sizeof(DirectionalSampleData));
            m_volumeContainer.push_back(dsd);
        }
        stream.read(reinterpret_cast<char*>(&m_splatSamples), sizeof(bool));
        stream.read(reinterpret_cast<char*>(&m_sceneBounds), sizeof(BBox));
    }

    private:

    void splatSample(DirectionalSampleData &sample, const BBox &splattingBounds, const Point2 &sample2D) const
    {
        const Vector3 boundsExtents = (splattingBounds.upper - splattingBounds.lower) * 0.5f;
        //float maxBoundsExtent = (boundsExtents.x > boundsExtents.y) ?   ((boundsExtents.x > boundsExtents.z) ? boundsExtents.x : boundsExtents.z):
        //                                                                ((boundsExtents.y > boundsExtents.z) ? boundsExtents.y : boundsExtents.z);
        const Vector3 sampleDisplacement = boundsExtents * squareToUniformSphere(sample2D);
        const Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
        Vector3 sampleDirection(sample.direction.x, sample.direction.y, sample.direction.z);
        
        Point3 splattedPosition = samplePosition + sampleDisplacement;
        if (!embree::inside(splattingBounds, splattedPosition))
        {
            Point3 sourcePosition = samplePosition + sampleDirection * sample.distance;
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

            sample.position.x = splattedPosition[0];
            sample.position.y = splattedPosition[1];
            sample.position.z = splattedPosition[2];

            sampleDirection = sourcePosition - splattedPosition;
            sample.distance = embree::length(sampleDirection);
            sampleDirection = sampleDirection / sample.distance;

            sample.direction.x = sampleDirection[0];
            sample.direction.y = sampleDirection[1];
            sample.direction.z = sampleDirection[2];

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
                objFile << "v " << sample.position.x << "\t" << sample.position.y << "\t"<< sample.position.z << std::endl;
                if (!pointsOnly)
                {
                    Vector3 dir(sample.direction.x, sample.direction.y, sample.direction.z);
                    Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
                    Point3 pos2 = samplePosition + dir * sample.distance;
                    objFile << "v " << pos2[0] << "\t" << pos2[1] << "\t"<< pos2[2] << std::endl;
                    objFile << "v " << sample.position.x << "\t" << sample.position.y << "\t"<< sample.position.z << std::endl;
                }
            }

            for (auto& sample : subSampledData)
            {
                Vector3 dir(sample.direction.x, sample.direction.y, sample.direction.z);
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