// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

#include "SampleData.h"

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_sort.h>

#define SAMPLE_DATA_STORAGE_FILE_HEADER_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_SAMPLE_STORAGE"

namespace openpgl
{

struct SampleDataStorage
{

    typedef tbb::concurrent_vector<SampleData> SampleDataContainer;
    SampleDataContainer m_surfaceContainer;
    SampleDataContainer m_volumeContainer;

    static SampleDataStorage* newSampleDataStorage()
    {
        return new openpgl::SampleDataStorage();
    }

    static SampleDataStorage* newSampleDataStorageFromFile(const std::string sampleDataStorageFileName)
    {
        std::filebuf fb;
        fb.open (sampleDataStorageFileName, std::ios::in | std::ios::binary);
        if (!fb.is_open()) throw std::runtime_error("error: couldn't open file");
        std::istream is(&fb);

        auto size = strlen(SAMPLE_DATA_STORAGE_FILE_HEADER_STRING) + 1;
        OPENPGL_ASSERT(size <= 256);
        char buf[256];
        is.read(&buf[0], size);
        if (!is) throw std::runtime_error("error: invalid file header");
#ifdef OPENPGL_STRICT_IO_VERSION_CHECKING
        for (auto i = 0; i < size; i++)
        {
            if (buf[i] != SAMPLE_DATA_STORAGE_FILE_HEADER_STRING[i])
                throw std::runtime_error("error: invalid file header");
        }
#endif
        openpgl::SampleDataStorage* gSampleStorage = new openpgl::SampleDataStorage();
        gSampleStorage->deserialize(is);

        fb.close();

        return gSampleStorage;
    }

    static void storeSampleDataStorageToFile(SampleDataStorage* gSampleDataStorage, const std::string sampleDataStorageFileName)
    {
        std::filebuf fb;
        fb.open (sampleDataStorageFileName, std::ios::out | std::ios::binary);
        if (!fb.is_open()) throw std::runtime_error("error: couldn't open file!");
        std::ostream os(&fb);

        os.write(SAMPLE_DATA_STORAGE_FILE_HEADER_STRING, strlen(SAMPLE_DATA_STORAGE_FILE_HEADER_STRING) + 1);

        gSampleDataStorage->serialize(os);

        os.flush();
        fb.close();
    }

    inline void addSample(SampleData sample)
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

    inline void addSamples(const SampleData* samples, int nSamples)
    {
        for (int i = 0; i<nSamples; i++)
        {
            addSample(samples[i]);
        }
    }

    inline void reserveSurface(const size_t &size)
    {
        m_surfaceContainer.reserve(size);
    }

    inline size_t sizeSurface() const
    {
        return m_surfaceContainer.size();
    }

    inline SampleData getSampleSurface(const int idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_surfaceContainer.size());
        
        SampleData sd;
        if(idx < m_surfaceContainer.size())
        {
            sd = m_surfaceContainer[idx];
        }
        return sd;
    }

    inline void clearSurface()
    {
        m_surfaceContainer.clear();
    }

    void sortSurface()
    {
        std::sort(m_surfaceContainer.begin(), m_surfaceContainer.end(), SampleDataLess);
    }

    inline void reserveVolume(const size_t &size)
    {
        m_volumeContainer.reserve(size);
    }

    inline size_t sizeVolume() const
    {
        return m_volumeContainer.size();
    }

    inline SampleData getSampleVolume(const int idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_volumeContainer.size());

        SampleData sd;
        if(idx < m_volumeContainer.size())
        {
            sd = m_volumeContainer[idx];
        }
        return sd;
    }

    inline void clearVolume()
    {
        m_volumeContainer.clear();
    }

    void sortVolume()
    {
        std::sort(m_volumeContainer.begin(), m_volumeContainer.end(), SampleDataLess);
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
            SampleData dsd = m_surfaceContainer[n];
            stream.write(reinterpret_cast<const char*>(&dsd), sizeof(SampleData));
        }

        size_t num_volume_samples = m_volumeContainer.size();
        stream.write(reinterpret_cast<const char*>(&num_volume_samples), sizeof(size_t));
        for ( size_t n = 0; n < num_volume_samples; n++)
        {
            SampleData dsd = m_volumeContainer[n];
            stream.write(reinterpret_cast<const char*>(&dsd), sizeof(SampleData));
        }
    }

    void deserialize(std::istream& stream)
    {
        size_t num_surface_samples;
        stream.read(reinterpret_cast<char*>(&num_surface_samples), sizeof(size_t));
        m_surfaceContainer.reserve(num_surface_samples);
        for ( size_t n = 0; n < num_surface_samples; n++)
        {
            SampleData dsd;
            stream.read(reinterpret_cast<char*>(&dsd), sizeof(SampleData));
            m_surfaceContainer.push_back(dsd);
        }

        size_t num_volume_samples;
        stream.read(reinterpret_cast<char*>(&num_volume_samples), sizeof(size_t));
        m_volumeContainer.reserve(num_volume_samples);
        for ( size_t n = 0; n < num_volume_samples; n++)
        {
            SampleData dsd;
            stream.read(reinterpret_cast<char*>(&dsd), sizeof(SampleData));
            m_volumeContainer.push_back(dsd);
        }
    }

bool operator==(const SampleDataStorage& b) const {
        std::vector<SampleData> surfaceSampleDataA;
        std::vector<SampleData> volumeSampleDataA;

        std::vector<SampleData> surfaceSampleDataB;
        std::vector<SampleData> volumeSampleDataB;

        surfaceSampleDataA.resize(m_surfaceContainer.size());        
        volumeSampleDataA.resize(m_volumeContainer.size());

        for (int i = 0; i < m_surfaceContainer.size(); i++){
            surfaceSampleDataA[i] = m_surfaceContainer[i];
        }
        tbb::parallel_sort(surfaceSampleDataA.begin(), surfaceSampleDataA.end(), SampleDataLess);

        for (int i = 0; i < m_volumeContainer.size(); i++){
            volumeSampleDataA[i] = m_volumeContainer[i];
        }
        tbb::parallel_sort(volumeSampleDataA.begin(), volumeSampleDataA.end(), SampleDataLess);

        surfaceSampleDataB.resize(b.m_surfaceContainer.size());        
        volumeSampleDataB.resize(b.m_volumeContainer.size());

        for (int i = 0; i < b.m_surfaceContainer.size(); i++){
            surfaceSampleDataB[i] = b.m_surfaceContainer[i];
        }
        tbb::parallel_sort(surfaceSampleDataB.begin(), surfaceSampleDataB.end(), SampleDataLess);

        for (int i = 0; i < b.m_volumeContainer.size(); i++){
            volumeSampleDataB[i] = b.m_volumeContainer[i];
        }
        tbb::parallel_sort(volumeSampleDataB.begin(), volumeSampleDataB.end(), SampleDataLess); 

        bool equal = true;
        int sizeB = surfaceSampleDataB.size();
        for (int i = 0; i < surfaceSampleDataA.size(); i++) {
            if( i< sizeB && !SampleDataEqual(surfaceSampleDataA[i], surfaceSampleDataB[i])){
                equal = false;
                //std::cout << "Non-equal surfaceSample[" << i << "]: " << std::endl << " left = " << toString(surfaceSampleDataA[i]) << std::endl << " right = " << toString(surfaceSampleDataB[i]) << std::endl;
            }
        }

        sizeB = volumeSampleDataB.size();
        for (int i = 0; i < volumeSampleDataA.size(); i++) {
            if( i< sizeB && !SampleDataEqual(volumeSampleDataA[i], volumeSampleDataB[i])){
                equal = false;
                //std::cout << "Non-equal volumeSample[" << i << "]: " << std::endl << " left = " << toString(volumeSampleDataA[i]) << std::endl << " right = " << toString(volumeSampleDataB[i]) << std::endl;
            }
        }

        return equal;
    }

    private:

    void exportSamplesToObj(std::ofstream &objFile, const SampleDataContainer &sampleContainer, bool pointsOnly = true)
    {
        std::vector<SampleData> subSampledData;
        subSampledData.reserve(sampleContainer.size());
        for (size_t i =0; i < sampleContainer.size(); i++)
        {
            subSampledData.push_back(sampleContainer[i]);
        }

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
    }
};

}
