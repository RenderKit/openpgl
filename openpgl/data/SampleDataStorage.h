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
    typedef tbb::concurrent_vector<InvalidSampleData> InvalidSampleDataContainer;
    struct SampleContainer{
        SampleDataContainer samples;
        InvalidSampleDataContainer invalidSamples;
    };
    
    SampleContainer m_surfaceContainer;
    SampleContainer m_volumeContainer;

    //
    //InvalidSampleDataContainer m_invalidSurfaceContainer;
    //InvalidSampleDataContainer m_volumeContainer.invalidSamples;

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

    inline void addSample(const SampleData& sample)
    {
        if(isInsideVolume(sample))
        {
            m_volumeContainer.samples.push_back(sample);
        }
        else
        {
            m_surfaceContainer.samples.push_back(sample);
        }
    }

    inline void addSamples(const SampleData* samples, int nSamples)
    {
        for (int i = 0; i<nSamples; i++)
        {
            addSample(samples[i]);
        }
    }

    inline void addInvalidSample(const InvalidSampleData& sample)
    {
        if(sample.volume)
        {
            m_volumeContainer.invalidSamples.push_back(sample);
        }
        else
        {
            m_surfaceContainer.invalidSamples.push_back(sample);
        }
    }

    inline void addInvalidSamples(const InvalidSampleData* samples, int nSamples)
    {
        for (int i = 0; i<nSamples; i++)
        {
            addInvalidSample(samples[i]);
        }
    }


    inline void reserveSurface(const size_t &size)
    {
        m_surfaceContainer.samples.reserve(size);
    }

    inline size_t sizeSurface() const
    {
        return m_surfaceContainer.samples.size();
    }

    inline SampleData getSampleSurface(const int idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_surfaceContainer.samples.size());
        
        SampleData sd;
        if(idx < m_surfaceContainer.samples.size())
        {
            sd = m_surfaceContainer.samples[idx];
        }
        return sd;
    }

    inline void clearSurface()
    {
        m_surfaceContainer.samples.clear();
    }

    void sortSurface()
    {
        std::sort(m_surfaceContainer.samples.begin(), m_surfaceContainer.samples.end(), SampleDataLess);
    }

    inline void reserveVolume(const size_t &size)
    {
        m_volumeContainer.samples.reserve(size);
    }

    inline size_t sizeVolume() const
    {
        return m_volumeContainer.samples.size();
    }

    inline SampleData getSampleVolume(const int idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_volumeContainer.samples.size());

        SampleData sd;
        if(idx < m_volumeContainer.samples.size())
        {
            sd = m_volumeContainer.samples[idx];
        }
        return sd;
    }

    inline void clearVolume()
    {
        m_volumeContainer.samples.clear();
    }

    void sortVolume()
    {
        std::sort(m_volumeContainer.samples.begin(), m_volumeContainer.samples.end(), SampleDataLess);
    }

    inline void reserveInvalidSurface(const size_t &size)
    {
        m_surfaceContainer.invalidSamples.reserve(size);
    }

    inline size_t sizeInvalidSurface() const
    {
        return m_surfaceContainer.invalidSamples.size();
    }

    inline InvalidSampleData getInvalidSampleSurface(const int idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_surfaceContainer.invalidSamples.size());
        
        InvalidSampleData isd;
        if(idx < m_surfaceContainer.invalidSamples.size())
        {
            isd = m_surfaceContainer.invalidSamples[idx];
        }
        return isd;
    }

    inline void clearInvalidSurface()
    {
        m_surfaceContainer.invalidSamples.clear();
    }

    void sortInvalidSurface()
    {
        std::sort(m_surfaceContainer.invalidSamples.begin(), m_surfaceContainer.invalidSamples.end(), InvalidSampleDataLess);
    }

    inline void reserveInvalidVolume(const size_t &size)
    {
        m_volumeContainer.invalidSamples.reserve(size);
    }

    inline size_t sizeInvalidVolume() const
    {
        return m_volumeContainer.invalidSamples.size();
    }

    inline InvalidSampleData getInvalidSampleVolume(const int idx) const
    {
        OPENPGL_ASSERT(idx >= 0);
        OPENPGL_ASSERT(idx < m_volumeContainer.invalidSamples.size());

        InvalidSampleData isd;
        if(idx < m_volumeContainer.invalidSamples.size())
        {
            isd = m_volumeContainer.invalidSamples[idx];
        }
        return isd;
    }

    inline void clearInvalidVolume()
    {
        m_volumeContainer.invalidSamples.clear();
    }

    void sortInvalidVolume()
    {
        std::sort(m_volumeContainer.invalidSamples.begin(), m_volumeContainer.invalidSamples.end(), InvalidSampleDataLess);
    }

    void exportSurfaceSamplesToObj(std::string objFileName, bool pointsOnly = true)
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());
        exportSamplesToObj(objFile, m_surfaceContainer.samples, pointsOnly);
        objFile.close();
    }

    void exportVolumeSamplesToObj(std::string objFileName, bool pointsOnly = true)
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());
        exportSamplesToObj(objFile, m_volumeContainer.samples, pointsOnly);
        objFile.close();
    }


    void serialize(std::ostream& stream) const
    {
        size_t num_surface_samples = m_surfaceContainer.samples.size();
        stream.write(reinterpret_cast<const char*>(&num_surface_samples), sizeof(size_t));
        for ( size_t n = 0; n < num_surface_samples; n++)
        {
            SampleData dsd = m_surfaceContainer.samples[n];
            stream.write(reinterpret_cast<const char*>(&dsd), sizeof(SampleData));
        }

        size_t num_volume_samples = m_volumeContainer.samples.size();
        stream.write(reinterpret_cast<const char*>(&num_volume_samples), sizeof(size_t));
        for ( size_t n = 0; n < num_volume_samples; n++)
        {
            SampleData dsd = m_volumeContainer.samples[n];
            stream.write(reinterpret_cast<const char*>(&dsd), sizeof(SampleData));
        }

        size_t num_invalid_surface_samples = m_surfaceContainer.invalidSamples.size();
        stream.write(reinterpret_cast<const char*>(&num_invalid_surface_samples), sizeof(size_t));
        for ( size_t n = 0; n < num_invalid_surface_samples; n++)
        {
            InvalidSampleData isd = m_surfaceContainer.invalidSamples[n];
            stream.write(reinterpret_cast<const char*>(&isd), sizeof(InvalidSampleData));
        }

        size_t num_invalid_volume_samples = m_volumeContainer.invalidSamples.size();
        stream.write(reinterpret_cast<const char*>(&num_invalid_volume_samples), sizeof(size_t));
        for ( size_t n = 0; n < num_invalid_volume_samples; n++)
        {
            InvalidSampleData isd = m_volumeContainer.invalidSamples[n];
            stream.write(reinterpret_cast<const char*>(&isd), sizeof(InvalidSampleData));
        }
    }

    void deserialize(std::istream& stream)
    {
        size_t num_surface_samples;
        stream.read(reinterpret_cast<char*>(&num_surface_samples), sizeof(size_t));
        m_surfaceContainer.samples.reserve(num_surface_samples);
        for ( size_t n = 0; n < num_surface_samples; n++)
        {
            SampleData dsd;
            stream.read(reinterpret_cast<char*>(&dsd), sizeof(SampleData));
            m_surfaceContainer.samples.push_back(dsd);
        }

        size_t num_volume_samples;
        stream.read(reinterpret_cast<char*>(&num_volume_samples), sizeof(size_t));
        m_volumeContainer.samples.reserve(num_volume_samples);
        for ( size_t n = 0; n < num_volume_samples; n++)
        {
            SampleData dsd;
            stream.read(reinterpret_cast<char*>(&dsd), sizeof(SampleData));
            m_volumeContainer.samples.push_back(dsd);
        }

        size_t num_invalid_surface_samples;
        stream.read(reinterpret_cast<char*>(&num_invalid_surface_samples), sizeof(size_t));
        m_surfaceContainer.invalidSamples.reserve(num_invalid_surface_samples);
        for ( size_t n = 0; n < num_invalid_surface_samples; n++)
        {
            InvalidSampleData isd;
            stream.read(reinterpret_cast<char*>(&isd), sizeof(InvalidSampleData));
            m_surfaceContainer.invalidSamples.push_back(isd);
        }

        size_t num_invalid_volume_samples;
        stream.read(reinterpret_cast<char*>(&num_invalid_volume_samples), sizeof(size_t));
        m_volumeContainer.invalidSamples.reserve(num_invalid_volume_samples);
        for ( size_t n = 0; n < num_invalid_volume_samples; n++)
        {
            InvalidSampleData isd;
            stream.read(reinterpret_cast<char*>(&isd), sizeof(InvalidSampleData));
            m_volumeContainer.invalidSamples.push_back(isd);
        }
    }

    bool validate() const {
        bool valid = true;
        for (int i = 0; i < m_surfaceContainer.samples.size(); i++){
            valid = valid && isValid(m_surfaceContainer.samples[i]);
        }

        for (int i = 0; i < m_volumeContainer.samples.size(); i++){
            valid = valid && isValid(m_volumeContainer.samples[i]);
        }
        return valid;
    }

    bool operator==(const SampleDataStorage& b) const {
        std::vector<SampleData> surfaceSampleDataA;
        std::vector<SampleData> volumeSampleDataA;

        std::vector<InvalidSampleData> surfaceInvalidSampleDataA;
        std::vector<InvalidSampleData> volumeInvalidSampleDataA;

        std::vector<SampleData> surfaceSampleDataB;
        std::vector<SampleData> volumeSampleDataB;

        std::vector<InvalidSampleData> surfaceInvalidSampleDataB;
        std::vector<InvalidSampleData> volumeInvalidSampleDataB;

        surfaceSampleDataA.resize(m_surfaceContainer.samples.size());        
        volumeSampleDataA.resize(m_volumeContainer.samples.size());

        for (int i = 0; i < m_surfaceContainer.samples.size(); i++){
            surfaceSampleDataA[i] = m_surfaceContainer.samples[i];
        }
        tbb::parallel_sort(surfaceSampleDataA.begin(), surfaceSampleDataA.end(), SampleDataLess);

        for (int i = 0; i < m_volumeContainer.samples.size(); i++){
            volumeSampleDataA[i] = m_volumeContainer.samples[i];
        }
        tbb::parallel_sort(volumeSampleDataA.begin(), volumeSampleDataA.end(), SampleDataLess);

        surfaceInvalidSampleDataA.resize(m_surfaceContainer.invalidSamples.size());        
        volumeInvalidSampleDataA.resize(m_volumeContainer.invalidSamples.size());

        for (int i = 0; i < m_surfaceContainer.invalidSamples.size(); i++){
            surfaceInvalidSampleDataA[i] = m_surfaceContainer.invalidSamples[i];
        }
        tbb::parallel_sort(surfaceInvalidSampleDataA.begin(), surfaceInvalidSampleDataA.end(), InvalidSampleDataLess);

        for (int i = 0; i < m_volumeContainer.invalidSamples.size(); i++){
            volumeInvalidSampleDataA[i] = m_volumeContainer.invalidSamples[i];
        }
        tbb::parallel_sort(volumeInvalidSampleDataA.begin(), volumeInvalidSampleDataA.end(), InvalidSampleDataLess);

        surfaceSampleDataB.resize(b.m_surfaceContainer.samples.size());        
        volumeSampleDataB.resize(b.m_volumeContainer.samples.size());

        for (int i = 0; i < b.m_surfaceContainer.samples.size(); i++){
            surfaceSampleDataB[i] = b.m_surfaceContainer.samples[i];
        }
        tbb::parallel_sort(surfaceSampleDataB.begin(), surfaceSampleDataB.end(), SampleDataLess);

        for (int i = 0; i < b.m_volumeContainer.samples.size(); i++){
            volumeSampleDataB[i] = b.m_volumeContainer.samples[i];
        }
        tbb::parallel_sort(volumeSampleDataB.begin(), volumeSampleDataB.end(), SampleDataLess); 

        surfaceInvalidSampleDataB.resize(b.m_surfaceContainer.invalidSamples.size());        
        volumeInvalidSampleDataB.resize(b.m_volumeContainer.invalidSamples.size());

        for (int i = 0; i < b.m_surfaceContainer.invalidSamples.size(); i++){
            surfaceInvalidSampleDataB[i] = b.m_surfaceContainer.invalidSamples[i];
        }
        tbb::parallel_sort(surfaceInvalidSampleDataB.begin(), surfaceInvalidSampleDataB.end(), InvalidSampleDataLess);

        for (int i = 0; i < b.m_volumeContainer.invalidSamples.size(); i++){
            volumeInvalidSampleDataB[i] = b.m_volumeContainer.invalidSamples[i];
        }
        tbb::parallel_sort(volumeInvalidSampleDataB.begin(), volumeInvalidSampleDataB.end(), InvalidSampleDataLess);

        bool equal = true;
        int sizeB = surfaceSampleDataB.size();
        for (int i = 0; i < surfaceSampleDataA.size(); i++) {
            if( i< sizeB && !SampleDataEqual(surfaceSampleDataA[i], surfaceSampleDataB[i])){
                equal = false;
            }
        }

        sizeB = surfaceInvalidSampleDataB.size();
        for (int i = 0; i < surfaceInvalidSampleDataA.size(); i++) {
            if( i< sizeB && !InvalidSampleDataEqual(surfaceInvalidSampleDataA[i], surfaceInvalidSampleDataB[i])){
                equal = false;
                //std::cout << "Non-equal surfaceSample[" << i << "]: " << std::endl << " left = " << toString(surfaceSampleDataA[i]) << std::endl << " right = " << toString(surfaceSampleDataB[i]) << std::endl;
            }
        }

        sizeB = volumeSampleDataB.size();
        for (int i = 0; i < volumeSampleDataA.size(); i++) {
            if( i< sizeB && !SampleDataEqual(volumeSampleDataA[i], volumeSampleDataB[i])){
                equal = false;
            }
        }

        sizeB = volumeInvalidSampleDataB.size();
        for (int i = 0; i < volumeInvalidSampleDataA.size(); i++) {
            if( i< sizeB && !InvalidSampleDataEqual(volumeInvalidSampleDataA[i], volumeInvalidSampleDataB[i])){
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
