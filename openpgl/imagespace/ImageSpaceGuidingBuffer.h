// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

#include "Denoiser.h"

#ifdef USE_EMBREE_PARALLEL
#define TASKING_TBB
#include <embreeSrc/common/algorithms/parallel_for.h>
#else
#include <tbb/parallel_for.h>
#endif

//Please include your own zlib-compatible API header before
//including `tinyexr.h` when you disable `TINYEXR_USE_MINIZ`
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_NANOZLIB 1
//#include "zlib.h"
//Or, if your project uses `stb_image[_write].h`, use their
//zlib implementation:
//#define TINYEXR_USE_STB_ZLIB 1
//#define TINYEXR_USE_THREAD 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>


namespace openpgl
{

struct ImageSpaceGuidingBuffer{

    struct Buffers {
        Buffers(pgl_point2i resolution): numPixels(resolution.x*resolution.y) {
            contribution = new pgl_vec3f[numPixels];
            secondMoment = new pgl_vec3f[numPixels];

            albedo = new pgl_vec3f[numPixels];
            normal = new pgl_vec3f[numPixels];
            spp = new float[numPixels];

            filteredContribution = new pgl_vec3f[numPixels];
            filteredSecondMoment = new pgl_vec3f[numPixels];
        } 
        
        ~Buffers() {
            delete[] contribution;
            delete[] secondMoment;

            delete[] albedo;
            delete[] normal;
            delete[] spp;

            delete[] filteredContribution;
            delete[] filteredSecondMoment;
        }

        int numPixels {0};

        pgl_vec3f* contribution {nullptr};
        pgl_vec3f* secondMoment {nullptr};

        pgl_vec3f* albedo {nullptr};
        pgl_vec3f* normal {nullptr};
        float* spp {nullptr};

        pgl_vec3f* filteredContribution {nullptr};
        pgl_vec3f* filteredSecondMoment {nullptr};
    };

    ImageSpaceGuidingBuffer(pgl_point2i resolution, bool useSecondMoment): m_useSecondMoment(useSecondMoment), m_resolution(resolution)
    {
        m_denoiser = new Denoiser(m_resolution, false);
        m_contributionEstimateBuffers = new Buffers(m_resolution);
        m_ready = false;
    }

    ImageSpaceGuidingBuffer(const std::string& fileName)
    {
        EXRVersion exrVersion;
        EXRHeader exrHeader;
        EXRImage exrImage;

        InitEXRHeader(&exrHeader);
        InitEXRImage(&exrImage);
        
        if (IsEXR(fileName.c_str()) == TINYEXR_SUCCESS)
        {
            m_ready = true;
            
            ParseEXRVersionFromFile(&exrVersion, fileName.c_str());
            ParseEXRHeaderFromFile(&exrHeader, &exrVersion,fileName.c_str(),nullptr);
            LoadEXRImageFromFile(&exrImage, &exrHeader, fileName.c_str(), nullptr);

            m_resolution.x = exrImage.width;
            m_resolution.y = exrImage.height;

            const size_t numPixels = m_resolution.x * m_resolution.y;

            m_denoiser = new Denoiser(m_resolution, false);
            m_contributionEstimateBuffers = new Buffers(m_resolution);

            std::vector<std::string> layer_names;
            tinyexr::GetLayers(exrHeader, layer_names);

            for (int i = 0; i< layer_names.size(); i++)
            {
                std::string layerName = layer_names[i];
                std::vector<tinyexr::LayerChannel> channels;
                tinyexr::ChannelsInLayer(
                    exrHeader, layerName, channels);

                int nChannels = channels.size();
                int idxR = -1;
                int idxG = -1;
                int idxB = -1;
                //int idxA = -1;

                int idxY = -1;

                for ( int j = 0; j < nChannels; j++)
                {
                    const tinyexr::LayerChannel &ch = channels[j];
                    if (ch.name == "R") {
                        idxR = int(ch.index);
                    } else if (ch.name == "G") {
                        idxG = int(ch.index);
                    } else if (ch.name == "B") {
                        idxB = int(ch.index);
                    //} else if (ch.name == "A") {
                    //    idxA = int(ch.index);
                    } else if (ch.name == "Y") {
                        idxY = int(ch.index);
                    }
                }


                float* bufferPtr = nullptr;
                if(layerName == "Contrib"){
                    bufferPtr = (float*) m_contributionEstimateBuffers->filteredContribution;
                }
                else if(layerName == "Contribt2nd")
                {
                    bufferPtr = (float*) m_contributionEstimateBuffers->filteredSecondMoment;
                }
                else if(layerName == "Spp")
                {
                    bufferPtr = (float*) m_contributionEstimateBuffers->spp;                   
                }
                else if(layerName == "ContribRaw")
                {
                    bufferPtr = (float*) m_contributionEstimateBuffers->contribution;
                }
                else if(layerName == "Contribt2ndRaw")
                {
                    bufferPtr = (float*) m_contributionEstimateBuffers->secondMoment;                
                }
                else if(layerName == "Albedo")
                {
                    bufferPtr = (float*) m_contributionEstimateBuffers->albedo;                   
                }
                else if(layerName == "N")
                {
                    bufferPtr = (float*) m_contributionEstimateBuffers->normal;                  
                }
                else
                {
                    
                }

#ifdef USE_EMBREE_PARALLEL
                embree::parallel_for(0,(int)numPixels, 1, [&] ( const embree::range<unsigned>& r ) {
                for (size_t pIdx=r.begin(); pIdx<r.end(); pIdx++)
#else
                tbb::parallel_for( tbb::blocked_range<int>(0,numPixels), [&](tbb::blocked_range<int> r)
                {
                for (int pIdx = r.begin(); pIdx<r.end(); ++pIdx)
#endif
                {
                    switch(nChannels){
                        case 1: {
                            const float val =
                            reinterpret_cast<float **>(exrImage.images)[idxY][pIdx];
                            
                            bufferPtr[pIdx*nChannels + 0] = val;
                            break;
                        }
                        case 3: {
                            const float valR =
                            reinterpret_cast<float **>(exrImage.images)[idxR][pIdx];
                            const float valG =
                            reinterpret_cast<float **>(exrImage.images)[idxG][pIdx];
                            const float valB =
                            reinterpret_cast<float **>(exrImage.images)[idxB][pIdx];
                            
                            bufferPtr[pIdx*nChannels + 0] = valR;
                            bufferPtr[pIdx*nChannels + 1] = valG;
                            bufferPtr[pIdx*nChannels + 2] = valB;
                            break;
                        }
                    }
                }
                });
            }
            m_ready = true;
        } 
        else 
        {
            m_ready = false;
        }
    }

    ~ImageSpaceGuidingBuffer()
    {
        delete m_denoiser;
        delete m_contributionEstimateBuffers;
    }

    void store(const std::string& fileName) const 
    {
        EXRHeader exrHeader;
        EXRImage exrImage;

        InitEXRHeader(&exrHeader);
        InitEXRImage(&exrImage);

        int width = m_resolution.x;
        int height = m_resolution.y;

        std::vector<int> numChannels;
        std::vector<tinyexr::LayerChannel> layerChannels;
        std::vector<const float*> channelValues;
        int cIdx = layerChannels.size();
        layerChannels.emplace_back(cIdx,"Contrib");
        numChannels.emplace_back(3);
        channelValues.emplace_back((const float*)m_contributionEstimateBuffers->filteredContribution);

        cIdx = layerChannels.size();
        layerChannels.emplace_back(cIdx,"Contribt2nd");
        numChannels.emplace_back(3);
        channelValues.emplace_back((const float*)m_contributionEstimateBuffers->filteredSecondMoment);

        cIdx = layerChannels.size();
        layerChannels.emplace_back(cIdx,"Spp");
        numChannels.emplace_back(1);
        channelValues.emplace_back((const float*)m_contributionEstimateBuffers->spp);

        cIdx = layerChannels.size();
        layerChannels.emplace_back(cIdx,"ContribRaw");
        numChannels.emplace_back(3);
        channelValues.emplace_back((const float*)m_contributionEstimateBuffers->contribution);

        cIdx = layerChannels.size();
        layerChannels.emplace_back(cIdx,"Contribt2ndRaw");
        numChannels.emplace_back(3);
        channelValues.emplace_back((const float*)m_contributionEstimateBuffers->secondMoment);

        cIdx = layerChannels.size();
        layerChannels.emplace_back(cIdx,"Albedo");
        numChannels.emplace_back(3);
        channelValues.emplace_back((const float*)m_contributionEstimateBuffers->albedo);

        cIdx = layerChannels.size();
        layerChannels.emplace_back(cIdx,"N");
        numChannels.emplace_back(3);
        channelValues.emplace_back((const float*)m_contributionEstimateBuffers->normal);

        int totalNumLayers = layerChannels.size();
        int totalNumChannels = 0;
        for (int i = 0; i < numChannels.size(); i++)
        {
            totalNumChannels += numChannels[i];
        }

        exrHeader.num_channels = totalNumChannels;
        exrImage.num_channels = totalNumChannels;
        exrImage.width = width;
        exrImage.height = height;

        const size_t numPixels = width * height;

        // Convert image data from AoS to SoA (i.e. one image per channel in each layer)
        std::vector<std::vector<float>> channelImages;
        for (int i = 0; i < totalNumLayers; i++)
        {
            for(int j=0; j< numChannels[i]; j++)
            {
                channelImages.emplace_back(width * height);
            }
            int offset = channelImages.size() - numChannels[i];

#ifdef USE_EMBREE_PARALLEL
            embree::parallel_for(0,(int)numPixels, 1, [&] ( const embree::range<unsigned>& r ) {
            for (size_t pIdx=r.begin(); pIdx<r.end(); pIdx++)
#else
            tbb::parallel_for( tbb::blocked_range<int>(0,numPixels), [&](tbb::blocked_range<int> r)
            {
            for (int pIdx = r.begin(); pIdx<r.end(); ++pIdx)
#endif
            {
                for(int c = 0; c < numChannels[i]; c++)
                {
                    channelImages[offset+c][pIdx] = channelValues[i][(pIdx)*numChannels[i] + c];
                }
            }
            });
        }

        // Set the channel names
        std::vector<EXRChannelInfo> channels(totalNumChannels);
        int offset = 0;
        for (int lay = 0; lay < totalNumLayers; ++lay) {
            char namePrefix[256];
            std::size_t prefixLen = 0;
            const char* layerName = layerChannels[lay].name.c_str();
            if (!layerName || layerName[0] == 0) {
                prefixLen = 0;
                namePrefix[0] = 0;
            } else {
                prefixLen = strlen(layerName) + 1;
                strncpy(namePrefix, layerName, 255);
                namePrefix[prefixLen - 1] = '.';
            }

            if (numChannels[lay] == 1) {
                strncpy(channels[offset + 0].name, namePrefix, 255);
                strncpy(channels[offset + 0].name + prefixLen, "Y", 255 - prefixLen);
            } else if (numChannels[lay] == 3) {
                strncpy(channels[offset + 0].name, namePrefix, 255);
                strncpy(channels[offset + 0].name + prefixLen, "R", 255 - prefixLen);

                strncpy(channels[offset + 1].name, namePrefix, 255);
                strncpy(channels[offset + 1].name + prefixLen, "G", 255 - prefixLen);

                strncpy(channels[offset + 2].name, namePrefix, 255);
                strncpy(channels[offset + 2].name + prefixLen, "B", 255 - prefixLen);
            /*
            } else if (numChannels[lay] == 4) {
                strncpy(channels[offset + 0].name, namePrefix, 255);
                strncpy(channels[offset + 0].name + prefixLen, "R", 255 - prefixLen);

                strncpy(channels[offset + 1].name, namePrefix, 255);
                strncpy(channels[offset + 1].name + prefixLen, "G", 255 - prefixLen);

                strncpy(channels[offset + 2].name, namePrefix, 255);
                strncpy(channels[offset + 2].name + prefixLen, "B", 255 - prefixLen);

                strncpy(channels[offset + 3].name, namePrefix, 255);
                strncpy(channels[offset + 3].name + prefixLen, "A", 255 - prefixLen);
            */
            } else {
                std::cerr << "ERROR while writing " << fileName
                        << ": images with " << totalNumChannels << " channels are currently not supported. "
                        << "no file has been written." << std::endl;
                return;
            }

            offset += numChannels[lay];
        }

        // Sort channels by the ASCII byte code of their names because thats what OpenEXR expects
        std::vector<int> channelIndices;
        for (int i = 0; i < totalNumChannels; ++i) channelIndices.emplace_back(i);
        std::sort(channelIndices.begin(), channelIndices.end(), [&channels] (int a, int b) {
            return strcmp(channels[a].name, channels[b].name) < 0;
        });

        std::vector<EXRChannelInfo> sortedChannels(totalNumChannels);
        float** imagePtr = (float **) alloca(sizeof(float*) * exrImage.num_channels);
        for (int i = 0; i < totalNumChannels; ++i) {
            sortedChannels[i] = channels[channelIndices[i]];
            imagePtr[i] = channelImages[channelIndices[i]].data();
        }
        exrHeader.channels = sortedChannels.data();
        exrImage.images = (unsigned char**)imagePtr;

        bool writeHalf = false;
        // Define pixel type of the buffer and requested output pixel type in the file
        exrHeader.pixel_types = (int*) alloca(sizeof(int) * exrHeader.num_channels);
        exrHeader.requested_pixel_types = (int*) alloca(sizeof(int) * exrHeader.num_channels);
        for (int i = 0; i < exrHeader.num_channels; i++) {
            exrHeader.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
            exrHeader.requested_pixel_types[i] = writeHalf ? TINYEXR_PIXELTYPE_HALF : TINYEXR_PIXELTYPE_FLOAT;
        }

        const char* errorMsg = nullptr;
        int retCode = 0;
        retCode = SaveEXRImageToFile(&exrImage, &exrHeader, fileName.c_str(), &errorMsg);
        if (retCode != TINYEXR_SUCCESS) {
            std::cerr << "TinyEXR error (" << retCode << "): " << errorMsg << std::endl;
            FreeEXRErrorMessage(errorMsg);
        }
    }

    void update()
    {
        if(m_useSecondMoment)
        {
            m_denoiser->denoise(m_contributionEstimateBuffers->contribution, m_contributionEstimateBuffers->secondMoment, m_contributionEstimateBuffers->normal, m_contributionEstimateBuffers->albedo, m_contributionEstimateBuffers->filteredContribution, m_contributionEstimateBuffers->filteredSecondMoment);
        }
        else
        {
            m_denoiser->denoise(m_contributionEstimateBuffers->contribution, m_contributionEstimateBuffers->normal, m_contributionEstimateBuffers->albedo, m_contributionEstimateBuffers->filteredContribution);
        }
        m_ready = true;
    }

    void addSample(const pgl_point2i pixel, const PGLImageSpaceSample& sample)
    {
        std::size_t pixelIdx = pixel.y * m_resolution.x + pixel.x;
        m_contributionEstimateBuffers->spp[pixelIdx] += 1;
        float alpha = 1.f / m_contributionEstimateBuffers->spp[pixelIdx];

        m_contributionEstimateBuffers->contribution[pixelIdx] = (1.f - alpha) * m_contributionEstimateBuffers->contribution[pixelIdx] + alpha * sample.color;
        m_contributionEstimateBuffers->albedo[pixelIdx] = (1.f - alpha) * m_contributionEstimateBuffers->albedo[pixelIdx] + alpha * sample.albedo;
        m_contributionEstimateBuffers->normal[pixelIdx] = (1.f - alpha) * m_contributionEstimateBuffers->normal[pixelIdx] + alpha * sample.normal;
        m_contributionEstimateBuffers->secondMoment[pixelIdx] = (1.f - alpha) * m_contributionEstimateBuffers->secondMoment[pixelIdx] + alpha * (sample.color * sample.color);
    }

    pgl_vec3f getContributionEstimate(const pgl_point2i pixel, const bool secondMoment = false) const{
        std::size_t pixelIdx = pixel.y * m_resolution.x + pixel.x;
        const pgl_vec3f c = !secondMoment ? m_contributionEstimateBuffers->filteredContribution[pixelIdx] : m_contributionEstimateBuffers->filteredSecondMoment[pixelIdx];
        return c;
    }

    bool isReady() const
    {
        return m_ready;
    }

    private:
    bool m_ready {false};
    bool m_useSecondMoment {false};
    pgl_point2i m_resolution;
    Denoiser* m_denoiser;

    Buffers *m_contributionEstimateBuffers {nullptr};
};

}