// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"
#include "Common.h"

#include <string>

namespace openpgl
{
namespace cpp
{
namespace util
{

struct ImageSpaceGuidingBuffer
{
    typedef PGLImageSpaceSample Sample;
    
    ImageSpaceGuidingBuffer(const Point2i resolution);

    ImageSpaceGuidingBuffer(const std::string& fileName);

    ~ImageSpaceGuidingBuffer();

    void Update();

    void AddSample(const Point2i pixel, const Sample& sample);

    void Store(const std::string& fileName) const;

    Vector3f GetContributionEstimate(const Point2i pixel) const;

    bool IsReady() const;

    private:
        PGLImageSpaceGuidingBuffer m_imageSpaceGuidingBufferHandle{nullptr};
};

OPENPGL_INLINE ImageSpaceGuidingBuffer::ImageSpaceGuidingBuffer(const Point2i resolution)
{
    m_imageSpaceGuidingBufferHandle = pglFieldNewImageSpaceGuidingBuffer(resolution);
}

OPENPGL_INLINE ImageSpaceGuidingBuffer::ImageSpaceGuidingBuffer(const std::string& fileName)
{
    m_imageSpaceGuidingBufferHandle = pglFieldNewImageSpaceGuidingBufferFromFile(fileName.c_str());
}

OPENPGL_INLINE ImageSpaceGuidingBuffer::~ImageSpaceGuidingBuffer()
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    if(m_imageSpaceGuidingBufferHandle)
        pglReleaseImageSpaceGuidingBuffer(m_imageSpaceGuidingBufferHandle);
    m_imageSpaceGuidingBufferHandle = nullptr;
}

OPENPGL_INLINE void ImageSpaceGuidingBuffer::Update()
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    pglImageSpaceGuidingBufferUpdate(m_imageSpaceGuidingBufferHandle);
}

OPENPGL_INLINE void ImageSpaceGuidingBuffer::AddSample(const Point2i pixel, const Sample& sample)
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    pglImageSpaceGuidingBufferAddSample(m_imageSpaceGuidingBufferHandle, pixel, sample);
}

OPENPGL_INLINE void ImageSpaceGuidingBuffer::Store(const std::string& fileName) const
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    pglImageSpaceGuidingBufferStore(m_imageSpaceGuidingBufferHandle, fileName.c_str());
}

OPENPGL_INLINE Vector3f ImageSpaceGuidingBuffer::GetContributionEstimate(const Point2i pixel) const
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    return pglImageSpaceGuidingBufferGetContributionEstimate(m_imageSpaceGuidingBufferHandle, pixel);
}

OPENPGL_INLINE bool ImageSpaceGuidingBuffer::IsReady() const
{
    OPENPGL_ASSERT(m_imageSpaceGuidingBufferHandle);
    return pglImageSpaceGuidingBufferIsReady(m_imageSpaceGuidingBufferHandle);
}

}
}
}