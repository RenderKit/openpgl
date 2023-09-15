// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

#define OPENPGL_INLINE_INTER inline

#define OPENPGL_HISTOGRAM_RESOLUTION 8
#define OPENPGL_HISTOGRAM_SIZE OPENPGL_HISTOGRAM_RESOLUTION * OPENPGL_HISTOGRAM_RESOLUTION

namespace openpgl
{

struct OutgoingRadianceHistogram
{
    OutgoingRadianceHistogram();
    
    openpgl::Vector3 getOugoingRadiance(openpgl::Vector3 dir) const;

    void addSample(openpgl::Vector3 dir, openpgl::Vector3 sample);

    void decay(const float alpha);

    void update(const SampleData* samples, const size_t numSamples);

    void serialize(std::ostream& stream) const;
        
    void deserialize(std::istream& stream);
private:

    openpgl::Point2 dirToCanonical(const openpgl::Vector3& d) const;

private:
    openpgl::Vector3 data[OPENPGL_HISTOGRAM_SIZE];
    float numSamples[OPENPGL_HISTOGRAM_SIZE];

};

OPENPGL_INLINE_INTER OutgoingRadianceHistogram::OutgoingRadianceHistogram()
{
    for(int i=0; i< OPENPGL_HISTOGRAM_SIZE;i++)
    {
        data[i] = openpgl::Vector3(0.f);
        numSamples[i]= 0.f;
    }
}

OPENPGL_INLINE_INTER openpgl::Vector3 OutgoingRadianceHistogram::getOugoingRadiance(openpgl::Vector3 dir) const
{
    const openpgl::Point2 p = dirToCanonical(dir);
    const int res = OPENPGL_HISTOGRAM_RESOLUTION;
    const int histIdx =
        std::min(int(p.x * res), res - 1) +
        std::min(int(p.y * res), res - 1) * res;
    return data[histIdx];
}

OPENPGL_INLINE_INTER void OutgoingRadianceHistogram::addSample(Vector3 dir, openpgl::Vector3 sample)
{
    const openpgl::Point2 p = dirToCanonical(dir);
    const int res = OPENPGL_HISTOGRAM_RESOLUTION;
    const int histIdx =
        std::min(int(p.x * res), res - 1) +
        std::min(int(p.y * res), res - 1) * res;
    numSamples[histIdx] += 1.f;
    data[histIdx] += (sample-data[histIdx]) / numSamples[histIdx];
}

OPENPGL_INLINE_INTER void OutgoingRadianceHistogram::decay(const float alpha){
    for(int i=0; i< OPENPGL_HISTOGRAM_SIZE; i++)
    {
        numSamples[i]*= alpha;
    }
}

OPENPGL_INLINE_INTER openpgl::Point2 OutgoingRadianceHistogram::dirToCanonical(const openpgl::Vector3& d) const {
    if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
        return {0, 0};
    }

    const float cosTheta = std::min(std::max(d.z, -1.0f), 1.0f);
    float phi = std::atan2(d.y, d.x);
    while (phi < 0)
        phi += 2.0f * M_PI;

    return {(cosTheta + 1.f) / 2.f, phi / (2.f * M_PI)};
}


OPENPGL_INLINE_INTER void OutgoingRadianceHistogram::update(const SampleData* samples, const size_t numSamples)
{
    for(int n = 0; n < numSamples; n++)
    {
        Vector3 dir = Vector3(samples[n].directionOut.x, samples[n].directionOut.y, samples[n].directionOut.z);
        Vector3 col = Vector3(samples[n].radianceOut.x, samples[n].radianceOut.y, samples[n].radianceOut.z);
        addSample(dir, col);
    }
    /*
    std::cout << "OutgoingRadianceHistogram: ";
    for (int i = 0; i < OPENPGL_HISTOGRAM_SIZE; i++)
    {
        std::cout << data[i].x << ", " << data[i].y << ", " << data[i].z << "\t\t";
    }
    std::cout << std::endl;
    */
}

OPENPGL_INLINE_INTER void OutgoingRadianceHistogram::serialize(std::ostream& stream) const
{
    stream.write(reinterpret_cast<const char*>(&data), sizeof(openpgl::Vector3) * OPENPGL_HISTOGRAM_SIZE);
    stream.write(reinterpret_cast<const char*>(&numSamples), sizeof(float) * OPENPGL_HISTOGRAM_SIZE);
}
    
OPENPGL_INLINE_INTER void OutgoingRadianceHistogram::deserialize(std::istream& stream)
{
    stream.read(reinterpret_cast<char*>(&data), sizeof(openpgl::Vector3) * OPENPGL_HISTOGRAM_SIZE);
    stream.read(reinterpret_cast<char*>(&numSamples), sizeof(float) * OPENPGL_HISTOGRAM_SIZE);
}
}