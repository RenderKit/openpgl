#pragma once

#ifndef OPENPGL_BUILD
#include <openpgl/defines.h>
#endif

#include "Common.h"

#define OPENPGL_GPU_HISTOGRAM_RESOLUTION 8
#define OPENPGL_GPU_HISTOGRAM_SIZE OPENPGL_GPU_HISTOGRAM_RESOLUTION *OPENPGL_GPU_HISTOGRAM_RESOLUTION

namespace openpgl
{
namespace gpu
{
template <int maxComponents>
struct FlatVMM
{
   public:
    float _weights[maxComponents];
    float _kappas[maxComponents];
    float _meanDirections[maxComponents][3];
    float _distances[maxComponents];
    float _pivotPosition[3];
    int _numComponents{maxComponents};
#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
    float _fluenceRGBWeights[maxComponents][3];
    float _fluenceRGB[3];
#endif
};

struct FieldData
{
    int m_numSurfaceTreeLets;
    int m_numVolumeTreeLets;

    void *m_surfaceTreeLets;
    void *m_volumeTreeLets;

    int m_numSurfaceDistributions;
    int m_numVolumeDistributions;

    void *m_surfaceDistributions;
    void *m_volumeDistributions;

    int m_numPhaseFunctionRepresentations;
    void *m_phaseFunctionRepresentations;

#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
    void *m_surfaceOutgoingRadianceHistogram;
    void *m_volumeOutgoingRadianceHistogram;
#endif
};

struct SurfaceSamplingDistributionData
{
    OPENPGL_GPU_CALLABLE SurfaceSamplingDistributionData() = default;
    const void *m_field{nullptr};
    pgl_point3f m_pos{0.f, 0.f, 0.f};
    int m_idx{-1};
};

struct VMMPhaseFunctionRepresentationData
{
    float g = 0.f;
    float meanCosines[4]{0.f, 0.f, 0.f, 0.f};
    float weights[4]{1.f / 4.f, 1.f / 4.f, 1.f / 4.f, 1.f / 4.f};
};

struct VolumeSamplingDistributionData
{
    OPENPGL_GPU_CALLABLE VolumeSamplingDistributionData() = default;
    const void *m_field{nullptr};
    pgl_point3f m_pos{0.f, 0.f, 0.f};
    int m_idx{-1};
    VMMPhaseFunctionRepresentationData m_phaseRep;
};

#if defined(OPENPGL_EF_RADIANCE_CACHES) || defined(OPENPGL_RADIANCE_CACHES)
struct OutgoingRadianceHistogramData
{
    float data[OPENPGL_GPU_HISTOGRAM_SIZE][3];
    // float numSamples[OPENPGL_HISTOGRAM_SIZE];
};
#endif
}  // namespace gpu
}  // namespace openpgl