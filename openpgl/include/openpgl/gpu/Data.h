#pragma once

namespace openpgl{
namespace gpu{
    template<int maxComponents> 
    struct FlatVMM {
        public:
        float _weights[maxComponents];
        float _kappas[maxComponents];
        float _meanDirections[maxComponents][3];
        float _distances[maxComponents];
        float _pivotPosition[3];
        int _numComponents{maxComponents};
    };

    struct FieldData {
        int m_numSurfaceTreeLets;
        int m_numVolumeTreeLets;

        void *m_surfaceTreeLets;
        void *m_volumeTreeLets;

        int m_numSurfaceDistributions;
        int m_numVolumeDistributions;

        void *m_surfaceDistributions;
        void *m_volumeDistributions;
    };

} // gpu
} // openpgl