#include "Device.h"

struct PathSegment{
    bool isDelta;
    bool volumeScatter;
    Point3 position;
    Normal3 normal;
    Vector3 scatterWeight;
    Vector3 directionIn;
    float pdfDirectionIn;
    Vector3 directionOut;
    Vector3 transmittanceWeight;
    Vector3 directContribution;
    Vector3 scatteredContribution;
    float miWeight;
    float rrProbability;
    float eta;
    float roughness;
};

struct PathSegmentStorage{
    PathSegment segments[10+1];
    uint32_t curDepth;
};

#include "PathSegmentStorage_soa.h"

struct PathSegmentStorageBuffer: public SOA<PathSegmentStorage> {

public:
    PathSegmentStorageBuffer() = default;
    PathSegmentStorageBuffer(int n, openpgl::gpu::Device *device) : SOA<PathSegmentStorage>(n, device){
        //std::cout << "PathSegmentStorage(int n, openpgl::gpu::Device *device)" << std::endl;
    }

    OPENPGL_GPU_CALLABLE
    void Reset(const int pixelIndex) {
        curDepth[pixelIndex] = 0;
        /*
        for (int idx = 0; idx < 10; idx++) {
            segments[idx].scatteredContribution[pixelIndex] = {0.f, 0.f, 0.f};
            segments[idx].directContribution[pixelIndex] = {0.f, 0.f, 0.f};
        }
        */
    }
};