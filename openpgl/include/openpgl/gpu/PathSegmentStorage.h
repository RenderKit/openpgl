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

    PathSegmentStorageBuffer(std::string fileName, openpgl::gpu::Device *device)
    {
        std::filebuf fb;
        fb.open(fileName, std::ios::in | std::ios::binary);
        if (!fb.is_open())
            throw std::runtime_error("error: couldn't open file");
        std::istream is(&fb);
        
        uint32_t n;
        bool managed;
        
        is.read(reinterpret_cast<char *>(&n), sizeof(n));
        is.read(reinterpret_cast<char *>(&managed), sizeof(bool));

        new (this) SOA<PathSegmentStorage>(n, device, managed);

        SampleData* stored_samples = new SampleData[(10+1)*n];
        uint32_t* stored_nSamples = new uint32_t[n];
        ZeroValueSampleData* stored_zvSamples = new ZeroValueSampleData[(10+1)*n];
        uint32_t* stored_nZVSamples = new uint32_t[n];
    
        is.read(reinterpret_cast<char *>(stored_samples), sizeof(SampleData) * (n * (10+1)));
        is.read(reinterpret_cast<char *>(stored_nSamples), sizeof(uint32_t) * n);
        is.read(reinterpret_cast<char *>(stored_zvSamples), sizeof(ZeroValueSampleData) * (n * (10+1)));
        is.read(reinterpret_cast<char *>(stored_nZVSamples), sizeof(uint32_t) * n);


        device->memcpyArrayToGPU<uint32_t>(nSamples, stored_nSamples, n);
        device->memcpyArrayToGPU<uint32_t>(nZVSamples, stored_nZVSamples, n);
        for (int i=0; i < 10+1; i++) {
            device->memcpyArrayToGPU<SampleData>(samples[i], &(stored_samples[i*n]), n);
            device->memcpyArrayToGPU<ZeroValueSampleData>(zvSamples[i], &(stored_zvSamples[i*n]), n);
        }
        device->wait();

        fb.close();

        delete[] stored_samples;
        delete[] stored_nSamples;
        delete[] stored_zvSamples;
        delete[] stored_nZVSamples;
    }

    OPENPGL_GPU_CALLABLE
    void PropagateSamples(const int pixelIndex, SampleDataStorageBuffer* sampleDataStorageBuffer) {
        uint32_t depth = curDepth[pixelIndex];
        Vector3 contrbution(0.f, 0.f , 0.f);

        Vector3 scatterWeights[10+1];
        Point3 positions[10+1];
        Vector3 directions[10+1];
        float pdfs[10+1];
        Vector3 contributions[10+1];

        for (int n = 0; n < depth; n++) {
            scatterWeights[n] = segments[n].scatterWeight[pixelIndex];
            positions[n] = segments[n].position[pixelIndex];
            directions[n] = segments[n].directionIn[pixelIndex];
            pdfs[n] = segments[n].pdfDirectionIn[pixelIndex];
            contributions[n] = segments[n].scatteredContribution[pixelIndex];
            contributions[n] += segments[n].directContribution[pixelIndex];
        }

        for (int n = depth-2; n >= 0; n--) {

            float distance = 1e6f;
            contrbution += contributions[n+1];
            //contrbution += contributions[n+1];
            if (contrbution[0] > 0.f || contrbution[1] > 0.f || contrbution[2] > 0.f) {
                sampleDataStorageBuffer->AddSampleData(pixelIndex, positions[n], directions[n], pdfs[n], distance, contrbution, false);
            }
            //else 
            //{
            ////} else if (contrbution[0] == -1.f){
            //    sampleDataStorageBuffer.AddZeroValueSampleData(pixelIndex, pos, direction, false);
            //}
            contrbution[0] *= scatterWeights[n][0];
            contrbution[1] *= scatterWeights[n][1];
            contrbution[2] *= scatterWeights[n][2];
        }
    }

    OPENPGL_GPU_CALLABLE
    void Reset(const int pixelIndex) {
        curDepth[pixelIndex] = 0;
    }
};