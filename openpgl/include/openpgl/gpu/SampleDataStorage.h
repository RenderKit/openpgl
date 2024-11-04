#include "Device.h"
#include "../cpp/SampleData.h"
#include "SampleDataStorage_soa.h"

struct SampleDataStorageBuffer: public SOA<SampleDataStorage> {
public:
    SampleDataStorageBuffer() = default;
    SampleDataStorageBuffer(int n, openpgl::gpu::Device *device) : SOA<SampleDataStorage>(n, device){}

    OPENPGL_GPU_CALLABLE
    void AddSampleData(const int pixelIndex, const Point3f& position, const Vector3f& direction, const Float pdf, const Float distance, const Vector3f contribution, const bool volume)
    {
        uint nSample = nSamples[pixelIndex];
        //if(nSample < 10) 
        {
            uint idx = nSample;
            uint flags = 0;
            flags = volume ? flags | openpgl::cpp::SampleData::Flags::EInsideVolume : flags;
            SampleData sd = {flags, position, direction, ((contribution[0] + contribution[1] + contribution[2]) / 3.f)/pdf, pdf, distance};
            //samples[pixelIndex][idx] = sd;
            samples[idx][pixelIndex] = sd;
            /*
            samples[idx].position[pixelIndex] = position;
            samples[idx].direction[pixelIndex] = direction;
            samples[idx].weight[pixelIndex] = (contribution[0] + contribution[1] + contribution[2]) / 3.f;
            samples[idx].pdf[pixelIndex] = pdf;
            samples[idx].distance[pixelIndex] = distance;
            samples[idx].flags[pixelIndex] = flags;
            */
            nSamples[pixelIndex] = nSample + 1;
            
        }
    }

    OPENPGL_GPU_CALLABLE
    void AddZeroValueSampleData(const int pixelIndex, const Point3f& position, const Vector3f& direction, const bool volume){
        uint nZVSample = nZVSamples[pixelIndex];
        //if(nZVSample < 10) 
        {
            uint idx = nZVSample;
            ZeroValueSampleData zvSD = {volume, position, direction};
            //zvSamples[pixelIndex][idx] = zvSD;
            zvSamples[idx][pixelIndex] = zvSD;
            /*
            zvSamples[idx].volume[pixelIndex] = volume;
            zvSamples[idx].position[pixelIndex] = position;
            zvSamples[idx].direction[pixelIndex] = direction;
            */
            nZVSamples[pixelIndex] = nZVSample + 1;
        }
    }

    OPENPGL_GPU_CALLABLE
    void Reset(const int pixelIndex) {
        nSamples[pixelIndex] = 0;
        nZVSamples[pixelIndex] = 0;
    }

    void CollectSampleData(openpgl::cpp::SampleStorage& sampleStorage){
        int maxQueueSize = this->nAlloc;
        Timer timer;
        ParallelFor(0, maxQueueSize, [&](int pixelIndex)
        {
            uint nSamples = this->nSamples[pixelIndex];
            for (int n = 0; n < nSamples; n++) {
                SampleData sd = this->samples[n][pixelIndex];
                openpgl::cpp::SampleData pg_sd;
                pg_sd.position = {sd.position.x, sd.position.y, sd.position.z};
                pg_sd.direction ={sd.direction.x, sd.direction.y, sd.direction.z};
                pg_sd.weight = sd.weight;
                pg_sd.pdf = sd.pdf;
                pg_sd.distance = sd.distance;
                pg_sd.flags = sd.flags;
                sampleStorage.AddSample(pg_sd);
            }

            uint nZeroValueSamples = this->nZVSamples[pixelIndex];
            for (int n = 0; n < nZeroValueSamples; n++) {
                uint idx = (n*this->nAlloc) + n;
                ZeroValueSampleData zvsd = this->zvSamples[n][pixelIndex];
                openpgl::cpp::ZeroValueSampleData pg_zvsd;
                pg_zvsd.position = {zvsd.position.x, zvsd.position.y, zvsd.position.z};
                pg_zvsd.direction = {zvsd.direction.x, zvsd.direction.y, zvsd.direction.z};
                pg_zvsd.volume = zvsd.volume;
                sampleStorage.AddZeroValueSample(pg_zvsd); 
            }
        }
        );
        std::cout << std::endl << "CollectSampleData: time(sec) = " << timer.ElapsedSeconds() << std::endl;
    }
};