// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//#ifdef __WIN32__
    #define _USE_MATH_DEFINES
//#endif

#include <string>
#include <list>
#include <fstream>
#include <iostream>

#include <algorithm>
#include <vector>
#include <cmath>
#include <random>

#include <regex>

#include <tbb/info.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <openpgl/cpp/OpenPGL.h>

#if defined(OPENPGL_GPU_SYCL)
    #include <sycl/sycl.hpp>
#endif
#include "../../openpgl/field/FieldGPU.h"

#include <string>
#include <type_traits>

#include "timer.h"

//Please include your own zlib-compatible API header before
//including `tinyexr.h` when you disable `TINYEXR_USE_MINIZ`
#define TINYEXR_USE_MINIZ 0
#include "zlib.h"
//Or, if your project uses `stb_image[_write].h`, use their
//zlib implementation:
//#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

// Setting up a device global variable for the Guiding Field
#if defined(OPENPGL_GPU_CUDA) && defined(__CUDACC__)
    __device__ __constant__ openpgl_gpu::FieldGPU global_device_field;
#elif defined(OPENPGL_GPU_SYCL)
    #include <sycl/sycl.hpp>
    sycl::ext::oneapi::experimental::device_global<const openpgl_gpu::FieldGPU> global_device_field;
#else
    openpgl_gpu::FieldGPU global_device_field;
#endif

// The CUDA function for running the test
#if defined(OPENPGL_GPU_CUDA) && defined(__CUDACC__)
__global__ void test(/*openpgl_gpu::FieldGPU field, openpgl_gpu::FieldGPU::Distribution* device_distributions,*/ pgl_vec3f* device_positionsSurface, int* device_ids, pgl_vec2f *device_random_samples, pgl_vec3f *device_random_directions, float *device_pdfs, int nSurfaceSamples)
{
    //openpgl_gpu::FieldGPU field(device_nodes);
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nSurfaceSamples)
    {
/*        
        float pos[3] = {device_positionsSurface[idx].x, device_positionsSurface[idx].y, device_positionsSurface[idx].z};
        device_ids[idx] = field.getDataIdxAtPos(pos);
        //device_random_directions[idx] = distributions[device_ids[idx]].sample(device_random_samples[idx]);
        //device_pdfs[idx] = distributions[device_ids[idx]].pdf(device_random_directions[idx]);
        device_random_directions[idx] = field.m_distributions[device_ids[idx]].samplePos(device_positionsSurface[idx], device_random_samples[idx]);
        device_pdfs[idx] = device_distributions[device_ids[idx]].pdfPos(device_positionsSurface[idx], device_random_directions[idx]);
*/
        float sample = -1.f;
        openpgl_gpu::SurfaceSamplingDistribution ssd;
        ssd.Init(&global_device_field, device_positionsSurface[idx], sample);
        device_ids[idx] = ssd.GetId();
        pgl_vec3f direction {0.f, 0.f, 0.f};
        device_pdfs[idx] = ssd.SamplePDF(device_random_samples[idx], direction);
        device_random_directions[idx] = direction;
    }    
}
#endif

enum BenchType{
    HELP=0,
    BENCH_LOOKUP_ID,
    NONE
};

inline bool file_exists(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());
    return file.good();
}

inline pgl_vec3f squareToUniformSphere(const float sampleX, const float sampleY){
    float z = 1.0f - 2.0f * sampleY;
    float r = std::sqrt(std::max(0.f,(1.0f - z*z)));
    float phi = 2.0f * M_PI * sampleX;
    float sinPhi = std::sin(phi);
    float cosPhi = std::cos(phi);
    pgl_vec3f sample;
    sample.x = r * cosPhi;
    sample.y = r * sinPhi;
    sample.z = z;
    return sample;
}

struct BenchParams {
    BenchType type {HELP};
    std::string field_file_name {""};
    std::string position_exr_file_name;
    std::string id_exr_file_name;
    //std::string out_file_name {""};

    unsigned int num_threads {0};

    PGL_DEVICE_TYPE device_type {PGL_DEVICE_TYPE_NONE};

    bool validate() {
        bool valid = true;

        switch(type) {
            case HELP:
                break;

            case BENCH_LOOKUP_ID:
                if(field_file_name == "" || 
                    !file_exists(field_file_name)) {
                    std::cout << "ERROR: Field file not set or does not exists: " << field_file_name << std::endl;
                    valid = false;
                }
                if(position_exr_file_name == "" || 
                    !file_exists(position_exr_file_name)) {
                    std::cout << "ERROR: Position EXR file not set or does not exists: " << position_exr_file_name << std::endl;
                    valid = false;
                }
                if(id_exr_file_name == "") {
                    std::cout << "ERROR: Id EXR output file not set " << std::endl;
                    valid = false;
                }
                if(device_type == PGL_DEVICE_TYPE_NONE){
                    std::cout << "ERROR: Device type not set." << std::endl;
                    valid = false;            
                }
                break;
            
            case NONE:
                valid = false;
                break;
            default:
                valid = false;
                break;
        }

        return valid;
    }
};

bool parseCommandLine(std::list<std::string> &args,
                        BenchParams &benchParams) {

    //bool collectSamples = false;
    
    for (auto it = args.begin(); it != args.end();) {
        const std::string arg = *it;

        if (arg == "-help") {
            //collectSamples = false; 
            benchParams.type = BenchType::HELP;
        } else if (arg == "-type") {
            //collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_type = *it;
                if(str_type == "benchLookUpId") {
                    benchParams.type = BenchType::BENCH_LOOKUP_ID;
                } else {
                    std::cout << "ERROR: Unknown type: " << str_type << std::endl;
                    std::cout << "       Valid types are: [benchLookUpId] "<< std::endl;
                    return false;
                }
            } else {
                return false;
            }
        } else if (arg == "-device") {
            //collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_type = *it;
                if(str_type == "CPU_4") {
                    benchParams.device_type = PGL_DEVICE_TYPE_CPU_4; 
                } else if(str_type == "CPU_8") {
                    benchParams.device_type = PGL_DEVICE_TYPE_CPU_8; 
                } else {
                    std::cout << "ERROR: Unknown device type: " << str_type << std::endl;
                    std::cout << "       Valid types are: [CPU_4 CPU_8] "<< std::endl;
                    return false;
                }
            } else {
                return false;
            }
        }else if (arg == "-field") {
            //collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_field = *it;
                benchParams.field_file_name = str_field;
            } else {
                return false;
            }
        } else if (arg == "-positionEXR") {
            //collectSamples = true; 
            ++it;
            if(it != args.end())
            {
                const std::string str_position_exr = *it;
                benchParams.position_exr_file_name = str_position_exr; 
            } else {
                return false;
            }
        } else if (arg == "-idEXR") {
            //collectSamples = true; 
            ++it;
            if(it != args.end())
            {
                const std::string str_id_exr = *it;
                benchParams.id_exr_file_name = str_id_exr; 
            } else {
                return false;
            }
        } else if (arg == "-threads") {
            ++it;
            if(it != args.end())
            {
                const std::string str_field = *it;
                benchParams.num_threads = std::stoi(str_field);
            } else {
                return false;
            }
        }
        ++it;
    }
    return true;
}

void print_help(){
    std::cout << "usage openpgl_gpu -type <benchLookUpId> [<options>]" << std::endl;
    std::cout << std::endl;
    std::cout << "type options:" << std::endl;
    std::cout << "  " << "benchLookUpId    " << "\t" << "Measures the average time of an initialization of a SurfaceSamplingDistribution" << std::endl;
    std::cout << "  " << "                 " << "\t" << "from a guiding Field. The positions form samples from a SampleStorage are used" << std::endl;
    std::cout << "  " << "                 " << "\t" << "during the initialization." << std::endl;
    std::cout << "  " << "                 " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                 " << "\t" << "\"openpgl_bench -type benchLookUp -samples ss0.st -field field.gf -device CPU_4\"" << std::endl;
    std::cout << std::endl;
    std::cout << "general options:" << std::endl;
    std::cout << "  -device <CPU_4 | CPU_8 | GPU_X>" << "\t SIMD width of the loaded Field or the Field that should be initialized." << std::endl;
    std::cout << "  -threads n             " << "\t Number of n threads that should be used during the measurements." << std::endl;
    std::cout << std::endl;
    std::cout << "benchLookUpId options:" << std::endl;
    std::cout << "  -field f0              " << "\t The stored Field object that is loaded for the test." << std::endl; 
    std::cout << "  -positionEXR p0        " << "\t The stored samples which positions are used for query/initialize" << std::endl;
    std::cout << "                         " << "\t the SurfaceSamplingDistributions."<< std::endl;
    std::cout << "  -idEXR i0              " << "\t The stored samples which positions are used for query/initialize" << std::endl;
    std::cout << "                         " << "\t the SurfaceSamplingDistributions."<< std::endl;
    std::cout << std::endl;
}

void bench_lookup_id(BenchParams &benchParams){
    std::uniform_real_distribution<float> distU(0.f, 1.f);
    int nThreads = benchParams.num_threads;

    int num_threads = 0;
    if (nThreads > 0){  
        num_threads = nThreads;
    }
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, num_threads);
    std::cout << "num_threads = " << num_threads << std::endl;
    
    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_4);
    openpgl::cpp::Field field(&device, benchParams.field_file_name);

    float* img; // width * height * RGBA
    int width;
    int height;
    const char* err = NULL; // or nullptr in C++11

    int ret = LoadEXR(&img, &width, &height, benchParams.position_exr_file_name.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
        if (err) {
            fprintf(stderr, "ERR : %s\n", err);
            FreeEXRErrorMessage(err); // release memory of error message.
        }
    } else {
        //free(out); // release memory of image data
    }
    //openpgl::cpp::SampleStorage sampleStorage(benchParams.samples_file_names[0]);

    std::cout << "Field::Validate: " << field.Validate() << std::endl;

    int nSurfaceSamples = width*height;
    openpgl_gpu::Device deviceGPU;
    int* ids = new int[nSurfaceSamples];
    int* host_ids = new int[nSurfaceSamples];
    int* device_ids = deviceGPU.mallocArray<int>(nSurfaceSamples);

    std::cout << "Prepare Data: START" << std::endl;
    pgl_vec3f* positionsSurface =  new pgl_vec3f [nSurfaceSamples];
    //pgl_vec3f* host_positionsSurface =  new pgl_vec3f [nSurfaceSamples];
    pgl_vec3f* device_positionsSurface =  deviceGPU.mallocArray<pgl_vec3f>(nSurfaceSamples);
    for (int i = 0; i < nSurfaceSamples; i++)
    {
        int idx = i;
        pgl_vec3f pos;
        pos.x = img[i*4+0];
        pos.y = img[i*4+1];
        pos.z = img[i*4+2];
        positionsSurface[idx] = pos;
    }
    deviceGPU.memcpyArrayToGPU(device_positionsSurface, positionsSurface, nSurfaceSamples);
    deviceGPU.wait();

    std::mt19937_64 rng(0);
    pgl_vec2f *random_samples = new pgl_vec2f[nSurfaceSamples];
    pgl_vec2f *device_random_samples = deviceGPU.mallocArray<pgl_vec2f>(nSurfaceSamples);
    pgl_vec3f *random_directions = new pgl_vec3f[nSurfaceSamples];
    pgl_vec3f *host_random_directions = new pgl_vec3f[nSurfaceSamples];
    pgl_vec3f *device_random_directions = deviceGPU.mallocArray<pgl_vec3f>(nSurfaceSamples);
    float *pdfs = new float[nSurfaceSamples];
    float *host_pdfs = new float[nSurfaceSamples];
    float *device_pdfs = deviceGPU.mallocArray<float>(nSurfaceSamples);

    for (int i = 0; i < nSurfaceSamples; i++)
    {
        random_samples[i].x = distU(rng);
        random_samples[i].y = distU(rng);
    }
    deviceGPU.memcpyArrayToGPU(device_random_samples, random_samples, nSurfaceSamples);

    std::cout << "Prepare Data: END" << std::endl;

    std::string foo;
    std::getline( std::cin, foo );

    std::cout << "Copying data to GPU\n";
    openpgl_gpu::KDTreeLet *device_nodes = deviceGPU.mallocArray<openpgl_gpu::KDTreeLet>(field.GetNumKDNodes());
    deviceGPU.memcpyArrayToGPU(device_nodes, (openpgl_gpu::KDTreeLet*)field.GetKdNodes(), field.GetNumKDNodes());

    openpgl_gpu::FieldGPU::Distribution *distributions = new openpgl_gpu::FieldGPU::Distribution[field.GetNumDistributions()];
    openpgl_gpu::FieldGPU::Distribution *device_distributions = deviceGPU.mallocArray<openpgl_gpu::FieldGPU::Distribution>(field.GetNumDistributions());
    field.CopyDistributions(distributions);
    deviceGPU.memcpyArrayToGPU(device_distributions, distributions, field.GetNumDistributions());
    // wait until all CPU -> GPU copies are done
    deviceGPU.wait();

    int num_nodes = field.GetNumKDNodes();
    std::cout << "Executing on GPU\n";
    openpgl_gpu::FieldGPU device_field(device_nodes, device_distributions);
#if defined(OPENPGL_GPU_SYCL) || defined(OPENPGL_GPU_CPU)
#if defined(OPENPGL_GPU_SYCL)
    deviceGPU.q.copy(&device_field, global_device_field).wait(); 
    deviceGPU.q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(nSurfaceSamples), [=] (sycl::id<1> i) {
            int idx = i.get(0);
#else
    global_device_field = device_field;
    tbb::parallel_for( tbb::blocked_range<int>(0,nSurfaceSamples), [&](tbb::blocked_range<int> r)
    {
        for (int n = r.begin(); n<r.end(); ++n)
        {
            int idx = n;
#endif                
            //for (int j = 0; j< 1000000; j++) 
            {
#if defined(OPENPGL_GPU_SYCL)
                idx = (i.get(0)+512) % nSurfaceSamples;
#else
                idx = (n+512) % nSurfaceSamples;
#endif
                float pos[3] = {device_positionsSurface[idx].x, device_positionsSurface[idx].y, device_positionsSurface[idx].z};
                /*
                device_ids[idx] = device_field.getDataIdxAtPos(pos);
                //device_random_directions[idx] = device_distributions[device_ids[idx]].sample(random_samples[idx]);
                //device_pdfs[idx] = device_distributions[device_ids[idx]].pdf(device_random_directions[idx]);
                device_random_directions[idx] = device_field.m_distributions[device_ids[idx]].samplePos(device_positionsSurface[idx], device_random_samples[idx]);
                device_pdfs[idx] = device_distributions[device_ids[idx]].pdfPos(device_positionsSurface[idx], device_random_directions[idx]);
                */
                float sample = -1.f;
                openpgl_gpu::SurfaceSamplingDistribution ssd;
#if defined(OPENPGL_GPU_SYCL)
                ssd.Init(&global_device_field.get(), device_positionsSurface[idx], sample);
#else
                ssd.Init(&global_device_field, device_positionsSurface[idx], sample);
#endif
                device_ids[idx] = ssd.GetId();
                pgl_vec3f direction {0.f, 0.f, 0.f};
                device_pdfs[idx] = ssd.SamplePDF(device_random_samples[idx], direction);
                device_random_directions[idx] = direction;
            }
#if defined(OPENPGL_GPU_SYCL)
        });
    });
#else
        }
    });
#endif
#elif defined(OPENPGL_GPU_CUDA) && defined(__CUDACC__)
    cudaMemcpyToSymbol(global_device_field, &device_field, sizeof(openpgl_gpu::FieldGPU), 0, cudaMemcpyHostToDevice);
    deviceGPU.wait();
    test<<< 1 +  nSurfaceSamples/256, 256>>>(/*device_field, device_distributions, */device_positionsSurface, device_ids, device_random_samples, device_random_directions, device_pdfs, nSurfaceSamples);
#endif
    deviceGPU.wait();
    //deviceGPU.freeArray(device_nodes);
    
    std::cout << "Download data from GPU\n";
    deviceGPU.memcpyArrayFromGPU(device_ids, host_ids, nSurfaceSamples);
    deviceGPU.memcpyArrayFromGPU(device_random_directions, host_random_directions, nSurfaceSamples);
    deviceGPU.memcpyArrayFromGPU(device_pdfs, host_pdfs, nSurfaceSamples);

    Timer timer;
    timer.reset();

    #if 1
    std::cout << "Executing on CPU\n";
    int step = nSurfaceSamples / num_threads;
    tbb::parallel_for( tbb::blocked_range<int>(0,num_threads), [&](tbb::blocked_range<int> r)
    {
        for (int n = r.begin(); n<r.end(); ++n)
        {
            openpgl::cpp::SurfaceSamplingDistribution ssd(&field);
            int tStep = n*step;
            for (int i = 0; i < nSurfaceSamples; i++)
            {
                int idx = (i+tStep) % nSurfaceSamples;
                float sample = -1.0f;
                ssd.Init(&field, positionsSurface[idx], sample);
                ids[idx] = ssd.GetId();
                random_directions[idx] = ssd.Sample(random_samples[idx]);
                pdfs[idx] = ssd.PDF(random_directions[idx]);
            }
        }
    });
    #endif
    std::cout << "done\n";

    float* outDiff = new float[nSurfaceSamples*4];
    float* outCPU = new float[nSurfaceSamples*4];
    float* outGPU = new float[nSurfaceSamples*4];
    for (int i = 0 ; i < nSurfaceSamples; i++){

#if 0
        outDiff[i*4+0] = fabsf(random_directions[i].x - host_random_directions[i].x);
        outDiff[i*4+1] = fabsf(random_directions[i].y - host_random_directions[i].y);
        outDiff[i*4+2] = fabsf(random_directions[i].z - host_random_directions[i].z);
        outCPU[i*4+0] = (1.0f + random_directions[i].x) / 2.f;
        outCPU[i*4+1] = (1.0f + random_directions[i].y) / 2.f;
        outCPU[i*4+2] = (1.0f + random_directions[i].z) / 2.f;
        outGPU[i*4+0] = (1.0f + host_random_directions[i].x) / 2.f;
        outGPU[i*4+1] = (1.0f + host_random_directions[i].y) / 2.f;
        outGPU[i*4+2] = (1.0f + host_random_directions[i].z) / 2.f;
#elif 1
        outDiff[i*4+0] = fabsf(pdfs[i] - host_pdfs[i]);
        outDiff[i*4+1] = fabsf(pdfs[i] - host_pdfs[i]);
        outDiff[i*4+2] = fabsf(pdfs[i] - host_pdfs[i]);
        outCPU[i*4+0] = (pdfs[i]);
        outCPU[i*4+1] = (pdfs[i]);
        outCPU[i*4+2] = (pdfs[i]);
        outGPU[i*4+0] = (host_pdfs[i]);
        outGPU[i*4+1] = (host_pdfs[i]);
        outGPU[i*4+2] = (host_pdfs[i]);
#elif 0
        outDiff[i*4+0] = (positionsSurface[i].x - host_random_directions[i].x);
        outDiff[i*4+1] = (positionsSurface[i].y - host_random_directions[i].y);
        outDiff[i*4+2] = (positionsSurface[i].z - host_random_directions[i].z);
        outCPU[i*4+0] = (positionsSurface[i].x);
        outCPU[i*4+1] = (positionsSurface[i].y);
        outCPU[i*4+2] = (positionsSurface[i].z);
        outGPU[i*4+0] = (host_random_directions[i].x);
        outGPU[i*4+1] = (host_random_directions[i].y);
        outGPU[i*4+2] = (host_random_directions[i].z);
#else
        //int id = device_ids[i] - ids[i];
        int id = host_ids[i] - ids[i];
        std::mt19937_64 genDiff(id);
        std::mt19937_64 genCPU(ids[i]);
        std::mt19937_64 genGPU(host_ids[i]);

        outDiff[i*4+0] = distU(genDiff);
        outDiff[i*4+1] = distU(genDiff);
        outDiff[i*4+2] = distU(genDiff);

        outCPU[i*4+0] = distU(genCPU);
        outCPU[i*4+1] = distU(genCPU);
        outCPU[i*4+2] = distU(genCPU);

        outGPU[i*4+0] = distU(genGPU);
        outGPU[i*4+1] = distU(genGPU);
        outGPU[i*4+2] = distU(genGPU);

#endif
        outDiff[i*4+3] = 1.0f;
        outCPU[i*4+3] = 1.0f;
        outGPU[i*4+3] = 1.0f;
    }
    SaveEXR(outDiff, width, height, 4, 0, std::regex_replace(benchParams.id_exr_file_name, std::regex("\\.exr"), "_diff.exr").c_str(), &err);
    SaveEXR(outCPU, width, height, 4, 0, std::regex_replace(benchParams.id_exr_file_name, std::regex("\\.exr"), "_cpu.exr").c_str(), &err);
    SaveEXR(outGPU, width, height, 4, 0, std::regex_replace(benchParams.id_exr_file_name, std::regex("\\.exr"), "_gpu.exr").c_str(), &err);
    delete[] outDiff;
    delete[] outCPU;
    delete[] outGPU;

    std::cout << "SurfaceSamplingDistribution::Init";
    std::cout << " time: "<< (timer.elapsed()/float(nSurfaceSamples)) << "µs" << "\t nThreads = " << num_threads << std::endl;

    delete[] distributions;
    delete[] ids;
    delete[] host_ids;
    delete[] pdfs;
    delete[] host_pdfs;
    delete[] random_directions;
    delete[] host_random_directions;
    delete[] random_samples;
    delete[] positionsSurface;

    deviceGPU.freeArray(device_distributions);
    deviceGPU.freeArray(device_ids);
    deviceGPU.freeArray(device_pdfs);
    deviceGPU.freeArray(device_random_directions);
    deviceGPU.freeArray(device_random_samples);
    deviceGPU.freeArray(device_positionsSurface);
}


int main (int argc, char *argv[]) {

    std::list<std::string> args(argv, argv + argc);

    BenchParams benchParams;
    bool success = parseCommandLine(args, benchParams);
    success = success ? benchParams.validate() : false;

    if(success){
        switch (benchParams.type)
        {
        case HELP:
            print_help();
            break;

        case BENCH_LOOKUP_ID:
            bench_lookup_id(benchParams);
            break;
                       
        default:
            print_help();
            break;
        }

    } else {

    } 
    return 0;
}