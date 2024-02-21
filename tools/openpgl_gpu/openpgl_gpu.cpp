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

#include <tbb/info.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <openpgl/cpp/OpenPGL.h>

#include <sycl/sycl.hpp>
#include </home/stefanwe/intel/libraries.graphics.renderkit.openpgl/openpgl/field/FieldGPU.h>
//#include </home/stefanwe/intel/libraries.graphics.renderkit.openpgl/openpgl/field/FieldGPU.h>

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
    int nRepetitions = 10;

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
    sycl::queue q;
    int* ids = new int[nSurfaceSamples];
    int* device_ids = sycl::malloc_shared<int>(nSurfaceSamples, q);

    std::cout << "Prepare Data: START" << std::endl;
    pgl_vec3f* positionsSurface = sycl::malloc_shared<pgl_vec3f>(nSurfaceSamples, q);
    for (int i = 0; i < nSurfaceSamples; i++)
    {
        int idx = i;
        pgl_vec3f pos;
        pos.x = img[i*4+0];
        pos.y = img[i*4+1];
        pos.z = img[i*4+2];
        positionsSurface[idx] = pos;
    }
    std::mt19937_64 rng(0);
    pgl_vec2f *random_samples = sycl::malloc_shared<pgl_vec2f>(nSurfaceSamples, q);
    pgl_vec3f *random_directions = new pgl_vec3f[nSurfaceSamples];
    pgl_vec3f *device_random_directions = sycl::malloc_shared<pgl_vec3f>(nSurfaceSamples, q);
    for (int i = 0; i < nSurfaceSamples; i++)
    {
        random_samples[i].x = distU(rng);
        random_samples[i].y = distU(rng);
    }

    std::cout << "Prepare Data: END" << std::endl;


    std::string foo;
    std::getline( std::cin, foo );

    std::cout << "Copying data to GPU\n";
    openpgl_gpu::KDTreeLet *device_nodes = sycl::malloc_device<openpgl_gpu::KDTreeLet>(field.GetNumKDNodes(), q);
    q.memcpy(device_nodes, field.GetKdNodes(), field.GetNumKDNodes() * sizeof(openpgl_gpu::KDTreeLet));

    openpgl_gpu::ParallaxAwareVonMisesFisherMixture<32> *distributions = sycl::malloc_shared<openpgl_gpu::ParallaxAwareVonMisesFisherMixture<32>>(field.GetNumDistributions(), q);
    field.CopyDistributions(distributions);

    q.wait();
    int num_nodes = field.GetNumKDNodes();
    std::cout << "Executing on GPU\n";
    openpgl_gpu::FieldGPU device_field(device_nodes);
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(nSurfaceSamples), [=] (sycl::id<1> i) {
                int idx = i.get(0);
                    float pos[3] = {positionsSurface[idx].x, positionsSurface[idx].y, positionsSurface[idx].z};
                    device_ids[idx] = device_field.getDataIdxAtPos(pos);
                   // auto foo = distributions[device_ids[idx]]._meanDirections[0];//.sample(random_samples[idx]);
                    //device_random_directions[idx] = {foo[0], foo[1], foo[2]};
                    device_random_directions[idx] = distributions[device_ids[idx]].sample(random_samples[idx]);
        });
    });
    q.wait();
    sycl:free(device_nodes, q);

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
            for(int m = 0; m < nRepetitions; m++)
            {
                int tStep = n*step;
                for (int i = 0; i < nSurfaceSamples; i++)
                {
                    int idx = (i+tStep) % nSurfaceSamples;
                    float sample = 0.0f;
                    ssd.Init(&field, positionsSurface[idx], sample);
                    ids[idx] = ssd.GetId();
                    random_directions[idx] = ssd.Sample(random_samples[idx]);
                }
            }
        }
    });
    #endif
    std::cout << "done\n";

    float* out = new float[nSurfaceSamples*4];
    for (int i = 0 ; i < nSurfaceSamples; i++){
        int id = device_ids[i] - ids[i];

        std::mt19937_64 gen(id);
#if 1
        out[i*4+0] = fabsf(random_directions[i].x - device_random_directions[i].x);
        out[i*4+1] = fabsf(random_directions[i].y - device_random_directions[i].y);
        out[i*4+2] = fabsf(random_directions[i].z - device_random_directions[i].z);
        // out[i*4+0] = fabsf(random_directions[i].x);
        // out[i*4+1] = fabsf(random_directions[i].y);
        // out[i*4+2] = fabsf(random_directions[i].z);
        // out[i*4+0] = fabsf(device_random_directions[i].x);
        // out[i*4+1] = fabsf(device_random_directions[i].y);
        // out[i*4+2] = fabsf(device_random_directions[i].z);
#else
        out[i*4+0] = distU(gen);
        out[i*4+1] = distU(gen);
        out[i*4+2] = distU(gen);
#endif
        out[i*4+3] = 1.0f;
    }
    SaveEXR(out, width, height, 4, 0, benchParams.id_exr_file_name.c_str(), &err);
    delete[] out;

    std::cout << "SurfaceSamplingDistribution::Init";
    std::cout << " time: "<< (timer.elapsed()/float(nSurfaceSamples*nRepetitions)) << "µs" << "\t nThreads = " << num_threads << std::endl;
    delete[] ids;

    sycl::free(distributions, q);
    sycl::free(device_ids, q);
    sycl::free(random_samples, q);
    sycl::free(positionsSurface, q);
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