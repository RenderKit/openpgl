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

#include "timer.h"

enum BenchType{
    HELP=0,
    INIT_FIELD,
    BENCH_LOOKUP_SAMPLE,
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
    BenchType type {NONE};
    std::string field_file_name {""};
    std::vector<std::string> samples_file_names;
    //std::string out_file_name {""};

    unsigned int num_threads {1};

    PGL_DEVICE_TYPE device_type {PGL_DEVICE_TYPE_NONE};

    bool validate() {
        bool valid = true;

        if(device_type == PGL_DEVICE_TYPE_NONE){
            std::cout << "ERROR: Device type not set." << std::endl;
            valid = false;            
        }

        switch(type) {
            case HELP:
                break;

            case INIT_FIELD:
                if(field_file_name == ""/* || 
                    !file_exists(field_file_name)*/) {
                    std::cout << "ERROR: Field output file not set." << std::endl;
                    valid = false;
                }
                if(samples_file_names.size() == 0 /*|| 
                    !file_exists(samples_file_name)*/) {
                        std::cout << "ERROR: Samples file not set" << std::endl;
                    valid = false;
                } else {
                    for (int i = 0; i < samples_file_names.size(); i++)
                    {
                        if (!file_exists(samples_file_names[i])) {
                            std::cout << "ERROR: Samples file does not exists: " << samples_file_names[i] << std::endl;
                            valid = false;
                        }
                    }
                }
                break;

            case BENCH_LOOKUP_SAMPLE:
                if(field_file_name == "" || 
                    !file_exists(field_file_name)) {
                    std::cout << "ERROR: Field file not set or does not exists: " << field_file_name << std::endl;
                    valid = false;
                }

//                if (num_threads == 0)
//                {
//
//                }

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

    bool collectSamples = false;
    
    for (auto it = args.begin(); it != args.end();) {
        const std::string arg = *it;

        if (arg == "-help") {
            collectSamples = false; 
            benchParams.type = BenchType::HELP;
        } else if (arg == "-type") {
            collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_type = *it;
                if(str_type == "initField") {
                    benchParams.type = BenchType::INIT_FIELD; 
                } else if(str_type == "benchLookUpSample") {
                    benchParams.type = BenchType::BENCH_LOOKUP_SAMPLE; 
                //}// else if(str_type == "validateSamples") {
                //    debugParams.type = DebugType::VALIDATE_SAMPLES; 
                //} else if(str_type == "update") {
                
                } else {
                    std::cout << "ERROR: Unknown type: " << str_type << std::endl;
                    std::cout << "       Valid types are: [initField benchLookUpSample] "<< std::endl;
                    return false;
                }
            } else {
                return false;
            }
        } else if (arg == "-device") {
            collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_type = *it;
                if(str_type == "CPU_4") {
                    benchParams.device_type = PGL_DEVICE_TYPE_CPU_4; 
                } else if(str_type == "CPU_8") {
                    benchParams.device_type = PGL_DEVICE_TYPE_CPU_8; 
                //}// else if(str_type == "validateSamples") {
                //    debugParams.type = DebugType::VALIDATE_SAMPLES; 
                //} else if(str_type == "update") {
                
                } else {
                    std::cout << "ERROR: Unknown device type: " << str_type << std::endl;
                    std::cout << "       Valid types are: [CPU_4 CPU_8] "<< std::endl;
                    return false;
                }
            } else {
                return false;
            }
        }else if (arg == "-field") {
            collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_field = *it;
                benchParams.field_file_name = str_field;
            } else {
                return false;
            }
        } else if (arg == "-samples") {
            collectSamples = true; 
            ++it;
            if(it != args.end())
            {
                const std::string str_samples = *it;
                benchParams.samples_file_names.push_back(str_samples); 
            } else {
                return false;
            }
        } else if (collectSamples) {
            const std::string str_samples = *it;
            benchParams.samples_file_names.push_back(str_samples); 
        } else if (arg == "-threads") {
            collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_field = *it;
                benchParams.num_threads = std::stoi(str_field);
            } else {
                return false;
            }
        }
        /*} else if (arg == "-out") {
            collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_samples = *it;
                benchParams.out_file_name = str_samples; 
            } else {
                return false;
            }
        */
        ++it;
/* */
    }
    return true;
}

void print_help(){

}

void init_field(BenchParams &benchParams){

    std::cout << "init_field" << std::endl;
    openpgl::cpp::Device device(benchParams.device_type);
    
    PGLFieldArguments fieldSettings;
    pglFieldArgumentsSetDefaults(fieldSettings,PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
    fieldSettings.deterministic = false;
    fieldSettings.debugArguments.fitRegions = false;
    openpgl::cpp::Field* field = new openpgl::cpp::Field(&device, fieldSettings);
    
    std::vector<openpgl::cpp::SampleStorage*> sampleStorages;
    for (int i = 0; i < benchParams.samples_file_names.size(); i++){

        std::cout << "sampleStorage["<< i << "]: " << benchParams.samples_file_names[i];
        openpgl::cpp::SampleStorage* sampleStorage = new openpgl::cpp::SampleStorage(benchParams.samples_file_names[i]);
        std::cout << "\t nSamples = " << sampleStorage->GetSizeSurface() << std::endl;
        sampleStorages.push_back(sampleStorage);
    }
    Timer overallTimer;
    overallTimer.reset();
    for (int i = 0; i < benchParams.samples_file_names.size(); i++){
       
        Timer updateTimer;
        updateTimer.reset();
        field->Update(*sampleStorages[i]);
        std::cout << "Field::Update() stepTime: "<< updateTimer.elapsed() * 1e-3f <<"ms"<< std::endl;
    }
    std::cout << "Field::Update() overallTime: "<< overallTimer.elapsed() * 1e-3f <<"ms"<< std::endl;
    field->Store(benchParams.field_file_name);
    delete field;
}

void validate_field(BenchParams &benchParams){
    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_8);
    openpgl::cpp::Field field(&device, benchParams.field_file_name);
    bool valid = field.Validate();
}

void bench_lookup_sample(BenchParams &benchParams){
    int nThreads = benchParams.num_threads;
    int nRepetitions = 10;

    int num_threads = 0; //tbb::info::default_concurrency();
    if (nThreads > 0){  
        num_threads = nThreads;
    }
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, num_threads);
    std::cout << "num_threads = " << num_threads << std::endl;
    
    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_4);
    openpgl::cpp::Field field(&device, benchParams.field_file_name);
    openpgl::cpp::SampleStorage sampleStorage(benchParams.samples_file_names[0]);

    std::cout << "Field::Validate: " << field.Validate() << std::endl;
    //std::cout << "Prepare Data: START" << << std::endl;

    int nSurfaceSamples = sampleStorage.GetSizeSurface();
    int nSurfaceSamplesNThreads = num_threads * nSurfaceSamples;
    std::cout << "Prepare Data: START" << std::endl;

    std::mt19937_64 gen(1337);
    std::uniform_real_distribution<float> distU(0.f, 1.f);

    float* samplesSurfaceU = new float[nSurfaceSamplesNThreads];
    pgl_vec2f* samplesSurfaceDirectionUV = new pgl_vec2f[nSurfaceSamplesNThreads];
    pgl_vec3f* sampleSurfaceNormal = new pgl_vec3f[nSurfaceSamplesNThreads];
    pgl_vec3f* sampleSurfaceSampledDirection = new pgl_vec3f[nSurfaceSamplesNThreads];
    pgl_vec3f* positionsSurface = new pgl_vec3f[nSurfaceSamplesNThreads];
    for (int j = 0; j < num_threads; j++)
    {
        for (int i = 0; i < nSurfaceSamples; i++)
        {
            int idx = j * nSurfaceSamples + i;
            //int idx = i;
            samplesSurfaceU[idx] = distU(gen);
            samplesSurfaceDirectionUV[idx].x = distU(gen);
            samplesSurfaceDirectionUV[idx].y = distU(gen);
            sampleSurfaceNormal[idx] = squareToUniformSphere(distU(gen),distU(gen));

            openpgl::cpp::SampleData sd = sampleStorage.GetSampleSurface(i);
            positionsSurface[idx] = sd.position;
        }
    }

    Timer timer;
    timer.reset();

    tbb::parallel_for( tbb::blocked_range<int>(0,num_threads), [&](tbb::blocked_range<int> r)
    {
        for (int n = r.begin(); n<r.end(); ++n)
        //for (int n = 0; n<num_threads; ++n)
        {
            openpgl::cpp::SurfaceSamplingDistribution ssd(&field);        
            for(int m = 0; m < nRepetitions; m++)
            {                
                for (int i = 0; i < nSurfaceSamples; i++)
                {
                    //int idx = n * nSurfaceSamples + i;
                    int idx = i;
                    ssd.Init(&field, positionsSurface[idx], samplesSurfaceU[idx]);
                    ssd.ApplyCosineProduct(sampleSurfaceNormal[idx]);
                    ssd.SamplePDF(samplesSurfaceDirectionUV[idx], sampleSurfaceSampledDirection[idx]);
                }
            }
        }
    });

    std::cout << "SurfaceSamplingDistribution::Init time: "<< timer.elapsed()/float(nSurfaceSamples*nRepetitions) << "\t nThreads = " << num_threads << std::endl;
}


int main (int argc, char *argv[]) {

    std::list<std::string> args(argv, argv + argc);

    BenchParams benchParams;
    bool success = parseCommandLine(args, benchParams);
    success = success ? benchParams.validate() : false;
/* */
    if(success){
        switch (benchParams.type)
        {
        case HELP:
            print_help();
            break;
        
        case INIT_FIELD:
            init_field(benchParams);
            break;

        case BENCH_LOOKUP_SAMPLE:
            bench_lookup_sample(benchParams);
            break;
                       
        default:
            print_help();
            break;
        }

    } else {

    } 
/* */
    return 0;
}