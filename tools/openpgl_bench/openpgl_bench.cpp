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
    BENCH_LOOKUP,
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
    BenchType type {HELP};
    std::string field_file_name {""};
    std::vector<std::string> samples_file_names;
    //std::string out_file_name {""};

    unsigned int num_threads {0};

    PGL_DEVICE_TYPE device_type {PGL_DEVICE_TYPE_NONE};

    bool validate() {
        bool valid = true;

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
                if(device_type == PGL_DEVICE_TYPE_NONE){
                    std::cout << "ERROR: Device type not set." << std::endl;
                    valid = false;            
                }

                break;

            case BENCH_LOOKUP_SAMPLE:
            case BENCH_LOOKUP:
                if(field_file_name == "" || 
                    !file_exists(field_file_name)) {
                    std::cout << "ERROR: Field file not set or does not exists: " << field_file_name << std::endl;
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
                } else if(str_type == "benchLookUp") {
                    benchParams.type = BenchType::BENCH_LOOKUP; 
                } else {
                    std::cout << "ERROR: Unknown type: " << str_type << std::endl;
                    std::cout << "       Valid types are: [initField benchLookUp benchLookUpSample] "<< std::endl;
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
        ++it;
    }
    return true;
}

void print_help(){
    std::cout << "usage openpgl_bench -type <initField | benchLookUp | benchLookUpSample> [<options>]" << std::endl;
    std::cout << std::endl;
    std::cout << "type options:" << std::endl;
    std::cout << "  " << "initField        " << "\t" << "Measures the time to build a guiding Field from set of samples "<< std::endl;
    std::cout << "  " << "                 " << "\t" << "(i.e., saved SampleStorage objects)." << std::endl;
    std::cout << "  " << "                 " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                 " << "\t" << "\"openpgl_bench -type initField -samples ss0.st ss1.st -field field.gf -device CPU_4\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << "benchLookUp      " << "\t" << "Measures the average time of an initialization of a SurfaceSamplingDistribution" << std::endl;
    std::cout << "  " << "                 " << "\t" << "from a guiding Field. The positions form samples from a SampleStorage are used" << std::endl;
    std::cout << "  " << "                 " << "\t" << "during the initialization." << std::endl;
    std::cout << "  " << "                 " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                 " << "\t" << "\"openpgl_bench -type benchLookUp -samples ss0.st -field field.gf -device CPU_4\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << "benchLookUpSample" << "\t" << "Measures the average time of an initialization of a SurfaceSamplingDistribution" << std::endl;
    std::cout << "  " << "                 " << "\t" << "from a guiding Field combined with applying the cosine product and generating" << std::endl;
    std::cout << "  " << "                 " << "\t" << "a directional samples. The positions form samples from a SampleStorage are used" << std::endl;
    std::cout << "  " << "                 " << "\t" << "during the initialization." << std::endl;
    std::cout << "  " << "                 " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                 " << "\t" << "\"openpgl_bench -type benchLookUpSample -samples ss0.st -field field.gf -device CPU_4\"" << std::endl;
    std::cout << std::endl;
    std::cout << "general options:" << std::endl;
    std::cout << "  -device <CPU_4 | CPU_8>" << "\t SIMD width of the loaded Field or the Field that should be initialized." << std::endl;
    std::cout << "  -threads n             " << "\t Number of n threads that should be used during the measurements." << std::endl;
    std::cout << std::endl;
    std::cout << "initField options:" << std::endl;
    std::cout << "  -samples <s0 s1 .. sn> " << "\t A list of stored samples used to initialize update the Field. "<< std::endl;
    std::cout << "                         " << "\t Each sample set represents one training/update step/iteration. "<< std::endl;
    std::cout << "  -field f0              " << "\t The file where the initialized Field is stored." << std::endl;
    std::cout << std::endl;
    std::cout << "benchLookUp | benchLookUpSample options:" << std::endl;
    std::cout << "  -field f0              " << "\t The stored Field object that is loaded for the test." << std::endl; 
    std::cout << "  -samples s0            " << "\t The stored samples which positions are used for query/initialize" << std::endl;
    std::cout << "                         " << "\t the SurfaceSamplingDistributions."<< std::endl;
    std::cout << std::endl;
}

void init_field(BenchParams &benchParams){

    int nThreads = benchParams.num_threads;

    int num_threads = 0; 
    if (nThreads > 0){  
        num_threads = nThreads;
    }
    std::cout << "init_field: threads = " << num_threads << std::endl;
    openpgl::cpp::Device device(benchParams.device_type, num_threads);
    
    PGLFieldArguments fieldSettings;
    pglFieldArgumentsSetDefaults(fieldSettings,PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
    fieldSettings.deterministic = true;
    fieldSettings.debugArguments.fitRegions = true;
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

    for (int n = 0; n < sampleStorages.size(); n++){
        delete sampleStorages[n];
        sampleStorages[n] = nullptr;
    }

}

void validate_field(BenchParams &benchParams){
    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_8);
    openpgl::cpp::Field field(&device, benchParams.field_file_name);
    bool valid = field.Validate();
}

void bench_lookup_sample(BenchParams &benchParams, bool applyCosine, bool sample){
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

    int nSurfaceSamples = sampleStorage.GetSizeSurface();

    std::cout << "Prepare Data: START" << std::endl;
    std::mt19937_64 gen(1337);
    std::uniform_real_distribution<float> distU(0.f, 1.f);

    float* samplesSurfaceU = new float[nSurfaceSamples];
    pgl_vec2f* samplesSurfaceDirectionUV = new pgl_vec2f[nSurfaceSamples];
    pgl_vec3f* sampleSurfaceNormal = new pgl_vec3f[nSurfaceSamples];
    pgl_vec3f* sampleSurfaceSampledDirection = new pgl_vec3f[nSurfaceSamples];
    pgl_vec3f* positionsSurface = new pgl_vec3f[nSurfaceSamples];

    for (int i = 0; i < nSurfaceSamples; i++)
    {
        int idx = i;
        samplesSurfaceU[idx] = distU(gen);
        samplesSurfaceDirectionUV[idx].x = distU(gen);
        samplesSurfaceDirectionUV[idx].y = distU(gen);
        sampleSurfaceNormal[idx] = squareToUniformSphere(distU(gen),distU(gen));

        openpgl::cpp::SampleData sd = sampleStorage.GetSampleSurface(i);
        positionsSurface[idx] = sd.position;
    }

    std::cout << "Prepare Data: END" << std::endl;

    Timer timer;
    timer.reset();
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
                    ssd.Init(&field, positionsSurface[idx], samplesSurfaceU[idx]);
                    if(applyCosine)
                        ssd.ApplyCosineProduct(sampleSurfaceNormal[idx]);
                    if(sample)
                        ssd.SamplePDF(samplesSurfaceDirectionUV[idx], sampleSurfaceSampledDirection[idx]);
                }
            }
        }
    });

    std::cout << "SurfaceSamplingDistribution::Init";
    if(applyCosine)
        std::cout << "+ApplyCosine";
    if(sample)
        std::cout << "+SamplePDF"; 
    std::cout << " time: "<< (timer.elapsed()/float(nSurfaceSamples*nRepetitions)) << "µs" << "\t nThreads = " << num_threads << std::endl;
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
        
        case INIT_FIELD:
            init_field(benchParams);
            break;

        case BENCH_LOOKUP:
            bench_lookup_sample(benchParams, false, false);
            break;

        case BENCH_LOOKUP_SAMPLE:
            bench_lookup_sample(benchParams, true, true);
            break;
                       
        default:
            print_help();
            break;
        }

    } else {

    } 
    return 0;
}