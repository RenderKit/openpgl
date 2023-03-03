// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openpgl/cpp/OpenPGL.h>

#include <string>
#include <list>
#include <fstream>
#include <iostream>

#include <vector>

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

struct BenchParams {
    BenchType type {NONE};
    std::string field_file_name {""};
    std::vector<std::string> samples_file_names;
    //std::string out_file_name {""};

    PGL_DEVICE_TYPE bench_type {PGL_DEVICE_TYPE_CPU_4};

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
                break;

            case BENCH_LOOKUP_SAMPLE:
                if(field_file_name == "" || 
                    !file_exists(field_file_name)) {
                    std::cout << "ERROR: Field file not set or does not exists: " << field_file_name << std::endl;
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

        if (collectSamples)
        {
            const std::string str_samples = *it;
            benchParams.samples_file_names.push_back(str_samples); 
        } else if (arg == "-help") {
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
            collectSamples = false; 
            ++it;
            if(it != args.end())
            {
                const std::string str_samples = *it;
                benchParams.samples_file_names.push_back(str_samples); 
            } else {
                return false;
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
        }
        ++it;
/* */
    }
    return true;
}

void print_help(){

}

void init_field(BenchParams &benchParams){

    std::cout << "init_field" << std::endl;
/* */
    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_4);
    
    PGLFieldArguments fieldSettings;
    pglFieldArgumentsSetDefaults(fieldSettings,PGL_SPATIAL_STRUCTURE_KDTREE, PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM);
    openpgl::cpp::Field* field = new openpgl::cpp::Field(&device, fieldSettings);

    for (int i = 0; i < benchParams.samples_file_names.size(); i++){
        openpgl::cpp::SampleStorage sampleStorage(benchParams.samples_file_names[i]);
        field->Update(sampleStorage);
    }

    field->Store(benchParams.field_file_name);
    delete field;
    //openpgl::cpp::Field field(&device, benchParams.field_file_name);
    //openpgl::cpp::SampleStorage sampleStorage(benchParams.samples_file_names[0]);
    //std::cout << "sampleStorage: numSurfaceSamples = " << sampleStorage.GetSizeSurface() << "\tnumVolumeSamples = " << sampleStorage.GetSizeVolume() << std::endl;
    //field.Validate();
    //field.Update(sampleStorage);
/* */
}

void validate_field(BenchParams &benchParams){
    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_8);
    openpgl::cpp::Field field(&device, benchParams.field_file_name);
    bool valid = field.Validate();
}

void validate_samples(BenchParams &benchParams){

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
/*
        case VALIDATE_FIELD:
            validate_field(debugParams);
            break;

        case VALIDATE_SAMPLES:
            validate_samples(debugParams);
            break;
*/                        
        default:
            print_help();
            break;
        }

    } else {

    } 
/* */
    return 0;
}