// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openpgl/cpp/OpenPGL.h>

#include <string>
#include <list>
#include <fstream>
#include <vector>
#include <iostream>

enum DebugType{
    HELP=0,
    UPDATE_FIELD,
    VALIDATE_FIELD,
    VALIDATE_SAMPLES,
    EXPORT_SAMPLES,
    NONE
};

inline bool file_exists(const std::string& file_name)
{
    std::ifstream file(file_name.c_str());
    return file.good();
}

struct DebugParams {
    DebugType type {NONE};
    std::string field_file_name {""};
    std::string samples_file_name {""};
    std::string obj_out_file_name {""};

    bool validate() {
        bool valid = true;
        switch(type) {
            case HELP:
                break;

            case UPDATE_FIELD:
                if(field_file_name == "" || 
                    !file_exists(field_file_name)) {
                    std::cout << "ERROR: Field file not set or does not exists: " << field_file_name << std::endl;
                    valid = false;
                }
                if(samples_file_name == "" || 
                    !file_exists(samples_file_name)) {
                        std::cout << "ERROR: Samples file not set or does not exists: " << samples_file_name << std::endl;
                    valid = false;
                }
                break;

            case VALIDATE_FIELD:
                if(field_file_name == "" || 
                    !file_exists(field_file_name)) {
                    std::cout << "ERROR: Field file not set or does not exists: " << field_file_name << std::endl;
                    valid = false;
                }
                break;

            case VALIDATE_SAMPLES:
                if(samples_file_name == "" || 
                    !file_exists(samples_file_name)) {
                        std::cout << "ERROR: Samples file not set or does not exists: " << samples_file_name << std::endl;
                    valid = false;
                }
                break;
            case EXPORT_SAMPLES:
                if(samples_file_name == "" || 
                    !file_exists(samples_file_name)) {
                        std::cout << "ERROR: Samples file not set or does not exists: " << samples_file_name << std::endl;
                    valid = false;
                }
                if(obj_out_file_name == "") {
                        std::cout << "ERROR: OBJ output file not set: " << obj_out_file_name << std::endl;
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
                        DebugParams &debugParams) {

    for (auto it = args.begin(); it != args.end();) {
        const std::string arg = *it;

        if (arg == "-help") {
           debugParams.type = DebugType::HELP; 
        } else if (arg == "-type") {
            ++it;
            if(it != args.end())
            {
                const std::string str_type = *it;
                if(str_type == "updateField") {
                    debugParams.type = DebugType::UPDATE_FIELD; 
                } else if(str_type == "validateField") {
                    debugParams.type = DebugType::VALIDATE_FIELD; 
                } else if(str_type == "validateSamples") {
                    debugParams.type = DebugType::VALIDATE_SAMPLES; 
                } else if(str_type == "exportSamplesToOBJ") {
                    debugParams.type = DebugType::EXPORT_SAMPLES; 
                } else {
                    std::cout << "ERROR: Unknown type: " << str_type << std::endl;
                    std::cout << "       Valid types are: [updateField validateField validateSamples exportSamplesToOBJ] "<< std::endl;
                    return false;
                }
            } else {
                return false;
            }
        }else if (arg == "-field") {
            ++it;
            if(it != args.end())
            {
                const std::string str_field = *it;
                debugParams.field_file_name = str_field;
            } else {
                return false;
            }
        } else if (arg == "-samples") {
            ++it;
            if(it != args.end())
            {
                const std::string str_samples = *it;
                debugParams.samples_file_name = str_samples; 
            } else {
                return false;
            }
        } else if (arg == "-out") {
            ++it;
            if(it != args.end())
            {
                const std::string str_samples = *it;
                debugParams.obj_out_file_name = str_samples; 
            } else {
                return false;
            }
        }
        ++it;
    }
    return true;
}

void print_help(){

}

void update_field(DebugParams &debugParams){
    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_4);
    openpgl::cpp::Field field(&device, debugParams.field_file_name);
    openpgl::cpp::SampleStorage sampleStorage(debugParams.samples_file_name);

    std::cout << "sampleStorage: numSurfaceSamples = " << sampleStorage.GetSizeSurface() << "\tnumVolumeSamples = " << sampleStorage.GetSizeVolume() << std::endl;

    field.Validate();
    field.Update(sampleStorage);
}

void validate_field(DebugParams &debugParams){
    openpgl::cpp::Device device(PGL_DEVICE_TYPE_CPU_8);
    openpgl::cpp::Field field(&device, debugParams.field_file_name);
    bool valid = field.Validate();
}

void validate_samples(DebugParams &debugParams){

}


void export_samples(DebugParams &debugParams){
    std::ofstream objFile;
    objFile.open(debugParams.obj_out_file_name.c_str());
    
    openpgl::cpp::SampleStorage sampleStorage(debugParams.samples_file_name);

    bool pointsOnly = true;
    std::vector<openpgl::cpp::SampleData> subSampledData;
    subSampledData.reserve(sampleStorage.GetSizeSurface() + sampleStorage.GetSizeVolume());
    for (size_t i =0; i < sampleStorage.GetSizeSurface(); i++)
    {
        subSampledData.push_back(sampleStorage.GetSampleSurface(i));
    }
    for (size_t i =0; i < sampleStorage.GetSizeVolume(); i++)
    {
        subSampledData.push_back(sampleStorage.GetSampleVolume(i));
    }

    for (auto& sample : subSampledData)
    {
        objFile << "v " << sample.position.x << "\t" << sample.position.y << "\t"<< sample.position.z << std::endl;
        if (!pointsOnly)
        {
            pgl_vec3f dir = sample.direction;
            pgl_vec3f samplePosition = sample.position;
            pgl_vec3f pos2;
            pos2.x = samplePosition.x + dir.x * sample.distance;
            pos2.y = samplePosition.y + dir.y * sample.distance;
            pos2.z = samplePosition.z + dir.z * sample.distance;
            objFile << "v " << pos2.x << "\t" << pos2.y << "\t"<< pos2.z << std::endl;
            objFile << "v " << sample.position.x << "\t" << sample.position.y << "\t"<< sample.position.z << std::endl;
        }
    }

    for (auto& sample : subSampledData)
    {
        pgl_vec3f dir = sample.direction;
        objFile << "vn " << dir.x << "\t" << dir.y << "\t"<< dir.z << std::endl;
        if (!pointsOnly)
        {
            objFile << "vn " << dir.x << "\t" << dir.y << "\t"<< dir.z << std::endl;
            objFile << "vn " << dir.x << "\t" << dir.y << "\t"<< dir.z << std::endl;
        }
    }

    if (!pointsOnly)
    {
        for (int i = 0; i < subSampledData.size(); i++)
        {
            objFile << "f " << i*3+1 << "\t" << i*3+2 << "\t"<< i*3+3 << std::endl;
        }
    }

    objFile.close();
}


int main (int argc, char *argv[]) {

    std::list<std::string> args(argv, argv + argc);

    DebugParams debugParams;
    bool success = parseCommandLine(args, debugParams);
    success = success ? debugParams.validate() : false;

    if(success){
        switch (debugParams.type)
        {
        case HELP:
            print_help();
            break;
        
        case UPDATE_FIELD:
            update_field(debugParams);
            break;

        case VALIDATE_FIELD:
            validate_field(debugParams);
            break;

        case VALIDATE_SAMPLES:
            validate_samples(debugParams);
            break;
         case EXPORT_SAMPLES:
            export_samples(debugParams);
            break;                       
        default:
            print_help();
            break;
        }

    } else {

    } 

    return 0;
}