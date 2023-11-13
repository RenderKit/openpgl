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
    COMPARE_SAMPLES,
    COMPARE_FIELDS,
    UPDATE_COMPARE_FIELDS,
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
    std::string field_file_name_comp {""};
    std::string samples_file_name {""};
    std::string samples_file_name_comp {""};
    std::string obj_out_file_name {""};

    PGL_DEVICE_TYPE device_type {PGL_DEVICE_TYPE_NONE};

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
                if(device_type == PGL_DEVICE_TYPE_NONE){
                    std::cout << "ERROR: Device type not set." << std::endl;
                    valid = false;            
                }
                break;

            case VALIDATE_FIELD:
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
            case COMPARE_SAMPLES:
                if(samples_file_name == "" || 
                    !file_exists(samples_file_name)) {
                        std::cout << "ERROR: Samples file not set or does not exists: " << samples_file_name << std::endl;
                    valid = false;
                }
                if(samples_file_name_comp == "" || 
                    !file_exists(samples_file_name_comp)) {
                        std::cout << "ERROR: Samples file not set or does not exists: " << samples_file_name_comp << std::endl;
                    valid = false;
                }
                break;
            case COMPARE_FIELDS:
                if(field_file_name == "" || 
                    !file_exists(field_file_name)) {
                        std::cout << "ERROR: Field file not set or does not exists: " << field_file_name << std::endl;
                    valid = false;
                }
                if(field_file_name_comp == "" || 
                    !file_exists(field_file_name_comp)) {
                        std::cout << "ERROR: Field file not set or does not exists: " << field_file_name_comp << std::endl;
                    valid = false;
                }
                if(device_type == PGL_DEVICE_TYPE_NONE){
                    std::cout << "ERROR: Device type not set." << std::endl;
                    valid = false;            
                }
                break;
            case UPDATE_COMPARE_FIELDS:
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
                if(field_file_name_comp == "" || 
                    !file_exists(field_file_name_comp)) {
                        std::cout << "ERROR: Field file not set or does not exists: " << field_file_name_comp << std::endl;
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
                } else if(str_type == "compareSamples") {
                    debugParams.type = DebugType::COMPARE_SAMPLES; 
                } else if(str_type == "compareFields") {
                    debugParams.type = DebugType::COMPARE_FIELDS; 
                } else if(str_type == "updateCompareFields") {
                    debugParams.type = DebugType::UPDATE_COMPARE_FIELDS; 
                } else {
                    std::cout << "ERROR: Unknown type: " << str_type << std::endl;
                    std::cout << "       Valid types are: [updateField validateField validateSamples exportSamplesToOBJ compareSamples compareFields] "<< std::endl;
                    return false;
                }
            } else {
                return false;
            }
        } else if (arg == "-device") {
            ++it;
            if(it != args.end())
            {
                const std::string str_type = *it;
                if(str_type == "CPU_4") {
                    debugParams.device_type = PGL_DEVICE_TYPE_CPU_4; 
                } else if(str_type == "CPU_8") {
                    debugParams.device_type = PGL_DEVICE_TYPE_CPU_8; 
                } else {
                    std::cout << "ERROR: Unknown device type: " << str_type << std::endl;
                    std::cout << "       Valid types are: [CPU_4 CPU_8] "<< std::endl;
                    return false;
                }
            } else {
                return false;
            }
        } else if (arg == "-field") {
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
        } else if (arg == "-samplesComp") {
            ++it;
            if(it != args.end())
            {
                const std::string str_samples = *it;
                debugParams.samples_file_name_comp = str_samples; 
            } else {
                return false;
            }
        } else if (arg == "-fieldComp") {
            ++it;
            if(it != args.end())
            {
                const std::string str_samples = *it;
                debugParams.field_file_name_comp = str_samples; 
            } else {
                return false;
            }        } else if (arg == "-out") {
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
    std::cout << "usage openpgl_debug -type <types> [<options>]" << std::endl;
    std::cout << std::endl;
    std::cout << "types:" << std::endl;
    std::cout << "  " << "validateSamples " << "\t" << "Checks if the samples stored in a SampleStorage object are valid" << std::endl;
    std::cout << "  " << "                " << "\t" << "(i.e., all values are in valid ranges)." << std::endl;
    std::cout << "  " << "                " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                " << "\t" << "\"openpgl_debug -type validateSamples -samples ss0.st\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << "compareSamples  " << "\t" << "Checks if the samples stored in two different SampleStorage objects" << std::endl;
    std::cout << "  " << "                " << "\t" << "are the same." << std::endl;
    std::cout << "  " << "                " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                " << "\t" << "\"openpgl_debug -type compareSamples -samples ss0.st -samplesComp ss1.st\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << "validateField   " << "\t" << "Checks if a Field object is valid (e.g., if the spatial and directional structures are valid)." << std::endl;
    std::cout << "  " << "                " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                " << "\t" << "\"openpgl_debug -type validateField -field field0.gf -device CPU_4\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << "compareFields   " << "\t" << "Checks if the guiding structures stored in two different Field objects" << std::endl;
    std::cout << "  " << "                " << "\t" << "are the same." << std::endl;
    std::cout << "  " << "                " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                " << "\t" << "\"openpgl_debug -type compareFields -field field0.gf -fieldComp field1.gf -device CPU_4\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << "updateField     " << "\t" << "Loads a pre-trained guiding structure (Field) and updates it using a set of samples"<< std::endl;
    std::cout << "  " << "                " << "\t" << "loaded from a SampleStorage object." << std::endl;
    std::cout << "  " << "                " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                " << "\t" << "\"openpgl_debug -type updateField -field field.gf -samples ss0.st -device CPU_4\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << "updateCompareFields " << "\t" << "Loads a pre-trained guiding structure (Field) and updates it using a set of samples loaded from a"<< std::endl;
    std::cout << "  " << "                    " << "\t" << "SampleStorage object and compares the resulting guiding structure to another pre-trained Field." << std::endl;
    std::cout << "  " << "                    " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                    " << "\t" << "\"openpgl_debug -type updateCompareFields -field field0.gf -samples ss0.st -fieldComp field1.gf -device CPU_4\"" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << "exportSamplesToOBJ  " << "\t" << "Loads samples from a SampleStorage object and stores them as \".obj\" file."<< std::endl;
    std::cout << "  " << "                    " << "\t" << "example:" << std::endl;
    std::cout << "  " << "                    " << "\t" << "\"openpgl_debug -type exportSamplesToOBJ -samples ss0.st  -out ss0.obj\"" << std::endl;
    std::cout << std::endl;
}

void update_field(DebugParams &debugParams){
    openpgl::cpp::Device device(debugParams.device_type);
    openpgl::cpp::Field field(&device, debugParams.field_file_name);
    openpgl::cpp::SampleStorage sampleStorage(debugParams.samples_file_name);

    std::cout << "Validate Field:" << std::endl;
    bool fieldValid = field.Validate();
    std::cout << "  -field: "<< debugParams.field_file_name << " is " << (fieldValid ? "valid": "NOT valid") << std::endl;

    std::cout << "Validate Samples:" << std::endl;
    bool samplesValidate = sampleStorage.Validate();
    std::cout << "  -samples: "<< debugParams.samples_file_name << " is " << (samplesValidate ? "valid" : "NOT valid") << std::endl;

    field.Update(sampleStorage);
    std::cout << "Validate Updated Field:" << std::endl;
    bool fieldUpdatedValid = field.Validate();
    std::cout << "  updated field: is " << (fieldUpdatedValid ? "valid" : "NOT valid") << std::endl;
}

void validate_field(DebugParams &debugParams){
    openpgl::cpp::Device device(debugParams.device_type);
    openpgl::cpp::Field field(&device, debugParams.field_file_name);
    std::cout << "Validate Field:" << std::endl;
    bool fieldValid = field.Validate();
    std::cout << "  -field: "<< debugParams.field_file_name << " is "<< (fieldValid ? "valid": "NOT valid") << std::endl;
}

void validate_samples(DebugParams &debugParams){
    openpgl::cpp::SampleStorage sampleStorage(debugParams.samples_file_name);
    std::cout << "Validate Samples:" << std::endl;
    bool samplesValid = sampleStorage.Validate();
    std::cout << "  -samples: "<< debugParams.samples_file_name << " is "<< (samplesValid ? "valid" : "NOT valid") << std::endl;
}

void compare_samples(DebugParams &debugParams){
    openpgl::cpp::SampleStorage sampleStorage(debugParams.samples_file_name);
    openpgl::cpp::SampleStorage sampleStorageComp(debugParams.samples_file_name_comp);

    std::cout << "Validate Samples:" << std::endl;
    bool samplesValid = sampleStorage.Validate();
    std::cout << "  -samples: "<< debugParams.samples_file_name << " is " << (samplesValid ? "valid" : "NOT valid") << std::endl;

    bool samplesCompValid = sampleStorageComp.Validate();
    std::cout << "  -samplesComp: "<< debugParams.samples_file_name_comp << " is " << (samplesCompValid ? "valid" : "NOT valid")<< std::endl;

    std::cout << "Compare Samples:" << std::endl;
    bool equal = (sampleStorage.operator==(sampleStorageComp));
    std::cout << "  Samples are: "<< (equal ? "EQUAL" : "NOT-EQUAL") << std::endl;
}

void compare_fields(DebugParams &debugParams){
    openpgl::cpp::Device device(debugParams.device_type);
    openpgl::cpp::Field field(&device, debugParams.field_file_name);
    openpgl::cpp::Field fieldComp(&device, debugParams.field_file_name_comp);

    std::cout << "Validate Fields:" << std::endl;
    bool fieldValid = field.Validate();
    std::cout << "  -field: "<< debugParams.field_file_name << " is "<< (fieldValid ? "valid" : "NOT valid") << std::endl;

    bool fieldCompValid = fieldComp.Validate();
    std::cout << "  -fieldComp: "<< debugParams.field_file_name_comp << " is "<< (fieldCompValid ? "valid": "NOT valid") << std::endl;

    std::cout << "Compare Fields:" << std::endl;
    bool equal = (field.operator==(fieldComp));
    std::cout << "  Fields are: "<< (equal ? "EQUAL": "NOT-EQUAL") << std::endl;
}

void update_compare_fields(DebugParams &debugParams){
    openpgl::cpp::Device device(debugParams.device_type);
    openpgl::cpp::Field field(&device, debugParams.field_file_name);
    openpgl::cpp::SampleStorage sampleStorage(debugParams.samples_file_name);
    openpgl::cpp::Field fieldComp(&device, debugParams.field_file_name_comp);

    std::cout << "Validate Fields:" << std::endl;
    bool fieldValid = field.Validate();
    std::cout << "  -field: "<< debugParams.field_file_name << " is " << (fieldValid ? "valid": "NOT valid") << std::endl;

    bool fieldCompValid = fieldComp.Validate();
    std::cout << "  -fieldComp: "<< debugParams.field_file_name_comp << " is " << (fieldCompValid ? "valid": "NOT valid") << std::endl;

    std::cout << "Validate Samples:" << std::endl;
    bool samplesValidate = sampleStorage.Validate();
    std::cout << "  -samples: "<< debugParams.samples_file_name << " is " << (samplesValidate ? "valid" : "NOT valid") << std::endl;


    field.Update(sampleStorage);
    std::cout << "Validate Updated Field:" << std::endl;
    bool fieldUpdatedValid = field.Validate();
    std::cout << "  updated field: is " << (fieldUpdatedValid ? "valid" : "NOT valid") << std::endl;

    std::cout << "Compare Fields:" << std::endl;
    bool equal = (field.operator==(fieldComp));
    std::cout << "  Fields are: " << (equal ? "EQUAL" : "NOT-EQUAL") << std::endl;
}

void export_samples(DebugParams &debugParams){

    std::cout << "Export Samples as OBJ:" << std::endl;
    std::cout << "  -samples " << debugParams.samples_file_name << std::endl;
    std::cout << "  -out     " << debugParams.obj_out_file_name << std::endl;
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
        case COMPARE_SAMPLES:
            compare_samples(debugParams);
            break;  
        case COMPARE_FIELDS:
            compare_fields(debugParams);
            break;    
        case UPDATE_COMPARE_FIELDS:
            update_compare_fields(debugParams);
            break;                  
        default:
            print_help();
            break;
        }
    } else {

    } 

    return 0;
}