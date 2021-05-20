// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"

#include "../include/openpgl/data.h"

#include <iostream>
#include <fstream>
#include <sstream>

namespace openpgl
{


    typedef PGLSampleData SampleData;
    enum SampleData_Flags
    {
        ESplatted = 1<<0,      // point does not represent any real scene intersection point
        EInsideVolume = 1<<1   // point does not represent any real scene intersection point
    };

    inline bool isValid(const SampleData& dsd)
    {
        bool valid = true;
        valid = valid && embree::isvalid(dsd.position.x);
        valid = valid && embree::isvalid(dsd.position.y);
        valid = valid && embree::isvalid(dsd.position.z);
        valid = valid && embree::isvalid(dsd.direction.x);
        valid = valid && embree::isvalid(dsd.direction.y);
        valid = valid && embree::isvalid(dsd.direction.z);
        valid = valid && dsd.weight >=0.f;
        valid = valid && dsd.pdf >0.f;
        valid = valid && dsd.distance >0.f;
        return valid;
    }

    inline bool isInsideVolume(const SampleData& dsd)
    {
        return (dsd.flags & EInsideVolume);
    }

    struct {
        inline bool operator() (const PGLSampleData &compA,  const PGLSampleData &compB )
        {
            return compA.weight < compB.weight ||
                    ( compA.weight        == compB.weight          &&  ( compA.pdf       < compB.pdf              ||
                    ( compA.pdf           == compB.pdf             &&  ( compA.distance < compB.distance          ||
                    ( compA.distance      == compB.distance        &&  ( compA.position.x < compB.position.x    ||
                    (compA.position.x    == compB.position.x    &&  ( compA.position.y < compB.position.y    ||
                    (compA.position.y    == compB.position.y     &&  ( compA.position.z < compB.position.z   ||
                    (compA.position.z    == compB.position.z     &&  ( compA.direction.x < compB.direction.x  ||
                    (compA.direction.x    == compB.direction.x   &&  ( compA.direction.y < compB.direction.y  ||
                    (compA.direction.y    == compB.direction.y   &&  ( compA.direction.z < compB.direction.z  ))))))))))))))));
        }
    } SampleDataLess;

inline SampleData *LoadSampleData(const std::string fileName, size_t &numData){

    std::ifstream file;
    file.open(fileName, std::ios::binary);
    file.read((char*)&numData, sizeof(size_t));

    SampleData *data = new SampleData[numData];
    file.read((char*)data, numData*sizeof(SampleData));
    file.close();

    return data;
}

inline void StoreSampleData(const std::string fileName, const SampleData *data, const size_t &numData){
    std::ofstream file;
    file.open(fileName, std::ios::binary);

    file.write((char*)&numData, sizeof(size_t));
    file.write((char*)&data, numData * sizeof(SampleData));
    file.close();
}


}