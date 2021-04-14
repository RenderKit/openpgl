// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl.h"

#include "../include/openpgl/data.h"

#include <iostream>
#include <fstream>
#include <sstream>

namespace openpgl
{


    typedef PGLDirectionalSampleData DirectionalSampleData;
    enum DirectionalSampleData_Flags
    {
        ESplatted = 1<<0,      // point does not represent any real scene intersection point
        EInsideVolume = 1<<1   // point does not represent any real scene intersection point
    };

    inline bool isValid(const DirectionalSampleData& dsd)
    {
        bool valid = true;
        valid &= embree::isvalid(dsd.position.x);
        valid &= embree::isvalid(dsd.position.y);
        valid &= embree::isvalid(dsd.position.z);
        valid &= embree::isvalid(dsd.direction.x);
        valid &= embree::isvalid(dsd.direction.y);
        valid &= embree::isvalid(dsd.direction.z);
        valid &= dsd.weight >=0.f;
        valid &= dsd.pdf >0.f;
        valid &= dsd.distance >0.f;
        return valid;
    }

    inline bool isInsideVolume(const DirectionalSampleData& dsd)
    {
        return (dsd.flags & EInsideVolume);
    }

    struct {
        inline bool operator() (const PGLDirectionalSampleData &compA,  const PGLDirectionalSampleData &compB )
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
    } DirectionalSampleDataLess;

inline DirectionalSampleData *LoadDirectionalSampleData(const std::string fileName, size_t &numData){

    std::ifstream file;
    file.open(fileName, std::ios::binary);

    //size_t numData;
    file.read((char*)&numData, sizeof(size_t));

    DirectionalSampleData *data = new DirectionalSampleData[numData];
    file.read((char*)data, numData*sizeof(DirectionalSampleData));
    file.close();

    return data;
}

inline void StoreDirectionalSampleData(const std::string fileName, const DirectionalSampleData *data, const size_t &numData){
    std::ofstream file;
    file.open(fileName, std::ios::binary);

    file.write((char*)&numData, sizeof(size_t));
    file.write((char*)&data, numData * sizeof(DirectionalSampleData));
    file.close();
}


}