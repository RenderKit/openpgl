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

    typedef PGLInvalidSampleData InvalidSampleData;
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
        OPENPGL_ASSERT(valid);
        valid = valid && embree::isvalid(dsd.direction.x);
        valid = valid && embree::isvalid(dsd.direction.y);
        valid = valid && embree::isvalid(dsd.direction.z);
        OPENPGL_ASSERT(valid);
        valid = valid && embree::isvalid(dsd.weight);
        valid = valid && dsd.weight >=0.f;
        OPENPGL_ASSERT(valid);
        valid = valid && embree::isvalid(dsd.pdf);
        valid = valid && dsd.pdf >0.f;
        OPENPGL_ASSERT(valid);
        valid = valid && embree::isvalid(dsd.distance);
        valid = valid && dsd.distance >0.f;
        OPENPGL_ASSERT(valid);
        return valid;
    }

    inline bool isInsideVolume(const SampleData& sd)
    {
        return (sd.flags & EInsideVolume);
    }

    inline std::string toString(const SampleData& sd)
    {
        std::stringstream ss;
        ss << "SampleData: "; 
        ss << "position = " << sd.position.x << "\t " << sd.position.y << "\t " << sd.position.z << "\t ";
        ss << "\t direction = " << sd.direction.x << "\t " << sd.direction.y << "\t " << sd.direction.z << "\t ";
        ss << "\t weight = " << sd.weight << "\t ";
        ss << "\t pdf = " << sd.pdf << "\t ";
        ss << "\t distance = " << sd.distance << "\t ";
        ss << "\t flags = " << sd.flags;
        ss << std::endl;
        
        return ss.str();   
    }

    inline bool SampleDataEqual(const PGLSampleData &compA,  const PGLSampleData &compB)
    {
        if(compA.position.x != compB.position.x || compA.position.y != compB.position.y ||
           compA.position.z != compB.position.z || compA.direction.x != compB.direction.x ||
           compA.direction.y != compB.direction.y || compA.direction.z != compB.direction.z ||
           compA.weight != compB.weight || compA.pdf != compB.pdf || 
           compA.distance != compB.distance || compA.flags != compB.flags)
        {
            return false;
        }
        return true;
    }

    inline bool SampleDataLess(const PGLSampleData &compA,  const PGLSampleData &compB )
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

    inline bool InvalidSampleDataEqual(const PGLInvalidSampleData &compA,  const PGLInvalidSampleData &compB)
    {
        if( compA.position.x != compB.position.x || 
            compA.position.y != compB.position.y ||
            compA.position.z != compB.position.z ||
            compA.direction.x != compB.direction.x || 
            compA.direction.y != compB.direction.y ||
            compA.direction.z != compB.direction.z)
        {
            return false;
        }
        return true;
    }

    inline bool InvalidSampleDataLess(const PGLInvalidSampleData &compA,  const PGLInvalidSampleData &compB )
    {
        return    compA.position.x < compB.position.x    ||   (compA.position.x    == compB.position.x    
            &&  ( compA.position.y < compB.position.y    ||   (compA.position.y    == compB.position.y     
            &&  ( compA.position.z < compB.position.z    ||   (compA.position.z    == compB.position.z
            &&  ( compA.direction.x < compB.direction.x    ||   (compA.direction.x    == compB.direction.x
            &&  ( compA.direction.y < compB.direction.y    ||   (compA.direction.y    == compB.direction.y
            &&  ( compA.direction.z < compB.direction.z    ||   (compA.direction.z    == compB.direction.z     
                )))))))))));
    }

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