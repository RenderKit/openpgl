// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#if defined(PGL_USE_DIRECTION_COMPRESSION) || defined(PGL_USE_COLOR_COMPRESSION)
#include "../include/openpgl/compression.h"
#endif

#include "../include/openpgl/data.h"

#include <iostream>
#include <fstream>
#include <sstream>

namespace openpgl
{

    typedef PGLZeroValueSampleData ZeroValueSampleData;
    typedef PGLSampleData SampleData;
    enum SampleData_Flags
    {
        EInsideVolume = 1<<0,   // point does not represent any real scene intersection point
        EDirectLight = 1<<1  // if the samples represents direct light from a light source
    };

    inline bool isValid(const SampleData& dsd)
    {
        bool valid = true;
        valid = valid && embree::isvalid(dsd.position.x);
        valid = valid && embree::isvalid(dsd.position.y);
        valid = valid && embree::isvalid(dsd.position.z);
        OPENPGL_ASSERT(valid);
#ifndef PGL_USE_DIRECTION_COMPRESSION
        valid = valid && embree::isvalid(dsd.direction.x);
        valid = valid && embree::isvalid(dsd.direction.y);
        valid = valid && embree::isvalid(dsd.direction.z);
        OPENPGL_ASSERT(valid);
#else 
        pgl_vec3f dir = dequantize_direction(dsd.direction);
        valid = valid && embree::isvalid(dir.x);
        valid = valid && embree::isvalid(dir.y);
        valid = valid && embree::isvalid(dir.z);
        OPENPGL_ASSERT(valid);
#endif
        valid = valid && embree::isvalid(dsd.weight);
        valid = valid && dsd.weight >=0.f;
        OPENPGL_ASSERT(valid);
        valid = valid && embree::isvalid(dsd.pdf);
        valid = valid && dsd.pdf >0.f;
        OPENPGL_ASSERT(valid);
        valid = valid && embree::isvalid(dsd.distance);
        valid = valid && dsd.distance >0.f;
        OPENPGL_ASSERT(valid);

#ifdef OPENPGL_EF_RADIANCE_CACHES
#ifndef PGL_USE_COLOR_COMPRESSION
        valid = valid && embree::isvalid(dsd.radianceIn.x);
        valid = valid && dsd.radianceIn.x >= 0.f;
        valid = valid && embree::isvalid(dsd.radianceIn.y);
        valid = valid && dsd.radianceIn.y >= 0.f;
        valid = valid && embree::isvalid(dsd.radianceIn.z);
        valid = valid && dsd.radianceIn.z >= 0.f;
        OPENPGL_ASSERT(valid);
#else
        pgl_vec3f radianceIn = rgbe2vec3f(dsd.radianceIn);
        valid = valid && embree::isvalid(radianceIn.x);
        valid = valid && radianceIn.x >= 0.f;
        valid = valid && embree::isvalid(radianceIn.y);
        valid = valid && radianceIn.y >= 0.f;
        valid = valid && embree::isvalid(radianceIn.z);
        valid = valid && radianceIn.z >= 0.f;
        OPENPGL_ASSERT(valid);
#endif
        valid = valid && embree::isvalid(dsd.radianceInMISWeight);
        valid = valid && dsd.radianceInMISWeight >0.f;
        OPENPGL_ASSERT(valid);
#ifndef PGL_USE_DIRECTION_COMPRESSION
        valid = valid && embree::isvalid(dsd.directionOut.x);
        valid = valid && embree::isvalid(dsd.directionOut.y);
        valid = valid && embree::isvalid(dsd.directionOut.z);
        OPENPGL_ASSERT(valid);
#else 
        pgl_vec3f dirOut = dequantize_direction(dsd.directionOut);
        valid = valid && embree::isvalid(dirOut.x);
        valid = valid && embree::isvalid(dirOut.y);
        valid = valid && embree::isvalid(dirOut.z);
        OPENPGL_ASSERT(valid);
#endif
#ifndef PGL_USE_COLOR_COMPRESSION
        valid = valid && embree::isvalid(dsd.radianceOut.x);
        valid = valid && dsd.radianceOut.x >= 0.f;
        valid = valid && embree::isvalid(dsd.radianceOut.y);
        valid = valid && dsd.radianceOut.y >= 0.f;
        valid = valid && embree::isvalid(dsd.radianceOut.z);
        valid = valid && dsd.radianceOut.z >= 0.f;
        OPENPGL_ASSERT(valid);
#else
        pgl_vec3f radianceOut = rgbe2vec3f(dsd.radianceOut);
        valid = valid && embree::isvalid(radianceOut.x);
        valid = valid && radianceOut.x >= 0.f;
        valid = valid && embree::isvalid(radianceOut.y);
        valid = valid && radianceOut.y >= 0.f;
        valid = valid && embree::isvalid(radianceOut.z);
        valid = valid && radianceOut.z >= 0.f;
        OPENPGL_ASSERT(valid);
#endif
#endif
        return valid;
    }

    inline bool isInsideVolume(const SampleData& sd)
    {
        return (sd.flags & EInsideVolume);
    }

    inline bool isDirectLight(const SampleData& sd)
    {
        return (sd.flags & EDirectLight);
    }

    inline std::string toString(const SampleData& sd)
    {
        std::stringstream ss;
        ss << "SampleData: "; 
        ss << "position = " << sd.position.x << "\t " << sd.position.y << "\t " << sd.position.z << "\t ";
#ifndef PGL_USE_DIRECTION_COMPRESSION
        ss << "\t direction = " << sd.direction.x << "\t " << sd.direction.y << "\t " << sd.direction.z << "\t ";
#else
        pgl_vec3f direction = dequantize_direction(sd.direction);
        ss << "\t direction = " << direction.x << "\t " << direction.y << "\t " << direction.z << "\t ";
#endif
        ss << "\t weight = " << sd.weight << "\t ";
        ss << "\t pdf = " << sd.pdf << "\t ";
        ss << "\t distance = " << sd.distance << "\t ";
        ss << "\t flags = " << sd.flags;
#ifdef OPENPGL_EF_RADIANCE_CACHES
#ifndef PGL_USE_COLOR_COMPRESSION
        ss << "\t radianceIn = " << sd.radianceIn.x << "\t " << sd.radianceIn.y << "\t " << sd.radianceIn.z << "\t ";
#else
        pgl_vec3f radianceIn = rgbe2vec3f(sd.radianceIn);
        ss << "\t radianceIn = " << radianceIn.x << "\t " << radianceIn.y << "\t " << radianceIn.z << "\t ";
#endif
        ss << "\t radianceInMISWeight = " << sd.radianceInMISWeight;
#ifndef PGL_USE_COLOR_COMPRESSION
        ss << "\t radianceOut = " << sd.radianceOut.x << "\t " << sd.radianceOut.y << "\t " << sd.radianceOut.z << "\t ";
#else
        pgl_vec3f radianceOut = rgbe2vec3f(sd.radianceOut);
        ss << "\t radianceOut = " << radianceOut.x << "\t " << radianceOut.y << "\t " << radianceOut.z << "\t ";
#endif
#ifndef PGL_USE_DIRECTION_COMPRESSION
        ss << "\t directionOut = " << sd.directionOut.x << "\t " << sd.directionOut.y << "\t " << sd.directionOut.z << "\t ";
#else
        pgl_vec3f directionOut = dequantize_direction(sd.directionOut);
        ss << "\t directionOut = " << directionOut.x << "\t " << directionOut.y << "\t " << directionOut.z << "\t ";
#endif
#endif
        ss << std::endl;
        
        return ss.str();   
    }

    inline bool SampleDataEqual(const PGLSampleData &compA,  const PGLSampleData &compB)
    {
        if(compA.position.x != compB.position.x || compA.position.y != compB.position.y ||
           compA.position.z != compB.position.z || 
#ifndef PGL_USE_DIRECTION_COMPRESSION
           compA.direction.x != compB.direction.x || compA.direction.y != compB.direction.y || compA.direction.z != compB.direction.z ||
#else
           compA.direction != compB.direction || 
#endif
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
                (compA.position.z    == compB.position.z     &&
#ifndef PGL_USE_DIRECTION_COMPRESSION  
                ( compA.direction.x < compB.direction.x  || (compA.direction.x    == compB.direction.x   &&  ( compA.direction.y < compB.direction.y  ||
                (compA.direction.y    == compB.direction.y   &&  ( compA.direction.z < compB.direction.z  )))))
#else
                compA.direction < compB.direction
#endif
                )))))))))));
    }

    inline bool ZeroValueSampleDataEqual(const PGLZeroValueSampleData &compA,  const PGLZeroValueSampleData &compB)
    {
        if( compA.position.x != compB.position.x || 
            compA.position.y != compB.position.y ||
            compA.position.z != compB.position.z ||
#ifndef PGL_USE_DIRECTION_COMPRESSION
            compA.direction.x != compB.direction.x || 
            compA.direction.y != compB.direction.y ||
            compA.direction.z != compB.direction.z
#else
            compA.direction != compB.direction
#endif
            )
        {
            return false;
        }
        return true;
    }

    inline bool ZeroValueSampleDataLess(const PGLZeroValueSampleData &compA,  const PGLZeroValueSampleData &compB )
    {
        return    compA.position.x < compB.position.x    ||   (compA.position.x    == compB.position.x    
            &&  ( compA.position.y < compB.position.y    ||   (compA.position.y    == compB.position.y     
            &&  ( compA.position.z < compB.position.z    ||   (compA.position.z    == compB.position.z
#ifndef PGL_USE_DIRECTION_COMPRESSION
            &&  ( compA.direction.x < compB.direction.x    ||   (compA.direction.x    == compB.direction.x
            &&  ( compA.direction.y < compB.direction.y    ||   (compA.direction.y    == compB.direction.y
            &&  ( compA.direction.z < compB.direction.z    ||   (compA.direction.z    == compB.direction.z     
                ))))))
#else
            && compA.direction < compB.direction
#endif
                )))));
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