// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openpgl/openpgl.h"
#include "../openpglTypes.h"

//#include "../field/Field.h"

using namespace openpgl;


#define THROW_IF_NULL(obj, name)                         \
  if (obj == nullptr)                                    \
  throw std::runtime_error(std::string("null ") + name + \
                           std::string(" provided to ") + __FUNCTION__)

// convenience macros for repeated use of the above
#define THROW_IF_NULL_OBJECT(obj) THROW_IF_NULL(obj, "handle")
#define THROW_IF_NULL_STRING(str) THROW_IF_NULL(str, "string")

///////////////////////////////////////////////////////////////////////////////
// Field //////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" PGLField pglNewField(PGLFieldArguments args)
{
    GuidingField* gField = new GuidingField();
    return (PGLField) gField;
}

extern "C" uint32_t pglFieldGetIteration(PGLField field)
{
    auto *gField = (GuidingField *)field;
    return gField->getIteration();
}
/*
extern "C" uint32_t pglGetTrainingIteration(PGLField field)
{
    auto *gField = (GuidingField *)field;
    return gField->getTrainingIteration();
}
*/
extern "C" uint32_t pglFieldGetTotalSPP(PGLField field)
{
    auto *gField = (GuidingField *)field;
    return gField->getTotalSPP();
}

extern "C" void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds)
{
    auto *gField = (GuidingField *)field;
    //return gField->getIteration();
}

extern "C" pgl_box3f pglFieldGetSceneBounds(PGLField field)
{
    auto *gField = (GuidingField *)field;
    pgl_box3f bounds;
    return bounds;
    //return gField->getIteration();
}


extern "C"  void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage, uint32_t numPerPixelSamples)
{
    auto *gField = (GuidingField *)field;
}

extern "C"  PGLRegion pglFieldGetSurfaceRegion(PGLField field, pgl_point3f position, PGLSampler sampler)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    
    auto *gField = (GuidingField *)field;
    const GuidingRegion* gRegion = gField->getSurfaceGuidingRegion(pos, nullptr);
    return (PGLRegion) gRegion;
}

extern "C"  PGLRegion pglFieldGetVolumeRegion(PGLField field, pgl_point3f position, PGLSampler sampler)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    
    auto *gField = (GuidingField *)field;
    const GuidingRegion* gRegion = gField->getVolumeGuidingRegion(pos, nullptr);
    return (PGLRegion) gRegion;
}

///////////////////////////////////////////////////////////////////////////////
// Region /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" bool pglRegionIsValid(PGLRegion region)
{
    auto *gRegion = (GuidingRegion *)region;
    return gRegion->isValid();
}

extern "C" PGLDistribution pglRegionGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool &useParallaxComp)
{
    const openpgl::Point3 samplePos(samplePosition.x, samplePosition.y, samplePosition.z);
    auto *gRegion = (GuidingRegion *)region; 

    GuidingDistribution gDistribution;
    gRegion->getDistribution(gDistribution, samplePos, useParallaxComp);
    return (PGLDistribution)nullptr;
}


///////////////////////////////////////////////////////////////////////////////
// SampleStorage //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, PGLSampleData sample)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
//    gSampleStorage->addSample2(sample); 
}

extern "C" void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, PGLSampleData* samples, uint32_t numSamples)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
//    gSampleStorage->addSamples2(sample); 
}

extern "C" void pglSampleStorageReserve(PGLSampleStorage sampleStorage, const size_t sizeSurface, const size_t sizeVolume)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->reserveSurface(sizeSurface);
    gSampleStorage->reserveVolume(sizeVolume);
}

extern "C" void pglSampleStorageClear(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->clearSurface();
    gSampleStorage->clearVolume();
}

///////////////////////////////////////////////////////////////////////////////
// PathSegmentStorage /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" PGLPathSegmentStorage pglNewPathSegmentStorage()
{
    openpgl::PathSegmentDataStorage* pathSegmentStorage = new openpgl::PathSegmentDataStorage();
    return (PGLPathSegmentStorage) pathSegmentStorage;
}

extern "C" void pglPathSegmentStorageReserve(PGLPathSegmentStorage pathSegmentStorage, size_t size)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->reserve(size);
}

extern "C" void pglPathSegmentStorageClear(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->clear();
}

extern "C" size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage, const bool useNEEMiWeights, const bool guideDirectLight)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return gPathSegmentStorage->prepareSamples();
}
extern "C" void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment pathSegment)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    //gPathSegmentStorage->
}

extern "C" void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
//    gPathSegmentStorage->addSample(sample, nullptr);
}

///////////////////////////////////////////////////////////////////////////////
// Distribution ///// /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" bool pglDistributionIsValid(PGLDistribution distribution)
{
    auto *gDistriubtion = (GuidingDistribution *)distribution;
    return gDistriubtion->isValid();
}