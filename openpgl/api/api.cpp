// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../include/openpgl/openpgl.h"
//#include "../openpglTypes.h"

#include "openpgl_common.h"

#include "data/PathSegmentDataStorage.h"
#include "data/PathSegmentData.h"
#include "data/SampleData.h"
#include "data/SampleDataStorage.h"

#include "field/ISurfaceVolumeField.h"
#include "field/SurfaceVolumeField.h"
#include "directional/ISurfaceSamplingDistribution.h"
#include "directional/IVolumeSamplingDistribution.h"
#include "directional/vmm/ParallaxAwareVMM.h"
#include "directional/vmm/AdaptiveSplitandMergeFactory.h"
#include "directional/vmm/VMMSurfaceSamplingDistribution.h"
#include "directional/vmm/VMMVolumeSamplingDistribution.h"
#include "sampler/Sampler.h"

#include "spatial/kdtree/KDTreeBuilder.h"

using namespace openpgl;

#define THROW_IF_NULL(obj, name)                         \
  if (obj == nullptr)                                    \
  throw std::runtime_error(std::string("null ") + name + \
                           std::string(" provided to ") + __FUNCTION__)

// convenience macros for repeated use of the above
#define THROW_IF_NULL_OBJECT(obj) THROW_IF_NULL(obj, "handle")
#define THROW_IF_NULL_STRING(str) THROW_IF_NULL(str, "string")

#define OPENPGL_CATCH_BEGIN try {
#define OPENPGL_CATCH_END(a)                                        \
  }                                                                 \
  catch (const std::bad_alloc &)                                    \
  {                                                                 \
    std::cout <<                                                    \
             "Open PGL was unable to allocate memory" << std::endl; \
    return a;                                                       \
  }                                                                 \
  catch (const std::exception &e)                                   \
  {                                                                 \
    std::cout << e.what() << std::endl;              \
    return a;                                                       \
  }                                                                 \
  catch (...)                                                       \
  {                                                                 \
    std::cout <<                                                    \
               "an unrecognized exception was caught" << std::endl; \
    return a;                                                       \
  }

typedef ISurfaceVolumeField IGuidingField;


///////////////////////////////////////////////////////////////////////////////
// Field //////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT PGLField pglNewField(PGLFieldArguments args)OPENPGL_CATCH_BEGIN
{
    
    IGuidingField* gField = nullptr;
    
    if (args.spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && 
        args.directionalDistributionType ==  PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM )
    {
        using DirectionalDistriubtionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<4, 32>>;
        using GuidingField = SurfaceVolumeField<DirectionalDistriubtionFactory, KDTreePartitionBuilder, VMMSurfaceSamplingDistribution<DirectionalDistriubtionFactory::Distribution>, VMMVolumeSamplingDistribution<DirectionalDistriubtionFactory::Distribution>>;
        
        GuidingField::Settings gFieldSettings;
        gFieldSettings.settings.decayOnSpatialSplit   = 0.25f;
        gFieldSettings.settings.deterministic         = false;

        PGLKDTreeArguments *spatialSturctureArguments = (PGLKDTreeArguments*)args.spatialSturctureArguments;
        gFieldSettings.settings.useStochasticNNLookUp = spatialSturctureArguments->knnLookup;
        gFieldSettings.settings.spatialSubdivBuilderSettings.minSamples = spatialSturctureArguments->minSamples;
        gFieldSettings.settings.spatialSubdivBuilderSettings.maxSamples = spatialSturctureArguments->maxSamples;
        gFieldSettings.settings.spatialSubdivBuilderSettings.maxDepth   = spatialSturctureArguments->maxDepth;

        PGLVMMFactoryArguments *directionalDistributionArguments = (PGLVMMFactoryArguments*)args.directionalDistributionArguments;
        gFieldSettings.distributionFactorySettings.weightedEMCfg.initK = directionalDistributionArguments->initK;
        gFieldSettings.distributionFactorySettings.weightedEMCfg.maxK = directionalDistributionArguments->maxK;
        gFieldSettings.distributionFactorySettings.weightedEMCfg.maxEMIterrations = directionalDistributionArguments->maxEMIterrations;

        gFieldSettings.distributionFactorySettings.weightedEMCfg.maxKappa = directionalDistributionArguments->maxKappa;
        gFieldSettings.distributionFactorySettings.weightedEMCfg.maxMeanCosine = openpgl::KappaToMeanCosine<float>(gFieldSettings.distributionFactorySettings.weightedEMCfg.maxKappa);
        gFieldSettings.distributionFactorySettings.weightedEMCfg.convergenceThreshold = directionalDistributionArguments->convergenceThreshold;
        gFieldSettings.distributionFactorySettings.weightedEMCfg.weightPrior = directionalDistributionArguments->weightPrior;
        gFieldSettings.distributionFactorySettings.weightedEMCfg.meanCosinePriorStrength = directionalDistributionArguments->meanCosinePriorStrength;
        gFieldSettings.distributionFactorySettings.weightedEMCfg.meanCosinePrior = directionalDistributionArguments->meanCosinePrior;

        gFieldSettings.distributionFactorySettings.splittingThreshold = directionalDistributionArguments->splittingThreshold;
        gFieldSettings.distributionFactorySettings.mergingThreshold = directionalDistributionArguments->mergingThreshold;


        gFieldSettings.distributionFactorySettings.partialReFit = directionalDistributionArguments->partialReFit;
        gFieldSettings.distributionFactorySettings.maxSplitItr = directionalDistributionArguments->maxSplitItr;

        gFieldSettings.distributionFactorySettings.useSplitAndMerge = directionalDistributionArguments->useSplitAndMerge;
        gFieldSettings.distributionFactorySettings.minSamplesForSplitting = directionalDistributionArguments->minSamplesForSplitting;
        gFieldSettings.distributionFactorySettings.minSamplesForPartialRefitting = directionalDistributionArguments->minSamplesForPartialRefitting;
        gFieldSettings.distributionFactorySettings.minSamplesForMerging = directionalDistributionArguments->minSamplesForMerging;

        gFieldSettings.useParallaxCompensation = args.useParallaxCompensation;
        gField = new GuidingField(gFieldSettings);
    }else if(args.spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && 
        args.directionalDistributionType ==  PGL_DIRECTIONAL_DISTRIBUTION_QUADTREE )
    {
        // TODO
    }
    
    

    return (PGLField) gField;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleaseField(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    delete gField;
}
OPENPGL_CATCH_END()

extern "C" OPENPGL_DLLEXPORT size_t pglFieldGetIteration(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    return gField->getIteration();
}
OPENPGL_CATCH_END(0)

extern "C" OPENPGL_DLLEXPORT size_t pglFieldGetTotalSPP(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    return gField->getTotalSPP();
}
OPENPGL_CATCH_END(0)

extern "C" OPENPGL_DLLEXPORT void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds)
{
    auto *gField = (IGuidingField *)field;
    openpgl::BBox sceneBounds;
    sceneBounds.lower = openpgl::Vector3(bounds.lower.x,bounds.lower.y,bounds.lower.z);
    sceneBounds.upper = openpgl::Vector3(bounds.upper.x,bounds.upper.y,bounds.upper.z);
    gField->setSceneBounds(sceneBounds);
}
/*
extern "C" OPENPGL_DLLEXPORT pgl_box3f pglFieldGetSceneBounds(PGLField field)
{
    auto *gField = (GuidingField *)field;
    openpgl::BBox sceneBounds = gField->getSceneBounds();
    pgl_box3f bounds;
    pglBox3f(bounds, sceneBounds.lower.x, sceneBounds.lower.y, sceneBounds.lower.z, sceneBounds.upper.x, sceneBounds.upper.y, sceneBounds.upper.z);
    return bounds;
    //return gField->getIteration();
}
*/

extern "C" OPENPGL_DLLEXPORT  void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage, size_t numPerPixelSamples)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    if (gField->getIteration() == 0)
    {
        gField->buildField(gSampleStorage->m_surfaceContainer, gSampleStorage->m_volumeContainer);
    }
    else
    {
        gField->updateField(gSampleStorage->m_surfaceContainer, gSampleStorage->m_volumeContainer);
    }

    gField->addTrainingIteration(numPerPixelSamples);
}
OPENPGL_CATCH_END()
/*
extern "C" OPENPGL_DLLEXPORT  PGLRegion pglFieldGetSurfaceRegion(PGLField field, pgl_point3f position, PGLSampler* sampler)OPENPGL_CATCH_BEGIN
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    SamplerC gSampler(sampler);
    float sample1D = gSampler.next1D();
    auto *gField = (GuidingField *)field;
    const IRegion* gRegion = gField->getSurfaceGuidingRegion(pos, sample1D);
    return (PGLRegion) gRegion;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT  PGLRegion pglFieldGetVolumeRegion(PGLField field, pgl_point3f position, PGLSampler* sampler)OPENPGL_CATCH_BEGIN
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    SamplerC gSampler(sampler);
    float sample1D = gSampler.next1D();
    auto *gField = (GuidingField *)field;
    const IRegion* gRegion = gField->getVolumeGuidingRegion(pos, sample1D);
    return (PGLRegion) gRegion;
}
OPENPGL_CATCH_END(nullptr)
*/

extern "C" OPENPGL_DLLEXPORT  PGLSurfaceSamplingDistribution pglFieldNewSurfaceSamplingDistribution(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    ISurfaceSamplingDistribution* surfaceSamplingDistribution = gField->newSurfaceSamplingDistribution();
    return (PGLSurfaceSamplingDistribution) surfaceSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT  bool pglFieldInitSurfaceSamplingDistriubtion(PGLField field, PGLSurfaceSamplingDistribution surfaceSamplingDistriubtion, pgl_point3f position, const float sample1D, const bool useParallaxComp)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    auto *gField = (IGuidingField *)field;
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution = (ISurfaceSamplingDistribution*)surfaceSamplingDistriubtion;
    return gField->initSurfaceSamplingDistribution(gSurfaceSamplingDistribution, pos, sample1D, useParallaxComp);
}
extern "C" OPENPGL_DLLEXPORT  PGLVolumeSamplingDistribution pglFieldNewVolumeSamplingDistribution(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (IGuidingField *)field;
    IVolumeSamplingDistribution* volumeSamplingDistribution = gField->newVolumeSamplingDistribution();
    return (PGLVolumeSamplingDistribution) volumeSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT  bool pglFieldInitVolumeSamplingDistriubtion(PGLField field, PGLVolumeSamplingDistribution volumeSamplingDistriubtion, pgl_point3f position, const float sample1D, const bool useParallaxComp)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    auto *gField = (IGuidingField *)field;
    IVolumeSamplingDistribution* gVolumeSamplingDistribution = (IVolumeSamplingDistribution*)volumeSamplingDistriubtion;
    return gField->initVolumeSamplingDistribution(gVolumeSamplingDistribution, pos, sample1D, useParallaxComp);
}


///////////////////////////////////////////////////////////////////////////////
// Region /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*
extern "C" OPENPGL_DLLEXPORT bool pglRegionGetValid(PGLRegion region)
{
    auto *gRegion = (IRegion *)region;
    return gRegion->valid;
}
*/
/*
extern "C" OPENPGL_DLLEXPORT PGLDistribution pglRegionGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool &useParallaxComp)
{
    const openpgl::Point3 samplePos(samplePosition.x, samplePosition.y, samplePosition.z);
    auto *gRegion = (IRegion *)region; 

    GuidingDistribution gDistribution;
    gRegion->getDistribution(gDistribution, samplePos, useParallaxComp);
    return (PGLDistribution)nullptr;
}
*/


///////////////////////////////////////////////////////////////////////////////
// SampleStorage //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


extern "C" OPENPGL_DLLEXPORT PGLSampleStorage pglNewSampleStorage()OPENPGL_CATCH_BEGIN
{
    openpgl::SampleDataStorage* sampleStorage = new openpgl::SampleDataStorage();
    return (PGLSampleStorage) sampleStorage;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleaseSampleStorage(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    delete gSampleStorage;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, PGLSampleData& sample)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    openpgl::SampleData opglSample = /**(openpgl::SampleData*)*/sample;
    gSampleStorage->addSample(opglSample); 
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData* samples, size_t numSamples)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;

    openpgl::SampleData* opglSamples = (openpgl::SampleData*)samples;
    for(size_t n =0; n < numSamples; n++)
    {
        openpgl::SampleData opglSample = opglSamples[n];
        gSampleStorage->addSample(opglSample); 
    }
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageReserve(PGLSampleStorage sampleStorage, const size_t sizeSurface, const size_t sizeVolume)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->reserveSurface(sizeSurface);
    gSampleStorage->reserveVolume(sizeVolume);
}

extern "C" OPENPGL_DLLEXPORT void pglSampleStorageClear(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    gSampleStorage->clearSurface();
    gSampleStorage->clearVolume();
}

extern "C" OPENPGL_DLLEXPORT size_t pglSampleStorageGetSizeSurface(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeSurface();
}

extern "C" OPENPGL_DLLEXPORT size_t pglSampleStorageGetSizeVolume(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeVolume();
}


///////////////////////////////////////////////////////////////////////////////
// SampleData /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*
extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetPosition(PGLSampleData sampleData, const pgl_point3f pos)
{
    sampleData->position = pos;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetDirection(PGLSampleData sampleData, const pgl_vec3f direction)
{
    sampleData->direction = direction;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetDistance(PGLSampleData sampleData, const float distance)
{
    sampleData->distance = distance;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetPDF(PGLSampleData sampleData, const float pdf)
{
    sampleData->pdf = pdf;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetWeight(PGLSampleData sampleData, const float weight)
{
    sampleData->weight = weight;
}

extern "C" OPENPGL_DLLEXPORT void pglSampleDataSetFlags(PGLSampleData sampleData, const int flags)
{
    sampleData->flags = flags;
}
*/

///////////////////////////////////////////////////////////////////////////////
// PathSegmentStorage /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT PGLPathSegmentStorage pglNewPathSegmentStorage()OPENPGL_CATCH_BEGIN
{
    openpgl::PathSegmentDataStorage* pathSegmentStorage = new openpgl::PathSegmentDataStorage();
    return (PGLPathSegmentStorage) pathSegmentStorage;
}
OPENPGL_CATCH_END(nullptr)

extern "C" OPENPGL_DLLEXPORT void pglReleasePathSegmentStorage(PGLPathSegmentStorage pathSegmentStorage)OPENPGL_CATCH_BEGIN
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    delete gPathSegmentStorage;
}
OPENPGL_CATCH_END()

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageReserve(PGLPathSegmentStorage pathSegmentStorage, size_t size)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->reserve(size);
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageClear(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->clear();
}

extern "C" OPENPGL_DLLEXPORT size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage,const bool &spaltSamples, PGLSampler* sampler,  const bool useNEEMiWeights, const bool guideDirectLight)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    SamplerC gSampler(sampler);
    return gPathSegmentStorage->prepareSamples(spaltSamples, &gSampler, useNEEMiWeights, guideDirectLight);
}

extern "C" OPENPGL_DLLEXPORT const PGLSampleData* pglPathSegmentStorageGetSamples(PGLPathSegmentStorage pathSegmentStorage, size_t &nSamples)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    const std::vector<openpgl::SampleData> &opglSamples = gPathSegmentStorage->getSamples();
    nSamples = opglSamples.size();
    return (PGLSampleData*)opglSamples.data();
}


/*
extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment pathSegment)
{
    //auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    //gPathSegmentStorage->
}
*/
extern "C" OPENPGL_DLLEXPORT void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    gPathSegmentStorage->addSample(sample);
}


extern "C" OPENPGL_DLLEXPORT PGLPathSegment pglPathSegmentNextSegment(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage *)pathSegmentStorage;
    return (PGLPathSegment)gPathSegmentStorage->next();

}
///////////////////////////////////////////////////////////////////////////////
// PathSegment ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetPosition(PGLPathSegment pathSegment, pgl_point3f position)
{
    pathSegment->position = position;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetNormal(PGLPathSegment pathSegment, pgl_vec3f normal)
{
    pathSegment->normal = normal;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetDirectionIn(PGLPathSegment pathSegment, pgl_vec3f directionIn)
{
    pathSegment->directionIn = directionIn;
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglPathSegmentGetDirectionIn(PGLPathSegment pathSegment)
{
    return pathSegment->directionIn;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetPDFDirectionIn(PGLPathSegment pathSegment, float pdfDirectionIn)
{
    pathSegment->pdfDirectionIn = pdfDirectionIn;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetDirectionOut(PGLPathSegment pathSegment, pgl_vec3f directionOut)
{
    pathSegment->directionOut = directionOut;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetVolumeScatter(PGLPathSegment pathSegment, bool volumeScatter)
{
    pathSegment->volumeScatter = volumeScatter;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetScatteringWeight(PGLPathSegment pathSegment, pgl_vec3f scatteringWeight)
{
    pathSegment->scatteringWeight = scatteringWeight;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetDirectContribution(PGLPathSegment pathSegment, pgl_vec3f directContribution)
{
    pathSegment->directContribution = directContribution;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentAddDirectContribution(PGLPathSegment pathSegment, pgl_vec3f directContribution)
{
    pathSegment->directContribution = directContribution;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetScatteredContribution(PGLPathSegment pathSegment, pgl_vec3f scatteredContribution)
{
    pathSegment->scatteredContribution = scatteredContribution;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentAddScatteredContribution(PGLPathSegment pathSegment, pgl_vec3f scatteredContribution)
{
    pathSegment->scatteredContribution = scatteredContribution;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetMiWeight(PGLPathSegment pathSegment, float miWeight)
{
    pathSegment->miWeight = miWeight;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetRussianRouletteProbability(PGLPathSegment pathSegment, float russianRouletteProbability)
{
    pathSegment->russianRouletteProbability = russianRouletteProbability;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetEta(PGLPathSegment pathSegment, float eta)
{
    pathSegment->eta = eta;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetIsDelta(PGLPathSegment pathSegment, bool isDelta)
{
    pathSegment->isDelta = isDelta;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetRoughness(PGLPathSegment pathSegment, float roughness)
{
    pathSegment->roughness = roughness;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetRegion(PGLPathSegment pathSegment, const PGLRegion region)
{
    pathSegment->regionPtr = region;
}

extern "C" OPENPGL_DLLEXPORT void pglPathSegmentSetTransmittanceWeight(PGLPathSegment pathSegment, pgl_vec3f transmittanceWeight)
{
    pathSegment->transmittanceWeight = transmittanceWeight;
}




///////////////////////////////////////////////////////////////////////////////
// Distribution ///// /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*
extern "C" OPENPGL_DLLEXPORT bool pglDistributionIsValid(PGLDistribution distribution)
{
    auto *gDistribution = (GuidingDistribution *)distribution;
    return gDistribution->isValid();
}
*/
/*
extern "C" OPENPGL_DLLEXPORT PGLSurfaceSamplingDistribution pglNewSurfaceSamplingDistribution()OPENPGL_CATCH_BEGIN
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  new GuidedSurfaceSamplingDistribution();
    return (PGLSurfaceSamplingDistribution)gSurfaceSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)
*/

extern "C" OPENPGL_DLLEXPORT void pglReleaseSurfaceSamplingDistribution(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)OPENPGL_CATCH_BEGIN
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    delete gSurfaceSamplingDistribution;
}
OPENPGL_CATCH_END()
/*
extern "C" OPENPGL_DLLEXPORT void pglSurfaceSamplingDistributionInit(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    IRegion *gRegion = (IRegion*)region;
    openpgl::Vector3 opglSamplePosition(samplePosition.x, samplePosition.y, samplePosition.z);
    GuidingDistribution distriubtion = gRegion->getDistribution(opglSamplePosition, useParallaxComp);
    gSurfaceSamplingDistribution->init(&distriubtion);
}
*/
extern "C" OPENPGL_DLLEXPORT void pglSurfaceSamplingDistributionApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    openpgl::Vector3 opglNormal(normal.x, normal.y, normal.z);
    gSurfaceSamplingDistribution->applyCosineProduct(opglNormal);
}

extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglSurfaceSamplingDistributionSample(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;   
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gSurfaceSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" OPENPGL_DLLEXPORT float pglSurfaceSamplingDistributionPDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" OPENPGL_DLLEXPORT bool pglSurfaceSamplingDistributionIsValid(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution; 
    return gSurfaceSamplingDistribution->valid();
}

extern "C" OPENPGL_DLLEXPORT void pglSurfaceSamplingDistributionClear(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    gSurfaceSamplingDistribution->clear();
}

extern "C" OPENPGL_DLLEXPORT PGLRegion pglSurfaceSamplingGetRegion(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    ISurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (ISurfaceSamplingDistribution*)surfaceSamplingDistribution;
    const IRegion* gRegion = gSurfaceSamplingDistribution->getRegion();
    return (PGLRegion) gRegion;
}

/*
extern "C" OPENPGL_DLLEXPORT PGLVolumeSamplingDistribution pglNewVolumeSamplingDistribution()OPENPGL_CATCH_BEGIN
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  new GuidedVolumeSamplingDistribution();
    return (PGLVolumeSamplingDistribution)gVolumeSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)
*/
extern "C" OPENPGL_DLLEXPORT void pglReleaseVolumeSamplingDistribution(PGLVolumeSamplingDistribution volumeSamplingDistribution)OPENPGL_CATCH_BEGIN
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    delete gVolumeSamplingDistribution;
}
OPENPGL_CATCH_END()
/*
extern "C" OPENPGL_DLLEXPORT void pglVolumeSamplingDistributionInit(PGLVolumeSamplingDistribution volumeSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    IRegion *gRegion = (IRegion*)region;
    openpgl::Vector3 opglSamplePosition(samplePosition.x, samplePosition.y, samplePosition.z);
    GuidingDistribution distriubtion = gRegion->getDistribution(opglSamplePosition, useParallaxComp);
    gVolumeSamplingDistribution->init(&distriubtion);
}
*/
extern "C" OPENPGL_DLLEXPORT pgl_vec3f pglVolumeSamplingDistributionSample(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point2f sample)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution; 
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gVolumeSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" OPENPGL_DLLEXPORT float pglVolumeSamplingDistributionPDF(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" OPENPGL_DLLEXPORT bool pglVolumeSamplingDistributionIsValid(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->valid();
}

extern "C" OPENPGL_DLLEXPORT void pglVolumeSamplingDistributionClear(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    IVolumeSamplingDistribution* gVolumeSamplingDistribution =  (IVolumeSamplingDistribution*)volumeSamplingDistribution;
    gVolumeSamplingDistribution->clear();
}


extern "C" OPENPGL_DLLEXPORT void pglFieldArgumentsSetDefaults(PGLFieldArguments &fieldArguments, const PGL_SPATIAL_STRUCTURE_TYPE spatialType, const PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalType)
{
    switch (spatialType)
    {
    case PGL_SPATIAL_STRUCTURE_TYPE::PGL_SPATIAL_STRUCTURE_KDTREE:
        fieldArguments.spatialStructureType = PGL_SPATIAL_STRUCTURE_KDTREE;
        fieldArguments.spatialSturctureArguments = new PGLKDTreeArguments();
        break;
    
    default:
        fieldArguments.spatialStructureType = PGL_SPATIAL_STRUCTURE_KDTREE;
        fieldArguments.spatialSturctureArguments = new PGLKDTreeArguments();
        break;
    }


    switch (directionalType)
    {
    case PGL_DIRECTIONAL_DISTRIBUTION_TYPE::PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM:
        fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM;
        fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments();
        fieldArguments.useParallaxCompensation = true;
        break;
    
    default:
        fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM;
        fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments();
        fieldArguments.useParallaxCompensation = true;
        break;
    }


}