// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../include/openpgl/openpgl.h"
#include "../openpglTypes.h"

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

///////////////////////////////////////////////////////////////////////////////
// Field //////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" PGLField pglNewField(PGLFieldArguments args)OPENPGL_CATCH_BEGIN
{
    
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
    GuidingField* gField = new GuidingField(gFieldSettings);
    return (PGLField) gField;
}
OPENPGL_CATCH_END(nullptr)

extern "C" void pglReleaseField(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (GuidingField *)field;
    delete gField;
}
OPENPGL_CATCH_END()

extern "C" size_t pglFieldGetIteration(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (GuidingField *)field;
    return gField->getIteration();
}
OPENPGL_CATCH_END(0)

extern "C" size_t pglFieldGetTotalSPP(PGLField field)OPENPGL_CATCH_BEGIN
{
    auto *gField = (GuidingField *)field;
    return gField->getTotalSPP();
}
OPENPGL_CATCH_END(0)

extern "C" void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds)
{
    auto *gField = (GuidingField *)field;
    openpgl::BBox sceneBounds;
    sceneBounds.lower = openpgl::Vector3(bounds.lower.x,bounds.lower.y,bounds.lower.z);
    sceneBounds.upper = openpgl::Vector3(bounds.upper.x,bounds.upper.y,bounds.upper.z);
    gField->setSceneBounds(sceneBounds);
}
/*
extern "C" pgl_box3f pglFieldGetSceneBounds(PGLField field)
{
    auto *gField = (GuidingField *)field;
    openpgl::BBox sceneBounds = gField->getSceneBounds();
    pgl_box3f bounds;
    pglBox3f(bounds, sceneBounds.lower.x, sceneBounds.lower.y, sceneBounds.lower.z, sceneBounds.upper.x, sceneBounds.upper.y, sceneBounds.upper.z);
    return bounds;
    //return gField->getIteration();
}
*/

extern "C"  void pglFieldUpdate(PGLField field, PGLSampleStorage sampleStorage, size_t numPerPixelSamples)OPENPGL_CATCH_BEGIN
{
    auto *gField = (GuidingField *)field;
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

extern "C"  PGLRegion pglFieldGetSurfaceRegion(PGLField field, pgl_point3f position, PGLSampler* sampler)OPENPGL_CATCH_BEGIN
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    GuidingSampler gSampler(sampler);
    auto *gField = (GuidingField *)field;
    const GuidingRegion* gRegion = gField->getSurfaceGuidingRegion(pos, &gSampler);
    return (PGLRegion) gRegion;
}
OPENPGL_CATCH_END(nullptr)

extern "C"  PGLRegion pglFieldGetVolumeRegion(PGLField field, pgl_point3f position, PGLSampler* sampler)OPENPGL_CATCH_BEGIN
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    GuidingSampler gSampler(sampler);
    auto *gField = (GuidingField *)field;
    const GuidingRegion* gRegion = gField->getVolumeGuidingRegion(pos, &gSampler);
    return (PGLRegion) gRegion;
}
OPENPGL_CATCH_END(nullptr)

///////////////////////////////////////////////////////////////////////////////
// Region /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" bool pglRegionGetValid(PGLRegion region)
{
    auto *gRegion = (GuidingRegion *)region;
    return gRegion->valid;
}
/*
extern "C" PGLDistribution pglRegionGetDistribution(PGLRegion region, pgl_point3f samplePosition, const bool &useParallaxComp)
{
    const openpgl::Point3 samplePos(samplePosition.x, samplePosition.y, samplePosition.z);
    auto *gRegion = (GuidingRegion *)region; 

    GuidingDistribution gDistribution;
    gRegion->getDistribution(gDistribution, samplePos, useParallaxComp);
    return (PGLDistribution)nullptr;
}
*/


///////////////////////////////////////////////////////////////////////////////
// SampleStorage //////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////


extern "C" PGLSampleStorage pglNewSampleStorage()OPENPGL_CATCH_BEGIN
{
    openpgl::SampleDataStorage* sampleStorage = new openpgl::SampleDataStorage();
    return (PGLSampleStorage) sampleStorage;
}
OPENPGL_CATCH_END(nullptr)

extern "C" void pglReleaseSampleStorage(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    delete gSampleStorage;
}

extern "C" void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, PGLSampleData& sample)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    openpgl::SampleData opglSample = /**(openpgl::SampleData*)*/sample;
    gSampleStorage->addSample(opglSample); 
}

extern "C" void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData* samples, size_t numSamples)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;

    openpgl::SampleData* opglSamples = (openpgl::SampleData*)samples;
    for(size_t n =0; n < numSamples; n++)
    {
        openpgl::SampleData opglSample = opglSamples[n];
        gSampleStorage->addSample(opglSample); 
    }
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

extern "C" size_t pglSampleStorageGetSizeSurface(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeSurface();
}

extern "C" size_t pglSampleStorageGetSizeVolume(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeVolume();
}


///////////////////////////////////////////////////////////////////////////////
// SampleData /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
/*
extern "C" void pglSampleDataSetPosition(PGLSampleData sampleData, const pgl_point3f pos)
{
    sampleData->position = pos;
}

extern "C" void pglSampleDataSetDirection(PGLSampleData sampleData, const pgl_vec3f direction)
{
    sampleData->direction = direction;
}

extern "C" void pglSampleDataSetDistance(PGLSampleData sampleData, const float distance)
{
    sampleData->distance = distance;
}

extern "C" void pglSampleDataSetPDF(PGLSampleData sampleData, const float pdf)
{
    sampleData->pdf = pdf;
}

extern "C" void pglSampleDataSetWeight(PGLSampleData sampleData, const float weight)
{
    sampleData->weight = weight;
}

extern "C" void pglSampleDataSetFlags(PGLSampleData sampleData, const int flags)
{
    sampleData->flags = flags;
}
*/

///////////////////////////////////////////////////////////////////////////////
// PathSegmentStorage /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" PGLPathSegmentStorage pglNewPathSegmentStorage()OPENPGL_CATCH_BEGIN
{
    openpgl::PathSegmentDataStorage<openpgl::GuidingRegion>* pathSegmentStorage = new openpgl::PathSegmentDataStorage<openpgl::GuidingRegion>();
    return (PGLPathSegmentStorage) pathSegmentStorage;
}
OPENPGL_CATCH_END(nullptr)

extern "C" void pglReleasePathSegmentStorage(PGLPathSegmentStorage pathSegmentStorage)OPENPGL_CATCH_BEGIN
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    delete gPathSegmentStorage;
}
OPENPGL_CATCH_END()

extern "C" void pglPathSegmentStorageReserve(PGLPathSegmentStorage pathSegmentStorage, size_t size)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    gPathSegmentStorage->reserve(size);
}

extern "C" void pglPathSegmentStorageClear(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    gPathSegmentStorage->clear();
}

extern "C" size_t pglPathSegmentStoragePrepareSamples(PGLPathSegmentStorage pathSegmentStorage,const bool &spaltSamples, PGLSampler* sampler,  const bool useNEEMiWeights, const bool guideDirectLight)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    GuidingSampler gSampler(sampler);
    return gPathSegmentStorage->prepareSamples(spaltSamples, &gSampler, useNEEMiWeights, guideDirectLight);
}

extern "C" const PGLSampleData* pglPathSegmentStorageGetSamples(PGLPathSegmentStorage pathSegmentStorage, size_t &nSamples)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    const std::vector<openpgl::SampleData> &opglSamples = gPathSegmentStorage->getSamples();
    nSamples = opglSamples.size();
    return (PGLSampleData*)opglSamples.data();
}


/*
extern "C" void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment pathSegment)
{
    //auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    //gPathSegmentStorage->
}
*/
extern "C" void pglPathSegmentStorageAddSample(PGLPathSegmentStorage pathSegmentStorage, PGLSampleData sample)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    gPathSegmentStorage->addSample(sample);
}


extern "C" PGLPathSegment pglPathSegmentNextSegment(PGLPathSegmentStorage pathSegmentStorage)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    return (PGLPathSegment)gPathSegmentStorage->next();

}
///////////////////////////////////////////////////////////////////////////////
// PathSegment ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" void pglPathSegmentSetPosition(PGLPathSegment pathSegment, pgl_point3f position)
{
    pathSegment->position = position;
}

extern "C" void pglPathSegmentSetNormal(PGLPathSegment pathSegment, pgl_vec3f normal)
{
    pathSegment->normal = normal;
}

extern "C" void pglPathSegmentSetDirectionIn(PGLPathSegment pathSegment, pgl_vec3f directionIn)
{
    pathSegment->directionIn = directionIn;
}

extern "C" pgl_vec3f pglPathSegmentGetDirectionIn(PGLPathSegment pathSegment)
{
    return pathSegment->directionIn;
}

extern "C" void pglPathSegmentSetPDFDirectionIn(PGLPathSegment pathSegment, float pdfDirectionIn)
{
    pathSegment->pdfDirectionIn = pdfDirectionIn;
}

extern "C" void pglPathSegmentSetDirectionOut(PGLPathSegment pathSegment, pgl_vec3f directionOut)
{
    pathSegment->directionOut = directionOut;
}

extern "C" void pglPathSegmentSetVolumeScatter(PGLPathSegment pathSegment, bool volumeScatter)
{
    pathSegment->volumeScatter = volumeScatter;
}

extern "C" void pglPathSegmentSetScatteringWeight(PGLPathSegment pathSegment, pgl_vec3f scatteringWeight)
{
    pathSegment->scatteringWeight = scatteringWeight;
}

extern "C" void pglPathSegmentSetDirectContribution(PGLPathSegment pathSegment, pgl_vec3f directContribution)
{
    pathSegment->directContribution = directContribution;
}

extern "C" void pglPathSegmentAddDirectContribution(PGLPathSegment pathSegment, pgl_vec3f directContribution)
{
    pathSegment->directContribution = directContribution;
}

extern "C" void pglPathSegmentSetScatteredContribution(PGLPathSegment pathSegment, pgl_vec3f scatteredContribution)
{
    pathSegment->scatteredContribution = scatteredContribution;
}

extern "C" void pglPathSegmentAddScatteredContribution(PGLPathSegment pathSegment, pgl_vec3f scatteredContribution)
{
    pathSegment->scatteredContribution = scatteredContribution;
}

extern "C" void pglPathSegmentSetMiWeight(PGLPathSegment pathSegment, float miWeight)
{
    pathSegment->miWeight = miWeight;
}

extern "C" void pglPathSegmentSetRussianRouletteProbability(PGLPathSegment pathSegment, float russianRouletteProbability)
{
    pathSegment->russianRouletteProbability = russianRouletteProbability;
}

extern "C" void pglPathSegmentSetEta(PGLPathSegment pathSegment, float eta)
{
    pathSegment->eta = eta;
}

extern "C" void pglPathSegmentSetIsDelta(PGLPathSegment pathSegment, bool isDelta)
{
    pathSegment->isDelta = isDelta;
}

extern "C" void pglPathSegmentSetRoughness(PGLPathSegment pathSegment, float roughness)
{
    pathSegment->roughness = roughness;
}

extern "C" void pglPathSegmentSetRegion(PGLPathSegment pathSegment, const PGLRegion region)
{
    pathSegment->regionPtr = region;
}

extern "C" void pglPathSegmentSetTransmittanceWeight(PGLPathSegment pathSegment, pgl_vec3f transmittanceWeight)
{
    pathSegment->transmittanceWeight = transmittanceWeight;
}




///////////////////////////////////////////////////////////////////////////////
// Distribution ///// /////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

extern "C" bool pglDistributionIsValid(PGLDistribution distribution)
{
    auto *gDistribution = (GuidingDistribution *)distribution;
    return gDistribution->isValid();
}



extern "C" PGLSurfaceSamplingDistribution pglNewSurfaceSamplingDistribution()OPENPGL_CATCH_BEGIN
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  new GuidedSurfaceSamplingDistribution();
    return (PGLSurfaceSamplingDistribution)gSurfaceSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)

extern "C" void pglReleaseSurfaceSamplingDistribution(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)OPENPGL_CATCH_BEGIN
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (GuidedSurfaceSamplingDistribution*)surfaceSamplingDistribution;
    delete gSurfaceSamplingDistribution;
}
OPENPGL_CATCH_END()

extern "C" void pglSurfaceSamplingDistributionInit(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp)
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (GuidedSurfaceSamplingDistribution*)surfaceSamplingDistribution;
    GuidingRegion *gRegion = (GuidingRegion*)region;
    openpgl::Vector3 opglSamplePosition(samplePosition.x, samplePosition.y, samplePosition.z);
    GuidingDistribution distriubtion = gRegion->getDistribution(opglSamplePosition, useParallaxComp);
    gSurfaceSamplingDistribution->init(distriubtion);
}

extern "C" void pglSurfaceSamplingDistributionApplyCosineProduct(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f normal)
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (GuidedSurfaceSamplingDistribution*)surfaceSamplingDistribution;
    openpgl::Vector3 opglNormal(normal.x, normal.y, normal.z);
    gSurfaceSamplingDistribution->applyCosineProduct(opglNormal);
}

extern "C" pgl_vec3f pglSurfaceSamplingDistributionSample(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_point2f sample)
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (GuidedSurfaceSamplingDistribution*)surfaceSamplingDistribution;   
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gSurfaceSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" float pglSurfaceSamplingDistributionPDF(PGLSurfaceSamplingDistribution surfaceSamplingDistribution, pgl_vec3f direction)
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (GuidedSurfaceSamplingDistribution*)surfaceSamplingDistribution;
    return gSurfaceSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" bool pglSurfaceSamplingDistributionIsValid(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (GuidedSurfaceSamplingDistribution*)surfaceSamplingDistribution; 
    return gSurfaceSamplingDistribution->valid();
}

extern "C" void pglSurfaceSamplingDistributionClear(PGLSurfaceSamplingDistribution surfaceSamplingDistribution)
{
    GuidedSurfaceSamplingDistribution* gSurfaceSamplingDistribution =  (GuidedSurfaceSamplingDistribution*)surfaceSamplingDistribution;
    gSurfaceSamplingDistribution->clear();
}

extern "C" PGLVolumeSamplingDistribution pglNewVolumeSamplingDistribution()OPENPGL_CATCH_BEGIN
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  new GuidedVolumeSamplingDistribution();
    return (PGLVolumeSamplingDistribution)gVolumeSamplingDistribution;
}
OPENPGL_CATCH_END(nullptr)

extern "C" void pglReleaseVolumeSamplingDistribution(PGLVolumeSamplingDistribution volumeSamplingDistribution)OPENPGL_CATCH_BEGIN
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  (GuidedVolumeSamplingDistribution*)volumeSamplingDistribution;
    delete gVolumeSamplingDistribution;
}
OPENPGL_CATCH_END()

extern "C" void pglVolumeSamplingDistributionInit(PGLVolumeSamplingDistribution volumeSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp)
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  (GuidedVolumeSamplingDistribution*)volumeSamplingDistribution;
    GuidingRegion *gRegion = (GuidingRegion*)region;
    openpgl::Vector3 opglSamplePosition(samplePosition.x, samplePosition.y, samplePosition.z);
    GuidingDistribution distriubtion = gRegion->getDistribution(opglSamplePosition, useParallaxComp);
    gVolumeSamplingDistribution->init(distriubtion);
}

extern "C" pgl_vec3f pglVolumeSamplingDistributionSample(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_point2f sample)
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  (GuidedVolumeSamplingDistribution*)volumeSamplingDistribution; 
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gVolumeSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" float pglVolumeSamplingDistributionPDF(PGLVolumeSamplingDistribution volumeSamplingDistribution, pgl_vec3f direction)
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  (GuidedVolumeSamplingDistribution*)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" bool pglVolumeSamplingDistributionIsValid(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  (GuidedVolumeSamplingDistribution*)volumeSamplingDistribution;
    return gVolumeSamplingDistribution->valid();
}

extern "C" void pglVolumeSamplingDistributionClear(PGLVolumeSamplingDistribution volumeSamplingDistribution)
{
    GuidedVolumeSamplingDistribution* gVolumeSamplingDistribution =  (GuidedVolumeSamplingDistribution*)volumeSamplingDistribution;
    gVolumeSamplingDistribution->clear();
}


extern "C" void pglFieldArgumentsSetDefaults(PGLFieldArguments &fieldArguments)
{
    fieldArguments.spatialStructureType = PGL_SPATIAL_STRUCTURE_KDTREE;
    fieldArguments.spatialSturctureArguments = new PGLKDTreeArguments();

    fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM;
    fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments();

    fieldArguments.useParallaxCompensation = true;
}