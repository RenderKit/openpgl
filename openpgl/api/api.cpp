// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../include/openpgl/openpgl.h"
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

extern "C" void pglReleaseField(PGLField field)
{
    //auto *gField = (GuidingField *)field;
    //delete gField;
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
/*
extern "C" void pglFieldSetSceneBounds(PGLField field, pgl_box3f bounds)
{
    auto *gField = (GuidingField *)field;
    openpgl::BBox sceneBounds;
    sceneBounds.lower = openpgl::Vector3(bounds.lower.x,bounds.lower.y,bounds.lower.z);
    sceneBounds.upper = openpgl::Vector3(bounds.upper.x,bounds.upper.y,bounds.upper.z);
    gField->setSceneBounds(sceneBounds);
}

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

extern "C"  void pglFieldUpdate(PGLField field, pgl_box3f bounds, PGLSampleStorage sampleStorage, uint32_t numPerPixelSamples)
{
    auto *gField = (GuidingField *)field;
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    if (gField->getIteration() == 0)
    {
        //pgl_box3f bounds = pglFieldGetSceneBounds(field);
        openpgl::BBox sceneBounds;
        sceneBounds.lower = openpgl::Vector3(bounds.lower.x,bounds.lower.y,bounds.lower.z);
        sceneBounds.upper = openpgl::Vector3(bounds.upper.x,bounds.upper.y,bounds.upper.z);
        gField->buildField(sceneBounds, gSampleStorage->m_surfaceContainer, gSampleStorage->m_volumeContainer);
    }
    else
    {
        gField->updateField(gSampleStorage->m_surfaceContainer, gSampleStorage->m_volumeContainer);
    }

    gField->addTrainingIteration(numPerPixelSamples);
}

extern "C"  PGLRegion pglFieldGetSurfaceRegion(PGLField field, pgl_point3f position, PGLSampler* sampler)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    GuidingSampler gSampler(sampler);
    auto *gField = (GuidingField *)field;
    const GuidingRegion* gRegion = gField->getSurfaceGuidingRegion(pos, &gSampler);
    return (PGLRegion) gRegion;
}

extern "C"  PGLRegion pglFieldGetVolumeRegion(PGLField field, pgl_point3f position, PGLSampler* sampler)
{
    const openpgl::Point3 pos(position.x, position.y, position.z);
    GuidingSampler gSampler(sampler);
    auto *gField = (GuidingField *)field;
    const GuidingRegion* gRegion = gField->getVolumeGuidingRegion(pos, &gSampler);
    return (PGLRegion) gRegion;
}

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


extern "C" PGLSampleStorage pglNewSampleStorage()
{
    openpgl::SampleDataStorage* sampleStorage = new openpgl::SampleDataStorage();
    return (PGLSampleStorage) sampleStorage;
}

extern "C" void pglReleaseSampleStorage(PGLSampleStorage sampleStorage)
{
    //auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
}

/*
extern "C" void pglSampleStorageSetSceneBounds(PGLSampleStorage sampleStorage, pgl_box3f bounds)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    openpgl::BBox sceneBound;
    sceneBound.lower = openpgl::Vector3(bounds.lower.x,bounds.lower.y,bounds.lower.z);
    sceneBound.upper = openpgl::Vector3(bounds.upper.x,bounds.upper.y,bounds.upper.z);
    //gSampleStorage->setSceneBounds(sceneBound);
}
*/
extern "C" void pglSampleStorageAddSample(PGLSampleStorage sampleStorage, PGLSampleData& sample)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    openpgl::DirectionalSampleData opglSample = /**(openpgl::DirectionalSampleData*)*/sample;
    gSampleStorage->addSample(opglSample); 
}

extern "C" void pglSampleStorageAddSamples(PGLSampleStorage sampleStorage, const PGLSampleData* samples, uint32_t numSamples)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;

    openpgl::DirectionalSampleData* opglSamples = (openpgl::DirectionalSampleData*)samples;
    for(uint32_t n =0; n < numSamples; n++)
    {
        openpgl::DirectionalSampleData opglSample = opglSamples[n];
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

extern "C" uint32_t pglSampleStorageGetSizeSurface(PGLSampleStorage sampleStorage)
{
    auto *gSampleStorage = (openpgl::SampleDataStorage *)sampleStorage;
    return gSampleStorage->sizeSurface();
}

extern "C" uint32_t pglSampleStorageGetSizeVolume(PGLSampleStorage sampleStorage)
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

extern "C" PGLPathSegmentStorage pglNewPathSegmentStorage()
{
    openpgl::PathSegmentDataStorage<openpgl::GuidingRegion>* pathSegmentStorage = new openpgl::PathSegmentDataStorage<openpgl::GuidingRegion>();
    return (PGLPathSegmentStorage) pathSegmentStorage;
}

extern "C" void pglReleasePathSegmentStorage(PGLPathSegmentStorage pathSegmentStorage)
{
    //auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    //return (PGLPathSegmentStorage) pathSegmentStorage;
}

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

extern "C" const PGLSampleData* pglPathSegmentStorageGetSamples(PGLPathSegmentStorage pathSegmentStorage, uint32_t &nSamples)
{
    auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    const std::vector<openpgl::DirectionalSampleData> &opglSamples = gPathSegmentStorage->getSamples();
    nSamples = opglSamples.size();
    return (PGLSampleData*)opglSamples.data();
}



extern "C" void pglPathSegmentStorageAddSegment(PGLPathSegmentStorage pathSegmentStorage, PGLPathSegment pathSegment)
{
    //auto *gPathSegmentStorage = (openpgl::PathSegmentDataStorage<openpgl::GuidingRegion> *)pathSegmentStorage;
    //gPathSegmentStorage->
}

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

extern "C" void pglPathSegmentSetIsDelta(PGLPathSegment pathSegment, float isDelta)
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



extern "C" PGLBSDFSamplingDistribution pglNewBSDFSamplingDistribution()
{
    GuidingBSDFSamplingDistribution* gBSDFSamplingDistribution =  new GuidingBSDFSamplingDistribution();
    return (PGLBSDFSamplingDistribution)gBSDFSamplingDistribution;
}

extern "C" void pglReleaseBSDFSamplingDistribution(PGLBSDFSamplingDistribution bsdfSamplingDistribution)
{
    GuidingBSDFSamplingDistribution* gBSDFSamplingDistribution =  (GuidingBSDFSamplingDistribution*)bsdfSamplingDistribution;
    delete gBSDFSamplingDistribution;
}

extern "C" void pglBSDFSamplingDistributionInit(PGLBSDFSamplingDistribution bsdfSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, pgl_vec3f normal, bool useParallaxComp, bool useCosine)
{
    GuidingBSDFSamplingDistribution* gBSDFSamplingDistribution =  (GuidingBSDFSamplingDistribution*)bsdfSamplingDistribution;
    GuidingRegion *gRegion = (GuidingRegion*)region;
    openpgl::Vector3 opglSamplePosition(samplePosition.x, samplePosition.y, samplePosition.z);
    openpgl::Vector3 opglNormal(normal.x, normal.y, normal.z);
    GuidingDistribution distriubtion = gRegion->getDistribution(opglSamplePosition, useParallaxComp);
    gBSDFSamplingDistribution->init(distriubtion, opglNormal, useCosine);
    //gBSDFSamplingDistribution->m_distributions[0] = gBSDFSamplingDistribution->m_liDistribution;
    //gBSDFSamplingDistribution->m_weights[0] = 1.0f;
    //gBSDFSamplingDistribution->m_numDistributions = 1;
    //gBSDFSamplingDistribution->m_productIntegral = 1.0f;

    //if ( useCosine )
    //{
    //    gBSDFSamplingDistribution->m_distributions[0].product(1.0f, openpgl::Vector3(normal.x, normal.y, normal.z), 2.18853f);
    //}
}

extern "C" pgl_vec3f pglBSDFSamplingDistributionSample(PGLBSDFSamplingDistribution bsdfSamplingDistribution, pgl_point2f sample)
{
    GuidingBSDFSamplingDistribution* gBSDFSamplingDistribution =  (GuidingBSDFSamplingDistribution*)bsdfSamplingDistribution;   
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gBSDFSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" float pglBSDFSamplingDistributionPDF(PGLBSDFSamplingDistribution bsdfSamplingDistribution, pgl_vec3f direction)
{
    GuidingBSDFSamplingDistribution* gBSDFSamplingDistribution =  (GuidingBSDFSamplingDistribution*)bsdfSamplingDistribution;
    return gBSDFSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" bool pglBSDFSamplingDistributionIsValid(PGLBSDFSamplingDistribution bsdfSamplingDistribution)
{
    GuidingBSDFSamplingDistribution* gBSDFSamplingDistribution =  (GuidingBSDFSamplingDistribution*)bsdfSamplingDistribution; 
    return gBSDFSamplingDistribution->valid();
}

extern "C" void pglBSDFSamplingDistributionClear(PGLBSDFSamplingDistribution bsdfSamplingDistribution)
{
    GuidingBSDFSamplingDistribution* gBSDFSamplingDistribution =  (GuidingBSDFSamplingDistribution*)bsdfSamplingDistribution;
    gBSDFSamplingDistribution->clear();
}

extern "C" PGLPhaseFunctionSamplingDistribution pglNewPhaseFunctionSamplingDistribution()
{
    GuidingPhaseFunctionSamplingDistribution* gPhaseFunctionSamplingDistribution =  new GuidingPhaseFunctionSamplingDistribution();
    return (PGLPhaseFunctionSamplingDistribution)gPhaseFunctionSamplingDistribution;
}

extern "C" void pglReleasePhaseFunctionSamplingDistribution(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution)
{
    GuidingPhaseFunctionSamplingDistribution* gPhaseFunctionSamplingDistribution =  (GuidingPhaseFunctionSamplingDistribution*)phaseFunctionSamplingDistribution;
    delete gPhaseFunctionSamplingDistribution;
}

extern "C" void pglPhaseFunctionSamplingDistributionInit(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution, PGLRegion region, pgl_point3f samplePosition, bool useParallaxComp)
{
    GuidingPhaseFunctionSamplingDistribution* gPhaseFunctionSamplingDistribution =  (GuidingPhaseFunctionSamplingDistribution*)phaseFunctionSamplingDistribution;
    GuidingRegion *gRegion = (GuidingRegion*)region;
    openpgl::Vector3 opglSamplePosition(samplePosition.x, samplePosition.y, samplePosition.z);
    //openpgl::Vector3 opglNormal(normal.x, normal.y, normal.z);
    GuidingDistribution distriubtion = gRegion->getDistribution(opglSamplePosition, useParallaxComp);
    gPhaseFunctionSamplingDistribution->init(distriubtion);
}

extern "C" pgl_vec3f pglPhaseFunctionSamplingDistributionSample(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution, pgl_point2f sample)
{
    GuidingPhaseFunctionSamplingDistribution* gPhaseFunctionSamplingDistribution =  (GuidingPhaseFunctionSamplingDistribution*)phaseFunctionSamplingDistribution; 
    openpgl::Point2 opglSample(sample.x, sample.y);
    openpgl::Vector3 opglDirection = gPhaseFunctionSamplingDistribution->sample(opglSample);
    pgl_vec3f pglDirection;
    pglVec3f(pglDirection, opglDirection.x, opglDirection.y, opglDirection.z);
    return pglDirection;
}

extern "C" float pglPhaseFunctionSamplingDistributionPDF(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution, pgl_vec3f direction)
{
    GuidingPhaseFunctionSamplingDistribution* gPhaseFunctionSamplingDistribution =  (GuidingPhaseFunctionSamplingDistribution*)phaseFunctionSamplingDistribution;
    return gPhaseFunctionSamplingDistribution->pdf(openpgl::Vector3(direction.x, direction.y, direction.z));
}

extern "C" bool pglPhaseFunctionSamplingDistributionIsValid(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution)
{
    GuidingPhaseFunctionSamplingDistribution* gPhaseFunctionSamplingDistribution =  (GuidingPhaseFunctionSamplingDistribution*)phaseFunctionSamplingDistribution;
    return gPhaseFunctionSamplingDistribution->valid();
}

extern "C" void pglPhaseFunctionSamplingDistributionClear(PGLPhaseFunctionSamplingDistribution phaseFunctionSamplingDistribution)
{
    GuidingPhaseFunctionSamplingDistribution* gPhaseFunctionSamplingDistribution =  (GuidingPhaseFunctionSamplingDistribution*)phaseFunctionSamplingDistribution;
    gPhaseFunctionSamplingDistribution->clear();
}


extern "C" void pglFieldArgumentsSetDefaults(PGLFieldArguments &fieldArguments)
{
    fieldArguments.spatialStructureType = PGL_SPATIAL_STRUCTURE_KDTREE;
    fieldArguments.spatialSturctureArguments = new PGLKDTreeArguments();

    fieldArguments.directionalDistributionType = PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM;
    fieldArguments.directionalDistributionArguments = new PGLVMMFactoryArguments();

    fieldArguments.useParallaxCompensation = true;
}