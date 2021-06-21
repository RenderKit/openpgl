#pragma once

#include "../include/openpgl/config.h"

#include "../openpgl_common.h"

#include "ISurfaceVolumeField.h"
#include "SurfaceVolumeField.h"

#include "directional/vmm/ParallaxAwareVMM.h"
#include "directional/vmm/AdaptiveSplitandMergeFactory.h"
#include "directional/vmm/VMMSurfaceSamplingDistribution.h"
#include "directional/vmm/VMMVolumeSamplingDistribution.h"

#include "spatial/kdtree/KDTreeBuilder.h"


#define FIELD_FILE_HEADER_STRING "OPENPGL_" OPENPGL_VERSION_STRING "_FIELD"

namespace openpgl {

struct FieldFactory {
    static ISurfaceVolumeField* newField(PGLFieldArguments args) {
        ISurfaceVolumeField* gField;

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
            delete spatialSturctureArguments;

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
            delete directionalDistributionArguments;

            gFieldSettings.useParallaxCompensation = args.useParallaxCompensation;
            gField = new GuidingField(gFieldSettings);
        } else {
            throw std::runtime_error("error: unrecognized field type");
        }

        return gField;
    }

    static ISurfaceVolumeField* newFieldFromFile(const std::string fieldFileName) {
        std::filebuf fb;
        fb.open (fieldFileName, std::ios::in | std::ios::binary);
        if (!fb.is_open()) throw std::runtime_error("error: couldn't open file");
        std::istream is(&fb);

        auto size = strlen(FIELD_FILE_HEADER_STRING) + 1;
        OPENPGL_ASSERT(size <= 256);
        char buf[256];
        is.read(&buf[0], size);
        if (!is) throw std::runtime_error("error: invalid file header");
        for (auto i = 0; i < size; i++)
        {
            if (buf[i] != FIELD_FILE_HEADER_STRING[i])
                throw std::runtime_error("error: invalid file header");
        }

        PGL_SPATIAL_STRUCTURE_TYPE spatialStructureType;
        is.read(reinterpret_cast<char*>(&spatialStructureType), sizeof(spatialStructureType));
        PGL_DIRECTIONAL_DISTRIBUTION_TYPE directionalDistributionType;
        is.read(reinterpret_cast<char*>(&directionalDistributionType), sizeof(directionalDistributionType));

        ISurfaceVolumeField* gField;

        if (spatialStructureType == PGL_SPATIAL_STRUCTURE_KDTREE && directionalDistributionType == PGL_DIRECTIONAL_DISTRIBUTION_PARALLAX_AWARE_VMM)
        {
            using DirectionalDistriubtionFactory = AdaptiveSplitAndMergeFactory<ParallaxAwareVonMisesFisherMixture<4, 32>>;
            using GuidingField = SurfaceVolumeField<DirectionalDistriubtionFactory, KDTreePartitionBuilder, VMMSurfaceSamplingDistribution<DirectionalDistriubtionFactory::Distribution>, VMMVolumeSamplingDistribution<DirectionalDistriubtionFactory::Distribution>>;

            gField = (ISurfaceVolumeField *)new GuidingField();
        } else {
            fb.close();
            throw std::runtime_error("error: unrecognized field type");
        }

        gField->deserialize(is);

        fb.close();

        return gField;
    }

    static void storeFieldToFile(ISurfaceVolumeField* gField, const std::string fieldFileName) {
        std::filebuf fb;
        fb.open (fieldFileName, std::ios::out | std::ios::binary);
        if (!fb.is_open()) throw std::runtime_error("error: couldn't open file!");
        std::ostream os(&fb);

        os.write(FIELD_FILE_HEADER_STRING, strlen(FIELD_FILE_HEADER_STRING) + 1);

        auto spatialStructureType = gField->getSpatialStructureType();
        os.write(reinterpret_cast<const char*>(&spatialStructureType), sizeof(spatialStructureType));
        auto directionalDistributionType = gField->getDirectionalDistributionType();
        os.write(reinterpret_cast<const char*>(&directionalDistributionType), sizeof(directionalDistributionType));

        gField->serialize(os);

        os.flush();
        fb.close();
    }
};

}

#undef FILE_HEADER_STRING