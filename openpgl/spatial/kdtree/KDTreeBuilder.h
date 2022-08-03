// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"
#include "KDTree.h"
#include "../../data/SampleStatistics.h"
#include "../../data/Range.h"
#include "../../include/openpgl/types.h"

/*
#if !defined(__WIN32__) and !defined(__MACOSX__)
    #include <tbb/task_scheduler_init.h>
#endif
*/
#include <tbb/concurrent_vector.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_invoke.h>
#include <iostream>
#include <limits>

namespace openpgl
{

template<typename TRegion, typename TContainer>
struct KDTreePartitionBuilder
{
    const static PGL_SPATIAL_STRUCTURE_TYPE SPATIAL_STRUCTURE_TYPE = PGL_SPATIAL_STRUCTURE_KDTREE;

    typedef KDTree SpatialStructure;

    struct Settings
    {
        size_t minSamples {100};
        size_t maxSamples {32000};
        size_t maxDepth{32};

        void serialize(std::ostream& stream) const;
        void deserialize(std::istream& stream);
        std::string toString() const;
    };

    void build(KDTree &kdTree, const BBox &bounds, TContainer &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings, const size_t &nCores) const
    {

        kdTree.init(bounds, 4096);
        dataStorage.resize(1);
        dataStorage[0].first.regionBounds = bounds;

        //KDNode &root kdTree.getRoot();
        updateTree(kdTree, samples, dataStorage, buildSettings, nCores);
    }

    void updateTree(KDTree &kdTree, TContainer &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings, const uint32_t &nCores) const
    {
        int numEstLeafs = dataStorage.size() + (samples.size()*2)/buildSettings.maxSamples+32;
        kdTree.m_nodes.reserve(4*numEstLeafs);
        dataStorage.reserve(2*numEstLeafs);

        KDNode &root = kdTree.getRoot();
        SampleStatistics sampleStats;
        sampleStats.clear();

        Range sampleRange;
        sampleRange.m_begin = 0;
        sampleRange.m_end = samples.size();

        size_t depth =1;


        if (root.isLeaf())
        {
            double x = 0.0f;
            double y = 0.0f;
            double z = 0.0f;

            for (const auto& sample : samples)
            {
                const Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
                sampleStats.addSample(samplePosition);
                x += samplePosition[0];
                y += samplePosition[1];
                z += samplePosition[2];
            }

            x /= double(samples.size());
            y /= double(samples.size());
            z /= double(samples.size());
        }
#ifdef OPENPGL_USE_OMP_THREADING
    #pragma omp parallel num_threads(nCores)
    #pragma omp single nowait
#else
/*
#if !defined(__WIN32__) and !defined(__MACOSX__)
        tbb::task_scheduler_init init(nCores);
#endif
*/
#endif
        updateTreeNode(&kdTree, root, depth, samples, sampleRange, sampleStats, &dataStorage, buildSettings);
        kdTree.finalize();
    }

    std::string toString() const;

private:

    inline typename TContainer::iterator pivotSplitSamples(typename TContainer::iterator begin, typename TContainer::iterator end,
                                                                        uint8_t splitDimension, float pivot) const
    {
        std::function<bool(typename TContainer::value_type)> pivotSplitPredicate
                = [splitDimension, pivot](typename TContainer::value_type sample) -> bool
        {
            const Vector3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
            return samplePosition[splitDimension] < pivot;

        };
        return std::partition(begin, end, pivotSplitPredicate);
    }


    inline typename TContainer::iterator pivotSplitSamplesWithStats(typename TContainer::iterator begin, typename TContainer::iterator end,
                                                                            uint8_t splitDimension, float pivot, SampleStatistics &statsLeft, SampleStatistics &statsRight) const
    {
        std::function<bool(typename TContainer::value_type)> pivotSplitPredicate
                = [splitDimension, pivot, &statsLeft, &statsRight](typename TContainer::value_type sample) -> bool
        {
            const Vector3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
            bool left = samplePosition[splitDimension] < pivot;
            if(left){
                statsLeft.addSample(samplePosition);
            }else{
                statsRight.addSample(samplePosition);
            }
            return left;
        };
        return std::partition(begin, end, pivotSplitPredicate);
    }


    inline void getSplitDimensionAndPosition(const SampleStatistics &sampleStats, uint8_t &splitDim, float &splitPos) const
    {
        const Vector3 sampleVariance = sampleStats.getVaraince();
        const Point3 sampleMean = sampleStats.getMean();

        auto maxDimension = [](const Vector3& v) -> uint8_t
        {
            return v[v[1] > v[0]] > v[2] ? v[1] > v[0] : 2;
        };

        splitDim = maxDimension(sampleVariance);
        splitPos = sampleMean[splitDim];
    }


    void updateTreeNode(KDTree *kdTree, KDNode &node, size_t depth, TContainer &samples, const Range sampleRange, const SampleStatistics sampleStats, tbb::concurrent_vector< std::pair<TRegion, Range> > *dataStorage, const Settings &buildSettings) const
    {
        if(sampleRange.size() == 0)
        {
            return;
        }
        uint8_t splitDim = {0};
        float splitPos = {0.0f};

        uint32_t nodeIdsLeftRight[2];
        Range sampleRangeLeftRight[2];
        SampleStatistics sampleStatsLeftRight[2];

        if (node.isLeaf())
        {
            uint32_t dataIdx = node.getDataIdx();
            std::pair<TRegion, Range> &regionAndRangeData = dataStorage->operator[](dataIdx);
            if(depth < buildSettings.maxDepth && regionAndRangeData.first.sampleStatistics.numSamples + sampleRange.size() > buildSettings.maxSamples)
            {
                SampleStatistics mergedSampleStats = regionAndRangeData.first.sampleStatistics;
                mergedSampleStats.merge( sampleStats );
                getSplitDimensionAndPosition(mergedSampleStats, splitDim, splitPos);

                //regionAndRangeData.first.onSplit();
                auto regionAndRangeDataRight = regionAndRangeData;

                // merge split handling
                regionAndRangeData.first.sampleStatistics.split(splitDim, splitPos, 0.25f, false);
                regionAndRangeDataRight.first.sampleStatistics.split(splitDim, splitPos, 0.25f, true);

                regionAndRangeData.first.splitFlag = true;
                regionAndRangeDataRight.first.splitFlag = true;

                regionAndRangeData.first.regionBounds.upper[splitDim] = splitPos;
                regionAndRangeDataRight.first.regionBounds.lower[splitDim] = splitPos;

                auto rigthDataItr = dataStorage->push_back(regionAndRangeDataRight);

                uint32_t rightDataIdx = std::distance(dataStorage->begin(), rigthDataItr);

                //we need to split the leaf node
                nodeIdsLeftRight[0] = kdTree->addChildrenPair();
                nodeIdsLeftRight[1] = nodeIdsLeftRight[0] + 1;
                node.setToInnerNode(splitDim, splitPos, nodeIdsLeftRight[0]);
                kdTree->getNode(nodeIdsLeftRight[0]).setDataNodeIdx(dataIdx);
                kdTree->getNode(nodeIdsLeftRight[1]).setDataNodeIdx(rightDataIdx);

                OPENPGL_ASSERT( kdTree->getNode(nodeIdsLeftRight[0]).isLeaf() );
                OPENPGL_ASSERT( kdTree->getNode(nodeIdsLeftRight[1]).isLeaf() );
            }
            else
            {
                regionAndRangeData.first.sampleStatistics.merge( sampleStats );
                regionAndRangeData.second = sampleRange;
                return;
            }
        }
        else
        {
            splitDim = node.getSplitDim();
            splitPos = node.getSplitPivot();
            nodeIdsLeftRight[0] = node.getLeftChildIdx();
            nodeIdsLeftRight[1] = nodeIdsLeftRight[0] + 1;
        }

        OPENPGL_ASSERT( !node.isLeaf() );
        OPENPGL_ASSERT (sampleRange.size() > 0);
        // TODO: update sample stats
        sampleStatsLeftRight[0].clear();
        sampleStatsLeftRight[1].clear();

        typename TContainer::iterator rPivotItr;

        auto begin = samples.begin() + sampleRange.m_begin, end = samples.begin() + sampleRange.m_end;
        if(kdTree->getNode(nodeIdsLeftRight[0]).isLeaf() || kdTree->getNode(nodeIdsLeftRight[1]).isLeaf() )
        {
            rPivotItr = pivotSplitSamplesWithStats(begin, end, splitDim, splitPos, sampleStatsLeftRight[0], sampleStatsLeftRight[1]);
        }
        else
        {
            rPivotItr = pivotSplitSamples(begin, end, splitDim, splitPos);
        }

        sampleRangeLeftRight[0] = Range(sampleRange.m_begin, std::distance(samples.begin(), rPivotItr));
        sampleRangeLeftRight[1] = Range(std::distance(samples.begin(), rPivotItr), sampleRange.m_end);

		/* This assert is a sanity check which is only valid with the assumption that the number of samples grows at same pace
		   as the number of spatial nodes: in practice this is not the case (e.g., after many 1spp iterations) 
		*/
        //OPENPGL_ASSERT(sampleRangeLeftRight[0].size() > 1);
        //OPENPGL_ASSERT(sampleRangeLeftRight[1].size() > 1);

#ifdef OPENPGL_USE_OMP_THREADING
    #pragma omp task mergeable
        updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[0]), depth + 1, sampleRangeLeftRight[0], sampleStatsLeftRight[0], dataStorage, buildSettings);
        updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[1]), depth + 1, sampleRangeLeftRight[1], sampleStatsLeftRight[1], dataStorage, buildSettings);
#else
    tbb::parallel_invoke(
        [&]{updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[0]), depth + 1, samples, sampleRangeLeftRight[0], sampleStatsLeftRight[0], dataStorage, buildSettings);},
        [&]{updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[1]), depth + 1, samples, sampleRangeLeftRight[1], sampleStatsLeftRight[1], dataStorage, buildSettings);}
    );
#endif
    }

};


template<class TRegion, typename TContainer>
inline std::string KDTreePartitionBuilder<TRegion, TContainer>::toString() const
{
    std::stringstream ss;
    ss << "KDTreePartitionBuilder" << std::endl;
    return ss.str();
}

template<class TRegion, typename TContainer>
inline std::string KDTreePartitionBuilder<TRegion, TContainer>::Settings::toString() const
{
    std::stringstream ss;
    ss << "KDTreePartitionBuilder::Settings:" << std::endl;
    ss << "  minSamples: " << minSamples << std::endl;
    ss << "  maxSamples: " << maxSamples << std::endl;
    ss << "  maxDepth: " << maxDepth << std::endl;

    return ss.str();
}


template<class TRegion, typename TContainer>
inline void KDTreePartitionBuilder<TRegion, TContainer>::Settings::serialize(std::ostream& stream)const
    {
        stream.write(reinterpret_cast<const char*>(&minSamples), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(&maxSamples), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(&maxDepth), sizeof(size_t));
    }

template<class TRegion, typename TContainer>
inline void KDTreePartitionBuilder<TRegion, TContainer>::Settings::deserialize(std::istream& stream)
    {
        stream.read(reinterpret_cast<char*>(&minSamples), sizeof(size_t));
        stream.read(reinterpret_cast<char*>(&maxSamples), sizeof(size_t));
        stream.read(reinterpret_cast<char*>(&maxDepth), sizeof(size_t));
    }
}