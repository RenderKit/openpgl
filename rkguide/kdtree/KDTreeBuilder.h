// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "KDTree.h"
#include "../data/SampleStatistics.h"
#include "../data/Range.h"

#include <tbb/concurrent_vector.h>

#include <iostream>
#include <limits>

namespace rkguide
{

template<typename TSamples, typename TRegion>
struct KDTreePartitionBuilder
{

    struct Settings
    {
        size_t minSamples {100};
        size_t maxSamples {32000};
        size_t maxDepth{32};
    };

    void build(KDTree &kdTree, const BBox &bound, std::vector<TSamples> &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings) const
    {

        kdTree.init(bound);
        dataStorage.resize(1);
        //KDNode &root kdTree.getRoot();
        updateTree(kdTree, samples, dataStorage, buildSettings);
    }

    void updateTree(KDTree &kdTree, std::vector<TSamples> &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings) const
    {
        KDNode &root = kdTree.getRoot();
        SampleStatistics sampleStats;
        sampleStats.clear();
        size_t numSamples = samples.size();

        Range sampleRange;
        sampleRange.start = 0;
        sampleRange.end = numSamples-1;

        size_t depth =1;


        if (root.isLeaf())
        {
            for (const auto& sample : samples)
            {
                sampleStats.addSample(sample.position);

            }
        }
        std::cout <<  std::numeric_limits<float>::max() << "\t" << sampleStats.getNumSamples() << std::endl;
        updateTreeNode(kdTree, root, depth, samples, sampleRange, sampleStats, dataStorage, buildSettings);

        std::cout << "KDTree: numNodes = " << kdTree.getNumNodes() << std::endl;
        std::cout << "DataStorage: numData = " << dataStorage.size() << std::endl;

    }

private:


    typename std::vector<TSamples>::iterator pivotSplitSamplesWithStats(typename std::vector<TSamples>::iterator begin,
                                                                           typename std::vector<TSamples>::iterator end,
                                                                            uint8_t splitDimension, float pivot, SampleStatistics &statsLeft, SampleStatistics &statsRight) const
    {
        std::function<bool(TSamples)> pivotSplitPredicate
                = [splitDimension, pivot, &statsLeft, &statsRight](TSamples sample) -> bool
        {
            bool left = sample.position[splitDimension] < pivot;
            if(left){
                statsLeft.addSample(sample.position);
            }else{
                statsRight.addSample(sample.position);
            }
            return left;
        };
        return std::partition(begin, end, pivotSplitPredicate);
    }


    void getSplitDimensionAndPosition(const SampleStatistics &sampleStats, uint8_t &splitDim, float &splitPos) const
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

    void updateTreeNode(KDTree &kdTree, KDNode &node, size_t depth, std::vector<TSamples> &samples, const Range &sampleRange, const SampleStatistics &sampleStats, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings) const
    {
        //std::cout << "updateTreeNode: " << "depth: " << depth << " \t sampleRange: " <<  sampleRange.start << " | " << sampleRange.end << std::endl;
        uint8_t splitDim = {0};
        float splitPos = {0.0f};

        uint32_t nodeIdsLeftRight[2];
        Range sampleRangeLeftRight[2];
        SampleStatistics sampleStatsLeftRight[2];

        if (node.isLeaf())
        {
            //std::cout << "\tisLeaf" << std::endl;
            uint32_t dataIdx = node.getDataIdx();
            std::pair<TRegion, Range> &regionAndRangeData = dataStorage[dataIdx];
            //std::cout << regionAndRangeData.first.sampleStatistics.numSamples + sampleRange.size() << std::endl;
            //std::cout << buildSettings.maxSamples << std::endl;
            //std::cout << buildSettings.maxDepth << std::endl;
            if(depth < buildSettings.maxDepth && regionAndRangeData.first.sampleStatistics.numSamples + sampleRange.size() > buildSettings.maxSamples)
            {
                //std::cout << "\t\tSplit" << std::endl;
                //regionAndRangeData.first.onSplit();
                regionAndRangeData.first.sampleStatistics.decay(0.5f);
                const auto rigthDataItr = dataStorage.push_back(regionAndRangeData);
                uint32_t rightDataIdx = std::distance(dataStorage.begin(), rigthDataItr);
                getSplitDimensionAndPosition(sampleStats, splitDim, splitPos);
                //we need to split the leaf node
                nodeIdsLeftRight[0] = kdTree.addChildrenPair();
                nodeIdsLeftRight[1] = nodeIdsLeftRight[0] + 1;
                node.setToInnerNode(splitDim, splitPos, nodeIdsLeftRight[0]);
                kdTree.getNode(nodeIdsLeftRight[0]).setDataNodeIdx(dataIdx);
                kdTree.getNode(nodeIdsLeftRight[1]).setDataNodeIdx(rightDataIdx);

                RKGUIDE_ASSERT( kdTree.getNode(nodeIdsLeftRight[0]).isLeaf() );
                RKGUIDE_ASSERT( kdTree.getNode(nodeIdsLeftRight[1]).isLeaf() );
            }
            else
            {
                //std::cout << "\t\tNo Split" << std::endl;
                std::cout << "dataId: " << dataIdx << "\tdepth: " << depth << "\tsize: " << sampleRange.size()  << std::endl;
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

        RKGUIDE_ASSERT( !node.isLeaf() );
        RKGUIDE_ASSERT (sampleRange.size() > 0);

        auto rBeginItr = samples.begin() + sampleRange.start;
        auto rEndItr = samples.begin() + sampleRange.end+1;
        // TODO: update sample stats
        sampleStatsLeftRight[0].clear();
        sampleStatsLeftRight[1].clear();
        auto rPivotItr = pivotSplitSamplesWithStats(rBeginItr, rEndItr, splitDim, splitPos, sampleStatsLeftRight[0], sampleStatsLeftRight[1]);

        size_t pivotOffSet  = std::distance( samples.begin(), rPivotItr );
        RKGUIDE_ASSERT (pivotOffSet > 0); 
        RKGUIDE_ASSERT (pivotOffSet > sampleRange.start); 
        RKGUIDE_ASSERT (pivotOffSet < sampleRange.end); 
        sampleRangeLeftRight[0].start = sampleRange.start;
        sampleRangeLeftRight[0].end = pivotOffSet-1;

        sampleRangeLeftRight[1].start = pivotOffSet;
        sampleRangeLeftRight[1].end = sampleRange.end;

        RKGUIDE_ASSERT(sampleRangeLeftRight[0].size() > 1);
        RKGUIDE_ASSERT(sampleRangeLeftRight[1].size() > 1);

        for (size_t i = 0; i < 2; i++)
        {
            if( sampleRangeLeftRight[i].size() >0 )
            {
                KDNode &nodeNext = kdTree.getNode(nodeIdsLeftRight[i]);
                updateTreeNode(kdTree, nodeNext, depth + 1, samples, sampleRangeLeftRight[i], sampleStatsLeftRight[i], dataStorage, buildSettings);
            }
        }
    }

};

}