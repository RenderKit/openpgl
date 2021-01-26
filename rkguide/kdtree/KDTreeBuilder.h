// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include "KDTree.h"
#include "../data/SampleStatistics.h"
#include "../data/Range.h"


#define USE_OMP_TASKS

#ifdef USE_TBB
    #define USE_TBB_CONCURRENT
#endif

#ifdef  USE_TBB_CONCURRENT
#include <tbb/concurrent_vector.h>
#else
#include <mitsuba/guiding/AtomicallyGrowingVector.h>
#endif
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <iostream>
#include <limits>

namespace rkguide
{

template<typename TRegion, typename TRange>
struct KDTreePartitionBuilder
{

    struct Settings
    {
        size_t minSamples {100};
        size_t maxSamples {32000};
        size_t maxDepth{32};
    };

#ifdef  USE_TBB_CONCURRENT
    void build(KDTree &kdTree, const BBox &bounds, typename TRange::Container &samples, tbb::concurrent_vector< std::pair<TRegion, TRange> > &dataStorage, const Settings &buildSettings, const size_t &nCores) const
#else
    void build(KDTree &kdTree, const BBox &bounds, typename TRange::Container &samples, AtomicallyGrowingVector< std::pair<TRegion, TRange> > &dataStorage, const Settings &buildSettings, const size_t &nCores) const

#endif
    {

        kdTree.init(bounds, 4096);
        dataStorage.resize(1);
        dataStorage[0].first.regionBounds = bounds;

        //KDNode &root kdTree.getRoot();
        updateTree(kdTree, samples, dataStorage, buildSettings, nCores);
    }
#ifdef  USE_TBB_CONCURRENT
    void updateTree(KDTree &kdTree, typename TRange::Container &samples, tbb::concurrent_vector< std::pair<TRegion, TRange> > &dataStorage, const Settings &buildSettings, const uint32_t &nCores) const
#else
    void updateTree(KDTree &kdTree, typename TRange::Container &samples, AtomicallyGrowingVector< std::pair<TRegion, TRange> > &dataStorage, const Settings &buildSettings, const uint32_t &nCores) const
#endif
    {
        int numEstLeafs = dataStorage.size() + (samples.size()*2)/buildSettings.maxSamples+32;
        kdTree.m_nodes.reserve(4*numEstLeafs);
        dataStorage.reserve(2*numEstLeafs);

        KDNode &root = kdTree.getRoot();
        SampleStatistics sampleStats;
        sampleStats.clear();

        TRange sampleRange(samples);
        size_t depth =1;


        if (root.isLeaf())
        {
//            mitsuba::ref<mitsuba::Timer> statsTimer = new mitsuba::Timer();

            double x = 0.0f;
            double y = 0.0f;
            double z = 0.0f;

            for (const auto& sample : samples)
            {
                sampleStats.addSample(sample.position);
                x += sample.position[0];
                y += sample.position[1];
                z += sample.position[2];
            }

            x /= double(samples.size());
            y /= double(samples.size());
            z /= double(samples.size());
/*
            std::cout <<  "Stats building: time: " <<  statsTimer->getSeconds() << std::endl;

            mitsuba::ref<mitsuba::Timer> statsTimerPar = new mitsuba::Timer();
            SampleStatistics sampleStatsIdentity;
            SampleStatistics sampleStats2 = tbb::parallel_reduce(
                tbb::blocked_range<int>(0,samples.size()), 
                sampleStatsIdentity, 
                [&](tbb::blocked_range<int> r, SampleStatistics running_total)
                {
                        for (int i=r.begin(); i<r.end(); ++i)
                        {
                            running_total.addSample(samples[i].position);
                        }
                        return running_total;
                    }, SampleStatistics());
            //std::cout <<  "StatsPar building: time: " <<  statsTimerPar->getSeconds() << std::endl;
            std::cout <<  "mean double: " << x << "\t" << y << "\t" << z << std::endl;
            std::cout <<  "sampleStats: " << sampleStats.toString() << std::endl;
            std::cout <<  "sampleStats2: " << sampleStats2.toString() << std::endl;

            //std::cout <<  "Tree building: time: " <<  statsTimer->getSeconds() << std::endl;
            sampleStats = sampleStats2;
*/
        }
        //std::cout <<  std::numeric_limits<float>::max() << "\t" << sampleStats.getNumSamples() << std::endl;
#ifdef USE_OMP_TASKS
#pragma omp parallel num_threads(nCores)
#pragma omp single nowait
#endif
        updateTreeNode(&kdTree, root, depth, sampleRange, sampleStats, &dataStorage, buildSettings);

    }

private:

    inline typename TRange::Container::iterator pivotSplitSamples(typename TRange::Container::iterator begin,
                                                                           typename TRange::Container::iterator end,
                                                                            uint8_t splitDimension, float pivot) const
    {
        std::function<bool(typename TRange::DataType)> pivotSplitPredicate
                = [splitDimension, pivot](typename TRange::DataType sample) -> bool
        {
            return sample.position[splitDimension] < pivot;

        };
        return std::partition(begin, end, pivotSplitPredicate);
    }


    inline typename TRange::Container::iterator pivotSplitSamplesWithStats(typename TRange::Container::iterator begin,
                                                                           typename TRange::Container::iterator end,
                                                                            uint8_t splitDimension, float pivot, SampleStatistics &statsLeft, SampleStatistics &statsRight) const
    {
        std::function<bool(typename TRange::DataType)> pivotSplitPredicate
                = [splitDimension, pivot, &statsLeft, &statsRight](typename TRange::DataType sample) -> bool
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

#ifdef  USE_TBB_CONCURRENT
    void updateTreeNode(KDTree *kdTree, KDNode &node, size_t depth, const TRange sampleRange, const SampleStatistics sampleStats, tbb::concurrent_vector< std::pair<TRegion, TRange> > *dataStorage, const Settings &buildSettings) const
#else
    void updateTreeNode(KDTree *kdTree, KDNode &node, size_t depth, const TRange sampleRange, const SampleStatistics sampleStats, AtomicallyGrowingVector< std::pair<TRegion, TRange> > *dataStorage, const Settings &buildSettings) const
#endif
    {
        if(sampleRange.size() == 0)
        {
            return;
        }
        uint8_t splitDim = {0};
        float splitPos = {0.0f};

        uint32_t nodeIdsLeftRight[2];
        TRange sampleRangeLeftRight[2];
        SampleStatistics sampleStatsLeftRight[2];

        if (node.isLeaf())
        {
            uint32_t dataIdx = node.getDataIdx();
            std::pair<TRegion, TRange> &regionAndRangeData = dataStorage->operator[](dataIdx);
            if(depth < buildSettings.maxDepth && regionAndRangeData.first.sampleStatistics.numSamples + sampleRange.size() > buildSettings.maxSamples)
            {
                getSplitDimensionAndPosition(sampleStats, splitDim, splitPos);

                //regionAndRangeData.first.onSplit();
                auto regionAndRangeDataRight = regionAndRangeData;

                // merge split handling
                regionAndRangeData.first.sampleStatistics.split(splitDim, splitPos, 0.25f, false);
                regionAndRangeDataRight.first.sampleStatistics.split(splitDim, splitPos, 0.25f, true);

                regionAndRangeData.first.splitFlag = true;
                regionAndRangeDataRight.first.splitFlag = true;

                regionAndRangeData.first.regionBounds.upper[splitDim] = splitPos;
                regionAndRangeDataRight.first.regionBounds.lower[splitDim] = splitPos;

#ifdef  USE_TBB_CONCURRENT
                auto rigthDataItr = dataStorage->push_back(regionAndRangeDataRight);
#else
                auto rigthDataItr = dataStorage->back_insert(regionAndRangeDataRight);
#endif
                uint32_t rightDataIdx = std::distance(dataStorage->begin(), rigthDataItr);

                //we need to split the leaf node
                nodeIdsLeftRight[0] = kdTree->addChildrenPair();
                nodeIdsLeftRight[1] = nodeIdsLeftRight[0] + 1;
                node.setToInnerNode(splitDim, splitPos, nodeIdsLeftRight[0]);
                kdTree->getNode(nodeIdsLeftRight[0]).setDataNodeIdx(dataIdx);
                kdTree->getNode(nodeIdsLeftRight[1]).setDataNodeIdx(rightDataIdx);

                RKGUIDE_ASSERT( kdTree->getNode(nodeIdsLeftRight[0]).isLeaf() );
                RKGUIDE_ASSERT( kdTree->getNode(nodeIdsLeftRight[1]).isLeaf() );
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

        RKGUIDE_ASSERT( !node.isLeaf() );
        RKGUIDE_ASSERT (sampleRange.size() > 0);
        // TODO: update sample stats
        sampleStatsLeftRight[0].clear();
        sampleStatsLeftRight[1].clear();

        typename TRange::Container::iterator rPivotItr;

        if(kdTree->getNode(nodeIdsLeftRight[0]).isLeaf() || kdTree->getNode(nodeIdsLeftRight[1]).isLeaf() )
        {
            rPivotItr = pivotSplitSamplesWithStats(sampleRange.begin(), sampleRange.end(), splitDim, splitPos, sampleStatsLeftRight[0], sampleStatsLeftRight[1]);
        }
        else
        {
            rPivotItr = pivotSplitSamples(sampleRange.begin(), sampleRange.end(), splitDim, splitPos);
        }

        sampleRangeLeftRight[0].m_start = sampleRange.begin();
        sampleRangeLeftRight[0].m_end = rPivotItr;

        sampleRangeLeftRight[1].m_start = rPivotItr;
        sampleRangeLeftRight[1].m_end = sampleRange.end();

        RKGUIDE_ASSERT(sampleRangeLeftRight[0].size() > 1);
        RKGUIDE_ASSERT(sampleRangeLeftRight[1].size() > 1);

#ifdef USE_OMP_TASKS
#pragma omp task mergeable
        updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[0]), depth + 1, sampleRangeLeftRight[0], sampleStatsLeftRight[0], dataStorage, buildSettings);
        updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[1]), depth + 1, sampleRangeLeftRight[1], sampleStatsLeftRight[1], dataStorage, buildSettings);
#else
#pragma omp parallel for num_threads(2)
        for (size_t i = 0; i < 2; i++)
        {
            updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[i]), depth + 1, sampleRangeLeftRight[i], sampleStatsLeftRight[i], dataStorage, buildSettings);
        }
#endif
    }

};

}