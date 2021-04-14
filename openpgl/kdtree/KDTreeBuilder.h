// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../openpgl_common.h"
#include "KDTree.h"
#include "../data/SampleStatistics.h"
#include "../data/Range.h"


#define USE_OMP_TASKS

#include <tbb/concurrent_vector.h>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include <iostream>
#include <limits>

namespace openpgl
{

template<typename TRegion, typename TRange>
struct KDTreePartitionBuilder
{

    struct Settings
    {
        size_t minSamples {100};
        size_t maxSamples {32000};
        size_t maxDepth{32};

        void serialize(std::ostream& stream) const;
        void deserialize(std::istream& stream);
        std::string toString() const;
    };

    void build(KDTree &kdTree, const BBox &bounds, typename TRange::Container &samples, tbb::concurrent_vector< std::pair<TRegion, TRange> > &dataStorage, const Settings &buildSettings, const size_t &nCores) const
    {

        kdTree.init(bounds, 4096);
        dataStorage.resize(1);
        dataStorage[0].first.regionBounds = bounds;

        //KDNode &root kdTree.getRoot();
        updateTree(kdTree, samples, dataStorage, buildSettings, nCores);
    }

    void updateTree(KDTree &kdTree, typename TRange::Container &samples, tbb::concurrent_vector< std::pair<TRegion, TRange> > &dataStorage, const Settings &buildSettings, const uint32_t &nCores) const
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
#ifdef USE_OMP_TASKS
#pragma omp parallel num_threads(nCores)
#pragma omp single nowait
#endif
        updateTreeNode(&kdTree, root, depth, sampleRange, sampleStats, &dataStorage, buildSettings);

    }

    std::string toString() const;

private:

    inline typename TRange::Container::iterator pivotSplitSamples(typename TRange::Container::iterator begin,
                                                                           typename TRange::Container::iterator end,
                                                                            uint8_t splitDimension, float pivot) const
    {
        std::function<bool(typename TRange::DataType)> pivotSplitPredicate
                = [splitDimension, pivot](typename TRange::DataType sample) -> bool
        {
            const Vector3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
            return samplePosition[splitDimension] < pivot;

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


    void updateTreeNode(KDTree *kdTree, KDNode &node, size_t depth, const TRange sampleRange, const SampleStatistics sampleStats, tbb::concurrent_vector< std::pair<TRegion, TRange> > *dataStorage, const Settings &buildSettings) const
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

        OPENPGL_ASSERT(sampleRangeLeftRight[0].size() > 1);
        OPENPGL_ASSERT(sampleRangeLeftRight[1].size() > 1);

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


template<class TRegion, typename TRange>
inline std::string KDTreePartitionBuilder<TRegion, TRange>::toString() const
{
    std::stringstream ss;
    ss << "KDTreePartitionBuilder" << std::endl;
    return ss.str();
}

template<class TRegion, typename TRange>
inline std::string KDTreePartitionBuilder<TRegion, TRange>::Settings::toString() const
{
    std::stringstream ss;
    ss << "KDTreePartitionBuilder::Settings:" << std::endl;
    ss << "  minSamples: " << minSamples << std::endl;
    ss << "  maxSamples: " << maxSamples << std::endl;
    ss << "  maxDepth: " << maxDepth << std::endl;

    return ss.str();
}


template<class TRegion, typename TRange>
inline void KDTreePartitionBuilder<TRegion, TRange>::Settings::serialize(std::ostream& stream)const
    {
        stream.write(reinterpret_cast<const char*>(&minSamples), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(&maxSamples), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(&maxDepth), sizeof(size_t));
    }

template<class TRegion, typename TRange>
inline void KDTreePartitionBuilder<TRegion, TRange>::Settings::deserialize(std::istream& stream)
    {
        stream.read(reinterpret_cast<char*>(&minSamples), sizeof(size_t));
        stream.read(reinterpret_cast<char*>(&maxSamples), sizeof(size_t));
        stream.read(reinterpret_cast<char*>(&maxDepth), sizeof(size_t));
    }
}