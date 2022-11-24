// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"
#include "KDTree.h"
#include "../../data/SampleStatistics.h"
#include "../../data/Range.h"
#include "../../include/openpgl/types.h"

#ifdef USE_EMBREE_PARALLEL
#define TASKING_TBB
#include <embreeSrc/common/algorithms/parallel_partition.h>
#include <embreeSrc/common/algorithms/parallel_reduce.h>
#endif
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

template<typename TRegion, typename TSamplesContainer, typename TInvalidSamplesContainer>
struct KDTreePartitionBuilder
{
    const static PGL_SPATIAL_STRUCTURE_TYPE SPATIAL_STRUCTURE_TYPE = PGL_SPATIAL_STRUCTURE_KDTREE;

    typedef KDTree SpatialStructure;

#ifdef USE_EMBREE_PARALLEL
    static const size_t PARALLEL_THRESHOLD = 4 * 1024;
    static const size_t PARALLEL_PARTITION_BLOCK_SIZE = 4 * 1024;
#endif

    struct Settings
    {
        size_t minSamples {100};
        size_t maxSamples {PGL_TREE_MAX_SAMPLE_PER_LEAF};
        size_t maxDepth {32};

        void serialize(std::ostream& stream) const;
        void deserialize(std::istream& stream);
        std::string toString() const;

        bool operator==(const Settings& b) const {
            bool equal = true;
            if(minSamples != b.minSamples || maxSamples != b.maxSamples ||
                maxDepth != b.maxDepth)
            {
                equal = false;
            }
            return equal;
        }
    };

    void build(KDTree &kdTree, const BBox &bounds, TSamplesContainer &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings) const
    {

        kdTree.init(bounds, 4096);
        dataStorage.resize(1);
        dataStorage[0].first.regionBounds = bounds;

        updateTree(kdTree, samples, dataStorage, buildSettings);
    }

    void updateTree(KDTree &kdTree, TSamplesContainer &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage, const Settings &buildSettings) const
    {
        int numEstLeafs = dataStorage.size() + (samples.size()*2)/buildSettings.maxSamples+32;
        kdTree.m_nodes.reserve(4*numEstLeafs);
        dataStorage.reserve(2*numEstLeafs);

        KDNode &root = kdTree.getRoot();
        SampleStatistics sampleStats;
        sampleStats.clear();

        BBox bounds = kdTree.getBounds();

        Range sampleRange;
        sampleRange.m_begin = 0;
        sampleRange.m_end = samples.size();

        size_t depth =1;

        if (root.isLeaf())
        {
#ifdef USE_EMBREE_PARALLEL
            IntegerSampleStatistics iSampleStats = embree::parallel_reduce(size_t(0), samples.size(), IntegerSampleStatistics(bounds), [&] (const embree::range<size_t>& r) -> IntegerSampleStatistics {  
              IntegerSampleStatistics stats(bounds);
              for (size_t i=r.begin(); i<r.end(); i++) {
                const PGLSampleData sample = samples[i];
                const Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
                stats.addSample(samplePosition);
              }
              return stats;
            }, [] (const IntegerSampleStatistics& a, const IntegerSampleStatistics& b) { return IntegerSampleStatistics::merge(a,b); });
#else
            IntegerSampleStatistics iSampleStats(bounds);
            for (const auto& sample : samples)
            {
                const Point3 samplePosition(sample.position.x, sample.position.y, sample.position.z);
                iSampleStats.addSample(samplePosition);
            }
#endif
            sampleStats = iSampleStats.getSampleStatistics();
        }
        updateTreeNode(&kdTree, root, depth, bounds, samples, sampleRange, sampleStats, &dataStorage, buildSettings);
        kdTree.finalize();
    }


void insertTree(KDTree &kdTree, TInvalidSamplesContainer &samples, tbb::concurrent_vector< std::pair<TRegion, Range> > &dataStorage) const
    {
        //int numEstLeafs = dataStorage.size() + (samples.size()*2)/buildSettings.maxSamples+32;
        //kdTree.m_nodes.reserve(4*numEstLeafs);
        //dataStorage.reserve(2*numEstLeafs);

        KDNode &root = kdTree.getRoot();
        //SampleStatistics sampleStats;
        //sampleStats.clear();

        Range sampleRange;
        sampleRange.m_begin = 0;
        sampleRange.m_end = samples.size();

        size_t depth =1;

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
        insertTreeNode(&kdTree, root, depth, samples, sampleRange, &dataStorage);
    }

    std::string toString() const;

private:
    template<class TContainer>
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


    inline typename TSamplesContainer::iterator pivotSplitSamplesWithStats(typename TSamplesContainer::iterator begin, typename TSamplesContainer::iterator end,
                                                                            uint8_t splitDimension, float pivot, SampleStatistics &statsLeft, SampleStatistics &statsRight) const
    {
        std::function<bool(typename TSamplesContainer::value_type)> pivotSplitPredicate
                = [splitDimension, pivot, &statsLeft, &statsRight](typename TSamplesContainer::value_type sample) -> bool
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

#ifdef USE_EMBREE_PARALLEL
    template<class DataType>
    inline size_t pivotSplitSamples2(DataType* samples, const size_t begin, const size_t end,
                                                                    uint8_t splitDimension, float pivot) const
    {
        auto isLeft = [&] (const DataType &sample) { return Vector3(sample.position.x, sample.position.y, sample.position.z)[splitDimension] < pivot; };
        size_t center = 0;
        bool parallel = (end-begin) < PARALLEL_THRESHOLD ? false : true;
        if (!parallel) {
            center = embree::serial_partitioning(samples, begin, end, isLeft);
        } else {
            center = embree::parallel_partitioning(samples, begin, end, isLeft, PARALLEL_PARTITION_BLOCK_SIZE);
        }
        return center;
    }

    inline size_t pivotSplitSamplesWithStats2(PGLSampleData* samples, const size_t begin, const size_t end,
                                                                    uint8_t splitDimension, float pivot, SampleStatistics &statsLeft, SampleStatistics &statsRight) const
    {
        auto isLeft = [&] (const PGLSampleData &sample) { const Vector3 v(sample.position.x, sample.position.y, sample.position.z); return v[splitDimension] < pivot; };
        size_t center = 0;
        bool parallel = (end-begin) < PARALLEL_THRESHOLD ? false : true;
        if (!parallel) {
            center = embree::serial_partitioning(samples, begin, end, statsLeft, statsRight, isLeft,
                                [] (SampleStatistics& sstats, const PGLSampleData& sample) { sstats.addSample(Vector3(sample.position.x, sample.position.y, sample.position.z)); });
        } else {    
            center = embree::parallel_partitioning(samples, begin, end, SampleStatistics(), 
                statsLeft, statsRight, isLeft, 
                [] (SampleStatistics& sstats, const PGLSampleData& sample) { sstats.addSample(Vector3(sample.position.x, sample.position.y, sample.position.z)); },
                [] (SampleStatistics& sstats0,const SampleStatistics& sstats1) { sstats0.merge(sstats1); },
            PARALLEL_PARTITION_BLOCK_SIZE);
        }
        return center;
    }

    inline size_t pivotSplitSamplesWithStats3(const BBox& bounds, PGLSampleData* samples, const size_t begin, const size_t end,
                                                                    uint8_t splitDimension, float pivot, SampleStatistics &statsLeft, SampleStatistics &statsRight, bool parallel = true) const
    {
        auto isLeft = [&] (const PGLSampleData &sample) { const Vector3 v(sample.position.x, sample.position.y, sample.position.z); return v[splitDimension] < pivot; };
        size_t center = 0;
        bool runParallel = (end-begin) < PARALLEL_THRESHOLD || parallel == false ? false : true;
        if (!runParallel) {
            IntegerSampleStatistics iStatsLeft(bounds);
            IntegerSampleStatistics iStatsRight(bounds);
            center = embree::serial_partitioning(samples, begin, end, iStatsLeft, iStatsRight, isLeft,
                                [] (IntegerSampleStatistics& sstats, const PGLSampleData& sample) { sstats.addSample(Vector3(sample.position.x, sample.position.y, sample.position.z)); });
            statsLeft = iStatsLeft.getSampleStatistics();
            statsRight = iStatsRight.getSampleStatistics();
        } else {    
            IntegerSampleStatistics iStatsLeft(bounds);
            IntegerSampleStatistics iStatsRight(bounds);
            center = embree::parallel_partitioning(samples, begin, end, IntegerSampleStatistics(bounds), 
                iStatsLeft, iStatsRight, isLeft, 
                [] (IntegerSampleStatistics& sstats, const PGLSampleData& sample) { sstats.addSample(Vector3(sample.position.x, sample.position.y, sample.position.z)); },
                [] (IntegerSampleStatistics& sstats0,const IntegerSampleStatistics& sstats1) { sstats0.merge(sstats1); },
            PARALLEL_PARTITION_BLOCK_SIZE);
            statsLeft = iStatsLeft.getSampleStatistics();
            statsRight = iStatsRight.getSampleStatistics();
        }
        return center;
    }
#endif

    inline void getSplitDimensionAndPosition(const SampleStatistics &sampleStats, uint8_t &splitDim, float &splitPos) const
    {
        const Vector3 sampleVariance = sampleStats.getVariance();
        const Point3 sampleMean = sampleStats.getMean();

        auto maxDimension = [](const Vector3& v) -> uint8_t
                {
            return v[v[1] > v[0]] > v[2] ? v[1] > v[0] : 2;
        };

        splitDim = maxDimension(sampleVariance);
        splitPos = sampleMean[splitDim];
    }


    void updateTreeNode(KDTree *kdTree, KDNode &node, size_t depth, const BBox bounds, TSamplesContainer &samples, const Range sampleRange, const SampleStatistics sampleStats, tbb::concurrent_vector< std::pair<TRegion, Range> > *dataStorage, const Settings &buildSettings, bool parallel = true) const
    {
        if(sampleRange.size() <= 0)
        {
            return;
        }
        uint8_t splitDim = {0};
        float splitPos = {0.0f};

        uint32_t nodeIdsLeftRight[2];
        Range sampleRangeLeftRight[2];
        SampleStatistics sampleStatsLeftRight[2];

        BBox bondsLeftRight[2];

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

                auto rightDataItr = dataStorage->push_back(regionAndRangeDataRight);

                uint32_t rightDataIdx = std::distance(dataStorage->begin(), rightDataItr);

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

        bondsLeftRight[0] = bondsLeftRight[1] = bounds;
        bondsLeftRight[0].upper[splitDim] = splitPos;
        bondsLeftRight[1].lower[splitDim] = splitPos;

#ifdef USE_EMBREE_PARALLEL 
        size_t rPivotItr = 0;
#else
        typename TSamplesContainer::iterator rPivotItr(nullptr);
        auto begin = samples.begin() + sampleRange.m_begin, end = samples.begin() + sampleRange.m_end;
#endif
        if(kdTree->getNode(nodeIdsLeftRight[0]).isLeaf() || kdTree->getNode(nodeIdsLeftRight[1]).isLeaf() )
        {
            //splitStats = true;
#ifdef USE_EMBREE_PARALLEL
#ifndef USE_INTEGER_ARITHMETIC_STATS
            rPivotItr = pivotSplitSamplesWithStats2(samples.data(), sampleRange.m_begin, sampleRange.m_end, splitDim, splitPos, sampleStatsLeftRight[0], sampleStatsLeftRight[1]);
#else
            rPivotItr = pivotSplitSamplesWithStats3(bounds, samples.data(), sampleRange.m_begin, sampleRange.m_end, splitDim, splitPos, sampleStatsLeftRight[0], sampleStatsLeftRight[1], parallel);
#endif
#else
            rPivotItr = pivotSplitSamplesWithStats(begin, end, splitDim, splitPos, sampleStatsLeftRight[0], sampleStatsLeftRight[1]);
#endif
        }
        else
        {
#ifdef USE_EMBREE_PARALLEL
            rPivotItr = pivotSplitSamples2<typename TSamplesContainer::value_type>(samples.data(), sampleRange.m_begin, sampleRange.m_end, splitDim, splitPos);
#else
            rPivotItr = pivotSplitSamples<TSamplesContainer>(begin, end, splitDim, splitPos);
#endif
        }

        
#ifdef USE_EMBREE_PARALLEL
        sampleRangeLeftRight[0] = Range(sampleRange.m_begin, rPivotItr);
        sampleRangeLeftRight[1] = Range(rPivotItr, sampleRange.m_end);
#else
        sampleRangeLeftRight[0] = Range(sampleRange.m_begin, std::distance(samples.begin(), rPivotItr));
        sampleRangeLeftRight[1] = Range(std::distance(samples.begin(), rPivotItr), sampleRange.m_end);
#endif
        tbb::parallel_invoke(
            [&]{updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[0]), depth + 1, bondsLeftRight[0], samples, sampleRangeLeftRight[0], sampleStatsLeftRight[0], dataStorage, buildSettings, true);},
            [&]{updateTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[1]), depth + 1, bondsLeftRight[1], samples, sampleRangeLeftRight[1], sampleStatsLeftRight[1], dataStorage, buildSettings, true);}
        );
    }

    void insertTreeNode(KDTree *kdTree, KDNode &node, size_t depth, TInvalidSamplesContainer &samples, const Range sampleRange, tbb::concurrent_vector< std::pair<TRegion, Range> > *dataStorage) const
    {
        if(sampleRange.size() == 0)
        {
            return;
        }
        uint8_t splitDim = {0};
        float splitPos = {0.0f};

        uint32_t nodeIdsLeftRight[2];
        Range sampleRangeLeftRight[2];

        if (node.isLeaf())
        {
            uint32_t dataIdx = node.getDataIdx();
            std::pair<TRegion, Range> &regionAndRangeData = dataStorage->operator[](dataIdx);
            regionAndRangeData.first.sampleStatistics.addNumInvalidSamples(sampleRange.size());
            regionAndRangeData.first.numInvalidSamples = sampleRange.size();
            return;
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

#ifdef USE_EMBREE_PARALLEL 
        size_t rPivotItr = 0;
#else
        typename TInvalidSamplesContainer::iterator rPivotItr;
        auto begin = samples.begin() + sampleRange.m_begin, end = samples.begin() + sampleRange.m_end;
#endif
#ifdef USE_EMBREE_PARALLEL
            rPivotItr = pivotSplitSamples2<typename TInvalidSamplesContainer::value_type>(samples.data(), sampleRange.m_begin, sampleRange.m_end, splitDim, splitPos);
#else
            rPivotItr = pivotSplitSamples<TInvalidSamplesContainer>(begin, end, splitDim, splitPos);
#endif


#ifdef USE_EMBREE_PARALLEL
        sampleRangeLeftRight[0] = Range(sampleRange.m_begin, rPivotItr);
        sampleRangeLeftRight[1] = Range(rPivotItr, sampleRange.m_end);
#else
        sampleRangeLeftRight[0] = Range(sampleRange.m_begin, std::distance(samples.begin(), rPivotItr));
        sampleRangeLeftRight[1] = Range(std::distance(samples.begin(), rPivotItr), sampleRange.m_end);
#endif
		/* This assert is a sanity check which is only valid with the assumption that the number of samples grows at same pace
		   as the number of spatial nodes: in practice this is not the case (e.g., after many 1spp iterations) 
		*/
        //OPENPGL_ASSERT(sampleRangeLeftRight[0].size() > 1);
        //OPENPGL_ASSERT(sampleRangeLeftRight[1].size() > 1);

        tbb::parallel_invoke(
            [&]{insertTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[0]), depth + 1, samples, sampleRangeLeftRight[0], dataStorage);},
            [&]{insertTreeNode(kdTree, kdTree->getNode(nodeIdsLeftRight[1]), depth + 1, samples, sampleRangeLeftRight[1], dataStorage);}
        );

    }

};


template<class TRegion, typename TSamplesContainer, typename TInvalidSamplesContainer>
inline std::string KDTreePartitionBuilder<TRegion, TSamplesContainer, TInvalidSamplesContainer>::toString() const
{
    std::stringstream ss;
    ss << "KDTreePartitionBuilder" << std::endl;
    return ss.str();
}

template<class TRegion, typename TSamplesContainer, typename TInvalidSamplesContainer>
inline std::string KDTreePartitionBuilder<TRegion, TSamplesContainer, TInvalidSamplesContainer>::Settings::toString() const
{
    std::stringstream ss;
    ss << "KDTreePartitionBuilder::Settings:" << std::endl;
    ss << "  minSamples: " << minSamples << std::endl;
    ss << "  maxSamples: " << maxSamples << std::endl;
    ss << "  maxDepth: " << maxDepth << std::endl;

    return ss.str();
}


template<class TRegion, typename TSamplesContainer, typename TInvalidSamplesContainer>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TInvalidSamplesContainer>::Settings::serialize(std::ostream& stream)const
    {
        stream.write(reinterpret_cast<const char*>(&minSamples), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(&maxSamples), sizeof(size_t));
        stream.write(reinterpret_cast<const char*>(&maxDepth), sizeof(size_t));
    }

template<class TRegion, typename TSamplesContainer, typename TInvalidSamplesContainer>
inline void KDTreePartitionBuilder<TRegion, TSamplesContainer, TInvalidSamplesContainer>::Settings::deserialize(std::istream& stream)
    {
        stream.read(reinterpret_cast<char*>(&minSamples), sizeof(size_t));
        stream.read(reinterpret_cast<char*>(&maxSamples), sizeof(size_t));
        stream.read(reinterpret_cast<char*>(&maxDepth), sizeof(size_t));
    }
}