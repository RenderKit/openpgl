// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"

//#ifdef USE_TBB
// #define USE_TBB_CONCURRENT_NODES
//#endif

#ifdef USE_TBB_CONCURRENT_NODES
#include <tbb/concurrent_vector.h>
#else
#include <mitsuba/guiding/AtomicallyGrowingVector.h>
#endif

#include <fstream>
#include <string>
#include <iostream>

//#define MERGE_SPLITDIM_AND_NODE_IDX

namespace rkguide
{

struct KDNode
{
    enum{
        ESPlitDimX = 0,
        ESPlitDimY = 1,
        ESPlitDimZ = 2,
        ELeafNode  = 3,
    };

    float splitPosition {0.0f};
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
    uint32_t splitDimAndNodeIdx{0};
#else
    uint8_t splitDim{0};
    uint32_t nodeIdx{0};
#endif
    /////////////////////////////
    // Child node functions
    /////////////////////////////
    bool isChild() const
    {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        return ( splitDimAndNodeIdx >> 30) < ELeafNode;
#else
    return splitDim < ELeafNode;
#endif

    }

    void setLeftChildIdx(const uint32_t &idx) {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        RKGUIDE_ASSERT(idx < (1U<<31));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        RKGUIDE_ASSERT( !isLeaf() );

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        RKGUIDE_ASSERT( idx == getLeftChildIdx());
#else
    splitDim = ELeafNode;
    nodeIdx = idx;
#endif
    }

    uint32_t getLeftChildIdx() const {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        RKGUIDE_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        RKGUIDE_ASSERT( !isLeaf() );

        return (splitDimAndNodeIdx << 2) >> 2;
#else
        return  nodeIdx;
#endif
        }

    /////////////////////////////
    // Inner node functions
    /////////////////////////////

    void setToInnerNode( const uint8_t &_splitDim, const float &_splitPos, const uint32_t &_leftChildIdx) {
        splitPosition = _splitPos;
        splitDim = _splitDim;
        nodeIdx = _leftChildIdx;
        //setSplitDim(splitDim);
        //setLeftChildIdx(leftChildIdx);
        RKGUIDE_ASSERT(_splitDim == getSplitDim());
        RKGUIDE_ASSERT(_leftChildIdx == getLeftChildIdx());
    }


    /////////////////////////////
    // Leaf node functions
    /////////////////////////////

    void setLeaf(){
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        splitDimAndNodeIdx = (3U<<30);
        RKGUIDE_ASSERT(isLeaf());
#else
        splitDim = ELeafNode;
#endif
    }

    bool isLeaf() const {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        return (splitDimAndNodeIdx >> 30) == 3;
#else
        return splitDim == ELeafNode;
#endif
    }

    void setChildNodeIdx(const uint32_t &idx) {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        RKGUIDE_ASSERT(idx < (1U<<31));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        RKGUIDE_ASSERT( !isLeaf() );

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        RKGUIDE_ASSERT( idx == getLeftChildIdx());
#else
        nodeIdx = idx;
#endif
    }

    void setDataNodeIdx(const uint32_t &idx)
    {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        RKGUIDE_ASSERT(idx < (1U<<31)); // checks if the idx is in the right range
        setLeaf();
        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        RKGUIDE_ASSERT( isLeaf());
        RKGUIDE_ASSERT( getDataIdx() == idx);
#else
        setLeaf();
        nodeIdx = idx;
#endif
    }

    uint32_t getDataIdx() const
    {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        RKGUIDE_ASSERT(isLeaf());
        return (splitDimAndNodeIdx << 2) >> 2;
#else
    return nodeIdx;
#endif
    }


    /////////////////////////////
    // Split dimension functions
    /////////////////////////////

    uint8_t getSplitDim() const
    {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        RKGUIDE_ASSERT((splitDimAndNodeIdx >> 30) < ELeafNode);
        return (splitDimAndNodeIdx >> 30);
#else
        return splitDim;
#endif
    }

    void setSplitDim(const uint8_t &splitAxis) {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        RKGUIDE_ASSERT(splitAxis<ELeafNode);
        splitDimAndNodeIdx = (uint32_t(splitAxis)<<30);
        RKGUIDE_ASSERT (splitAxis == getSplitDim());
#else
        splitDim = splitAxis;
#endif
    }

    float getSplitPivot() const {
        return splitPosition;
    }

    void setSplitPivot(const float &pos){
        splitPosition = pos;
    }
};



struct KDTree
{
    // bounding box
    KDTree() = default;

    inline void init(const BBox &bounds, size_t numNodesReseve = 0)
    {
        m_bounds = bounds;
        m_nodes.clear();
        if (numNodesReseve > 0)
        {
            m_nodes.reserve(numNodesReseve);
        }
        clear();
        m_isInit = true;
    }

    inline void clear()
    {
        m_nodes.resize(1);
        m_nodes[0].setLeaf();
        m_nodes[0].setDataNodeIdx(0);
    }

    KDNode& getRoot()
    {
        return getNode(0);
    }

    KDNode& getNode( const size_t &idx )
    {
        //RKGUIDE_ASSERT( m_isInit );
        RKGUIDE_ASSERT( m_nodes.size() > idx );
        return m_nodes[idx];
    }


    const KDNode& getRoot() const
    {
        return getNode(0);
    }

    const KDNode& getNode( const size_t &idx ) const
    {
        //RKGUIDE_ASSERT( m_isInit );
        RKGUIDE_ASSERT( m_nodes.size() > idx );
        return m_nodes[idx];
    }

    size_t getNumNodes() const
    {
        return m_nodes.size();
    }

    uint32_t addChildrenPair()
    {
#ifdef USE_TBB_CONCURRENT_NODES
       auto firstChildItr = m_nodes.grow_by(2);
#else
        auto firstChildItr = m_nodes.back_insert(2, KDNode());
#endif
       return std::distance(m_nodes.begin(), firstChildItr);
    }

    const BBox &getBounds() const
    {
        return m_bounds;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss.precision(5);
        ss << "KDTree" << std::endl;

        return ss.str();
    }

    uint32_t getDataIdxAtPos(const Vector3 &pos, BBox &bbox) const
    {
        RKGUIDE_ASSERT(m_isInit);
        RKGUIDE_ASSERT(embree::inside(m_bounds, pos));
        bbox = m_bounds;

        uint32_t nodeIdx = 0;
        while(!m_nodes[nodeIdx].isLeaf())
        {
            uint8_t splitDim = m_nodes[nodeIdx].getSplitDim();
            float pivot = m_nodes[nodeIdx].getSplitPivot();
            if (pos[splitDim] < pivot)
                nodeIdx = m_nodes[nodeIdx].getLeftChildIdx();
            else
                nodeIdx = m_nodes[nodeIdx].getLeftChildIdx()+1;
        }

        return m_nodes[nodeIdx].getDataIdx();
    }

    void exportKDTreeStructureToObj(std::string objFileName) const
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());

        BBox rootBBox = m_bounds;
        const KDNode &root = getRoot();
        uint32_t vertexOffset = 0;
        exportKDNodeToObj(objFile, root, rootBBox,vertexOffset);
        objFile.close();
    }


    void exportKDNodeToObj(std::ofstream &objFile, const KDNode &node, BBox bbox, uint32_t &vertexIDOffset) const
    {
        if(!node.isLeaf())
        {
            const uint32_t leftNodeId = node.getLeftChildIdx();
            const uint8_t splitDim = node.getSplitDim();
            const float splitPosition =node.getSplitPivot();
            BBox bboxLeft = bbox;
            bboxLeft.upper[splitDim] = splitPosition;
            BBox bboxRight = bbox;
            bboxRight.lower[splitDim] = splitPosition;

            const KDNode &nodeLeft = m_nodes[leftNodeId];
            const KDNode &nodeRight = m_nodes[leftNodeId+1];

            exportKDNodeToObj(objFile, nodeLeft, bboxLeft, vertexIDOffset);
            exportKDNodeToObj(objFile, nodeRight, bboxRight, vertexIDOffset);
        }
        else
        {
            const uint32_t leafNodeId = node.getDataIdx();
            objFile << "# KDLeafNode"<< leafNodeId << std::endl;

            objFile << "v " << bbox.lower[0] << "\t" << bbox.lower[1] << "\t" << bbox.lower[2] << std::endl;
            objFile << "v " << bbox.lower[0] << "\t" << bbox.upper[1] << "\t" << bbox.lower[2] << std::endl;
            objFile << "v " << bbox.upper[0] << "\t" << bbox.upper[1] << "\t" << bbox.lower[2] << std::endl;
            objFile << "v " << bbox.upper[0] << "\t" << bbox.lower[1] << "\t" << bbox.lower[2] << std::endl;

            objFile << "v " << bbox.lower[0] << "\t" << bbox.lower[1] << "\t" << bbox.upper[2] << std::endl;
            objFile << "v " << bbox.lower[0] << "\t" << bbox.upper[1] << "\t" << bbox.upper[2] << std::endl;
            objFile << "v " << bbox.upper[0] << "\t" << bbox.upper[1] << "\t" << bbox.upper[2] << std::endl;
            objFile << "v " << bbox.upper[0] << "\t" << bbox.lower[1] << "\t" << bbox.upper[2] << std::endl;


            objFile << "f " << vertexIDOffset+1 + 0 << "\t" << vertexIDOffset+1 + 1 << "\t" << vertexIDOffset+1 + 2 << "\t" << vertexIDOffset+1 + 3 << std::endl;
            objFile << "f " << vertexIDOffset+1 + 7 << "\t" << vertexIDOffset+1 + 6 << "\t" << vertexIDOffset+1 + 5 << "\t" << vertexIDOffset+1 + 4 << std::endl;


            objFile << "f " << vertexIDOffset+1 + 0 << "\t" << vertexIDOffset+1 + 3 << "\t" << vertexIDOffset+1 + 7 << "\t" << vertexIDOffset+1 + 4 << std::endl;
            objFile << "f " << vertexIDOffset+1 + 5 << "\t" << vertexIDOffset+1 + 6 << "\t" << vertexIDOffset+1 + 2 << "\t" << vertexIDOffset+1 + 1 << std::endl;

            objFile << "f " << vertexIDOffset+1 + 5 << "\t" << vertexIDOffset+1 + 1 << "\t" << vertexIDOffset+1 + 0 << "\t" << vertexIDOffset+1 + 4 << std::endl;
            objFile << "f " << vertexIDOffset+1 + 7 << "\t" << vertexIDOffset+1 + 3 << "\t" << vertexIDOffset+1 + 2 << "\t" << vertexIDOffset+1 + 6 << std::endl;

            vertexIDOffset += 8;
            //std::cout << "BBox: lower = [" << bbox.lower[0] << ",\t"<< bbox.lower[1] << ",\t"<< bbox.lower[2] << "] \t upper = ["<< bbox.upper[0] << ",\t"<< bbox.upper[1] << ",\t"<< bbox.upper[2] << std::endl;
        }
    }

    template<typename TRegion, typename TRange>
#ifdef  USE_TBB
    void exportSampleBoundsToObj(std::string objFileName, const tbb::concurrent_vector< std::pair<TRegion, TRange> > &dataStorage) const
#else
    void exportSampleBoundsToObj(std::string objFileName, const AtomicallyGrowingVector< std::pair<TRegion, TRange> > &dataStorage) const
#endif
    {
        std::ofstream objFile;
        objFile.open(objFileName.c_str());

        BBox rootBBox = m_bounds;
        const KDNode &root = getRoot();
        uint32_t vertexOffset = 0;
        exportSampleBoundToObj<TRegion, TRange>(objFile, root, rootBBox,vertexOffset, dataStorage);
        objFile.close();
    }

    template<typename TRegion, typename TRange>
#ifdef  USE_TBB
    void exportSampleBoundToObj(std::ofstream &objFile, const KDNode &node, BBox bbox, uint32_t &vertexIDOffset, const tbb::concurrent_vector< std::pair<TRegion, TRange> > &dataStorage) const
#else
    void exportSampleBoundToObj(std::ofstream &objFile, const KDNode &node, BBox bbox, uint32_t &vertexIDOffset, const AtomicallyGrowingVector< std::pair<TRegion, TRange> > &dataStorage) const
#endif
    {
        if(!node.isLeaf())
        {
            const uint32_t leftNodeId = node.getLeftChildIdx();
            const uint8_t splitDim = node.getSplitDim();
            const float splitPosition =node.getSplitPivot();
            BBox bboxLeft = bbox;
            bboxLeft.upper[splitDim] = splitPosition;
            BBox bboxRight = bbox;
            bboxRight.lower[splitDim] = splitPosition;

            const KDNode &nodeLeft = m_nodes[leftNodeId];
            const KDNode &nodeRight = m_nodes[leftNodeId+1];

            exportSampleBoundToObj<TRegion, TRange>(objFile, nodeLeft, bboxLeft, vertexIDOffset, dataStorage);
            exportSampleBoundToObj<TRegion, TRange>(objFile, nodeRight, bboxRight, vertexIDOffset, dataStorage);
        }
        else
        {
            const uint32_t leafNodeId = node.getDataIdx();

            const std::pair<TRegion, TRange> &regionAndRange = dataStorage[leafNodeId];
            BBox sampleBound = regionAndRange.first.sampleStatistics.sampleBounds;
            objFile << "# SampleBound"<< leafNodeId << std::endl;

            objFile << "v " << sampleBound.lower[0] << "\t" << sampleBound.lower[1] << "\t" << sampleBound.lower[2] << std::endl;
            objFile << "v " << sampleBound.lower[0] << "\t" << sampleBound.upper[1] << "\t" << sampleBound.lower[2] << std::endl;
            objFile << "v " << sampleBound.upper[0] << "\t" << sampleBound.upper[1] << "\t" << sampleBound.lower[2] << std::endl;
            objFile << "v " << sampleBound.upper[0] << "\t" << sampleBound.lower[1] << "\t" << sampleBound.lower[2] << std::endl;

            objFile << "v " << sampleBound.lower[0] << "\t" << sampleBound.lower[1] << "\t" << sampleBound.upper[2] << std::endl;
            objFile << "v " << sampleBound.lower[0] << "\t" << sampleBound.upper[1] << "\t" << sampleBound.upper[2] << std::endl;
            objFile << "v " << sampleBound.upper[0] << "\t" << sampleBound.upper[1] << "\t" << sampleBound.upper[2] << std::endl;
            objFile << "v " << sampleBound.upper[0] << "\t" << sampleBound.lower[1] << "\t" << sampleBound.upper[2] << std::endl;


            objFile << "f " << vertexIDOffset+1 + 0 << "\t" << vertexIDOffset+1 + 1 << "\t" << vertexIDOffset+1 + 2 << "\t" << vertexIDOffset+1 + 3 << std::endl;
            objFile << "f " << vertexIDOffset+1 + 7 << "\t" << vertexIDOffset+1 + 6 << "\t" << vertexIDOffset+1 + 5 << "\t" << vertexIDOffset+1 + 4 << std::endl;


            objFile << "f " << vertexIDOffset+1 + 0 << "\t" << vertexIDOffset+1 + 3 << "\t" << vertexIDOffset+1 + 7 << "\t" << vertexIDOffset+1 + 4 << std::endl;
            objFile << "f " << vertexIDOffset+1 + 5 << "\t" << vertexIDOffset+1 + 6 << "\t" << vertexIDOffset+1 + 2 << "\t" << vertexIDOffset+1 + 1 << std::endl;

            objFile << "f " << vertexIDOffset+1 + 5 << "\t" << vertexIDOffset+1 + 1 << "\t" << vertexIDOffset+1 + 0 << "\t" << vertexIDOffset+1 + 4 << std::endl;
            objFile << "f " << vertexIDOffset+1 + 7 << "\t" << vertexIDOffset+1 + 3 << "\t" << vertexIDOffset+1 + 2 << "\t" << vertexIDOffset+1 + 6 << std::endl;

            vertexIDOffset += 8;
            //std::cout << "BBox: lower = [" << sampleBound.lower[0] << ",\t"<< sampleBound.lower[1] << ",\t"<< sampleBound.lower[2] << "] \t upper = ["<< sampleBound.upper[0] << ",\t"<< sampleBound.upper[1] << ",\t"<< sampleBound.upper[2] << std::endl;
        }
    }

public:
    bool m_isInit { false };

    BBox m_bounds;

    //node storage
#ifdef USE_TBB_CONCURRENT_NODES
    tbb::concurrent_vector<KDNode> m_nodes;
#else
    AtomicallyGrowingVector<KDNode> m_nodes;
#endif
};

}