// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../openpgl_common.h"

#include <tbb/concurrent_vector.h>


#include <fstream>
#include <string>
#include <iostream>

#define MERGE_SPLITDIM_AND_NODE_IDX

#define USE_TREELETS

namespace openpgl
{

#if defined(MERGE_SPLITDIM_AND_NODE_IDX) 
struct KDNodeOld
{
    enum{
        ESPlitDimX = 0,
        ESPlitDimY = 1,
        ESPlitDimZ = 2,
        ELeafNode  = 3,
    };
    
    float splitPosition {0.0f};
    uint8_t splitDim{0};
    uint32_t nodeIdx{0};


    bool isLeaf() const {
        return splitDim == ELeafNode;
    }

    void deserialize(std::istream& stream)
    {
        stream.read(reinterpret_cast<char*>(&splitPosition), sizeof(float));
        stream.read(reinterpret_cast<char*>(&splitDim), sizeof(uint8_t));
        stream.read(reinterpret_cast<char*>(&nodeIdx), sizeof(uint32_t));

        OPENPGL_ASSERT(splitDim <=ELeafNode);
    }
};
#endif

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
    //uint8_t splitDim{0};
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
        OPENPGL_ASSERT(idx < (1U<<31));
        OPENPGL_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        OPENPGL_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        OPENPGL_ASSERT( !isLeaf() );

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        OPENPGL_ASSERT( idx == getLeftChildIdx());
#else
    splitDim = ELeafNode;
    nodeIdx = idx;
#endif
    }

    uint32_t getLeftChildIdx() const {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        OPENPGL_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        OPENPGL_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        OPENPGL_ASSERT( !isLeaf() );

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
    #ifdef MERGE_SPLITDIM_AND_NODE_IDX
        splitDimAndNodeIdx = 0;
        splitDimAndNodeIdx = (uint32_t(_splitDim)<<30);
        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|_leftChildIdx;
    #else
        splitDim = _splitDim;
        nodeIdx = _leftChildIdx;
    #endif
        //setSplitDim(splitDim);
        //setLeftChildIdx(leftChildIdx);
        OPENPGL_ASSERT(_splitDim == getSplitDim());
        OPENPGL_ASSERT(_leftChildIdx == getLeftChildIdx());
    }


    /////////////////////////////
    // Leaf node functions
    /////////////////////////////

    void setLeaf(){
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        splitDimAndNodeIdx = (3U<<30);
        OPENPGL_ASSERT(isLeaf());
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
        OPENPGL_ASSERT(idx < (1U<<31));
        OPENPGL_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        OPENPGL_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        OPENPGL_ASSERT( !isLeaf() );

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        OPENPGL_ASSERT( idx == getLeftChildIdx());
#else
        nodeIdx = idx;
#endif
    }

    void setDataNodeIdx(const uint32_t &idx)
    {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        OPENPGL_ASSERT(idx < (1U<<31)); // checks if the idx is in the right range
        setLeaf();
        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        OPENPGL_ASSERT( isLeaf());
        OPENPGL_ASSERT( getDataIdx() == idx);
#else
        setLeaf();
        nodeIdx = idx;
#endif
    }

    uint32_t getDataIdx() const
    {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        OPENPGL_ASSERT(isLeaf());
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
        //OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) < ELeafNode);
        return (splitDimAndNodeIdx >> 30);
#else
        return splitDim;
#endif
    }

    void setSplitDim(const uint8_t &splitAxis) {
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        OPENPGL_ASSERT(splitAxis<ELeafNode);
        splitDimAndNodeIdx = (uint32_t(splitAxis)<<30);
        OPENPGL_ASSERT (splitAxis == getSplitDim());
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

    void serialize(std::ostream& stream)const
    {
        stream.write(reinterpret_cast<const char*>(&splitPosition), sizeof(float));
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        stream.write(reinterpret_cast<const char*>(&splitDimAndNodeIdx), sizeof(uint32_t));
#else
        stream.write(reinterpret_cast<const char*>(&splitDim), sizeof(uint8_t));
        stream.write(reinterpret_cast<const char*>(&nodeIdx), sizeof(uint32_t));
#endif
    }

    void deserialize(std::istream& stream)
    {
        stream.read(reinterpret_cast<char*>(&splitPosition), sizeof(float));
#ifdef MERGE_SPLITDIM_AND_NODE_IDX
        stream.read(reinterpret_cast<char*>(&splitDimAndNodeIdx), sizeof(uint32_t));
#else
        stream.read(reinterpret_cast<char*>(&splitDim), sizeof(uint8_t));
        stream.read(reinterpret_cast<char*>(&nodeIdx), sizeof(uint32_t));

        OPENPGL_ASSERT(splitDim >= 0);
        OPENPGL_ASSERT(splitDim <=ELeafNode);
#endif
    }

};

struct KDTreeLet
{
    KDNode nodes[8];
};



struct KDTree
{
    KDTree() = default;

    ~KDTree()
    {
        if(m_nodesPtr)
            delete[] m_nodesPtr;
    }

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
        //OPENPGL_ASSERT( m_isInit );
        OPENPGL_ASSERT( m_nodes.size() > idx );
        return m_nodes[idx];
    }


    const KDNode& getRoot() const
    {
        return getNode(0);
    }

    const KDNode& getNode( const size_t &idx ) const
    {
        //OPENPGL_ASSERT( m_isInit );
        OPENPGL_ASSERT( m_nodes.size() > idx );
        return m_nodes[idx];
    }

    size_t getNumNodes() const
    {
        return m_nodes.size();
    }

    uint32_t addChildrenPair()
    {
       auto firstChildItr = m_nodes.grow_by(2);
       return std::distance(m_nodes.begin(), firstChildItr);
    }

    const BBox &getBounds() const
    {
        return m_bounds;
    }

    void finalize()
    {
        if(m_nodesPtr){
            delete[] m_nodesPtr;
            m_nodesPtr = nullptr;
        }
        size_t nNodes = m_nodes.size();
        if (nNodes > 0)
        {
            m_nodesPtr = new KDNode[nNodes];
            for(int n = 0; n < nNodes; n++)
            {
                m_nodesPtr[n] = m_nodes[n];
            }
#ifdef USE_TREELETS
            buildTreeLets();
#endif
        }     
    }

    void buildTreeLets()
    {
        std::cout << "buildTreeLets: sizeof(KDNode) = " << sizeof(KDNode) << "\t sizeof(KDTreeLet) = " << sizeof(KDTreeLet) << std::endl;
        
        if(m_treeLets){
            delete[] m_treeLets;
            m_treeLets = nullptr;
            m_numTreeLets = 0;
        }

        std::vector<KDTreeLet> treeLets;
        treeLets.push_back(KDTreeLet());
        insertNode(m_nodesPtr[0], 0, 0, 0, treeLets);

        m_numTreeLets = treeLets.size();
        m_treeLets = new KDTreeLet[m_numTreeLets];
        for ( int i = 0; i < m_numTreeLets; i++){
            m_treeLets[i] = treeLets[i];
        }

        std::cout << "buildTreeLets: m_numTreeLets = " << m_numTreeLets << std::endl;
    }

    uint32_t insertNode(const KDNode& node, uint32_t nodeIdx, uint32_t treeLetIdx, uint32_t treeDepth, std::vector<KDTreeLet>& treeLets) {
        uint32_t treeLetLevel = treeDepth % 3;
        uint32_t globalNodeId = (treeLetIdx * 8) + nodeIdx;
        treeLets[treeLetIdx].nodes[nodeIdx] = node;
        if (!node.isLeaf()) {
            if(treeLetLevel == 0) {
                uint32_t childIdx = node.getLeftChildIdx();
                uint32_t newChildIdx = insertNode(m_nodesPtr[childIdx], 1, treeLetIdx, treeDepth+1,treeLets);
                insertNode(m_nodesPtr[childIdx+1], 2, treeLetIdx, treeDepth+1,treeLets);
                treeLets[treeLetIdx].nodes[nodeIdx].setLeftChildIdx(newChildIdx);
#ifndef MERGE_SPLITDIM_AND_NODE_IDX
                treeLets[treeLetIdx].nodes[nodeIdx].setSplitDim(node.getSplitDim());
#endif
                OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                OPENPGL_ASSERT(node.getSplitPivot() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitPivot());
                OPENPGL_ASSERT(newChildIdx == treeLets[treeLetIdx].nodes[nodeIdx].getLeftChildIdx());
            } else if (treeLetLevel == 1) {
                if(nodeIdx == 1)
                {
                    uint32_t childIdx = node.getLeftChildIdx();
                    uint32_t newChildIdx = insertNode(m_nodesPtr[childIdx], 3, treeLetIdx, treeDepth+1,treeLets);
                    insertNode(m_nodesPtr[childIdx+1], 4, treeLetIdx, treeDepth+1,treeLets);
                    treeLets[treeLetIdx].nodes[nodeIdx].setLeftChildIdx(newChildIdx);
#ifndef MERGE_SPLITDIM_AND_NODE_IDX
                    treeLets[treeLetIdx].nodes[nodeIdx].setSplitDim(node.getSplitDim());
#endif
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    OPENPGL_ASSERT(node.getSplitPivot() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitPivot());
                    OPENPGL_ASSERT(newChildIdx == treeLets[treeLetIdx].nodes[nodeIdx].getLeftChildIdx());
                } else if (nodeIdx == 2) {
                    uint32_t childIdx = node.getLeftChildIdx();
                    uint32_t newChildIdx = insertNode(m_nodesPtr[childIdx], 5, treeLetIdx, treeDepth+1,treeLets);
                    insertNode(m_nodesPtr[childIdx+1], 6, treeLetIdx, treeDepth+1,treeLets);
                    treeLets[treeLetIdx].nodes[nodeIdx].setLeftChildIdx(newChildIdx);
#ifndef MERGE_SPLITDIM_AND_NODE_IDX
                    treeLets[treeLetIdx].nodes[nodeIdx].setSplitDim(node.getSplitDim());
#endif
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    OPENPGL_ASSERT(node.getSplitPivot() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitPivot());
                    OPENPGL_ASSERT(newChildIdx == treeLets[treeLetIdx].nodes[nodeIdx].getLeftChildIdx());
                } else {
                    std::cout << "ERROR" << std::endl;
                }
            } else if (treeLetLevel == 2) {
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    uint32_t childIdx = node.getLeftChildIdx();
                    treeLets.push_back(KDTreeLet());
                    treeLets.push_back(KDTreeLet());
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    uint32_t leftTreeLetIdx = treeLets.size() - 2;
                    insertNode(m_nodesPtr[childIdx], 0, leftTreeLetIdx, treeDepth+1,treeLets);
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    insertNode(m_nodesPtr[childIdx+1], 0, leftTreeLetIdx+1, treeDepth+1,treeLets);
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    treeLets[treeLetIdx].nodes[nodeIdx].setLeftChildIdx(leftTreeLetIdx);
#ifndef MERGE_SPLITDIM_AND_NODE_IDX
                    treeLets[treeLetIdx].nodes[nodeIdx].setSplitDim(node.getSplitDim());
#endif
                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    //return lefTreeLetIdx;

                    OPENPGL_ASSERT(node.getSplitDim() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitDim());
                    OPENPGL_ASSERT(node.getSplitPivot() == treeLets[treeLetIdx].nodes[nodeIdx].getSplitPivot());
                    OPENPGL_ASSERT(leftTreeLetIdx == treeLets[treeLetIdx].nodes[nodeIdx].getLeftChildIdx());

            } else {
                std::cout << "ERROR" << std::endl;
            }
        }
        return globalNodeId;
    }
#ifndef USE_TREELETS
    uint32_t getDataIdxAtPos(const Vector3 &pos) const
    {
        OPENPGL_ASSERT(m_isInit);
        OPENPGL_ASSERT(embree::inside(m_bounds, pos));

        uint32_t nodeIdx = 0;
        while(!m_nodesPtr[nodeIdx].isLeaf())
        {
            uint8_t splitDim = m_nodesPtr[nodeIdx].getSplitDim();
            float pivot = m_nodesPtr[nodeIdx].getSplitPivot();

            nodeIdx = m_nodesPtr[nodeIdx].getLeftChildIdx();
            nodeIdx += pos[splitDim] >= pivot ? 1 : 0;
        }

//        std::cout << "dataID = " << m_nodesPtr[nodeIdx].getDataIdx() << std::endl;
        return m_nodesPtr[nodeIdx].getDataIdx();
    }

#else
    //uint32_t getDataIdxAtPosTreeLets(const Vector3 &pos) const
    uint32_t getDataIdxAtPos(const Vector3 &pos) const
    {
        OPENPGL_ASSERT(m_isInit);
        OPENPGL_ASSERT(embree::inside(m_bounds, pos));

        uint32_t treeIdx = 0;
        uint32_t nodeIdx = 0;
        uint32_t depth = 0;
        KDTreeLet treeLet = m_treeLets[treeIdx];
        //uint8_t splitDim = treeLet.nodes[nodeIdx].getSplitDim();
        //uint32_t childIdx = treeLet.nodes[nodeIdx].getLeftChildIdx();
        
        while(!treeLet.nodes[nodeIdx].isLeaf())
        //while(splitDim != KDNode::ELeafNode)
        {
            uint8_t splitDim = treeLet.nodes[nodeIdx].getSplitDim();
            uint32_t childIdx = treeLet.nodes[nodeIdx].getLeftChildIdx();
            float pivot = treeLet.nodes[nodeIdx].getSplitPivot();
            
            if(depth % 3 == 2){
                nodeIdx = 0;
                treeIdx = childIdx;
                treeIdx += pos[splitDim] >= pivot ? 1 : 0;
                treeLet = m_treeLets[treeIdx];
            } else {
                nodeIdx = childIdx - (treeIdx * 8);
                nodeIdx += pos[splitDim] >= pivot ? 1 : 0;
            }
            //splitDim = treeLet.nodes[nodeIdx].getSplitDim();
            //childIdx = splitDim != 3 ? treeLet.nodes[nodeIdx].getLeftChildIdx() : treeLet.nodes[nodeIdx].getDataIdx();
            depth++;
        }

        //std::cout << "dataID = " << treeLet.nodes[nodeIdx].getDataIdx() << "\t treeIdx = " << treeIdx << "\t nodeIdx = " << nodeIdx << std::endl;
        //return childIdx;
        return treeLet.nodes[nodeIdx].getDataIdx();
    }
#endif
    uint32_t getMaxNodeDepth(const KDNode& node) const
    {
        if(node.isLeaf())
        {
            return 1;
        } 
        else 
        {
            const uint32_t leftNodeId = node.getLeftChildIdx();
            const uint32_t leftMaxNodeDepth = getMaxNodeDepth(m_nodes[leftNodeId]);
            const uint32_t rightMaxNodeDepth = getMaxNodeDepth(m_nodes[leftNodeId+1]);
            return 1 + std::max(leftMaxNodeDepth, rightMaxNodeDepth);
        }
    }    

    uint32_t getMaxTreeDepth() const
    {
        if(m_nodes.size()>0)
        {
            return getMaxNodeDepth(m_nodes[0]);
        }
        return 0;
    }

    std::string toString() const
    {
        std::stringstream ss;
        ss.precision(5);
        ss << "KDTree::" << std::endl;
        ss << "  isInit: " << m_isInit <<  std::endl;
        ss << "  bounds: " << m_bounds<< std::endl;
        ss << "  maxDepth: " << getMaxTreeDepth() << std::endl;
        ss << "  numNodes: " << m_nodes.size() << std::endl;
        return ss.str();
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
    void exportSampleBoundsToObj(std::string objFileName, const tbb::concurrent_vector< std::pair<TRegion, TRange> > &dataStorage) const
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

    void exportSampleBoundToObj(std::ofstream &objFile, const KDNode &node, BBox bbox, uint32_t &vertexIDOffset, const tbb::concurrent_vector< std::pair<TRegion, TRange> > &dataStorage) const
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

    void serialize(std::ostream& stream)const
    {
        stream.write(reinterpret_cast<const char*>(&m_isInit), sizeof(bool));
        stream.write(reinterpret_cast<const char*>(&m_bounds), sizeof(BBox));
        size_t num_nodes = m_nodes.size();
        stream.write(reinterpret_cast<const char*>(&num_nodes), sizeof(size_t));
        for (size_t n = 0; n < num_nodes; n++)
        {
            m_nodes[n].serialize(stream);
        }
    }

    void deserialize(std::istream& stream)
    {
        stream.read(reinterpret_cast<char*>(&m_isInit), sizeof(bool));
        stream.read(reinterpret_cast<char*>(&m_bounds), sizeof(BBox));
        size_t num_nodes = 0;
        stream.read(reinterpret_cast<char*>(&num_nodes), sizeof(size_t));
        m_nodes.reserve(num_nodes);
        m_nodesPtr = new KDNode[num_nodes];
        //std::cout << "KDNode = " << sizeof(KDNode) << "\t KDNodeOld = " << sizeof(KDNodeOld) << std::endl;
        for (size_t n = 0; n < num_nodes; n++)
        {
            KDNode node;
#if defined(MERGE_SPLITDIM_AND_NODE_IDX) 
            KDNodeOld nodeOld;
            nodeOld.deserialize(stream);
            if (nodeOld.isLeaf()){
                node.setLeaf();
                node.setDataNodeIdx(nodeOld.nodeIdx);
            } else {
                node.splitPosition = nodeOld.splitPosition;
                node.setSplitDim(nodeOld.splitDim);
                node.setChildNodeIdx(nodeOld.nodeIdx);
            }
            
#else
            node.deserialize(stream);
#endif
            m_nodes.push_back(node);
            m_nodesPtr[n] = node;
        }

#ifdef USE_TREELETS
        if(num_nodes > 0)
            buildTreeLets();
#endif
    }

public:
    bool m_isInit { false };

    // bounds of the spatial region covered by the KDTree
    BBox m_bounds;

    // node storage used during build
    tbb::concurrent_vector<KDNode> m_nodes;

    // node storage used during querying
    KDNode* m_nodesPtr {nullptr};

    KDTreeLet* m_treeLets {nullptr};
    int m_numTreeLets {0};
};

}