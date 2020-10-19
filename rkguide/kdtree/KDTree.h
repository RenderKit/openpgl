// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../rkguide.h"
#include <tbb/concurrent_vector.h>

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
    uint32_t splitDimAndNodeIdx{0};

    /////////////////////////////
    // Child node functions
    /////////////////////////////
    bool isChild() const
    {
        return ( splitDimAndNodeIdx >> 30) < ELeafNode;
    }

    void setLeftChildIdx(const uint32_t &idx) {
        RKGUIDE_ASSERT(idx < (1U<<31));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        RKGUIDE_ASSERT( !isLeaf() );

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        RKGUIDE_ASSERT( idx == getLeftChildIdx());
    }

    uint32_t getLeftChildIdx() const {
        RKGUIDE_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        RKGUIDE_ASSERT( !isLeaf() );

        return (splitDimAndNodeIdx << 2) >> 2;
    }

    /////////////////////////////
    // Inner node functions
    /////////////////////////////

    void setToInnerNode( const uint8_t &splitDim, const float &splitPos, const uint32_t &leftChildIdx) {
        splitPosition = splitPos;
        setSplitDim(splitDim);
        setLeftChildIdx(leftChildIdx);
        RKGUIDE_ASSERT(splitDim == getSplitDim());
        RKGUIDE_ASSERT(leftChildIdx == getLeftChildIdx());
    }


    /////////////////////////////
    // Leaf node functions
    /////////////////////////////

    void setLeaf(){
        splitDimAndNodeIdx = (3U<<30);
        RKGUIDE_ASSERT(isLeaf());
    }

    bool isLeaf() const {
        return (splitDimAndNodeIdx >> 30) == 3;
    }

    void setChildNodeIdx(const uint32_t &idx) {
        RKGUIDE_ASSERT(idx < (1U<<31));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx & (3U << 30)) != (3U<<30));
        RKGUIDE_ASSERT( (splitDimAndNodeIdx >> 30)  != 3);
        RKGUIDE_ASSERT( !isLeaf() );

        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        RKGUIDE_ASSERT( idx == getLeftChildIdx());
    }

    void setDataNodeIdx(const uint32_t &idx)
    {
        RKGUIDE_ASSERT(idx < (1U<<31)); // checks if the idx is in the right range
        setLeaf();
        splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30)|idx;
        RKGUIDE_ASSERT( isLeaf());
        RKGUIDE_ASSERT( getDataIdx() == idx);
    }

    uint32_t getDataIdx() const
    {
        RKGUIDE_ASSERT(isLeaf());
        return (splitDimAndNodeIdx << 2) >> 2;
    }


    /////////////////////////////
    // Split dimension functions
    /////////////////////////////

    uint8_t getSplitDim() const
    {
        RKGUIDE_ASSERT((splitDimAndNodeIdx >> 30) < ELeafNode);
        return (splitDimAndNodeIdx >> 30);
    }

    void setSplitDim(const uint8_t &splitAxis) {
        RKGUIDE_ASSERT(splitAxis<ELeafNode);
        splitDimAndNodeIdx = (uint32_t(splitAxis)<<30);
        RKGUIDE_ASSERT (splitAxis == getSplitDim());
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
       auto firstChildItr = m_nodes.grow_by(2);
       return std::distance(m_nodes.begin(), firstChildItr);
    }

    const BBox &getBounds() const
    {
        return m_bounds;
    }

private:
    bool m_isInit { false };

    BBox m_bounds;

    //node storage
    tbb::concurrent_vector<KDNode> m_nodes;
};

}