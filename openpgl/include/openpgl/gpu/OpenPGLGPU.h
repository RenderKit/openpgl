#ifndef OPENPGL_GPU_H
#define OPENPGL_GPU_H

#include "../common.h"
#include "../cpp/Field.h"
#include "Device.h"

#include "Common.h"
#include "Distribution.h"

namespace openpgl
{
namespace gpu {
#if defined(OPENPGL_GPU_SYCL)
namespace sycl {
#elif defined(OPENPGL_GPU_CUDA)
namespace cuda {
#else
namespace cpu {
#endif

    struct Region
    {
    public:
    };

    struct KDNode
    {
        enum
        {
            ESPlitDimX = 0,
            ESPlitDimY = 1,
            ESPlitDimZ = 2,
            ELeafNode = 3,
        };

        float splitPosition{0.0f};
        uint32_t splitDimAndNodeIdx{0};

        /////////////////////////////
        // Child node functions
        /////////////////////////////
        bool isChild() const
        {
            return (splitDimAndNodeIdx >> 30) < ELeafNode;
        }

        void setLeftChildIdx(const uint32_t &idx)
        {
            //OPENPGL_ASSERT(idx < (1U << 31));
            //OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
            //OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
            //OPENPGL_ASSERT(!isLeaf());

            splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
            //OPENPGL_ASSERT(idx == getLeftChildIdx());
        }

        OPENPGL_GPU_CALLABLE uint32_t getLeftChildIdx() const
        {
            //OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
            //OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
            //OPENPGL_ASSERT(!isLeaf());

            return (splitDimAndNodeIdx << 2) >> 2;
        }

        /////////////////////////////
        // Inner node functions
        /////////////////////////////

        void setToInnerNode(const uint8_t &_splitDim, const float &_splitPos, const uint32_t &_leftChildIdx)
        {
            splitPosition = _splitPos;
            splitDimAndNodeIdx = 0;
            splitDimAndNodeIdx = (uint32_t(_splitDim) << 30);
            splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | _leftChildIdx;

            //OPENPGL_ASSERT(_splitDim == getSplitDim());
            //OPENPGL_ASSERT(_leftChildIdx == getLeftChildIdx());
        }

        /////////////////////////////
        // Leaf node functions
        /////////////////////////////

        void setLeaf()
        {
            splitDimAndNodeIdx = (3U << 30);
            //OPENPGL_ASSERT(isLeaf());
        }

        OPENPGL_GPU_CALLABLE bool isLeaf() const
        {
            return (splitDimAndNodeIdx >> 30) == 3;
        }

        void setChildNodeIdx(const uint32_t &idx)
        {
            //OPENPGL_ASSERT(idx < (1U << 31));
            //OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
            //OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
            //OPENPGL_ASSERT(!isLeaf());

            splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
            //OPENPGL_ASSERT(idx == getLeftChildIdx());
        }

        void setDataNodeIdx(const uint32_t &idx)
        {
            //OPENPGL_ASSERT(idx < (1U << 31)); // checks if the idx is in the right range
            setLeaf();
            splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
            //OPENPGL_ASSERT(isLeaf());
            //OPENPGL_ASSERT(getDataIdx() == idx);
        }

        OPENPGL_GPU_CALLABLE uint32_t getDataIdx() const
        {
            //OPENPGL_ASSERT(isLeaf());
            return (splitDimAndNodeIdx << 2) >> 2;
        }

        /////////////////////////////
        // Split dimension functions
        /////////////////////////////

        OPENPGL_GPU_CALLABLE uint8_t getSplitDim() const
        {
            return (splitDimAndNodeIdx >> 30);
        }

        void setSplitDim(const uint8_t &splitAxis)
        {
            //OPENPGL_ASSERT(splitAxis < ELeafNode);
            splitDimAndNodeIdx = (uint32_t(splitAxis) << 30);
            //OPENPGL_ASSERT(splitAxis == getSplitDim());
        }

        OPENPGL_GPU_CALLABLE float getSplitPivot() const
        {
            return splitPosition;
        }

        void setSplitPivot(const float &pos)
        {
            splitPosition = pos;
        }

        void serialize(std::ostream &stream) const
        {
            stream.write(reinterpret_cast<const char *>(&splitPosition), sizeof(float));
            stream.write(reinterpret_cast<const char *>(&splitDimAndNodeIdx), sizeof(uint32_t));
        }

        void deserialize(std::istream &stream)
        {
            stream.read(reinterpret_cast<char *>(&splitPosition), sizeof(float));
            stream.read(reinterpret_cast<char *>(&splitDimAndNodeIdx), sizeof(uint32_t));
        }

        bool operator==(const KDNode &b) const
        {
            bool equal = true;
            if (splitPosition != b.splitPosition || isLeaf() != b.isLeaf())
            {
                equal = false;
            }
            if (!isLeaf())
            {
                if (getSplitDim() != b.getSplitDim() ||
                    getLeftChildIdx() != b.getLeftChildIdx())
                {
                    equal = false;
                }
            }
            else
            {
                if (getDataIdx() != b.getDataIdx())
                {
                    equal = false;
                }
            }
            return equal;
        }
    };

    struct KDTreeLet
    {
        KDNode nodes[8];
    };

    struct FieldGPU
    {
        using Distribution = ParallaxAwareVonMisesFisherMixture<32>; 
        
        OPENPGL_GPU_CALLABLE FieldGPU() = default;

        FieldGPU(openpgl::gpu::Device* device, const openpgl::cpp::Field* field) {
            int numSurfaceNodes = field->GetNumSurfaceKDNodes();
            if(numSurfaceNodes > 0) {
                m_surfaceTreeLets = (KDTreeLet*) device->mallocArray<KDTreeLet>(numSurfaceNodes);
                device->memcpyArrayToGPU(m_surfaceTreeLets, (KDTreeLet*)field->GetSurfaceKdNodes(), numSurfaceNodes);
            }
            int numSurfaceDistributions = field->GetNumSurfaceDistributions();
            Distribution *surfaceDistributions = nullptr;
            if(numSurfaceDistributions > 0) {
                surfaceDistributions = new Distribution[numSurfaceDistributions];
                m_surfaceDistributions = device->mallocArray<Distribution>(numSurfaceDistributions);
                field->CopySurfaceDistributions(surfaceDistributions);
                device->memcpyArrayToGPU(m_surfaceDistributions, (const Distribution*) surfaceDistributions, numSurfaceDistributions);
            }
            
            int numVolumeNodes = field->GetNumVolumeKDNodes();
            if(numVolumeNodes > 0) {
                m_volumeTreeLets = (KDTreeLet*) device->mallocArray<KDTreeLet>(numVolumeNodes);
                device->memcpyArrayToGPU(m_volumeTreeLets, (KDTreeLet*)field->GetVolumeKdNodes(), numVolumeNodes);
            }
            int numVolumeDistributions = field->GetNumVolumeDistributions();
            Distribution *volumeDistributions = nullptr;
            if(numVolumeDistributions > 0) {
                volumeDistributions = new Distribution[numVolumeDistributions];
                m_volumeDistributions = device->mallocArray<Distribution>(numVolumeDistributions);
                field->CopyVolumeDistributions(volumeDistributions);
                device->memcpyArrayToGPU(m_volumeDistributions, (const Distribution*) volumeDistributions, numVolumeDistributions);
            }
            
            // wait until all CPU -> GPU copies are done
            device->wait();
            delete[] surfaceDistributions;
            delete[] volumeDistributions;
        }

        OPENPGL_GPU_CALLABLE uint32_t getDataIdxAtPos(const float *pos, const KDTreeLet *treeLets) const
        {
#ifdef USE_TREELETS
            uint32_t treeIdx = 0;
            uint32_t nodeIdx = 0;
            uint32_t depth = 0;
            KDTreeLet treeLet = treeLets[treeIdx];

            while (!treeLet.nodes[nodeIdx].isLeaf())
            {
                uint8_t splitDim = treeLet.nodes[nodeIdx].getSplitDim();
                uint32_t childIdx = treeLet.nodes[nodeIdx].getLeftChildIdx();
                float pivot = treeLet.nodes[nodeIdx].getSplitPivot();

                if (depth % 3 == 2)
                {
                    nodeIdx = 0;
                    treeIdx = childIdx;
                    treeIdx += pos[splitDim] >= pivot ? 1 : 0;
                    treeLet = treeLets[treeIdx];
                }
                else
                {
                    nodeIdx = childIdx - (treeIdx * 8);
                    nodeIdx += pos[splitDim] >= pivot ? 1 : 0;
                }
                depth++;
            }
            return treeLet.nodes[nodeIdx].getDataIdx();
#else
            uint32_t nodeIdx = 0;
            while (!m_nodesPtr[nodeIdx].isLeaf())
            {
                uint8_t splitDim = m_nodesPtr[nodeIdx].getSplitDim();
                float pivot = m_nodesPtr[nodeIdx].getSplitPivot();

                nodeIdx = m_nodesPtr[nodeIdx].getLeftChildIdx();
                nodeIdx += pos[splitDim] >= pivot ? 1 : 0;
            }
            return m_nodesPtr[nodeIdx].getDataIdx();
#endif
        }

        OPENPGL_GPU_CALLABLE uint32_t getSurfaceDistributionIdxAtPos(const float *pos) const
        {
            return getDataIdxAtPos(pos, m_surfaceTreeLets);
        }

        OPENPGL_GPU_CALLABLE uint32_t getVolumeDistributionIdxAtPos(const float *pos) const
        {
            return getDataIdxAtPos(pos, m_volumeTreeLets);
        }

#ifdef USE_TREELETS
/* Need to disable these direct initializations to pass through the std::is_trivially_default_constructible_v test. If not the following error is throughen:
*  error: static assertion failed due to requirement 'std::is_trivially_default_constructible_v<const openpgl_gpu::FieldGPU>': Type T must be trivially default constructable (until C++20 consteval is supported and enabled.)
*/
        KDTreeLet *m_surfaceTreeLets;//{nullptr};
        KDTreeLet *m_volumeTreeLets;//{nullptr};
#else
        KDNode *m_surfaceNodesPtr;//{nullptr};
#endif
        Distribution *m_surfaceDistributions;// {nullptr};
        Distribution *m_volumeDistributions;// {nullptr};

    };

    struct SurfaceSamplingDistributionData
    {
        OPENPGL_GPU_CALLABLE SurfaceSamplingDistributionData() = default;
        const FieldGPU* m_field {nullptr};
        pgl_point3f m_pos {0.f, 0.f, 0.f};
        int m_idx {-1};
    };

    struct SurfaceSamplingDistribution: public SurfaceSamplingDistributionData
    {
        OPENPGL_GPU_CALLABLE SurfaceSamplingDistribution() = default;
        
        OPENPGL_GPU_CALLABLE bool Init(const FieldGPU* field, const pgl_point3f& pos, float& sample1D)
        {
            m_pos = pos;
            m_field = field;
            float _pos[3] = {pos.x, pos.y, pos.z};
            m_idx = m_field->getSurfaceDistributionIdxAtPos(_pos);

            return m_idx >= 0;
        }

        OPENPGL_GPU_CALLABLE void Clear()
        {
            m_field = nullptr;
            m_pos = {0.f, 0.f, 0.f};
            m_idx = -1;
        }

        OPENPGL_GPU_CALLABLE pgl_vec3f Sample(const pgl_point2f& sample2D)const
        {
            return m_field->m_surfaceDistributions[m_idx].samplePos(m_pos, sample2D);
        }

        OPENPGL_GPU_CALLABLE float PDF(const pgl_vec3f& direction) const
        {
            return m_field->m_surfaceDistributions[m_idx].pdfPos(m_pos, direction);
        }

        OPENPGL_GPU_CALLABLE float SamplePDF(const pgl_point2f& sample2D, pgl_vec3f& direction) const
        {
            direction = m_field->m_surfaceDistributions[m_idx].samplePos(m_pos, sample2D);
            return m_field->m_surfaceDistributions[m_idx].pdfPos(m_pos, direction);
        }

        OPENPGL_GPU_CALLABLE float IncomingRadiancePDF(const pgl_vec3f& direction) const
        {
            return m_field->m_surfaceDistributions[m_idx].pdfPos(m_pos, direction);
        }

        OPENPGL_GPU_CALLABLE bool SupportsApplyCosineProduct() const
        {
            return false;
        }

        OPENPGL_GPU_CALLABLE uint32_t GetId() const
        {
            return m_idx;
        }
    };


    struct VolumeSamplingDistributionData
    {
        OPENPGL_GPU_CALLABLE VolumeSamplingDistributionData() = default;
        const FieldGPU* m_field {nullptr};
        pgl_point3f m_pos {0.f, 0.f, 0.f};
        int m_idx {-1};
    };

    struct VolumeSamplingDistribution: public VolumeSamplingDistributionData
    {

        OPENPGL_GPU_CALLABLE VolumeSamplingDistribution() = default;
        
        OPENPGL_GPU_CALLABLE bool Init(const FieldGPU* field, const pgl_point3f& pos, float& sample1D)
        {
            m_pos = pos;
            m_field = field;
            float _pos[3] = {pos.x, pos.y, pos.z};
            m_idx = m_field->getVolumeDistributionIdxAtPos(_pos);

            return m_idx >= 0;
        }

        OPENPGL_GPU_CALLABLE void Clear()
        {
            m_field = nullptr;
            m_pos = {0.f, 0.f, 0.f};
            m_idx = -1;
        }

        OPENPGL_GPU_CALLABLE pgl_vec3f Sample(const pgl_point2f& sample2D)const
        {
            return m_field->m_volumeDistributions[m_idx].samplePos(m_pos, sample2D);
        }

        OPENPGL_GPU_CALLABLE float PDF(const pgl_vec3f& direction) const
        {
            return m_field->m_volumeDistributions[m_idx].pdfPos(m_pos, direction);
        }

        OPENPGL_GPU_CALLABLE float SamplePDF(const pgl_point2f& sample2D, pgl_vec3f& direction) const
        {
            direction = m_field->m_volumeDistributions[m_idx].samplePos(m_pos, sample2D);
            return m_field->m_volumeDistributions[m_idx].pdfPos(m_pos, direction);
        }

        OPENPGL_GPU_CALLABLE float IncomingRadiancePDF(const pgl_vec3f& direction) const
        {
            return m_field->m_volumeDistributions[m_idx].pdfPos(m_pos, direction);
        }

        OPENPGL_GPU_CALLABLE uint32_t GetId() const
        {
            return m_idx;
        }
    };

} // sycl/cuda/cpu
} // gpu
} // openpgl

#endif