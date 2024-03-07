#define USE_TREELETS
#define ONE_OVER_FOUR_PI 0.07957747154594767

#if defined(OPENPGL_GPU_CPU)
    #include "../../third-party/embreeSrc/common/math/vec2.h"
    #include "../../third-party/embreeSrc/common/math/vec3.h"
#endif

namespace openpgl_gpu
{
    void pgl_sincosf(float x, float *sin, float *cos)
    {
        #ifdef SYCL_LANGUAGE_VERSION
        *sin = sycl::sincos(x, cos);
        #else
        sincosf(x, sin, cos);
        #endif
    }

#if defined(OPENPGL_GPU_SYCL)
    typedef sycl::float2 Vector2;
    typedef sycl::float3 Vector3;
    typedef sycl::float2 Point2;
    typedef sycl::float3 Point3;
    #define GPUNS sycl
#else
    typedef embree::Vec2<float> Vector2;
    typedef embree::Vec3<float> Vector3;
    typedef embree::Vec2<float> Point2;
    typedef embree::Vec3<float> Point3;
    #define GPUNS embree
#endif
    template<int maxComponents> struct FlatVMM {
    };

    inline Vector3 sphericalDirection(const float &cosTheta, const float &sinTheta, const float &cosPhi, const float &sinPhi)
    {
        return Vector3(sinTheta * cosPhi,
                       sinTheta * sinPhi,
                       cosTheta);
    };

    inline Vector3 sphericalDirection(const float &theta, const float &phi)
    {
        const float cosTheta = std::cos(theta);
        const float sinTheta = std::sin(theta);
        const float cosPhi = std::cos(phi);
        const float sinPhi = std::sin(phi);

        return sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
    };

    inline Vector3 squareToUniformSphere(const pgl_vec2f sample)
    {
        float z = 1.0f - 2.0f * sample.y;
        float r = std::sqrt(std::max(0.f, (1.0f - z * z)));
        float sinPhi, cosPhi;
        pgl_sincosf(2.0f * float(M_PI)* sample.x, &sinPhi, &cosPhi);
        return Vector3(r * cosPhi, r * sinPhi, z);
    }

    template <int maxComponents>
    struct ParallaxAwareVonMisesFisherMixture : public FlatVMM<maxComponents>
    {
    public:
        float _weights[maxComponents];
        float _kappas[maxComponents];
        float _meanDirections[maxComponents][3];
        float _distances[maxComponents];
        float _pivotPosition[3];
        int _numComponents{maxComponents};
        ParallaxAwareVonMisesFisherMixture()
        {
        }

    private:
        inline uint32_t selectComponent(float &sample) const
        {
            uint32_t selectedComponent{0};            
            float searched = sample;
            float sumWeights = 0.0f;
            float cdf = 0.0f;

            while (true)
            {
                cdf = _weights[selectedComponent];
                if (sumWeights + cdf >= searched || selectedComponent+1 >= _numComponents)
                {
                    break;
                }
                else
                {
                    sumWeights += cdf;
                    selectedComponent++;
                }
            }

            sample = std::min(1.0f - std::numeric_limits<float>::epsilon(), (searched - sumWeights) / cdf);
            return selectedComponent;
        }

    public:

        pgl_vec3f sample(const pgl_vec2f sample) const
        {

            uint32_t selectedComponent{0};
            // First, identify component we want to sample

            pgl_vec2f _sample = sample;
            selectedComponent = selectComponent(_sample.y);

            Vector3 sampledDirection(0.f, 0.f, 1.f);
            // Second, sample selected component
            const float sKappa = _kappas[selectedComponent];
            const float sEMinus2Kappa = expf(-2.0f * sKappa);
            Vector3 meanDirection(_meanDirections[selectedComponent][0], _meanDirections[selectedComponent][1], _meanDirections[selectedComponent][2]);

            if (sKappa == 0.0f)
            {
                sampledDirection = squareToUniformSphere(_sample);
            }
            else
            {
                float cosTheta = 1.f + logf(1.0f + ((sEMinus2Kappa - 1.f) * _sample.x)) / sKappa;

                // safeguard for numerical imprecisions (if sample[0] is 0.999999999)
                cosTheta = std::min(1.0f, std::max(cosTheta, -1.f));

                const float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

                const float phi = 2.f * float(M_PI) * _sample.y;

                float sinPhi, cosPhi;
                pgl_sincosf(phi, &sinPhi, &cosPhi);
                sampledDirection = sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
            }

            const Vector3 dx0(0.0f, meanDirection[2], -meanDirection[1]);
            const Vector3 dx1(-meanDirection[2], 0.0f, meanDirection[0]);
            const Vector3 dx = normalize(dot(dx0, dx0) > dot(dx1, dx1) ? dx0 : dx1);
            const Vector3 dy = normalize(cross(meanDirection, dx));

            Vector3 out = dx * sampledDirection[0] + dy * sampledDirection[1] + meanDirection * sampledDirection[2];
            return {out[0], out[1], out[2]};
        }

        pgl_vec3f samplePos(const pgl_vec3f pos, const pgl_vec2f sample) const
        {

            uint32_t selectedComponent{0};
            // First, identify component we want to sample
            pgl_vec2f _sample = sample;
            selectedComponent = selectComponent(_sample.y);

            Vector3 sampledDirection(0.f, 0.f, 1.f);
            // Second, sample selected component
            const float sKappa = _kappas[selectedComponent];
            const float sEMinus2Kappa = expf(-2.0f * sKappa);
            Vector3 meanDirection(_meanDirections[selectedComponent][0], _meanDirections[selectedComponent][1], _meanDirections[selectedComponent][2]);
            // parallax shift
            Vector3 _pos = {pos.x, pos.y, pos.z};
            const Vector3 relativePivotShift = {_pivotPosition[0] - _pos[0], _pivotPosition[1] - _pos[1], _pivotPosition[2] - _pos[2]};
            meanDirection *= _distances[selectedComponent];
            meanDirection += relativePivotShift;
            float length = GPUNS::length(meanDirection);
            meanDirection /= length;

            if (sKappa == 0.0f)
            {
                sampledDirection = squareToUniformSphere(_sample);
            }
            else
            {
                float cosTheta = 1.f + logf(1.0f + ((sEMinus2Kappa - 1.f) * _sample.x)) / sKappa;

                // safeguard for numerical imprecisions (if sample[0] is 0.999999999)
                cosTheta = std::min(1.0f, std::max(cosTheta, -1.f));

                const float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

                const float phi = 2.f * float(M_PI) * _sample.y;

                float sinPhi, cosPhi;
                pgl_sincosf(phi, &sinPhi, &cosPhi);
                sampledDirection = sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
            }

            const Vector3 dx0(0.0f, meanDirection[2], -meanDirection[1]);
            const Vector3 dx1(-meanDirection[2], 0.0f, meanDirection[0]);
            const Vector3 dx = normalize(dot(dx0, dx0) > dot(dx1, dx1) ? dx0 : dx1);
            const Vector3 dy = normalize(cross(meanDirection, dx));

            Vector3 out = dx * sampledDirection[0] + dy * sampledDirection[1] + meanDirection * sampledDirection[2];
            return {out[0], out[1], out[2]};
        }

        float pdf(const pgl_vec3f dir) const
        {
            const Vector3 _dir = {dir.x, dir.y, dir.z};
            float pdf {0.f};
            for (int k =0; k < _numComponents; k++)
            {
                const Vector3 meanDirection = {_meanDirections[k][0], _meanDirections[k][1], _meanDirections[k][2]};
                const float kappaK = _kappas[k];
                float norm = kappaK > 0.f ? kappaK / (2.f * M_PI * (1.f - expf(-2.f * kappaK))) : ONE_OVER_FOUR_PI;
                const float cosThetaK =  _dir[0] * meanDirection[0] + _dir[1] * meanDirection[1] + _dir[2] * meanDirection[2];
                const float costThetaMinusOneK = std::min(cosThetaK - 1.f, 0.f);
                pdf += _weights[k] * norm * expf(kappaK * costThetaMinusOneK);
            }
            return pdf;
        }

        float pdfPos(const pgl_vec3f pos, const pgl_vec3f dir) const
        {
            const Vector3 _dir = {dir.x, dir.y, dir.z};
            const Vector3 _pos = {pos.x, pos.y, pos.z};
            const Vector3 relativePivotShift = {_pivotPosition[0] - _pos[0], _pivotPosition[1] - _pos[1], _pivotPosition[2] - _pos[2]};
            
            float pdf {0.f};
            for (int k =0; k < _numComponents; k++)
            {
                Vector3 meanDirection = {_meanDirections[k][0], _meanDirections[k][1], _meanDirections[k][2]};
                meanDirection *= _distances[k];
                meanDirection += relativePivotShift;
                float length = GPUNS::length(meanDirection);
                meanDirection /= length;
                
                const float kappaK = _kappas[k];
                float norm = kappaK > 0.f ? kappaK / (2.f * M_PI * (1.f - expf(-2.f * kappaK))) : ONE_OVER_FOUR_PI;
                const float cosThetaK =  _dir[0] * meanDirection[0] + _dir[1] * meanDirection[1] + _dir[2] * meanDirection[2];
                const float costThetaMinusOneK = std::min(cosThetaK - 1.f, 0.f);
                pdf += _weights[k] * norm * expf(kappaK * costThetaMinusOneK);
            }
            return pdf;
        }
    };

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
            OPENPGL_ASSERT(idx < (1U << 31));
            OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
            OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
            OPENPGL_ASSERT(!isLeaf());

            splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
            OPENPGL_ASSERT(idx == getLeftChildIdx());
        }

        uint32_t getLeftChildIdx() const
        {
            OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
            OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
            OPENPGL_ASSERT(!isLeaf());

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

            OPENPGL_ASSERT(_splitDim == getSplitDim());
            OPENPGL_ASSERT(_leftChildIdx == getLeftChildIdx());
        }

        /////////////////////////////
        // Leaf node functions
        /////////////////////////////

        void setLeaf()
        {
            splitDimAndNodeIdx = (3U << 30);
            OPENPGL_ASSERT(isLeaf());
        }

        bool isLeaf() const
        {
            return (splitDimAndNodeIdx >> 30) == 3;
        }

        void setChildNodeIdx(const uint32_t &idx)
        {
            OPENPGL_ASSERT(idx < (1U << 31));
            OPENPGL_ASSERT((splitDimAndNodeIdx & (3U << 30)) != (3U << 30));
            OPENPGL_ASSERT((splitDimAndNodeIdx >> 30) != 3);
            OPENPGL_ASSERT(!isLeaf());

            splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
            OPENPGL_ASSERT(idx == getLeftChildIdx());
        }

        void setDataNodeIdx(const uint32_t &idx)
        {
            OPENPGL_ASSERT(idx < (1U << 31)); // checks if the idx is in the right range
            setLeaf();
            splitDimAndNodeIdx = ((splitDimAndNodeIdx >> 30) << 30) | idx;
            OPENPGL_ASSERT(isLeaf());
            OPENPGL_ASSERT(getDataIdx() == idx);
        }

        uint32_t getDataIdx() const
        {
            OPENPGL_ASSERT(isLeaf());
            return (splitDimAndNodeIdx << 2) >> 2;
        }

        /////////////////////////////
        // Split dimension functions
        /////////////////////////////

        uint8_t getSplitDim() const
        {
            return (splitDimAndNodeIdx >> 30);
        }

        void setSplitDim(const uint8_t &splitAxis)
        {
            OPENPGL_ASSERT(splitAxis < ELeafNode);
            splitDimAndNodeIdx = (uint32_t(splitAxis) << 30);
            OPENPGL_ASSERT(splitAxis == getSplitDim());
        }

        float getSplitPivot() const
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
        FieldGPU(const KDTreeLet *treelet) : m_treeLets(treelet) {}

        uint32_t getDataIdxAtPos(const float *pos) const
        {
#ifdef USE_TREELETS
            uint32_t treeIdx = 0;
            uint32_t nodeIdx = 0;
            uint32_t depth = 0;
            KDTreeLet treeLet = m_treeLets[treeIdx];

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
                    treeLet = m_treeLets[treeIdx];
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

#ifdef USE_TREELETS
        const KDTreeLet *m_treeLets{nullptr};
#else
        KDNode *m_nodesPtr{nullptr};
#endif
    };

    struct Device
    {

#if defined(OPENPG_GPU_SYCL)
        sycl::queue q;
#endif
        Device() {}

        ~Device() {}
        


        template<class T>
        T* mallocArray(size_t numElements)
        {
#if defined(OPENPG_GPU_SYCL)
            return sycl::malloc_shared<T>(numElements, q);
#else
            return new T[numElements];
#endif
        }
        
        template<class T>
        void freeArray(T* ptr)
        {
#if defined(OPENPG_GPU_SYCL)
            sycl:free(ptr, q);
#else
            delete[] ptr;
#endif
        }

        template<class T>
        void memcpyArrayToGPU(T* devicePtr, T* hostPtr, size_t numElements)
        {
#if defined(OPENPG_GPU_SYCL)
            q.memcpy(devicePtr, hostPtr, numElements * sizeof(T));
#else
            std::memcpy(devicePtr, hostPtr, numElements * sizeof(T));      
#endif
        }

        template<class T>
        void memcpyArrayFromGPU()
        {
#if defined(OPENPG_GPU_SYCL)

#else

#endif
        }

        void wait()
        {
#if defined(OPENPG_GPU_SYCL)
            q.wait();
#else

#endif
        }
    };
}
