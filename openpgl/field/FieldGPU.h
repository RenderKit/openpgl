#define USE_TREELETS
#define ONE_OVER_FOUR_PI 0.07957747154594767

#if defined(OPENPGL_GPU_CPU)
    #include "../../third-party/embreeSrc/common/math/vec2.h"
    #include "../../third-party/embreeSrc/common/math/vec3.h"
#endif


#if defined(OPENPGL_GPU_SYCL)
    #define FLT_EPSILON 1.19209290E-07F
    #define OPENPGL_GPU_CALLABLE
#elif defined(OPENPGL_GPU_CUDA) && defined(__CUDACC__)
    #define FLT_EPSILON 1.19209290E-07F
    #define OPENPGL_GPU_CALLABLE __host__ __device__
#else
    #define OPENPGL_GPU_CALLABLE
#endif

namespace openpgl_gpu
{
    OPENPGL_GPU_CALLABLE void pgl_sincosf(float x, float *sin, float *cos)
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
    using namespace sycl;

#elif defined(OPENPGL_GPU_CUDA)

    union Vector2
    {
        float2 vec;
        float data[2];
        OPENPGL_GPU_CALLABLE Vector2(float x, float y)
        {
            vec = {x, y};
        }
        
        OPENPGL_GPU_CALLABLE float& operator[](std::size_t idx)       { return data[idx]; }
        OPENPGL_GPU_CALLABLE const float& operator[](std::size_t idx) const { return data[idx]; }

        OPENPGL_GPU_CALLABLE const Vector2& operator*=(const Vector2& b){
            this->vec.x *= b.vec.x;
            this->vec.y *= b.vec.y;
            return *this;
        }

        OPENPGL_GPU_CALLABLE const Vector2& operator*=(const float b){
            this->vec.x *= b;
            this->vec.y *= b;
            return *this;
        }

        OPENPGL_GPU_CALLABLE const Vector2& operator/=(const Vector2& b){
            this->vec.x /= b.vec.x;
            this->vec.y /= b.vec.y;
            return *this;
        }

        OPENPGL_GPU_CALLABLE const Vector2& operator/=(const float b){
            this->vec.x /= b;
            this->vec.y /= b;
            return *this;
        }
    };

    OPENPGL_GPU_CALLABLE const Vector2 operator*(Vector2 lhs, const Vector2& rhs)
    {
        return lhs *= rhs;
    }

    OPENPGL_GPU_CALLABLE const Vector2 operator*(Vector2 lhs, const float f)
    {
        return lhs *= f;
    }

    OPENPGL_GPU_CALLABLE const Vector2 operator/(Vector2 lhs, const Vector2& rhs)
    {
        return lhs /= rhs;
    }

    OPENPGL_GPU_CALLABLE const Vector2 operator/(Vector2 lhs, const float f)
    {
        return lhs /= f;
    }

    OPENPGL_GPU_CALLABLE float dot(const Vector2 a, const Vector2 b)
    {
        return a[0]*b[0] + a[1]*b[1];
    }


    OPENPGL_GPU_CALLABLE float length(const Vector2 &a)
    {
        return sqrtf(dot(a,a));
    }


    OPENPGL_GPU_CALLABLE Vector2 normalize(const Vector2 &a)
    {
        return a*rsqrt(dot(a,a));
    }

    union Vector3
    {
        float3 vec;
        float data[3];
        OPENPGL_GPU_CALLABLE Vector3(float x, float y, float z)
        {
            vec = {x, y, z};
        }
        
        OPENPGL_GPU_CALLABLE float& operator[](std::size_t idx)       { return data[idx]; }
        OPENPGL_GPU_CALLABLE const float& operator[](std::size_t idx) const { return data[idx]; }

        OPENPGL_GPU_CALLABLE const Vector3& operator*=(const Vector3& b){
            this->vec.x *= b.vec.x;
            this->vec.y *= b.vec.y;
            this->vec.z *= b.vec.z;
            return *this;
        }
        OPENPGL_GPU_CALLABLE const Vector3& operator*=(const float b){
            this->vec.x *= b;
            this->vec.y *= b;
            this->vec.z *= b;
            return *this;
        }

        OPENPGL_GPU_CALLABLE const Vector3& operator/=(const Vector3& b){
            this->vec.x /= b.vec.x;
            this->vec.y /= b.vec.y;
            this->vec.z /= b.vec.z;
            return *this;
        }
        OPENPGL_GPU_CALLABLE const Vector3& operator/=(const float b){
            this->vec.x /= b;
            this->vec.y /= b;
            this->vec.z /= b;
            return *this;
        }

        OPENPGL_GPU_CALLABLE const Vector3& operator+=(const Vector3& b){
            this->vec.x += b.vec.x;
            this->vec.y += b.vec.y;
            this->vec.z += b.vec.z;
            return *this;
        }
        OPENPGL_GPU_CALLABLE const Vector3& operator+=(const float b){
            this->vec.x += b;
            this->vec.y += b;
            this->vec.z += b;
            return *this;
        }
    };

    OPENPGL_GPU_CALLABLE const Vector3 operator*(Vector3 lhs, const Vector3& rhs)
    {
        return lhs *= rhs;
    }

    OPENPGL_GPU_CALLABLE const Vector3 operator*(Vector3 lhs, const float f)
    {
        return lhs *= f;
    }

    OPENPGL_GPU_CALLABLE const Vector3 operator/(Vector3 lhs, const Vector3& rhs)
    {
        return lhs /= rhs;
    }

    OPENPGL_GPU_CALLABLE const Vector3 operator/(Vector3 lhs, const float f)
    {
        return lhs /= f;
    }

    OPENPGL_GPU_CALLABLE const Vector3 operator+(Vector3 lhs, const Vector3& rhs)
    {
        return lhs += rhs;
    }

    OPENPGL_GPU_CALLABLE const Vector3 operator+(Vector3 lhs, const float f)
    {
        return lhs += f;
    }

    OPENPGL_GPU_CALLABLE float dot(const Vector3 &a, const Vector3 &b)
    {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    OPENPGL_GPU_CALLABLE float length(const Vector3 &a)
    {
        return sqrtf(dot(a,a));
    }

    OPENPGL_GPU_CALLABLE Vector3 normalize(const Vector3 &a)
    {
        return a*rsqrt(dot(a,a));
    }

    OPENPGL_GPU_CALLABLE Vector3 cross(const Vector3 &a, const Vector3 &b)
    {
        return Vector3( a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] );
    }

    typedef Vector2 Point2;
    typedef Vector3 Point3;

#else
    typedef embree::Vec2<float> Vector2;
    typedef embree::Vec3<float> Vector3;
    typedef embree::Vec2<float> Point2;
    typedef embree::Vec3<float> Point3;
    #define GPUNS embree
#endif
    template<int maxComponents> struct FlatVMM {
    };

    OPENPGL_GPU_CALLABLE inline Vector3 sphericalDirection(const float &cosTheta, const float &sinTheta, const float &cosPhi, const float &sinPhi)
    {
        return Vector3(sinTheta * cosPhi,
                       sinTheta * sinPhi,
                       cosTheta);
    };

    OPENPGL_GPU_CALLABLE inline Vector3 sphericalDirection(const float &theta, const float &phi)
    {
        const float cosTheta = std::cos(theta);
        const float sinTheta = std::sin(theta);
        const float cosPhi = std::cos(phi);
        const float sinPhi = std::sin(phi);

        return sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
    };

    OPENPGL_GPU_CALLABLE inline Vector3 squareToUniformSphere(const pgl_vec2f sample)
    {
        float z = 1.0f - 2.0f * sample.y;
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
        float r = std::sqrt(std::max(0.f, (1.0f - z * z)));
#else
        float r = std::sqrt(std::fmaxf(0.f, (1.0f - z * z)));
#endif
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
        OPENPGL_GPU_CALLABLE inline uint32_t selectComponent(float &sample) const
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
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
            sample = std::min(1.0f - FLT_EPSILON, (searched - sumWeights) / cdf);
#else
            sample = std::fminf(1.0f - FLT_EPSILON, (searched - sumWeights) / cdf);
#endif
            return selectedComponent;
        }

    public:

        OPENPGL_GPU_CALLABLE pgl_vec3f sample(const pgl_vec2f sample) const
        {

            uint32_t selectedComponent{0};
            // First, identify component we want to sample

            pgl_vec2f _sample = sample;
            selectedComponent = selectComponent(_sample.y);

            Vector3 sampledDirection = Vector3(0.f, 0.f, 1.f);
            // Second, sample selected component
            const float sKappa = _kappas[selectedComponent];
            const float sEMinus2Kappa = expf(-2.0f * sKappa);
            Vector3 meanDirection = Vector3(_meanDirections[selectedComponent][0], _meanDirections[selectedComponent][1], _meanDirections[selectedComponent][2]);

            if (sKappa == 0.0f)
            {
                sampledDirection = squareToUniformSphere(_sample);
            }
            else
            {
                float cosTheta = 1.f + logf(1.0f + ((sEMinus2Kappa - 1.f) * _sample.x)) / sKappa;
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
                // safeguard for numerical imprecisions (if sample[0] is 0.999999999)
                cosTheta = std::min(1.0f, std::max(cosTheta, -1.f));
#else
                cosTheta = std::fminf(1.0f, std::fmaxf(cosTheta, -1.f));
#endif
                const float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

                const float phi = 2.f * float(M_PI) * _sample.y;

                float sinPhi, cosPhi;
                pgl_sincosf(phi, &sinPhi, &cosPhi);
                sampledDirection = sphericalDirection(cosTheta, sinTheta, cosPhi, sinPhi);
            }

            const Vector3 dx0 = Vector3(0.0f, meanDirection[2], -meanDirection[1]);
            const Vector3 dx1 = Vector3(-meanDirection[2], 0.0f, meanDirection[0]);
            const Vector3 dx = normalize(dot(dx0, dx0) > dot(dx1, dx1) ? dx0 : dx1);
            const Vector3 dy = normalize(cross(meanDirection, dx));

            Vector3 out = dx * sampledDirection[0] + dy * sampledDirection[1] + meanDirection * sampledDirection[2];
            return {out[0], out[1], out[2]};
        }

        OPENPGL_GPU_CALLABLE pgl_vec3f samplePos(const pgl_vec3f pos, const pgl_vec2f sample) const
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
            float flength = length(meanDirection);
            meanDirection /= flength;

            if (sKappa == 0.0f)
            {
                sampledDirection = squareToUniformSphere(_sample);
            }
            else
            {
                float cosTheta = 1.f + logf(1.0f + ((sEMinus2Kappa - 1.f) * _sample.x)) / sKappa;

// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
                // safeguard for numerical imprecisions (if sample[0] is 0.999999999)
                cosTheta = std::min(1.0f, std::max(cosTheta, -1.f));
#else
                cosTheta = std::fminf(1.0f, std::fmaxf(cosTheta, -1.f));
#endif
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

        OPENPGL_GPU_CALLABLE float pdf(const pgl_vec3f dir) const
        {
            const Vector3 _dir = {dir.x, dir.y, dir.z};
            float pdf {0.f};
            for (int k =0; k < _numComponents; k++)
            {
                const Vector3 meanDirection = {_meanDirections[k][0], _meanDirections[k][1], _meanDirections[k][2]};
                const float kappaK = _kappas[k];
                float norm = kappaK > 0.f ? kappaK / (2.f * M_PI * (1.f - expf(-2.f * kappaK))) : ONE_OVER_FOUR_PI;
                const float cosThetaK =  _dir[0] * meanDirection[0] + _dir[1] * meanDirection[1] + _dir[2] * meanDirection[2];
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
                const float costThetaMinusOneK = std::min(cosThetaK - 1.f, 0.f);
#else
                const float costThetaMinusOneK = std::fminf(cosThetaK - 1.f, 0.f);
#endif
                pdf += _weights[k] * norm * expf(kappaK * costThetaMinusOneK);
            }
            return pdf;
        }

        OPENPGL_GPU_CALLABLE float pdfPos(const pgl_vec3f pos, const pgl_vec3f dir) const
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
                float flength = length(meanDirection);
                meanDirection /= flength;
                
                const float kappaK = _kappas[k];
                float norm = kappaK > 0.f ? kappaK / (2.f * M_PI * (1.f - expf(-2.f * kappaK))) : ONE_OVER_FOUR_PI;
                const float cosThetaK =  _dir[0] * meanDirection[0] + _dir[1] * meanDirection[1] + _dir[2] * meanDirection[2];
// TODO: Fix
#if !defined(OPENPGL_GPU_CUDA)
                const float costThetaMinusOneK = std::min(cosThetaK - 1.f, 0.f);
#else
                const float costThetaMinusOneK = std::fminf(cosThetaK - 1.f, 0.f);
#endif
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

        OPENPGL_GPU_CALLABLE uint32_t getLeftChildIdx() const
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

        OPENPGL_GPU_CALLABLE bool isLeaf() const
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

        OPENPGL_GPU_CALLABLE uint32_t getDataIdx() const
        {
            OPENPGL_ASSERT(isLeaf());
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
            OPENPGL_ASSERT(splitAxis < ELeafNode);
            splitDimAndNodeIdx = (uint32_t(splitAxis) << 30);
            OPENPGL_ASSERT(splitAxis == getSplitDim());
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
        using Distribution = openpgl_gpu::ParallaxAwareVonMisesFisherMixture<32>; 
        
        OPENPGL_GPU_CALLABLE FieldGPU() = default;

        OPENPGL_GPU_CALLABLE FieldGPU(const KDTreeLet *treelet, const Distribution *distributions) : m_treeLets(treelet), m_distributions(distributions) {}

        OPENPGL_GPU_CALLABLE uint32_t getDataIdxAtPos(const float *pos) const
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
/* Need to disable these direct initializations to pass through the std::is_trivially_default_constructible_v test. If not the following error is throughen:
*  error: static assertion failed due to requirement 'std::is_trivially_default_constructible_v<const openpgl_gpu::FieldGPU>': Type T must be trivially default constructable (until C++20 consteval is supported and enabled.)
*/
        const KDTreeLet *m_treeLets;//{nullptr};
#else
        KDNode *m_nodesPtr;//{nullptr};
#endif
        const Distribution *m_distributions;// {nullptr};

    };

    struct SurfaceSamplingDistribution
    {
        const FieldGPU* m_field {nullptr};
        pgl_point3f m_pos {0.f, 0.f, 0.f};
        int m_idx {-1};

        OPENPGL_GPU_CALLABLE bool Init(const FieldGPU* field, const pgl_point3f& pos, float& sample1D)
        {
            m_pos = pos;
            m_field = field;
            float _pos[3] = {pos.x, pos.y, pos.z};
            m_idx = m_field->getDataIdxAtPos(_pos);

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
            return m_field->m_distributions[m_idx].samplePos(m_pos, sample2D);
        }

        OPENPGL_GPU_CALLABLE float PDF(const pgl_vec3f& direction) const
        {
            return m_field->m_distributions[m_idx].pdfPos(m_pos, direction);
        }

        OPENPGL_GPU_CALLABLE float SamplePDF(const pgl_point2f& sample2D, pgl_vec3f& direction) const
        {
            direction = m_field->m_distributions[m_idx].samplePos(m_pos, sample2D);
            return m_field->m_distributions[m_idx].pdfPos(m_pos, direction);
        }

        OPENPGL_GPU_CALLABLE float IncomingRadiancePDF(const pgl_vec3f& direction) const
        {
            return m_field->m_distributions[m_idx].pdfPos(m_pos, direction);
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


    struct Device
    {

#if defined(OPENPGL_GPU_SYCL)
        sycl::queue q;
#endif
        Device() {}

        ~Device() {}
        
        template<class T>
        T* mallocArray(size_t numElements)
        {
#if defined(OPENPGL_GPU_SYCL)
            return sycl::malloc_shared<T>(numElements, q);
#elif defined(OPENPGL_GPU_CUDA)
            void *devPtr;
            cudaMalloc(&devPtr, numElements * sizeof(T));
            return (T*)devPtr;
#else
            return new T[numElements];
#endif
        }
        
        template<class T>
        void freeArray(T* ptr)
        {
#if defined(OPENPGL_GPU_SYCL)
            sycl:free(ptr, q);
#elif defined(OPENPGL_GPU_CUDA)
            cudaFree(ptr);
#else
            delete[] ptr;
#endif
        }

        template<class T>
        void memcpyArrayToGPU(T* devicePtr, T* hostPtr, size_t numElements)
        {
#if defined(OPENPGL_GPU_SYCL)
            q.memcpy(devicePtr, hostPtr, numElements * sizeof(T));
#elif defined(OPENPGL_GPU_CUDA)
            cudaMemcpy(devicePtr, hostPtr, numElements * sizeof(T), cudaMemcpyHostToDevice);
            //cudaMemcpyAsync(devicePtr, hostPtr, numElements * sizeof(T), cudaMemcpyHostToDevice);
#else
            std::memcpy(devicePtr, hostPtr, numElements * sizeof(T));      
#endif
        }

        template<class T>
        void memcpyArrayFromGPU(T* devicePtr, T* hostPtr, size_t numElements)
        {
#if defined(OPENPGL_GPU_SYCL)
            q.memcpy(hostPtr, devicePtr, numElements * sizeof(T));
#elif defined(OPENPGL_GPU_CUDA)
            cudaMemcpy(hostPtr, devicePtr, numElements * sizeof(T), cudaMemcpyDeviceToHost);
            //cudaMemcpyAsync(hostPtr, devicePtr, numElements * sizeof(T), cudaMemcpyDeviceToHost);
#else
            std::memcpy(hostPtr, devicePtr, numElements * sizeof(T));
#endif
        }

        void wait()
        {
#if defined(OPENPGL_GPU_SYCL)
            q.wait();
#elif defined(OPENPGL_GPU_CUDA)
            cudaDeviceSynchronize();
#else

#endif
        }
    };
}
