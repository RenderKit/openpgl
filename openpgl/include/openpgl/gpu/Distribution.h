#ifndef OPENPGL_DISTRIBUTION_GPU_H
#define OPENPGL_DISTRIBUTION_GPU_H

#include "Common.h"

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
        pgl_sincosf(2.0f * float(M_PIf)* sample.x, &sinPhi, &cosPhi);
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

                const float phi = 2.f * float(M_PIf) * _sample.y;

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

                const float phi = 2.f * float(M_PIf) * _sample.y;

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
                float norm = kappaK > 0.f ? kappaK / (2.f * M_PIf * (1.f - expf(-2.f * kappaK))) : ONE_OVER_FOUR_PI;
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
                float norm = kappaK > 0.f ? kappaK / (2.f * M_PIf * (1.f - expf(-2.f * kappaK))) : ONE_OVER_FOUR_PI;
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

} // sycl/cuda/cpu
} // gpu
} // openpgl


#endif //OPENPGL_DISTRIBUTION_GPU_H